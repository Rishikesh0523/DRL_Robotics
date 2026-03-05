#!/usr/bin/env python3
"""
Gymnasium environment for mobile-robot navigation (mecanum holonomic)
NON-VISUAL (LiDAR-BASED) VERSION - 16D Observation Space.
Includes the final, fine-tuned reward structure to prevent reversing/getting stuck.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer
# from sensor_msgs.msg import Image # REMOVED: No longer need Image
# from cv_bridge import CvBridge # REMOVED: No longer need CvBridge
# import torch # REMOVED: No longer need PyTorch/CNN
# import torch.nn as nn # REMOVED
import math
import threading
import time

# ---------- REMOVED: MiniCNN Class is no longer needed ----------

# ========================= ENV CLASS =========================
class RobotNavigationEnv(Node, gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, "drl_nav_env_node")

        # 3-D holonomic action (vx, vy, ω)
        self.action_space = spaces.Box(
            low=np.array([-0.75, -0.75, -1.5], dtype=np.float32),
            high=np.array([0.75, 0.75, 1.5], dtype=np.float32),
            dtype=np.float32,
        )
        
        # ... (Observation space definition remains the same) ...

        # 16-D observation definition (10 + 2 + 4)
        self.lidar_sectors = 10
        # 10-D LiDAR ranges (0.0m to 10.0m)
        lidar_low = np.zeros(self.lidar_sectors, dtype=np.float32)
        lidar_high = np.full(self.lidar_sectors, 10.0, dtype=np.float32)
        # 2-D Relative Goal Info (distance, bearing_to_goal)
        goal_low = np.array([0.0, -math.pi], dtype=np.float32)
        goal_high = np.array([10.0, math.pi], dtype=np.float32)
        # 4-D Velocity components: [vx, vy, w, v_mag] 
        vel_low = np.array([-2.0, -2.0, -math.pi, 0.0], dtype=np.float32) 
        vel_high = np.array([2.0, 2.0, math.pi, 2.0], dtype=np.float32) 
        self.observation_space = spaces.Box(
            low=np.concatenate([lidar_low, goal_low, vel_low]),
            high=np.concatenate([lidar_high, goal_high, vel_high]),
            dtype=np.float32,
        )

        # ---------- ROS ----------
        qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.lidar_sub = Subscriber(self, LaserScan, "/lidar")
        self.odom_sub = Subscriber(self, Odometry, "/odom")
        
        self.ts = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.odom_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self._sensor_cb)
        
        # ---------- fixed positions (Unchanged) ----------
        self.start_position = np.array([-2.0, -2.0], dtype=np.float32)   
        self.goal_position = np.array([0.0, 0.0], dtype=np.float32)     
        self.current_position = np.zeros(2, dtype=np.float32)
        self.current_yaw = 0.0
        self.initial_distance = float("inf")
        self.last_distance = float("inf")
        self.episode_step = 0
        self.max_steps = 300
        self.total_reward = 0.0
        self.position_history = []
        
        # Observation storage
        self.latest_obs = None
        self._obs_ready = False
        
        # ---------- threading ----------
        self._lock = threading.Lock()
        self.get_logger().info(" Environment node started - waiting for sensors...")
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(self), daemon=True)
        self._spin_thread.start()
        self._wait_until_ready()

    def _wait_until_ready(self, timeout=10):
        """Wait for sensor data to be available"""
        t0 = time.time()
        while self.latest_obs is None and (time.time() - t0 < timeout):
            time.sleep(0.1)
        if self.latest_obs is None:
            self.get_logger().warn(" Sensor timeout - proceeding anyway")
        else:
            self.get_logger().info(" Sensors ready!")

    # MODIFIED: Removed 'img: Image' argument
    def _sensor_cb(self, scan: LaserScan, odom: Odometry):
        """Process synchronized sensor data (LiDAR and Odometry only)"""
        try:
            # 1. lidar sectors
            ranges = np.array(scan.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=10.0)
            sector_size = len(ranges) // self.lidar_sectors
            sectors = []
            for i in range(self.lidar_sectors):
                sector = ranges[i * sector_size: (i + 1) * sector_size]
                sectors.append(sector.min() if sector.size else 10.0)
            lidar_vec = np.array(sectors, dtype=np.float32)

            # 2. odom (vx, vy, w, yaw)
            vx = odom.twist.twist.linear.x
            vy = odom.twist.twist.linear.y
            w = odom.twist.twist.angular.z
            q = odom.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            v_mag = math.hypot(vx, vy) # Linear velocity magnitude

            with self._lock:
                self.current_position[0] = odom.pose.pose.position.x
                self.current_position[1] = odom.pose.pose.position.y
                self.current_yaw = yaw
                
                self.position_history.append((self.current_position[0], self.current_position[1]))
                if len(self.position_history) > 50:
                    self.position_history.pop(0)
            
            # 3. Goal Vector (RELATIVE) - 2D
            dist_to_goal = self.get_distance_to_goal()
            bearing_to_goal = self._get_bearing_to_goal()
            goal_vec = np.array([dist_to_goal, bearing_to_goal], dtype=np.float32)
            
            # 4. Velocity Vector - 4D: [vx, vy, w, v_mag]
            vel_vec = np.array([vx, vy, w, v_mag], dtype=np.float32) 

            # 5. camera latent (REMOVED)
            
            # Concatenated vector is now 10 + 2 + 4 = 16D.
            full_obs = np.concatenate([lidar_vec, goal_vec, vel_vec]).astype(np.float32) # MODIFIED
            
            with self._lock:
                self.latest_obs = full_obs
            
        except Exception as e:
            self.get_logger().error(f"Sensor callback error: {e}")

    # --------------- helpers (Unchanged) ---------------
    def get_distance_to_goal(self):
        return np.linalg.norm(self.goal_position - self.current_position)

    def _get_bearing_to_goal(self):
        goal_angle_global = math.atan2(
            self.goal_position[1] - self.current_position[1],
            self.goal_position[0] - self.current_position[0]
        )
        bearing = self._normalize_angle(goal_angle_global - self.current_yaw)
        return bearing

    def is_stuck(self):
        if len(self.position_history) < 20:
            return False
        recent_positions = self.position_history[-20:]
        total_movement = 0.0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            total_movement += math.hypot(dx, dy)
        avg_movement = total_movement / (len(recent_positions) - 1)
        return avg_movement < 0.02 and self.episode_step > 30

    def avoid_wall(self, action):
        if self.latest_obs is None:
            return action
        
        # LiDAR data is now the first 10 elements of the 16D vector
        lidar_data = self.latest_obs[:self.lidar_sectors]
        front_idx = self.lidar_sectors // 2
        front_dist = lidar_data[front_idx] 
        
        #if front_dist < 0.35:
        #    left_avg = np.mean(lidar_data[:front_idx]) if front_idx > 0 else 10.0
        #    right_avg = np.mean(lidar_data[front_idx+1:self.lidar_sectors]) if front_idx < self.lidar_sectors-1 else 10.0
        #    
        #    if left_avg > right_avg:
        #        return np.array([0.0, 0.3, 0.0], dtype=np.float32)  
        #    else:
        #        return np.array([0.0, -0.3, 0.0], dtype=np.float32)  

        if front_dist < 0.5:
            left_avg = np.mean(lidar_data[:front_idx])
            right_avg = np.mean(lidar_data[front_idx+1:])
            strafe_adjust = 0.3 * (0.35 - front_dist)/0.35  # scale with proximity
            rotate_adjust = 0.3 * (0.35 - front_dist)/0.35
            action[0] *= 0.8  # slightly reduce forward speed
            action[1] += strafe_adjust if left_avg > right_avg else -strafe_adjust 
            action[2] += rotate_adjust if left_avg > right_avg else -rotate_adjust 
        
                
        return action

    def send_action(self, action):
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], -1, 1))
        twist.linear.y = float(np.clip(action[1], -1, 1))
        twist.angular.z = float(np.clip(action[2], -2, 2))
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        time.sleep(0.1)

    # --------------- gym interface (Modified Reset and Step) ---------------
    def reset(self, seed=None, options=None):
        """Reset environment WITHOUT using Gazebo teleport service (manual safe reset)."""
        super().reset(seed=seed)
    
        self.episode_step = 0
        self.total_reward = 0.0
        self.position_history.clear()
        self.stop_robot()
        self.get_logger().info("Resetting environment (manual mode)...")
    
        # --- Safe random start sampling inside the 4x4 arena (avoid walls/pillars) ---
        # We sample from a safe radius away from walls: x,y in [-2.5, 2.5]
        # You can make this deterministic by setting fixed_start = True
        fixed_start = False
    
        if fixed_start:
            start = np.array([0.0, -3.0], dtype=np.float32)  # example safe fixed start
        else:
            # try a few samples until we get a safe one (not inside known pillar regions)
            def is_safe(pt):
                x, y = pt[0], pt[1]
                # simple safety: keep at least 0.7 m away from arena walls at +/-4 and pillars
                if abs(x) > 3.2 or abs(y) > 3.2:
                    return False
                # Avoid pillar areas (from your SDF: pillar_1 at -1.0,1.5 ; pillar_2 at 1.3,1.0 ; pillar_4 at 0.8,-1.6)
                pillar_centers = [(-1.0, 1.5), (1.3, 1.0), (0.8, -1.6)]
                for cx, cy in pillar_centers:
                    if math.hypot(x - cx, y - cy) < 0.6:  # 0.6 safety radius
                        return False
                return True
    
            start = None
            for _ in range(20):
                cand = np.random.uniform(low=-2.5, high=2.5, size=(2,)).astype(np.float32)
                if is_safe(cand):
                    start = cand
                    break
            if start is None:
                # fallback safe center
                start = np.array([0.0, 0.0], dtype=np.float32)
    
        # --- Apply manual pose to internal state (no Gazebo call) ---
        with self._lock:
            self.current_position[:] = start
            self.current_yaw = 0.0  # or sample small random yaw: np.random.uniform(-0.2,0.2)
            # Optionally set a farther goal (adjust to your needs)
            # Example: 3.5 m diagonal target in arena bounds
            self.goal_position[:] = np.array([2.5, 2.5], dtype=np.float32)
            self.last_distance = np.linalg.norm(self.goal_position - self.current_position)
            self.initial_distance = self.last_distance
    
        # Ensure latest_obs exists and is valid 16D; if not, provide a safe dummy obs
        if self.latest_obs is None or (hasattr(self.latest_obs, "shape") and self.latest_obs.shape[0] != 16):
            # Create a safe dummy LiDAR: far distances, goal vector, zero velocities
            lidar_dummy = np.full(self.lidar_sectors, 10.0, dtype=np.float32)
            goal_dummy = np.array([self.last_distance, 0.0], dtype=np.float32)
            vel_dummy = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            dummy_obs = np.concatenate([lidar_dummy, goal_dummy, vel_dummy]).astype(np.float32)
            with self._lock:
                self.latest_obs = dummy_obs.copy()
            return dummy_obs, {}
        else:
            return self.latest_obs.copy(), {}
    

    def step(self, action):
        """Execute one time step with fine-tuned reward structure"""
        self.episode_step += 1
        
        # Apply wall avoidance and send action
        action = self.avoid_wall(action)
        self.send_action(action)
        time.sleep(0.12)
        
        # Get observation (must be 16D)
        with self._lock:
            if self.latest_obs is not None and self.latest_obs.shape[0] == 16:
                obs = self.latest_obs.copy()
                current_pos = self.current_position.copy()
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                current_pos = self.current_position.copy()

        # Calculate rewards (FINE-TUNED SCALING)
        dist_to_goal = self.get_distance_to_goal()
        min_lidar = obs[:self.lidar_sectors].min() if obs is not None else 10.0
        reward = 0.0
        
        # 1. Progress Reward (Potential-based: Prevents overshooting, incentivizes deceleration)
        P_old = -math.pow(self.last_distance, 2)
        P_new = -math.pow(dist_to_goal, 2)
        reward_progress = (P_new - P_old) * 2.0 # Increased scaling (2.0)
        reward += reward_progress
        
        # 2. Goal Reward
        if dist_to_goal < 0.1:  
            reward += 10.0
            done = True
            self.get_logger().info(" GOAL REACHED! +10 reward")
        else:
            done = False
            
        # 3. Collision Penalty
        if min_lidar < 0.15: 
            reward -= 10.0  # Severe hard penalty (equal to goal reward)
            done = True
            self.get_logger().warn(" COLLISION! -10.0 reward")
        elif min_lidar < 0.25:  
            # CRITICAL: Very gentle soft penalty (0.05 multiplier) to avoid reversing
            reward -= 0.05 * (0.25 - min_lidar) *0.2

        # 4. Time/Step Penalty (Encourages efficiency)
        reward -= 0.0005 * (1.0 + dist_to_goal * 0.1) 

        # 5. Stuck Penalty
        if self.is_stuck():
            reward -= 5.0 
            done = True
            self.get_logger().info(" STUCK! -5.0 reward")
            
        # 6. Directional Movement Penalty/Bonus (Strongly discourages reversing)
        # Penalizes linear_vx * cos(bearing) if the result is negative (moving away from goal)
        bearing_to_goal = self._get_bearing_to_goal() 
        cos_bearing = math.cos(bearing_to_goal)
        reward_alignment = (action[0]*math.cos(bearing_to_goal) + action[1]*math.sin(bearing_to_goal)) * 0.5        
        reward += reward_alignment
        
        # 7. Absolute Linear Movement Bonus (Ensures time penalty doesn't lead to stagnation)
        linear_mag_action = math.hypot(action[0], action[1])
        reward_move_bonus = linear_mag_action * 0.005 
        reward += reward_move_bonus

        # Update and termination
        self.last_distance = dist_to_goal
        self.total_reward += reward
        
        if self.episode_step >= self.max_steps:
            done = True
            if dist_to_goal < 1.0: 
                reward += (1.0 - dist_to_goal) * 0.5 
            self.get_logger().info(" Max steps reached")


        # Logging (Unchanged)
        if done or self.episode_step % 25 == 0:
            status = "DONE" if done else f"Step {self.episode_step}"
            self.get_logger().info(
                f"{status}: Pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                f"Dist={dist_to_goal:.2f} | Reward={reward:.2f} | "
                f"Total={self.total_reward:.2f}"
            )
            
        if done:
            success = "SUCCESS" if dist_to_goal < 0.3 else "FAILED"
            self.get_logger().info(
                f" Episode finished: {success} | "
                f"Final distance: {dist_to_goal:.2f}m | "
                f"Total reward: {self.total_reward:.2f}"
            )

        return obs, reward, done, False, {}

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def render(self, mode="human"):
        pass

    def close(self):
        """Cleanup environment"""
        self.get_logger().info(" Closing environment - stopping robot")
        self.stop_robot()
        if rclpy.ok():
            self.destroy_node()