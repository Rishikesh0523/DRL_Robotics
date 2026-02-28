"""
ROS2 Environment for Robot Navigation with MR LOGI Camera
"""

import os
import time
import math
import threading
from typing import Optional
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge

import gymnasium as gym
from gymnasium import spaces

from config import TrainingConfig, RobotConfig


class RobotNavigationEnv(Node, gym.Env):
    """ROS2 Environment for Robot Navigation using SAC with MR LOGI Camera"""

    def __init__(self):
        # Initialize both Node and gym.Env
        Node.__init__(self, 'drl_nav_env_node')
        gym.Env.__init__(self)

        # Load configuration
        self.config = RobotConfig()
        self.train_config = TrainingConfig()

        # UPDATE: Add camera features to state dimension
        self.CAMERA_FEATURES = 100  # Simple feature vector size
        self.UPDATED_STATE_DIM = self.config.STATE_DIM + self.CAMERA_FEATURES

        # UPDATE: Expand observation space to include camera
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.UPDATED_STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.ACTION_DIM,), dtype=np.float32
        )

        # ROS2 Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # UPDATE: Add camera subscriber
        self.lidar_sub = self.create_subscription(LaserScan, '/lidar', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/mr_logi/camera/processed', self.camera_callback, 10)

        # ADD: CV Bridge for camera processing
        self.bridge = CvBridge()

        # State variables
        self.latest_lidar = None
        self.odom_pose = None
        self.odom_yaw = 0.0
        self.odom_twist = None
        self.goal_pose = None
        self.last_action = [0.0, 0.0]

        # ADD: Camera state variable
        self.camera_features = np.zeros(self.CAMERA_FEATURES, dtype=np.float32)
        self.latest_camera_image = None

        # Episode variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.initial_distance = 0.0
        self.current_distance = 0.0
        self.previous_distance = 0.0
        self.start_position = None
        self.goal_position = None

        # Thread safety
        self._lock = threading.Lock()

        self.get_logger().info(" Robot Navigation Environment with MR LOGI Camera initialized")
        self.get_logger().info(" Listening to /lidar, /odom, and /mr_logi/camera/processed topics...")

    # ---------- UPDATED Callbacks ----------
    def lidar_callback(self, msg):
        with self._lock:
            self.latest_lidar = msg

    def odom_callback(self, msg):
        with self._lock:
            self.odom_pose = msg.pose.pose
            self.odom_twist = msg.twist.twist

            # Extract yaw from orientation
            orientation = msg.pose.pose.orientation
            _, _, self.odom_yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])

    # ADD: Camera callback
    def camera_callback(self, msg):
        """Process MR LOGI camera data for DRL"""
        with self._lock:
            try:
                # Convert ROS image to OpenCV
                self.latest_camera_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
                
                # Extract simple features for DRL
                self.camera_features = self.extract_camera_features(self.latest_camera_image)
                
            except Exception as e:
                self.get_logger().error(f"MR LOGI Camera processing error: {e}")

    def extract_camera_features(self, image):
        """Extract simple features from camera image for DRL"""
        # Simple downsampling: take every 4th pixel in both dimensions
        # From 160x120 -> 40x30 = 1200 features, then take first 100
        features = image[::4, ::4].flatten()
        
        # Take first 100 features and normalize to [0, 1]
        features = features[:self.CAMERA_FEATURES].astype(np.float32) / 255.0
        
        return features

    # ---------- UPDATED Core gym.Env methods ----------
    def get_observation(self):
        """Get current observation state with LiDAR, Odometry, and Camera data"""
        with self._lock:
            # UPDATE: Check for camera data too
            if self.latest_lidar is None or self.odom_pose is None or self.latest_camera_image is None:
                if self.episode_step == 0:
                    self.get_logger().info(" Waiting for first sensor data (LiDAR, Odom, Camera)...")
                return np.zeros(self.UPDATED_STATE_DIM, dtype=np.float32), False

            # If we have sensor data but no goal yet, return default observation
            if self.goal_pose is None:
                return np.zeros(self.UPDATED_STATE_DIM, dtype=np.float32), False

            # Get LiDAR data (360 degree scan)
            laser_ranges = list(self.latest_lidar.ranges)

            # DEBUG: Check LiDAR data
            valid_ranges = [r for r in laser_ranges if not np.isinf(r) and r > 0.05]
            if len(valid_ranges) == 0:
                self.get_logger().warn(f" No valid LiDAR ranges! Total: {len(laser_ranges)}, Valid: {len(valid_ranges)}")
                return np.zeros(self.UPDATED_STATE_DIM, dtype=np.float32), False

            min_lidar = min(valid_ranges) if valid_ranges else self.config.LIDAR_MAX_RANGE

            # Get robot and goal positions
            robot_x = self.odom_pose.position.x
            robot_y = self.odom_pose.position.y
            goal_x = self.goal_pose.position.x
            goal_y = self.goal_pose.position.y

            # Calculate goal information
            dx = goal_x - robot_x
            dy = goal_y - robot_y
            distance = math.sqrt(dx**2 + dy**2)

            # Calculate heading angle to goal
            goal_angle = math.atan2(dy, dx)
            heading_error = self.normalize_angle(goal_angle - self.odom_yaw)

            cos_heading = math.cos(heading_error)
            sin_heading = math.sin(heading_error)

            # Get current velocity
            velocity = self.get_robot_velocity()

            # Check collision from LiDAR
            collision = min_lidar < self.train_config.COLLISION_THRESHOLD

            # Check goal reached - ONLY if we're actually close to goal
            goal_reached = distance < self.train_config.GOAL_TOLERANCE

            # UPDATE: Prepare base state (LiDAR + Odom)
            base_state = self.prepare_base_state(
                latest_scan=laser_ranges,
                distance=distance,
                cos=cos_heading,
                sin=sin_heading,
                velocity=velocity,
                collision=collision,
                goal=goal_reached,
                action=self.last_action
            )

            # COMBINE: Add camera features to create full state
            full_state = np.concatenate([base_state, self.camera_features])

            # DEBUG: Log sensor status
            self.get_logger().info(
                f" Sensors - LiDAR: {min_lidar:.2f}m | "
                f"Cam: {np.mean(self.camera_features):.3f} | "
                f"Dist: {distance:.2f}m"
            )

            return full_state, (collision or goal_reached)

    def prepare_base_state(self, latest_scan, distance, cos, sin, velocity, collision, goal, action):
        """Prepare base state vector from LiDAR and odometry data (your existing logic)"""
        latest_scan = np.array(latest_scan)

        # Handle infinite values in LiDAR - use actual range_max from your LiDAR
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = self.config.LIDAR_MAX_RANGE

        # Filter out zero and very small values that might be errors
        valid_scan = latest_scan[latest_scan > 0.05]
        if len(valid_scan) == 0:
            valid_scan = np.array([self.config.LIDAR_MAX_RANGE])  # Default safe value

        # Bin LiDAR readings
        max_bins = self.config.LIDAR_BINS
        bin_size = max(1, len(valid_scan) // max_bins)
        min_values = []

        for i in range(0, len(valid_scan), bin_size):
            if i + bin_size <= len(valid_scan):
                bin_data = valid_scan[i:i + bin_size]
                min_val = np.min(bin_data) / self.config.LIDAR_MAX_RANGE
                min_values.append(min_val)

        # Pad if necessary
        while len(min_values) < max_bins:
            min_values.append(1.0)  # Far away (safe)

        # Normalize features
        norm_distance = min(distance / 15.0, 1.0)
        norm_cos = (cos + 1) / 2
        norm_sin = (sin + 1) / 2
        norm_lin_vel = (action[0] + 1) / 2
        norm_ang_vel = (action[1] + 1) / 2

        # Combine base features
        base_state = min_values + [norm_distance, norm_cos, norm_sin, norm_lin_vel, norm_ang_vel]

        return np.array(base_state, dtype=np.float32)

    # ---------- UPDATED Reward Function ----------
    def compute_reward(self, state, done):
        """Compute reward with camera-based bonuses"""
        reward = 0.0

        # Extract features from state
        distance = state[20] * 15.0  # Denormalize
        min_lidar = min(state[:20]) * self.config.LIDAR_MAX_RANGE
        
        # ADD: Extract camera features (last 100 elements)
        camera_features = state[-self.CAMERA_FEATURES:]

        # Progress reward (getting closer to goal)
        progress = (self.previous_distance - distance) * 3.0
        reward += progress

        # Goal reward (with safety bonus)
        if distance < self.train_config.GOAL_TOLERANCE:
            if min_lidar > self.train_config.COLLISION_THRESHOLD + 0.1:
                reward += 15.0
                self.get_logger().info(" GOAL REACHED SAFELY! +15 reward")
            else:
                reward += 5.0
                self.get_logger().info(" Goal reached (risky) +5 reward")

        # Collision penalty
        if min_lidar < self.train_config.COLLISION_THRESHOLD:
            reward -= 3.0
            self.get_logger().warn(f" COLLISION! Min LiDAR: {min_lidar:.2f}m -3 reward")

        # ADD: Camera-based rewards
        camera_reward = self.compute_camera_reward(camera_features, min_lidar)
        reward += camera_reward

        # Time penalty
        reward -= 0.01

        # Movement encouragement
        if abs(self.last_action[0]) > 0.05:
            reward += 0.005

        # Goal alignment bonus
        cos_heading = state[21] * 2 - 1
        if cos_heading > 0.9:
            reward += 0.01

        return reward

    def compute_camera_reward(self, camera_features, min_lidar):
        """Compute additional rewards based on camera features"""
        camera_reward = 0.0
        
        # Reward for visual diversity (indicates interesting environment)
        feature_variance = np.var(camera_features)
        if feature_variance > 0.1:
            camera_reward += 0.01
            
        # Small bonus for having camera data at all
        if np.mean(camera_features) > 0.1:
            camera_reward += 0.005
            
        return camera_reward

    # ---------- UPDATED Reset Method ----------
    def reset(self, seed=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Stop robot
        self.stop_robot()

        # Reset episode variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.last_action = [0.0, 0.0]
        
        # ADD: Reset camera features
        self.camera_features = np.zeros(self.CAMERA_FEATURES, dtype=np.float32)
        self.latest_camera_image = None

        # Wait longer for sensors to be ready
        self.get_logger().info(" Resetting environment...")
        time.sleep(1.0)

        # UPDATE: Check for camera data too
        if self.odom_pose is None or self.latest_lidar is None or self.latest_camera_image is None:
            self.get_logger().warn(" No sensor data available - cannot set goal")
            self.goal_pose = None
            obs = np.zeros(self.UPDATED_STATE_DIM, dtype=np.float32)
            return obs, {}

        # Set random goal
        robot_x = self.odom_pose.position.x
        robot_y = self.odom_pose.position.y

        # Create a goal that's definitely away from the robot
        angle = np.random.uniform(0, 2 * math.pi)
        distance = np.random.uniform(3.0, 6.0)

        goal_x = robot_x + distance * math.cos(angle)
        goal_y = robot_y + distance * math.sin(angle)

        # Ensure goal is within reasonable bounds
        goal_x = np.clip(goal_x, -4.0, 4.0)
        goal_y = np.clip(goal_y, -4.0, 4.0)

        self.set_goal(goal_x, goal_y)

        # Calculate initial distance
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        self.initial_distance = math.sqrt(dx**2 + dy**2)
        self.current_distance = self.initial_distance
        self.previous_distance = self.initial_distance

        self.get_logger().info(f" New episode - Start: ({robot_x:.2f}, {robot_y:.2f})")
        self.get_logger().info(f" Goal: ({goal_x:.2f}, {goal_y:.2f}) - Distance: {self.initial_distance:.2f}m")

        obs, _ = self.get_observation()
        return obs, {}

    # ---------- Keep all your existing methods unchanged ----------
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def set_goal(self, goal_x, goal_y):
        """Set a new goal position and log start/goal info"""
        goal_msg = PoseStamped()
        goal_msg.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.goal_pose = goal_msg.pose
        self.goal_position = (goal_x, goal_y)

        # Store start position
        if self.odom_pose:
            self.start_position = (self.odom_pose.position.x, self.odom_pose.position.y)

        self.get_logger().info(f" START: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
        self.get_logger().info(f" GOAL: ({goal_x:.2f}, {goal_y:.2f})")
        self.get_logger().info(f" Initial distance: {self.initial_distance:.2f}m")

    def get_robot_velocity(self):
        """Get current robot velocity"""
        if self.odom_twist:
            return {
                'linear_x': self.odom_twist.linear.x,
                'linear_y': self.odom_twist.linear.y,
                'angular_z': self.odom_twist.angular.z
            }
        return {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}

    def execute_action(self, action):
        """Send velocity command to robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0] * self.train_config.MAX_LINEAR_SPEED
        cmd_vel.angular.z = action[1] * self.train_config.MAX_ANGULAR_SPEED
        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
        time.sleep(0.1)

    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        self.last_action = np.clip(action, -1.0, 1.0).tolist()
        self.execute_action(self.last_action)

        obs, done = self.get_observation()
        reward = self.compute_reward(obs, done)

        self.episode_step += 1
        self.episode_reward += reward
        self.previous_distance = self.current_distance
        if self.odom_pose and self.goal_pose:
            rx = self.odom_pose.position.x
            ry = self.odom_pose.position.y
            gx = self.goal_pose.position.x
            gy = self.goal_pose.position.y
            self.current_distance = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

        # episode length limit
        truncated = self.episode_step >= self.train_config.MAX_EPISODE_STEPS
        terminated = done or truncated

        info = {}
        if terminated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_step,
            }
            self.get_logger().info(
                f" Episode finished → steps: {self.episode_step}  reward: {self.episode_reward:.2f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        """Cleanup environment"""
        self.stop_robot()
        self.get_logger().info(" Environment closed")

    def __del__(self):
        """Make sure the robot stops if the object is destroyed."""
        self.close()