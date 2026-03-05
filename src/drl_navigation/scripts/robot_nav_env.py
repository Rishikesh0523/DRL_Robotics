import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import threading
import time
import math


class RobotNavigationEnv(Node, gym.Env):
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, 'drl_training_node')

        self.metadata = {'render_modes': ['human']}

        # 3-D holonomic action space [vx, vy, w]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 12-D observation: [goal_unit_x, goal_unit_y, heading_err, 10 lidar]
        self.lidar_sectors = 10
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -math.pi] + [0.0] * self.lidar_sectors, dtype=np.float32),
            high=np.array([1.0, 1.0, math.pi] + [10.0] * self.lidar_sectors, dtype=np.float32),
            dtype=np.float32
        )

        # ---------- ROS 2 ----------
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/lidar', self.lidar_callback, 10)

        # ---------- state ----------
        self.current_position = np.array([0.0, 0.0])
        self.current_orientation = np.array([1.0, 0.0])
        self.goal_position = np.array([2.8, 2.8])
        self.lidar_ranges = None
        self.lidar_available = False
        self.min_lidar_distance = 10.0

        # ---------- episode ----------
        self.episode_step = 0
        self.max_steps = 300
        self.last_distance = float('inf')
        self.initial_distance = float('inf')
        self.total_reward = 0.0
        self.position_history = []

        print(" Robot Navigation Environment initialized")
        self.ros_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        self.wait_for_initial_data()

    # --------------- ROS helpers ---------------
    def ros_spin(self):
        rclpy.spin(self)

    def wait_for_initial_data(self):
        print("⏳ Waiting for sensor data...")
        timeout = time.time() + 10.0
        while time.time() < timeout:
            if (np.any(self.current_position != 0) and self.lidar_ranges is not None):
                self.initial_distance = self.get_distance_to_goal()
                self.last_distance = self.initial_distance
                print(f" All sensors ready! Initial distance: {self.initial_distance:.2f}")
                return
            time.sleep(0.1)
        print("  Some sensor data not received, but proceeding...")

    # --------------- callbacks ---------------
    def odom_callback(self, msg):
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y
        ox, oy, oz, ow = (msg.pose.pose.orientation.x,
                          msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z,
                          msg.pose.pose.orientation.w)
        yaw = math.atan2(2.0 * (ow * oz + ox * oy),
                         1.0 - 2.0 * (oy * oy + oz * oz))
        self.current_orientation[0] = math.cos(yaw)
        self.current_orientation[1] = math.sin(yaw)

        self.position_history.append(
            (self.current_position[0], self.current_position[1]))
        if len(self.position_history) > 20:
            self.position_history.pop(0)

    def lidar_callback(self, msg):
        self.lidar_ranges = list(msg.ranges)
        self.lidar_available = True
        valid = [r for r in self.lidar_ranges if 0.1 < r < 10.0]
        self.min_lidar_distance = min(valid) if valid else 10.0

    # --------------- observation ---------------
    def get_distance_to_goal(self):
        return np.linalg.norm(self.goal_position - self.current_position)

    def get_angle_to_goal(self):
        dx, dy = self.goal_position - self.current_position
        goal_angle = math.atan2(dy, dx)
        current_yaw = math.atan2(
            self.current_orientation[1], self.current_orientation[0])
        angle_diff = goal_angle - current_yaw
        return math.atan2(math.sin(angle_diff), math.cos(angle_diff))

    def process_lidar_data(self):
        if self.lidar_ranges is None:
            return [10.0] * self.lidar_sectors
        n = len(self.lidar_ranges)
        sector = n // self.lidar_sectors
        mins = []
        for i in range(self.lidar_sectors):
            rng = self.lidar_ranges[i * sector:(i + 1) * sector]
            valid = [v for v in rng if 0.1 < v < 10.0]
            mins.append(min(valid) if valid else 10.0)
        return mins

    def get_observation(self):
        disp = self.goal_position - self.current_position
        disp_norm = disp / (np.linalg.norm(disp) + 1e-8)
        angle = self.get_angle_to_goal()
        lidar_data = self.process_lidar_data()
        obs = np.concatenate(([disp_norm[0], disp_norm[1], angle], lidar_data))
        return obs.astype(np.float32)

    # --------------- reward / done ---------------
    #def is_stuck(self):
    #    if len(self.position_history) < 10:
    #        return False
    #    dist = 0
    #    for i in range(1, len(self.position_history)):
    #        dx = self.position_history[i][0] - self.position_history[i - 1][0]
    #        dy = self.position_history[i][1] - self.position_history[i - 1][1]
    #        dist += math.sqrt(dx * dx + dy * dy)
    #    avg = dist / (len(self.position_history) - 1)
    #    return avg < 0.02 and self.episode_step > 100


    def is_stuck(self):
      return False  


    def calculate_reward(self):
        cur_dist = self.get_distance_to_goal()
        #angle = abs(self.get_angle_to_goal())

        progress = (self.last_distance - cur_dist) * 50.0        # softer
      #  angle_pen = -angle * 0.5
      #  coll = 0.0
      #  if self.min_lidar_distance < 0.3:
      #      coll = -5.0
      #  elif self.min_lidar_distance < 0.5:
      #      coll = -2.0
      #  stuck = -1.0 if self.is_stuck() else 0.0
      #  success = 100.0 if cur_dist < 0.3 else 0.0
        time_pen = -0.05

        reward = progress + time_pen
        self.last_distance = cur_dist
        return reward

    def is_done(self):
        d = self.get_distance_to_goal()
        if d < 0.3:
            print(" GOAL REACHED!")
            return True
        if self.min_lidar_distance < 0.2:
            print(" COLLISION!")
            return True
        if self.is_stuck():
            print(" STUCK!")
            return True
        if self.episode_step >= self.max_steps:
            print(" MAX STEPS!")
            return True
        return False




    # --------------- control ---------------
    def send_velocity_command(self, action):
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], -0.5, 0.5))
        twist.linear.y = float(np.clip(action[1], -0.5, 0.5))
        twist.angular.z = float(np.clip(action[2], -1.0, 1.0))
        self.cmd_vel_pub.publish(twist)
       
       
        self.get_logger().info(f'CMD: vx={twist.linear.x:.2f} vy={twist.linear.y:.2f} w={twist.angular.z:.2f}')

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)

    # --------------- gym interface ---------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n Resetting environment")
        self.episode_step = 0
        self.position_history.clear()
        self.total_reward = 0.0
        self.stop_robot()
        time.sleep(0.5)
        self.last_distance = self.get_distance_to_goal()
        self.initial_distance = self.last_distance
        print(f" Start: {self.current_position}")
        print(f" Goal:  {self.goal_position}")
        print(f" Distance: {self.last_distance:.2f}")
        return self.get_observation(), {}

    def step(self, action):
        self.episode_step += 1
        self.send_velocity_command(action)
        time.sleep(0.1)
        obs = self.get_observation()
        reward = self.calculate_reward()
        terminated = self.is_done()
        self.total_reward += reward

        if terminated or self.episode_step % 25 == 0 or self.episode_step <= 5:
            print(f"Step {self.episode_step}: "
                  f"Action=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] "
                  f"RealPos=({self.current_position[0]:.2f}, {self.current_position[1]:.2f}) "
                  f"Dist={self.get_distance_to_goal():.2f} "
                  f"Rew={reward:.2f}")
        if terminated:
            print(f" Episode reward = {self.total_reward:.2f}")
        return obs, reward, terminated, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop_robot()
        self.destroy_node()