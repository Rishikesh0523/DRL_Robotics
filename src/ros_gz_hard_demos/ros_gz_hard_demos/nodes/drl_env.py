#!/usr/bin/env python3
"""
ROS 2 Jazzy Hardware DRL Environment with LiDAR Integration
Enhanced version of your existing environment
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
from typing import Dict, Any

# Your existing message imports
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray, Bool
import gymnasium as gym
from gymnasium import spaces

class JazzyHardwareDRLEnv(Node, gym.Env):
    def __init__(self):
        Node.__init__(self, 'jazzy_hardware_drl_env')
        gym.Env.__init__(self)
        
        # ROS 2 Jazzy optimized QoS
        fast_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', fast_qos)
        self.processed_lidar_pub = self.create_publisher(Float32MultiArray, '/drl/processed_lidar', reliable_qos)
        self.collision_pub = self.create_publisher(Bool, '/drl/collision_warning', reliable_qos)
        
        # Subscribers with Jazzy QoS
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, fast_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, fast_qos)
        self.camera_sub = self.create_subscription(
            Image, '/hardware/camera/processed', self.camera_callback, fast_qos)
        
        self.bridge = CvBridge()
        
        # Hardware-specific parameters for Jazzy
        self.declare_parameter('real_world_safety', True)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('lidar_min_range', 0.1)
        self.declare_parameter('lidar_max_range', 10.0)
        self.declare_parameter('collision_threshold', 0.4)
        self.declare_parameter('num_lidar_beams', 180)  # Reduced for efficiency
        
        # LiDAR processing parameters
        self.lidar_min_range = self.get_parameter('lidar_min_range').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.collision_threshold = self.get_parameter('collision_threshold').value
        self.num_lidar_beams = self.get_parameter('num_lidar_beams').value
        
        # Thread locks for sensor data
        self.lidar_lock = threading.Lock()
        self.odom_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        
        # Sensor data storage
        self.raw_lidar_data = None
        self.processed_lidar_data = np.zeros(self.num_lidar_beams, dtype=np.float32)
        self.odom_pose = None
        self.latest_camera = None
        self.collision_warning = False
        
        # Enhanced observation space with LiDAR
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(self.num_lidar_beams,), 
                dtype=np.float32
            ),
            'odometry': spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.pi]), 
                high=np.array([np.inf, np.inf, np.pi]), 
                shape=(3,), 
                dtype=np.float32
            ),
            'camera_features': spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(32,),  # Your existing camera features
                dtype=np.float32
            ),
            'goal_info': spaces.Box(
                low=-10.0, 
                high=10.0, 
                shape=(2,), 
                dtype=np.float32
            )
        })
        
        # Action space (linear velocity, angular velocity)
        max_linear = self.get_parameter('max_linear_speed').value
        max_angular = self.get_parameter('max_angular_speed').value
        
        self.action_space = spaces.Box(
            low=np.array([-max_linear, -max_angular]),
            high=np.array([max_linear, max_angular]), 
            dtype=np.float32
        )
        
        # Episode management
        self.episode_step = 0
        self.max_episode_steps = 1000
        self.episode_reward = 0.0
        
        # Goal position (you can make this dynamic)
        self.goal_position = np.array([3.0, 3.0])
        
        self.get_logger().info(' Jazzy Hardware DRL Environment with LiDAR Ready')
        self.get_logger().info(f' LiDAR beams: {self.num_lidar_beams}')
        self.get_logger().info(f' Collision threshold: {self.collision_threshold}m')
    
    def lidar_callback(self, msg: LaserScan):
        """Process LiDAR data with hardware-optimized preprocessing"""
        try:
            with self.lidar_lock:
                self.raw_lidar_data = msg
                self.process_lidar_data()
                
        except Exception as e:
            self.get_logger().error(f'LiDAR callback error: {e}')
    
    def process_lidar_data(self):
        """Hardware-optimized LiDAR processing for real-time DRL"""
        if self.raw_lidar_data is None:
            return
            
        try:
            # Convert to numpy and handle invalid values
            ranges = np.array(self.raw_lidar_data.ranges, dtype=np.float32)
            
            # Handle inf/nan values for hardware robustness
            ranges = np.nan_to_num(
                ranges, 
                nan=self.lidar_max_range, 
                posinf=self.lidar_max_range, 
                neginf=self.lidar_min_range
            )
            
            # Apply hardware-specific range limits
            ranges = np.clip(ranges, self.lidar_min_range, self.lidar_max_range)
            
            # Efficient downsampling for real-time processing
            if len(ranges) != self.num_lidar_beams:
                if len(ranges) > self.num_lidar_beams:
                    # Smart downsampling - preserve critical frontal sectors
                    ranges = self.smart_lidar_downsampling(ranges)
                else:
                    # Linear interpolation for sparse data
                    ranges = np.interp(
                        np.linspace(0, len(ranges)-1, self.num_lidar_beams),
                        np.arange(len(ranges)),
                        ranges
                    )
            
            # Normalize for DRL stability
            normalized_ranges = (ranges - self.lidar_min_range) / (self.lidar_max_range - self.lidar_min_range)
            self.processed_lidar_data = normalized_ranges.astype(np.float32)
            
            # Publish processed data for monitoring
            self.publish_processed_lidar()
            
            # Real-time collision detection
            self.detect_collisions(ranges)
            
        except Exception as e:
            self.get_logger().error(f'LiDAR processing error: {e}')
    
    def smart_lidar_downsampling(self, ranges: np.ndarray) -> np.ndarray:
        """Smart downsampling that preserves frontal obstacle details"""
        total_beams = len(ranges)
        target_beams = self.num_lidar_beams
        
        # Preserve higher resolution in front (critical for navigation)
        front_angle = np.pi / 3  # ±60 degrees frontal cone
        angle_increment = self.raw_lidar_data.angle_increment
        front_beams = int(front_angle / angle_increment)
        
        # Calculate downsampling ratios
        front_ratio = 2  # Keep more frontal beams
        side_ratio = max(1, (total_beams - front_beams) // (target_beams - front_beams // front_ratio))
        
        downsampled = []
        
        # High-resolution frontal sector
        front_indices = list(range(total_beams//2 - front_beams//2, total_beams//2 + front_beams//2))
        downsampled.extend(ranges[front_indices[::front_ratio]])
        
        # Lower resolution side sectors
        side_indices = list(range(0, total_beams//2 - front_beams//2)) + \
                      list(range(total_beams//2 + front_beams//2, total_beams))
        downsampled.extend(ranges[side_indices[::side_ratio]])
        
        # Ensure exact target size
        return np.array(downsampled[:target_beams], dtype=np.float32)
    
    def detect_collisions(self, ranges: np.ndarray):
        """Real-time collision detection with multiple safety zones"""
        # Front collision zone (critical)
        front_zone = self.num_lidar_beams // 3
        front_ranges = np.concatenate([
            ranges[:front_zone//2],  # Left front
            ranges[-front_zone//2:]  # Right front
        ])
        
        min_front_distance = np.min(front_ranges)
        
        # Immediate collision warning
        collision_imminent = min_front_distance < self.collision_threshold
        
        # Emergency stop if very close
        emergency_stop = min_front_distance < (self.collision_threshold * 0.5)
        
        if emergency_stop:
            self.get_logger().warn(f' EMERGENCY STOP: {min_front_distance:.2f}m')
            self.publish_emergency_stop()
        
        self.collision_warning = collision_imminent or emergency_stop
        
        # Publish collision warning
        collision_msg = Bool()
        collision_msg.data = self.collision_warning
        self.collision_pub.publish(collision_msg)
    
    def publish_processed_lidar(self):
        """Publish processed LiDAR data for monitoring/debugging"""
        lidar_msg = Float32MultiArray()
        lidar_msg.data = self.processed_lidar_data.tolist()
        self.processed_lidar_pub.publish(lidar_msg)
    
    def publish_emergency_stop(self):
        """Emergency stop command for hardware safety"""
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)
    
    def odom_callback(self, msg: Odometry):
        """Process odometry data"""
        try:
            with self.odom_lock:
                self.odom_pose = msg
        except Exception as e:
            self.get_logger().error(f'Odom callback error: {e}')
    
    def camera_callback(self, msg: Image):
        """Process camera data for DRL"""
        try:
            with self.camera_lock:
                self.latest_camera = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')
    
    def get_odometry_state(self) -> np.ndarray:
        """Extract odometry state for DRL"""
        if self.odom_pose is None:
            return np.zeros(3, dtype=np.float32)
        
        # Extract position
        x = self.odom_pose.pose.pose.position.x
        y = self.odom_pose.pose.pose.position.y
        
        # Extract orientation (quaternion to yaw)
        orientation = self.odom_pose.pose.pose.orientation
        yaw = self.quaternion_to_yaw(
            orientation.x, orientation.y, 
            orientation.z, orientation.w
        )
        
        return np.array([x, y, yaw], dtype=np.float32)
    
    def quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle"""
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    def extract_camera_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from camera image for DRL"""
        if image is None:
            return np.zeros(32, dtype=np.float32)
        
        # Simple grid-based features
        h, w = image.shape
        features = []
        grid_size = 8
        
        for i in range(0, h, h//grid_size):
            for j in range(0, w, w//grid_size):
                patch = image[i:min(i+h//grid_size, h), j:min(j+w//grid_size, w)]
                features.append(np.mean(patch) / 255.0)
        
        return np.array(features[:32], dtype=np.float32)  # Fixed size
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get complete observation with LiDAR, odom, and camera"""
        # LiDAR data
        with self.lidar_lock:
            lidar_obs = self.processed_lidar_data.copy()
        
        # Odometry data
        odom_obs = self.get_odometry_state()
        
        # Camera features
        with self.camera_lock:
            if self.latest_camera is not None:
                camera_obs = self.extract_camera_features(self.latest_camera)
            else:
                camera_obs = np.zeros(32, dtype=np.float32)
        
        # Relative goal position
        current_pos = odom_obs[:2]
        goal_obs = self.goal_position - current_pos
        
        return {
            'lidar': lidar_obs,
            'odometry': odom_obs,
            'camera_features': camera_obs,
            'goal_info': goal_obs.astype(np.float32)
        }
    
    def step(self, action: np.ndarray):
        """Execute one time step with hardware safety"""
        self.episode_step += 1
        
        # Apply hardware safety limits
        safe_action = self.apply_safety_limits(action)
        
        # Execute the action
        self.execute_action(safe_action)
        
        # Wait for action to take effect (hardware response time)
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward(observation, safe_action)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self.check_termination(observation)
        truncated = self.episode_step >= self.max_episode_steps
        
        info = {
            'episode_reward': self.episode_reward,
            'steps': self.episode_step,
            'collision': self.collision_warning,
            'goal_reached': self.check_goal_reached(observation)
        }
        
        return observation, reward, terminated, truncated, info
    
    def apply_safety_limits(self, action: np.ndarray) -> np.ndarray:
        """Apply real-world safety limits"""
        max_linear = self.get_parameter('max_linear_speed').value
        max_angular = self.get_parameter('max_angular_speed').value
        
        # Gradual acceleration limits for hardware safety
        safe_linear = np.clip(action[0], -max_linear, max_linear)
        safe_angular = np.clip(action[1], -max_angular, max_angular)
        
        # Additional safety: reduce speed when collision is imminent
        if self.collision_warning:
            safe_linear = min(safe_linear, max_linear * 0.3)
            self.get_logger().warn('  Reduced speed due to collision warning')
        
        return np.array([safe_linear, safe_angular], dtype=np.float32)
    
    def execute_action(self, action: np.ndarray):
        """Send velocity commands to hardware"""
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def calculate_reward(self, observation: Dict[str, np.ndarray], action: np.ndarray) -> float:
        """Calculate reward with LiDAR-based safety rewards"""
        reward = 0.0
        
        # Base survival reward
        reward += 0.01
        
        # LiDAR-based collision penalty
        min_lidar_distance = np.min(observation['lidar']) * (self.lidar_max_range - self.lidar_min_range)
        
        if self.collision_warning:
            reward -= 2.0
        elif min_lidar_distance < 0.5:  # Close to obstacle
            reward -= 0.5
        
        # Goal-oriented rewards
        goal_distance = np.linalg.norm(observation['goal_info'])
        reward += 0.1 * (1.0 / (goal_distance + 0.1))
        
        # Action smoothness penalty
        action_penalty = 0.01 * np.sum(np.square(action))
        reward -= action_penalty
        
        # Success bonus
        if goal_distance < 0.3:  # Goal reached
            reward += 10.0
        
        return reward
    
    def check_termination(self, observation: Dict[str, np.ndarray]) -> bool:
        """Check if episode should terminate"""
        # Collision termination
        if self.collision_warning:
            self.get_logger().warn(' Episode terminated: Collision')
            return True
        
        # Goal reached
        if self.check_goal_reached(observation):
            self.get_logger().info(' Episode terminated: Goal reached!')
            return True
        
        # Out of bounds (optional)
        current_pos = observation['odometry'][:2]
        if np.any(np.abs(current_pos) > 8.0):
            self.get_logger().warn(' Episode terminated: Out of bounds')
            return True
        
        return False
    
    def check_goal_reached(self, observation: Dict[str, np.ndarray]) -> bool:
        """Check if goal is reached"""
        goal_distance = np.linalg.norm(observation['goal_info'])
        return goal_distance < 0.3
    
    def reset(self, seed: int = None, options: Dict = None) -> tuple:
        """Reset the environment for new episode"""
        super().reset(seed=seed)
        
        self.get_logger().info('Resetting environment...')
        
        # Stop the robot
        self.publish_emergency_stop()
        
        # Reset episode variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.collision_warning = False
        
        # Wait for hardware to stabilize
        for _ in range(10):  # 1 second stabilization
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Get initial observation
        observation = self.get_observation()
        info = {}
        
        self.get_logger().info(' Environment reset complete')
        
        return observation, info
    
    def close(self):
        """Clean shutdown"""
        self.publish_emergency_stop()
        self.get_logger().info(' Environment shutdown complete')

def main():
    rclpy.init()
    node = JazzyHardwareDRLEnv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()