import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
import threading
import time

class RobotGazeboEnv(gym.Env):
    def __init__(self):
        super(RobotGazeboEnv, self).__init__()
        
        # Initialize ROS 2 node if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create a node for this environment
        self.node = Node('drl_robot_training')
        
        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # min linear, angular velocity
            high=np.array([1.0, 1.0]),   # max linear, angular velocity
            dtype=np.float32
        )
        
        # Observation space: laser scan data (simplified to 10 values) + position
        self.observation_space = spaces.Box(
            low=np.array([0] * 10 + [-10, -10, -3.14]), 
            high=np.array([10] * 10 + [10, 10, 3.14]), 
            dtype=np.float32
        )
        
        # ROS 2 publishers and subscribers
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.odom = None
        self.scan_data = None
        self.position = np.zeros(3)  # x, y, theta
        
        # Thread for spinning the node
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        
        # Wait for first messages
        self.node.get_logger().info("Waiting for sensor data...")
        start_time = time.time()
        while self.odom is None or self.scan_data is None:
            if time.time() - start_time > 10.0:  # 10 second timeout
                self.node.get_logger().error("Timeout waiting for sensor data!")
                break
            time.sleep(0.1)
        
        if self.odom is not None and self.scan_data is not None:
            self.node.get_logger().info("Sensor data received!")
        else:
            self.node.get_logger().error("Failed to receive sensor data!")
        
    def odom_callback(self, data):
        self.odom = data
        # Extract position and orientation
        self.position[0] = data.pose.pose.position.x
        self.position[1] = data.pose.pose.position.y
        
        # Convert quaternion to Euler angles
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.position[2] = yaw
        
    def scan_callback(self, data):
        self.scan_data = data.ranges
        
    def step(self, action):
        # Execute action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0] * 0.5)  # Scale to reasonable values
        vel_cmd.angular.z = float(action[1] * 1.0)
        self.cmd_vel_pub.publish(vel_cmd)
        
        # Wait for state update
        time.sleep(0.1)  # 100ms delay
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._check_done()
        
        info = {}  # Additional info if needed
        
        return observation, reward, done, info
        
    def reset(self, seed=None, options=None):
        # Reset robot position
        self.node.get_logger().info("Resetting environment...")
        
        # Stop the robot
        vel_cmd = Twist()
        self.cmd_vel_pub.publish(vel_cmd)
        time.sleep(1)
        
        # For now, we'll just reset the position tracking
        # In a real scenario, you'd call a service to reset the robot position
        self.position = np.zeros(3)
        
        # Wait for sensor data to update
        time.sleep(0.5)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Process sensor data for observation
        obs = np.zeros(13)  # 10 laser values + 3 position values
        
        if self.scan_data is not None:
            # Simplify laser data to 10 values
            num_scans = len(self.scan_data)
            if num_scans > 0:
                step = max(1, num_scans // 10)
                scan_obs = np.array(self.scan_data[::step])[:10]
                # Replace inf values with max range
                scan_obs[scan_obs == float('inf')] = 10.0
                obs[:10] = scan_obs
        
        # Add position information
        obs[10:] = self.position
        
        return obs
        
    def _calculate_reward(self):
        # Implement your reward function
        reward = 0.1  # Small reward for surviving
        
        # Penalty for being too close to obstacles
        if self.scan_data is not None and min(self.scan_data) < 0.5:
            reward -= 5
            
        # Reward forward movement
        if self.odom is not None:
            reward += abs(self.odom.twist.twist.linear.x) * 0.2
            
        # Additional reward for moving away from walls
        if self.scan_data is not None and min(self.scan_data) > 1.0:
            reward += 0.1
            
        return reward
        
    def _check_done(self):
        # Check if episode should end
        if self.scan_data is not None and min(self.scan_data) < 0.3:  # Collision
            self.node.get_logger().info("Collision detected! Ending episode.")
            return True
            
        # End episode after a certain time or condition
        # You can add more termination conditions here
        
        return False
        
    def close(self):
        # Clean up ROS resources
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()