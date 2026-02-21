#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
                
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach()
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DRLNavigationNode(Node):
    def __init__(self):
        super().__init__('drl_navigation_node')
        
        # Parameters
        self.declare_parameter('goal_x', 5.0)
        self.declare_parameter('goal_y', 5.0)
        self.declare_parameter('training', True)
        
        # Get parameters
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.training = self.get_parameter('training').value
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/model/vehicle_blue/odometry', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Initialize variables
        self.lidar_data = None
        self.robot_pose = None
        self.goal_position = [self.goal_x, self.goal_y]
        
        # DRL Agent
        state_size = 360 + 4  # lidar readings + robot position + goal position
        action_size = 5  # discrete actions: stop, forward, left, right, backward
        self.agent = DQNAgent(state_size, action_size)
        
        # Training parameters
        self.batch_size = 32
        self.episode = 0
        self.max_episodes = 1000
        
        # Action mappings (discrete to continuous)
        self.action_map = {
            0: [0.0, 0.0],    # stop
            1: [0.3, 0.0],    # forward
            2: [0.1, 0.5],    # left
            3: [0.1, -0.5],   # right
            4: [-0.2, 0.0]    # backward
        }
        
        self.get_logger().info("DRL Navigation Node Started")
        
    def lidar_callback(self, msg):
        # Process lidar data
        self.lidar_data = msg.ranges
        # Handle inf values
        self.lidar_data = [min(10.0, r) if not math.isinf(r) else 10.0 for r in self.lidar_data]
        
    def odom_callback(self, msg):
        # Process odometry data
        self.robot_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        
    def get_state(self):
        if self.lidar_data is None or self.robot_pose is None:
            return None
            
        # Combine lidar data and position information
        lidar_processed = np.array(self.lidar_data)
        position_data = np.array([
            self.robot_pose[0], 
            self.robot_pose[1],
            self.goal_position[0],
            self.goal_position[1]
        ])
        
        return np.concatenate([lidar_processed, position_data])
        
    def calculate_reward(self):
        if self.robot_pose is None or self.lidar_data is None:
            return 0
            
        # Distance to goal
        dx = self.robot_pose[0] - self.goal_position[0]
        dy = self.robot_pose[1] - self.goal_position[1]
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Check for collision
        min_distance = min(self.lidar_data)
        collision_penalty = -100 if min_distance < 0.3 else 0
        
        # Reward for making progress toward goal
        progress_reward = -distance_to_goal * 0.1
        
        # Combine rewards
        reward = progress_reward + collision_penalty
        
        # Additional reward for reaching goal
        if distance_to_goal < 0.5:
            reward += 500
            
        return reward
        
    def check_done(self):
        if self.robot_pose is None or self.lidar_data is None:
            return False
            
        # Check if robot reached goal
        dx = self.robot_pose[0] - self.goal_position[0]
        dy = self.robot_pose[1] - self.goal_position[1]
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Check for collision
        min_distance = min(self.lidar_data)
        
        return distance_to_goal < 0.5 or min_distance < 0.3
        
    def execute_action(self, action_idx):
        action = self.action_map[action_idx]
        twist = Twist()
        twist.linear.x = action[0]
        twist.angular.z = action[1]
        self.cmd_vel_pub.publish(twist)
        
    def train_episode(self):
        state = self.get_state()
        if state is None:
            return
            
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            action_idx = self.agent.act(state)
            self.execute_action(action_idx)
            
            # Wait for action to take effect
            rclpy.spin_once(self, timeout_sec=0.1)
            
            next_state = self.get_state()
            if next_state is None:
                break
                
            reward = self.calculate_reward()
            done = self.check_done()
            
            self.agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(self.agent.memory) > self.batch_size:
                self.agent.replay(self.batch_size)
                
        self.episode += 1
        self.get_logger().info(f"Episode: {self.episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        # Reset robot position (you might need a reset service)
        if done:
            self.execute_action(0)  # stop
            
    def run(self):
        if self.training:
            while self.episode < self.max_episodes and rclpy.ok():
                self.train_episode()
            # Save trained model
            torch.save(self.agent.model.state_dict(), 'drl_navigation_model.pth')
        else:
            # Load trained model for inference
            self.agent.model.load_state_dict(torch.load('drl_navigation_model.pth'))
            self.agent.epsilon = 0.0  # No exploration
            
            state = self.get_state()
            while state is not None and rclpy.ok():
                action_idx = self.agent.act(state)
                self.execute_action(action_idx)
                rclpy.spin_once(self, timeout_sec=0.1)
                state = self.get_state()

def main(args=None):
    rclpy.init(args=args)
    node = DRLNavigationNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()