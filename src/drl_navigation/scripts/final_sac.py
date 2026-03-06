#!/usr/bin/env python3
"""
SAC-based Robot Navigation Training - FINAL CORRECTED VERSION
Matched to your exact ROS2 topic structures
"""

import os
import time
import math
import traceback
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu, Image
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# ========================= ROS2 ENVIRONMENT =========================
class RobotNavigationEnv(Node, gym.Env):
    """ROS2 Environment for Robot Navigation using SAC"""
    
    def __init__(self):
        # Initialize both Node and gym.Env
        Node.__init__(self, 'drl_nav_env_node')
        gym.Env.__init__(self)
        
        # ROS2 Parameters
        self.declare_parameter('max_episode_steps', 300)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('collision_threshold', 0.25)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 0.8)
        
        self.max_steps = self.get_parameter('max_episode_steps').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.collision_threshold = self.get_parameter('collision_threshold').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        
        # State dimensions (20 LiDAR bins + 5 features)
        self.state_dim = 25
        self.action_dim = 2
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # ROS2 Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.lidar_sub = self.create_subscription(LaserScan, '/lidar', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # State variables
        self.latest_lidar = None
        self.odom_pose = None
        self.odom_yaw = 0.0
        self.goal_pose = None
        self.last_action = [0.0, 0.0]
        
        # Episode variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.initial_distance = 0.0
        self.current_distance = 0.0
        self.previous_distance = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Wait for first messages
        self.get_logger().info("Waiting for sensor data...")
        while (self.latest_lidar is None or self.odom_pose is None) and rclpy.ok():
            time.sleep(0.1)
        
        self.get_logger().info("Robot Navigation Environment initialized")
        
    def lidar_callback(self, msg):
        with self._lock:
            self.latest_lidar = msg
    
    def odom_callback(self, msg):
        with self._lock:
            self.odom_pose = msg.pose.pose
            orientation = msg.pose.pose.orientation
            _, _, self.odom_yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
    
    def set_goal(self, goal_x, goal_y):
        """Set a new goal position"""
        goal_msg = PoseStamped()
        goal_msg.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        self.goal_pose = goal_msg.pose
        self.get_logger().info(f"New goal set: ({goal_x:.2f}, {goal_y:.2f})")
    
    def get_observation(self):
        """Get current observation state"""
        with self._lock:
            if self.latest_lidar is None or self.odom_pose is None or self.goal_pose is None:
                return np.zeros(self.state_dim, dtype=np.float32), False
            
            # Get LiDAR data (360 degree scan)
            laser_ranges = list(self.latest_lidar.ranges)
            
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
            
            # Check collision from LiDAR
            valid_ranges = [r for r in laser_ranges if not np.isinf(r) and r > 0.05]
            min_lidar = min(valid_ranges) if valid_ranges else 10.0
            collision = min_lidar < self.collision_threshold
            
            # Check goal reached
            goal_reached = distance < self.goal_tolerance
            
            # Prepare state vector
            state = self.prepare_state(
                latest_scan=laser_ranges,
                distance=distance,
                cos=cos_heading,
                sin=sin_heading,
                collision=collision,
                goal=goal_reached,
                action=self.last_action
            )
            
            return state, (collision or goal_reached)
    
    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """Prepare state vector from sensor data"""
        latest_scan = np.array(latest_scan)
        
        # Handle infinite values in LiDAR - your LiDAR has range_max=10.0
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 10.0  # Use actual max range from your LiDAR
        
        # Bin LiDAR readings (20 bins from 360 degree scan)
        max_bins = 20
        bin_size = len(latest_scan) // max_bins
        min_values = []
        
        for i in range(0, len(latest_scan), bin_size):
            if i + bin_size <= len(latest_scan):
                bin_data = latest_scan[i:i + bin_size]
                min_val = np.min(bin_data) / 10.0  # Normalize using max range 10.0
                min_values.append(min_val)
        
        # Pad if necessary
        while len(min_values) < max_bins:
            min_values.append(1.0)  # Far away (safe)
        
        # Normalize other features
        norm_distance = min(distance / 15.0, 1.0)  # Max 15m
        norm_cos = (cos + 1) / 2  # [-1,1] -> [0,1]
        norm_sin = (sin + 1) / 2  # [-1,1] -> [0,1]
        norm_lin_vel = (action[0] + 1) / 2  # [-1,1] -> [0,1]
        norm_ang_vel = (action[1] + 1) / 2  # [-1,1] -> [0,1]
        
        # Combine all features
        state = min_values + [norm_distance, norm_cos, norm_sin, norm_lin_vel, norm_ang_vel]
        
        return np.array(state, dtype=np.float32)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def compute_reward(self, state, done):
        """Compute reward based on current state"""
        reward = 0.0
        
        # Extract features from state
        distance = state[20] * 15.0  # Denormalize (max 15m)
        min_lidar = min(state[:20]) * 10.0  # Denormalize (max 10m)
        
        # Progress reward (getting closer to goal)
        progress = (self.previous_distance - distance) * 3.0
        reward += progress
        
        # Goal reward (only if safe)
        if distance < self.goal_tolerance:
            if min_lidar > self.collision_threshold + 0.1:  # Safe arrival
                reward += 15.0
                self.get_logger().info(" GOAL REACHED SAFELY! +15 reward")
            else:  # Risky arrival
                reward += 5.0
                self.get_logger().info(" Goal reached (risky) +5 reward")
        
        # Collision penalty
        if min_lidar < self.collision_threshold:
            reward -= 3.0
            self.get_logger().warn(f" COLLISION! Min LiDAR: {min_lidar:.2f}m -3 reward")
        elif min_lidar < 0.5:  # Warning zone
            reward -= 0.1  # Small penalty for getting too close
        
        # Time penalty (encourage efficiency)
        reward -= 0.01
        
        # Movement encouragement (prevent getting stuck)
        if abs(self.last_action[0]) > 0.05:  # If moving forward/backward
            reward += 0.005
        
        # Goal alignment bonus
        cos_heading = state[21] * 2 - 1  # Denormalize to [-1,1]
        if cos_heading > 0.9:  # Heading directly towards goal
            reward += 0.01
        
        return reward
    
    def reset(self, seed=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Stop robot
        self.stop_robot()
        
        # Reset episode variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.last_action = [0.0, 0.0]
        
        # Set random goal (3-8m away in random direction)
        if self.odom_pose:
            robot_x = self.odom_pose.position.x
            robot_y = self.odom_pose.position.y
            
            # Safe goal placement (adjust based on your map)
            map_limits = (-8.0, 8.0)  # Typical Gazebo world size
            safety_margin = 1.0
            
            max_attempts = 10
            for attempt in range(max_attempts):
                angle = np.random.uniform(0, 2 * math.pi)
                distance = np.random.uniform(3.0, 6.0)
                
                goal_x = robot_x + distance * math.cos(angle)
                goal_y = robot_y + distance * math.sin(angle)
                
                # Check if goal is within reasonable bounds
                if (map_limits[0] + safety_margin < goal_x < map_limits[1] - safety_margin and
                    map_limits[0] + safety_margin < goal_y < map_limits[1] - safety_margin):
                    break
            else:
                # Fallback goal
                goal_x = robot_x + 4.0
                goal_y = robot_y + 4.0
            
            self.set_goal(goal_x, goal_y)
            
            # Calculate initial distance
            dx = goal_x - robot_x
            dy = goal_y - robot_y
            self.initial_distance = math.sqrt(dx**2 + dy**2)
            self.current_distance = self.initial_distance
            self.previous_distance = self.initial_distance
        
        # Wait for sensors to update
        time.sleep(0.5)
        
        obs, _ = self.get_observation()
        self.get_logger().info(f"Environment reset - Goal: {self.initial_distance:.2f}m away")
        
        return obs, {}
    
    def step(self, action):
        """Execute one time step"""
        self.episode_step += 1
        
        # Store action
        self.last_action = action
        
        # Execute action
        self.execute_action(action)
        
        # Wait for physics to update
        time.sleep(0.1)
        
        # Get new observation
        obs, done = self.get_observation()
        
        # Compute reward
        reward = self.compute_reward(obs, done)
        self.episode_reward += reward
        
        # Update distances
        self.previous_distance = self.current_distance
        self.current_distance = obs[20] * 15.0  # Denormalize
        
        # Check episode termination
        if self.episode_step >= self.max_steps:
            done = True
            self.get_logger().info(" Max steps reached")
        
        # Logging
        if done or self.episode_step % 25 == 0:
            status = "DONE" if done else f"Step {self.episode_step}"
            min_lidar = min(obs[:20]) * 10.0
            self.get_logger().info(
                f"{status} | Dist: {self.current_distance:.2f}m | "
                f"MinLiDAR: {min_lidar:.2f}m | Reward: {reward:.3f} | "
                f"Total: {self.episode_reward:.2f}"
            )
        
        if done:
            success = self.current_distance < self.goal_tolerance
            status = "SUCCESS" if success else "FAILED"
            self.get_logger().info(
                f"Episode {status} | Final distance: {self.current_distance:.2f}m | "
                f"Steps: {self.episode_step} | Total reward: {self.episode_reward:.2f}"
            )
        
        return obs, reward, done, False, {}
    
    def execute_action(self, action):
        """Send velocity command to robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0] * self.max_linear_speed
        cmd_vel.angular.z = action[1] * self.max_angular_speed
        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
        time.sleep(0.1)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        """Cleanup environment"""
        self.stop_robot()
        self.get_logger().info("Environment closed")

# ========================= SAC NETWORK ARCHITECTURE =========================
class DoubleQCritic(torch.nn.Module):
    """Twin Q-network for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        
        # Q1 network
        self.q1_layers = torch.nn.ModuleList()
        self.q1_layers.append(torch.nn.Linear(obs_dim + action_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.q1_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.q1_out = torch.nn.Linear(hidden_dim, 1)
        
        # Q2 network  
        self.q2_layers = torch.nn.ModuleList()
        self.q2_layers.append(torch.nn.Linear(obs_dim + action_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.q2_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.q2_out = torch.nn.Linear(hidden_dim, 1)
        
        self.activation = torch.nn.ReLU()
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        
        # Q1 forward
        q1 = x
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_out(q1)
        
        # Q2 forward
        q2 = x
        for layer in self.q2_layers:
            q2 = self.activation(layer(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2

class DiagGaussianActor(torch.nn.Module):
    """Gaussian policy network for SAC"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, hidden_depth=2, log_std_bounds=[-5, 2]):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_bounds = log_std_bounds
        
        # Shared trunk
        self.trunk_layers = torch.nn.ModuleList()
        self.trunk_layers.append(torch.nn.Linear(obs_dim, hidden_dim))
        for _ in range(hidden_depth - 1):
            self.trunk_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Output heads
        self.mu_layer = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_dim, action_dim)
        
        self.activation = torch.nn.ReLU()
        
    def forward(self, obs):
        x = obs
        for layer in self.trunk_layers:
            x = self.activation(layer(x))
            
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.tanh(log_std)
        
        # Apply log std bounds
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        
        std = torch.exp(log_std)
        
        return torch.distributions.Normal(mu, std)

# ========================= SAC ALGORITHM =========================
class SAC:
    """Soft Actor-Critic implementation"""
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        actor_lr=3e-4,
        critic_lr=3e-4,
        learnable_temperature=True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.learnable_temperature = learnable_temperature
        
        # Networks
        self.critic = DoubleQCritic(state_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = DiagGaussianActor(state_dim, action_dim).to(device)
        
        # Temperature (alpha)
        self.log_alpha = torch.tensor(np.log(0.1)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def act(self, obs, sample=True):
        """Get action from policy"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(-self.max_action, self.max_action)
            return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        """Single SAC update step"""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_dist = self.actor(next_states)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_probs
            target_Q = rewards + (1 - dones) * self.discount * target_V
        
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        dist = self.actor(states)
        new_actions = dist.rsample()
        log_probs = dist.log_prob(new_actions).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(states, new_actions)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        actor_loss = (self.alpha.detach() * log_probs - actor_Q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_probs - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Update target critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'average_reward': rewards.mean().item()
        }

# ========================= REPLAY BUFFER =========================
class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
        
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx].flatten(),
            self.next_states[idx],
            self.dones[idx].flatten()
        )

# ========================= TRAINING CONFIG =========================
class Config:
    # Training parameters
    TOTAL_TIMESTEPS = 100_000  # Reduced for faster testing
    BATCH_SIZE = 128
    BUFFER_SIZE = 50000
    LEARNING_STARTS = 1000
    
    # SAC parameters
    LEARNING_RATE = 3e-4
    DISCOUNT = 0.99
    TAU = 0.005
    
    # Network architecture
    HIDDEN_DIM = 256
    HIDDEN_DEPTH = 2
    
    # Directories
    LOG_DIR = "./sac_training_logs/"
    MODEL_DIR = "./sac_models/"

# ========================= MAIN TRAINING =========================
def make_env():
    """Create and wrap environment"""
    def _init():
        env = RobotNavigationEnv()
        env = Monitor(env)
        return env
    return _init

def train_sac():
    """Main training function"""
    print("=== Custom SAC Training ===")
    rclpy.init()
    
    # Create directories
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    vec_env = None
    try:
        # Create environment
        vec_env = DummyVecEnv([make_env()])
        env = vec_env.envs[0].env
        
        # Get dimensions from environment
        state_dim = env.observation_space.shape[0]  # Should be 25
        action_dim = env.action_space.shape[0]      # Should be 2
        max_action = float(env.action_space.high[0]) # Should be 1.0
        
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Max action: {max_action}")
        
        # Initialize SAC agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            max_action=max_action,
            discount=Config.DISCOUNT,
            tau=Config.TAU,
            actor_lr=Config.LEARNING_RATE,
            critic_lr=Config.LEARNING_RATE,
            alpha_lr=Config.LEARNING_RATE,
        )
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(state_dim, action_dim, Config.BUFFER_SIZE)
        
        # Training loop
        state = vec_env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        
        print(f"\nStarting training for {Config.TOTAL_TIMESTEPS} timesteps...")
        start_time = time.time()
        
        for t in range(Config.TOTAL_TIMESTEPS):
            # Select action
            if t < Config.LEARNING_STARTS:
                action = vec_env.action_space.sample()
                if t % 1000 == 0:
                    print(f"Step {t}: Using random exploration...")
            else:
                action = agent.act(state[0])
            
            # Take step
            next_state, reward, done, info = vec_env.step([action])
            
            # Store transition
            replay_buffer.add(state[0], action, reward[0], next_state[0], done[0])
            
            state = next_state
            episode_reward += reward[0]
            episode_length += 1
            
            # Train agent
            if t >= Config.LEARNING_STARTS and replay_buffer.size > Config.BATCH_SIZE:
                metrics = agent.update(replay_buffer, Config.BATCH_SIZE)
                
                # Log training metrics
                if t % 1000 == 0:
                    print(f"Step {t}: "
                          f"Critic Loss: {metrics['critic_loss']:.3f}, "
                          f"Actor Loss: {metrics['actor_loss']:.3f}, "
                          f"Alpha: {metrics['alpha']:.3f}, "
                          f"Avg Reward: {metrics['average_reward']:.3f}")
            
            # Episode end
            if done[0]:
                print(f"Episode {episode_num}: "
                      f"Steps: {episode_length}, "
                      f"Reward: {episode_reward:.2f}")
                
                episode_reward = 0
                episode_length = 0
                episode_num += 1
                state = vec_env.reset()
            
            # Save model periodically
            if t % 5000 == 0 and t > 0:
                model_path = f"{Config.MODEL_DIR}/sac_model_step_{t}.pth"
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'critic_target_state_dict': agent.critic_target.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'step': t
                }, model_path)
                print(f"Model saved: {model_path}")
        
        # Save final model
        final_path = f"{Config.MODEL_DIR}/sac_model_final.pth"
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'critic_target_state_dict': agent.critic_target.state_dict(),
            'log_alpha': agent.log_alpha,
            'step': Config.TOTAL_TIMESTEPS
        }, final_path)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Final model saved: {final_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
    finally:
        if vec_env is not None:
            vec_env.close()
        rclpy.shutdown()

# ========================= TESTING =========================
def test_sac(model_path, episodes=10):
    """Test the trained SAC model"""
    print(f"\n=== Testing SAC Model: {model_path} ===")
    rclpy.init()
    
    vec_env = None
    try:
        # Create environment
        vec_env = DummyVecEnv([make_env()])
        env = vec_env.envs[0].env
        
        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Load agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = SAC(state_dim, action_dim, device, max_action)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.log_alpha = checkpoint['log_alpha']
        
        print(f"Model loaded from step {checkpoint['step']}")
        
        # Test episodes
        successes = 0
        total_rewards = []
        episode_lengths = []
        
        for ep in range(episodes):
            state = vec_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 300:
                action = agent.act(state[0], sample=False)  # No exploration
                state, reward, done, info = vec_env.step([action])
                episode_reward += reward[0]
                steps += 1
            
            success = steps < 300  # Didn't timeout
            if success:
                successes += 1
                
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            print(f"Episode {ep+1}: "
                  f"Steps: {steps}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Success: {success}")
        
        # Summary
        success_rate = (successes / episodes) * 100
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\nTest Summary:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Length: {avg_length:.1f} steps")
        
    except Exception as e:
        print(f"Testing failed: {e}")
        traceback.print_exc()
    finally:
        if vec_env is not None:
            vec_env.close()
        rclpy.shutdown()

# ========================= MAIN =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom SAC Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='train: train new model, test: evaluate model')
    parser.add_argument('--model', type=str, 
                       default='./sac_models/sac_model_final.pth',
                       help='Model path for testing')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_sac()
    elif args.mode == 'test':
        test_sac(args.model, args.episodes)