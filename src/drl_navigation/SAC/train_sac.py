"""
Main training script for SAC robot navigation with MR LOGI Camera
"""

import os
import time
import traceback
import argparse
import numpy as np
import torch
import rclpy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_nav_env import RobotNavigationEnv
from sac_agent import SAC
from replay_buffer import ReplayBuffer
from config import TrainingConfig

def make_env():
    """Create and wrap environment"""
    def _init():
        env = RobotNavigationEnv()
        env = Monitor(env)
        return env
    return _init

def train_sac():
    """Main training function with camera support"""
    print("=== Custom SAC Training with MR LOGI Camera ===")
    
    # Initialize ROS2 FIRST
    rclpy.init()
    
    # Create directories
    config = TrainingConfig()
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    vec_env = None
    try:
        # Create environment
        print(" Creating Robot Navigation Environment...")
        print(" This environment now includes MR LOGI camera data!")
        vec_env = DummyVecEnv([make_env()])
        
        # Wait longer for ROS2 topics to initialize (camera needs time)
        print(" Waiting for ROS2 topics to initialize (LiDAR, Odom, Camera)...")
        time.sleep(5.0)  # Increased wait time for camera
        
        env = vec_env.envs[0].env
        
        # Get dimensions from environment (now includes camera features)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        print(f" State dimension: {state_dim} (includes {state_dim - 25} camera features)")
        print(f" Action dimension: {action_dim}")
        print(f"⚡ Max action: {max_action}")
        
        # Initialize SAC agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {device}")
        
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            max_action=max_action,
            discount=config.DISCOUNT,
            tau=config.TAU,
            actor_lr=config.LEARNING_RATE,
            critic_lr=config.LEARNING_RATE,
            alpha_lr=config.LEARNING_RATE,
        )
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(state_dim, action_dim, config.BUFFER_SIZE)
        
        # Training loop
        state = vec_env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        
        print(f"\n Starting training for {config.TOTAL_TIMESTEPS} timesteps...")
        print(" First steps will use random exploration")
        start_time = time.time()
        
        for t in range(config.TOTAL_TIMESTEPS):
            # Select action
            if t < config.LEARNING_STARTS:
                action = vec_env.action_space.sample()
                if t % 1000 == 0:
                    print(f" Step {t}: Using random exploration...")
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
            if t >= config.LEARNING_STARTS and replay_buffer.size > config.BATCH_SIZE:
                metrics = agent.update(replay_buffer, config.BATCH_SIZE)
                
                # Log training metrics
                if t % 1000 == 0:
                    print(f" Step {t}: "
                          f"Critic Loss: {metrics['critic_loss']:.3f}, "
                          f"Actor Loss: {metrics['actor_loss']:.3f}, "
                          f"Alpha: {metrics['alpha']:.3f}, "
                          f"Avg Reward: {metrics['average_reward']:.3f}")
            
            # Episode end
            if done[0]:
                print(f" Episode {episode_num}: "
                      f"Steps: {episode_length}, "
                      f"Reward: {episode_reward:.2f}")
                
                episode_reward = 0
                episode_length = 0
                episode_num += 1
                state = vec_env.reset()
            
            # Save model periodically
            if t % 5000 == 0 and t > 0:
                model_path = f"{config.MODEL_DIR}/sac_model_step_{t}.pth"
                agent.save(model_path)
                print(f" Model saved: {model_path}")
        
        # Save final model
        final_path = f"{config.MODEL_DIR}/sac_model_final.pth"
        agent.save(final_path)
        
        training_time = time.time() - start_time
        print(f"\n Training completed in {training_time:.1f} seconds")
        print(f" Final model saved: {final_path}")
        
    except Exception as e:
        print(f" Training failed: {e}")
        traceback.print_exc()
    finally:
        if vec_env is not None:
            vec_env.close()
        # Shutdown ROS2
        rclpy.shutdown()

def test_sac(model_path, episodes=10):
    """Test the trained SAC model with camera"""
    print(f"\n=== Testing SAC Model with MR LOGI Camera: {model_path} ===")
    
    # Initialize ROS2 FIRST
    rclpy.init()
    
    vec_env = None
    try:
        # Create environment
        vec_env = DummyVecEnv([make_env()])
        env = vec_env.envs[0].env
        
        # Get dimensions (now includes camera)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Load agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = SAC(state_dim, action_dim, device, max_action)
        agent.load(model_path)
        
        print(f" Model loaded: {model_path}")
        print(f" State dimension: {state_dim} (trained with camera)")
        
        # Wait for sensors to initialize
        print(" Waiting for sensors (LiDAR, Odom, Camera)...")
        time.sleep(3.0)
        
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
                action = agent.act(state[0], sample=False)
                state, reward, done, info = vec_env.step([action])
                episode_reward += reward[0]
                steps += 1
            
            success = steps < 300  # Didn't timeout
            if success:
                successes += 1
                
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            print(f" Episode {ep+1}: "
                  f"Steps: {steps}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Success: {success}")
        
        # Summary
        success_rate = (successes / episodes) * 100
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n Test Summary:")
        print(f" Success Rate: {success_rate:.1f}%")
        print(f" Average Reward: {avg_reward:.2f}")
        print(f" Average Length: {avg_length:.1f} steps")
        
    except Exception as e:
        print(f" Testing failed: {e}")
        traceback.print_exc()
    finally:
        if vec_env is not None:
            vec_env.close()
        # Shutdown ROS2
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom SAC Training with Camera')
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