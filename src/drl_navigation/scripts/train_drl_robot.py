#!/usr/bin/env python3
"""
PPO-based Robot Navigation Training Script with LiDAR Integration

This script trains a robot navigation policy using Proximal Policy Optimization (PPO)
with LiDAR sensor data for obstacle avoidance. The robot learns to navigate to a goal
position while avoiding obstacles in a simulated Gazebo environment.

Author: Robot Navigation Team
Date: 2025
License: MIT

Requirements:
    - ROS 2 (Jazzy)
    - Gazebo simulation running
    - LiDAR data published to /lidar topic
    - stable-baselines3
    - gymnasium

Usage:
    # Training:
    python3 train_ppo_robot.py
    
    # Testing (uncomment test_trained_model() call at the end):
    python3 train_ppo_robot.py
"""

import os
import time
import traceback
from typing import Optional, Tuple

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from robot_nav_env import RobotNavigationEnv


# ========================= Configuration Constants =========================

# Training Configuration
TOTAL_TIMESTEPS = 50000
PHASE1_TIMESTEPS = 10000
PHASE2_TIMESTEPS = 20000
PHASE3_TIMESTEPS = 20000

# Model Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 64
N_EPOCHS = 1000
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
NETWORK_ARCH = [128, 128]

# File Paths
TRAINING_LOGS_DIR = "./training_logs/"
TENSORBOARD_LOGS_DIR = "./tensorboard_logs_lidar/"
FINAL_MODEL_NAME = "trained_robot_model_lidar"
INTERRUPTED_MODEL_NAME = "trained_robot_model_lidar_interrupted"

# Environment Settings
GOAL_POSITION = (2.8, 2.8)
LIDAR_SECTORS = 10
MAX_STEPS_PER_EPISODE = 300

# Test Configuration
TEST_EPISODES = 3
TEST_MAX_STEPS = 200


# ========================= Helper Functions =========================

def create_directories() -> None:
    """
    Create necessary directories for training logs and model checkpoints.
    
    Creates:
        - training_logs/: For model checkpoints
        - tensorboard_logs_lidar/: For TensorBoard logs
    """
    os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
    print(f"✓ Created directory: {TRAINING_LOGS_DIR}")


def print_training_header(env: DummyVecEnv) -> None:
    """
    Print detailed training configuration and important information.
    
    Args:
        env: The vectorized environment containing observation and action spaces
    """
    print("=" * 70)
    print(" Starting DRL Training with LiDAR Integration")
    print("=" * 70)
    print(f"\n Goal: Navigate to position {GOAL_POSITION} while avoiding obstacles\n")
    
    print("  Training Configuration:")
    print(f"   • Observation space: {env.observation_space.shape}")
    print(f"   • Action space: {env.action_space}")
    print(f"   • LiDAR sectors: {LIDAR_SECTORS}")
    print(f"   • Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"   • Total training steps: {TOTAL_TIMESTEPS:,}")
    
    print("\n  IMPORTANT:")
    print("   • Stop any manual control (teleop_twist_keyboard)")
    print("   • Ensure Gazebo simulation is running")
    print("   • Verify LiDAR data is being published to /lidar")
    print("=" * 70 + "\n")


def create_ppo_model(env: DummyVecEnv) -> PPO:
    """
    Create and configure the PPO model with optimized hyperparameters.
    
    Args:
        env: The vectorized environment for training
        
    Returns:
        Configured PPO model ready for training
        
    Hyperparameters:
        - learning_rate: Step size for gradient descent (3e-4)
        - n_steps: Number of steps to collect before update (1024)
        - batch_size: Minibatch size for training (64)
        - n_epochs: Number of epochs when optimizing (10)
        - gamma: Discount factor (0.99)
        - gae_lambda: Factor for trade-off of bias vs variance (0.95)
        - clip_range: Clipping parameter for PPO (0.2)
        - ent_coef: Entropy coefficient for exploration (0.01)
        - vf_coef: Value function coefficient (0.5)
        - max_grad_norm: Maximum gradient norm (0.5)
    """
    print(" Creating PPO model with LiDAR-optimized hyperparameters...")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY_COEF,
        vf_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(
            net_arch=NETWORK_ARCH  # Two hidden layers with 128 neurons each
        ),
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGS_DIR
    )


#
#   # NEW  (drop-in)
#   from stable_baselines3 import SAC
#    model = SAC(
#          policy="MlpPolicy",
#          env=env,
#          learning_rate=3e-4,
#          buffer_size=200_000,
#          learning_starts=5_000,
#          batch_size=256,
#          tau=0.005,
#          gamma=0.98,
#          ent_coef="auto",
#          policy_kwargs=dict(net_arch=[512, 512, 256]),
#          verbose=1,
#          tensorboard_log=TENSORBOARD_LOGS_DIR
#    )


  
    print("✓ PPO model created successfully!\n")
    return model


def train_phase(
    model: PPO,
    phase_num: int,
    timesteps: int,
    checkpoint_callback: CheckpointCallback,
    reset_timesteps: bool = False
) -> None:
    """
    Train the model for a specific phase.
    
    Args:
        model: The PPO model to train
        phase_num: Phase number (1, 2, or 3)
        timesteps: Number of timesteps for this phase
        checkpoint_callback: Callback for saving checkpoints
        reset_timesteps: Whether to reset timestep counter
    """
    phase_descriptions = {
        1: "Basic navigation",
        2: "Refined obstacle avoidance",
        3: "Final training"
    }
    
    print(f"\n{'='*70}")
    print(f" Phase {phase_num}: {phase_descriptions[phase_num]} ({timesteps:,} steps)")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"lidar_phase{phase_num}",
        reset_num_timesteps=reset_timesteps
    )
    
    # Save phase checkpoint
    phase_model_path = f"{TRAINING_LOGS_DIR}ppo_robot_lidar_phase{phase_num}"
    model.save(phase_model_path)
    print(f"\n✓ Phase {phase_num} completed! Model saved to: {phase_model_path}")


# ========================= Main Training Function =========================

def main() -> None:
    """
    Main training function that orchestrates the entire training process.
    
    Training is divided into three phases:
        1. Phase 1 (10k steps): Basic navigation learning
        2. Phase 2 (20k steps): Refined obstacle avoidance
        3. Phase 3 (20k steps): Final policy refinement
    
    The trained model is saved at each phase and at the end of training.
    Supports graceful interruption via Ctrl+C.
    """
    # Initialize ROS 2
    print(" Initializing ROS 2...")
    rclpy.init()
    print("✓ ROS 2 initialized\n")
    
    try:
        # Create environment
        print(" Creating robot navigation environment...")
        env = RobotNavigationEnv()
        env = Monitor(env)  # Wrap for episode statistics
        env = DummyVecEnv([lambda: env])  # Vectorize for stable-baselines3
        print("✓ Environment created successfully!\n")
        
        # Create directories
        create_directories()
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,
            save_path=TRAINING_LOGS_DIR,
            name_prefix="ppo_robot_lidar"
        )
        
        # Create PPO model
        model = create_ppo_model(env)
        
        # Print training information
        print_training_header(env)
        
        # Display training start message
        print(" Starting training...")
        print("   → The robot will learn to navigate with obstacle avoidance!")
        print("   → Watch the simulation to see learning progress!")
        print("   → Press Ctrl+C to stop training gracefully.\n")
        
        # Phase 1: Initial training
        train_phase(
            model=model,
            phase_num=1,
            timesteps=PHASE1_TIMESTEPS,
            checkpoint_callback=checkpoint_callback,
            reset_timesteps=True
        )
        
        # Phase 2: Refined training
        train_phase(
            model=model,
            phase_num=2,
            timesteps=PHASE2_TIMESTEPS,
            checkpoint_callback=checkpoint_callback,
            reset_timesteps=False
        )
        
        # Phase 3: Final training
        train_phase(
            model=model,
            phase_num=3,
            timesteps=PHASE3_TIMESTEPS,
            checkpoint_callback=checkpoint_callback,
            reset_timesteps=False
        )
        
        # Save final model
        model.save(FINAL_MODEL_NAME)
        
        # Print success message
        print("\n" + "="*70)
        print(" Training completed successfully!")
        print("="*70)
        print("\n Models saved:")
        print(f"   • {FINAL_MODEL_NAME} (final model)")
        print(f"   • {TRAINING_LOGS_DIR}ppo_robot_lidar_phase1")
        print(f"   • {TRAINING_LOGS_DIR}ppo_robot_lidar_phase2")
        print(f"   • {TRAINING_LOGS_DIR}ppo_robot_lidar_phase3")
        print("\n To test the model, uncomment test_trained_model() at the end of this script.\n")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        print(" Saving current model...")
        model.save(INTERRUPTED_MODEL_NAME)
        print(f"✓ Model saved as: '{INTERRUPTED_MODEL_NAME}'\n")
        
    except Exception as e:
        print(f"\n Training error: {e}")
        print("\n Full traceback:")
        traceback.print_exc()
        
    finally:
        # Clean up resources
        print("\n Cleaning up...")
        env.close()
        rclpy.shutdown()
        print("✓ Cleanup complete.\n")


# ========================= Testing Function =========================

def test_trained_model(model_path: Optional[str] = None) -> None:
    """
    Test the trained model by running multiple episodes and reporting performance.
    
    Args:
        model_path: Path to the trained model. If None, uses FINAL_MODEL_NAME
        
    The function will:
        1. Load the trained model
        2. Run TEST_EPISODES episodes
        3. Report steps taken and reward for each episode
        4. Calculate and display average performance
    """
    print("\n" + "="*70)
    print(" Testing Trained Model")
    print("="*70 + "\n")
    
    # Initialize ROS 2
    print("Initializing ROS 2...")
    rclpy.init()
    
    try:
        # Create environment
        print(" Creating test environment...")
        env = RobotNavigationEnv()
        print("✓ Environment created\n")
        
        # Load trained model
        model_path = model_path or FINAL_MODEL_NAME
        print(f" Loading model from: {model_path}")
        
        try:
            model = PPO.load(model_path)
            print("✓ Model loaded successfully!\n")
        except FileNotFoundError:
            print(f" Error: No trained model found at '{model_path}'")
            print("Please train the model first by running: python3 train_ppo_robot.py\n")
            return
        
        # Test the model
        total_reward = 0.0
        successful_episodes = 0
        
        print(f" Running {TEST_EPISODES} test episodes...\n")
        
        for episode in range(TEST_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            
            print(f" Test Episode {episode + 1}/{TEST_EPISODES}")
            
            while not done and steps < TEST_MAX_STEPS:
                # Use deterministic policy for testing
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                if done:
                    if info.get('is_success', False):
                        successful_episodes += 1
                    break
            
            total_reward += episode_reward
            
            # Print episode results
            status = "✓ Success" if info.get('is_success', False) else "✗ Failed"
            print(f"   {status} | Steps: {steps:3d} | Reward: {episode_reward:7.2f}\n")
        
        # Print summary statistics
        avg_reward = total_reward / TEST_EPISODES
        success_rate = (successful_episodes / TEST_EPISODES) * 100
        
        print("="*70)
        print(" Test Results Summary")
        print("="*70)
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Success Rate: {success_rate:.1f}% ({successful_episodes}/{TEST_EPISODES})")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n Testing error: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        print(" Cleaning up test environment...")
        env.close()
        rclpy.shutdown()
        print("✓ Test complete.\n")


# ========================= Entry Point =========================

if __name__ == "__main__":
    main()
    
