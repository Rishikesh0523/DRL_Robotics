#!/usr/bin/env python3

import rclpy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from drl_navigation.envs.robot_env import RobotGazeboEnv

def main():
    # Initialize ROS 2
    rclpy.init()
    
    # Create environment
    env = RobotGazeboEnv()
    env = DummyVecEnv([lambda: env])
    
    # Set up callback to save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="robot_ppo"
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Train the model
    try:
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    
    # Save the final model
    model.save("ppo_robot_navigation")
    print("Model saved as ppo_robot_navigation")
    
    # Close environment
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()