#!/usr/bin/env python3

import rclpy
import os
import time
from datetime import datetime

def main():
    rclpy.init()
    
    try:
        from drl_navigation.envs.robot_env import RobotGazeboEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        import torch
        
        print(" Starting PPO Training for Gazebo Robot")
        print("=" * 50)
        
        # Create environment
        print("Creating environment...")
        env = RobotGazeboEnv()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        print(" Environment created successfully!")
        print(f" Observation space: {env.observation_space}")
        print(f" Action space: {env.action_space}")
        
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.expanduser(f"~/Downloads/gz_ws/models/ppo_robot_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f" Models will be saved to: {model_dir}")
        
        # Custom callback for training progress
        class TrainingProgressCallback(BaseCallback):
            def __init__(self, check_freq=1000, verbose=1):
                super(TrainingProgressCallback, self).__init__(verbose)
                self.check_freq = check_freq
                
            def _on_step(self):
                if self.n_calls % self.check_freq == 0:
                    if len(self.model.ep_info_buffer) > 0:
                        latest_episode = self.model.ep_info_buffer[-1]
                        if 'r' in latest_episode and 'l' in latest_episode:
                            reward = latest_episode['r']
                            length = latest_episode['l']
                            print(f" Step {self.n_calls}: Episode Reward: {reward:.2f}, Length: {length}")
                return True
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # Save every 10,000 steps
            save_path=model_dir,
            name_prefix="robot_ppo"
        )
        
        progress_callback = TrainingProgressCallback(check_freq=5000)
        
        # PPO Configuration
        print("\n Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.expanduser("~/Downloads/gz_ws/tb_logs/"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256]
            )
        )
        
        print(" PPO model initialized!")
        print(f" Total parameters: {sum(p.numel() for p in model.policy.parameters())}")
        
        # Start training
        print("\n Starting training...")
        print("   Press Ctrl+C to stop training early")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=100000,  # Start with 100k timesteps
                callback=[checkpoint_callback, progress_callback],
                tb_log_name=f"ppo_robot_{timestamp}",
                reset_num_timesteps=False
            )
        except KeyboardInterrupt:
            print("\n⏹Training interrupted by user")
        except Exception as e:
            print(f"\n Training error: {e}")
            import traceback
            traceback.print_exc()
        
        # Save final model
        training_time = time.time() - start_time
        model.save(os.path.join(model_dir, "ppo_robot_final"))
        
        print("=" * 50)
        print(f" Training completed!")
        print(f" Total training time: {training_time/60:.2f} minutes")
        print(f" Final model saved to: {os.path.join(model_dir, 'ppo_robot_final')}")
        print(f" TensorBoard logs: ~/Downloads/gz_ws/tb_logs/ppo_robot_{timestamp}")
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("\n Make sure to install required packages:")
        print("   pip install stable-baselines3 gymnasium torch")
        
    except Exception as e:
        print(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()