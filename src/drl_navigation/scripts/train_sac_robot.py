#!/usr/bin/env python3
"""
SAC-based Robot Navigation Training Script - OPTIMIZED VERSION
Includes support for loading previous weights to continue training.
Compatible with RobotNavigationEnv (LiDAR-based 16D observation space)
"""

import rclpy
import argparse

import os
import time
import traceback
from typing import Optional
import numpy as np
import argparse # Import argparse for command-line arguments

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# -------------- environment --------------
from robot_nav_env2 import RobotNavigationEnv   # <-- your final env file

# ========================= CONFIG =========================
TOTAL_TIMESTEPS   = 500_000         # Reduced for faster testing
PHASE1_STEPS      = 50_000
PHASE2_STEPS      = 200_000
PHASE3_STEPS      = 250_000         # Adjusted steps to match total

LEARNING_RATE     = 3e-4
BUFFER_SIZE       = 500_000
BATCH_SIZE        = 256
TAU               = 0.005
GAMMA             = 0.99
ENT_COEF          = "auto"
LEARNING_STARTS   = 5_000
TRAIN_FREQ        = 1
GRADIENT_STEPS    = 1
POLICY_KWARGS     = dict(net_arch=[256, 256])

# Dirs
TB_LOG_DIR        = "./tensorboard_logs_sac/"
CKPT_DIR          = "./training_logs/sac/"
FINAL_MODEL       = "trained_sac_robot"

# ========================= CALLBACKS =========================
class DebugCallback(BaseCallback):
    """ Custom callback for tracking training progress and debugging """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.n_calls % 500 == 0:
            if len(self.model.ep_info_buffer) > 0:
                recent_episodes = [ep for ep in self.model.ep_info_buffer if 'r' in ep and 'l' in ep]
                
                if recent_episodes:
                    ep_lens = [ep['l'] for ep in recent_episodes]
                    ep_rewards = [ep['r'] for ep in recent_episodes]
                    
                    mean_reward = np.mean(ep_rewards)
                    mean_length = np.mean(ep_lens)
                    
                    self.logger.record("debug/mean_episode_length", mean_length)
                    self.logger.record("debug/mean_episode_reward", mean_reward)
                    self.logger.record("debug/episode_count", len(recent_episodes))
                    
                    print(f" Step {self.n_calls}: "
                          f"Mean Reward: {mean_reward:.2f}, "
                          f"Mean Length: {mean_length:.1f}, "
                          f"Episodes: {len(recent_episodes)}")
                    
                    if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 1000:
                        buffer_size = self.model.replay_buffer.size()
                        self.logger.record("debug/replay_buffer_size", buffer_size)
                        # print(f"   Replay Buffer: {buffer_size} samples")
        
        return True
    
class SuccessCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Note: You need to ensure your Monitor wrapper captures 'is_success' in info
        # This often requires manual logging within the env's step() info return.
        return True

# ========================= UTILS =========================
def make_env():
    def _init():
        env = RobotNavigationEnv()
        env = Monitor(env)  # episode stats
        return env
    return _init

def create_dirs():
    """Create necessary directories"""
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f" Created directories: {TB_LOG_DIR}, {CKPT_DIR}")

def quick_test_model(model, episodes=1):
    """ Quick test without reloading model between phases """
    print(f"\n Quick Test ({episodes} episodes)")
    env = model.get_env()
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 50:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
        print(f"   Test Ep {ep+1}: Steps={steps}, Reward={total_reward:.2f}")
    
    return total_reward






def main(args=None):
    # 1. Initialize rclpy and let it strip out ROS-specific arguments
    #    This is the critical step that removes the unrecognized --ros-args
    rclpy.init(args=args) 

    # 2. Get the arguments remaining after rclpy.init()
    #    rclpy.init() returns the non-ROS arguments, which should only be your script's args
    parser = argparse.ArgumentParser(description='SAC Robot Training.')
    parser.add_argument('--model', type=str, default='sac_robot_interrupted.zip', help='Model file name to load or save.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode.')

    # Use rclpy.init()'s output for parsing
    args_known, args_unknown = parser.parse_known_args()

# ========================= TRAIN (MODIFIED) =========================
def train(model_path: Optional[str] = None):
    print("===  SAC Robot Navigation Training ===")
    if model_path:
        print(f"Loading previous weights from: {model_path}")
        is_continuing = True
    else:
        print("Creating fresh model - no previous weights loaded")
        is_continuing = False
        
    rclpy.init()
    create_dirs()

    vec_env = None
    model = None

    try:
        # Create environment
        print(" Creating environment...")
        vec_env = DummyVecEnv([make_env()])
        success_cb = SuccessCallback(vec_env)

        # Setup callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=2000,
            save_path=CKPT_DIR,
            name_prefix="sac_robot",
            verbose=1
        )
        debug_cb = DebugCallback()

        # CREATE OR LOAD MODEL
        if is_continuing:
            print(f" Loading existing SAC model from {model_path}...")
            # NOTE: We load the full model, which includes the optimizers and environment.
            model = SAC.load(
                path=model_path,
                env=vec_env,
                custom_objects=None, 
                verbose=1
            )
        else:
            print(" Creating new SAC model...")
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                tau=TAU,
                gamma=GAMMA,
                ent_coef=ENT_COEF,
                learning_starts=LEARNING_STARTS,
                train_freq=TRAIN_FREQ,
                gradient_steps=GRADIENT_STEPS,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log=TB_LOG_DIR,
                seed=42
            )

        print(" Model created/loaded successfully!")
        print(f" TensorBoard logs: {TB_LOG_DIR}")
        print(f" Checkpoints: {CKPT_DIR}")

        # --- TRAINING PHASES (Unified) ---
        
        # Determine total steps to run *after* loading (if continuing)
        # If continuing, we run the full 500k steps *again* until performance stabilizes.
        steps_to_run = [PHASE1_STEPS, PHASE2_STEPS, PHASE3_STEPS]
        phase_names = ["Basic Movement", "Goal-Oriented Navigation", "Policy Refinement"]
        
        def run_phase(steps, phase, description):
            print(f"\n{'='*50}")
            print(f" Phase {phase}: {description}")
            print(f" Training for {steps} steps...")
            print(f"{'='*50}")
            
            start_time = time.time()
            model.learn(
                total_timesteps=steps,
                callback=[checkpoint_cb, debug_cb, success_cb],
                tb_log_name=f"sac_phase{phase}",
                # CRITICAL: Only reset timesteps on a fresh start (Phase 1)
                reset_num_timesteps=(phase == 1 and not is_continuing)
            )
            phase_time = time.time() - start_time
            
            phase_model_path = f"{CKPT_DIR}sac_robot_phase{phase}"
            model.save(phase_model_path)
            print(f" Saved phase {phase} model: {phase_model_path}")
            print(f"⏱  Phase {phase} completed in {phase_time:.1f} seconds")
            
            if phase < len(steps_to_run):
                quick_test_model(model, episodes=1)
            
            return phase_time

        # 3-PHASE TRAINING EXECUTION
        print("\n Starting 3-phase training strategy...")
        for i, steps in enumerate(steps_to_run):
            run_phase(steps, i + 1, phase_names[i])


        # FINAL SAVE
        print(f"\n Saving final model: {FINAL_MODEL}.zip")
        model.save(FINAL_MODEL)
        
        print(f"\n Training completed successfully!")
        print(f" Final model saved: {FINAL_MODEL}.zip")

    except KeyboardInterrupt:
        print("\n⏹  Training interrupted by user")
        if model is not None:
            print(" Saving interrupted model...")
            model.save("sac_robot_interrupted")
            print(" Saved: sac_robot_interrupted.zip")
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        traceback.print_exc()
        if model is not None:
            print(" Attempting to save current model...")
            model.save("sac_robot_error_backup")
            print(" Saved: sac_robot_error_backup.zip")
    finally:
        if vec_env is not None:
            print(" Closing environment...")
            vec_env.close()
        print(" Shutting down ROS...")
        rclpy.shutdown()
        print(" Cleanup completed")

# ========================= MAIN EXECUTION =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAC DRL Trainer for Robot Navigation")
    
    # Argument for loading a model (optional)
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to a model file (.zip) to load and continue training (e.g., sac_robot_interrupted.zip).')
    
    # Add a mode if you want to switch between train/test/etc.
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test'], 
                        help='Choose mode: "train" or "test" (requires test_sac_model function).')

    args, _ = parser.parse_known_args()
    
    if args.mode == 'train':
        # Pass the model argument to the train function
        train(model_path=args.model) 
    elif args.mode == 'test':
        # You would need to ensure the test_sac_model function is present and imported/defined
        # test_sac_model(model_path=args.model) 
        print("Mode 'test' selected. Please use a separate script or define test_sac_model()")
    else:
        print(f"Invalid mode selected: {args.mode}")