#!/usr/bin/env python3
import rclpy
from robot_nav_env import RobotNavigationEnv
from stable_baselines3 import PPO
import time

def main():
    rclpy.init()
    
    # Load environment
    env = RobotNavigationEnv()
    
    # Load trained model
    try:
        model = PPO.load("trained_robot_model")
        print(" Loaded trained model successfully!")
    except FileNotFoundError:
        print(" No trained model found. Please train first using:")
        print("   python3 train_drl_robot.py")
        env.close()
        rclpy.shutdown()
        return
    except Exception as e:
        print(f" Error loading model: {e}")
        env.close()
        rclpy.shutdown()
        return
    
    print(" Testing trained model for 3 episodes...")
    print(" The trained model will control the robot autonomously.")
    
    success_count = 0
    total_steps = 0
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        print(f"Start: ({obs[0]:.2f}, {obs[1]:.2f}), Goal: ({obs[4]:.2f}, {obs[5]:.2f})")
        
        while not done and steps < 300:  # Reduced max steps for testing
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if steps % 30 == 0:
                print(f"   Step {steps}: Pos({obs[0]:6.2f}, {obs[1]:6.2f}), "
                      f"Dist: {obs[6]:5.2f}, Reward: {reward:6.2f}")
        
        if terminated and obs[6] < 0.5:  # Success condition
            success_count += 1
            print(f" SUCCESS! Reached goal in {steps} steps.")
        else:
            print(f" Episode ended: Steps={steps}, Total Reward={total_reward:.2f}")
        
        total_steps += steps
        
        # Small pause between episodes
        time.sleep(1.0)
    
    print(f"\n Test Summary:")
    print(f"   Success rate: {success_count}/3 episodes")
    print(f"   Average steps per episode: {total_steps/3:.1f}")
    
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()