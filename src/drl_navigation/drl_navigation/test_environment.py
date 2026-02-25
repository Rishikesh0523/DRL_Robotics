#!/usr/bin/env python3

import rclpy
from drl_navigation.envs.robot_env import RobotGazeboEnv

def main():
    rclpy.init()
    
    try:
        print("Creating RobotGazeboEnv...")
        env = RobotGazeboEnv()
        
        print("=== Environment Details ===")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Reset successful! Observation type: {type(obs)}, length: {len(obs)}")
        print(f"First 5 observation values: {obs[:5]}")
        
        # Test a few steps
        print("\nTesting steps with random actions...")
        for i in range(5):
            action = env.action_space.sample()
            print(f"Step {i+1}: Action = {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print("  Episode ended!")
                obs, info = env.reset()
        
        print("\n Environment test completed successfully!")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()