#!/usr/bin/env python3
import rclpy
from robot_nav_env import RobotNavigationEnv
import time

def main():
    rclpy.init()
    
    print("🧪 Testing Robot Navigation Environment with Gymnasium...")
    env = RobotNavigationEnv()
    
    print("\n1. Testing reset...")
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   First few values: {obs[:4]}")
    print(f"   Reset info: {info}")
    
    print("\n2. Testing action space...")
    print(f"   Action space: {env.action_space}")
    print(f"   Sample action: {env.action_space.sample()}")
    
    print("\n3. Testing random actions for 10 steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()  # Random action
        print(f"   Step {i}: Action = [{action[0]:.2f}, {action[1]:.2f}]")
        
        # Gymnasium step returns: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"     → Reward: {reward:7.3f}, Terminated: {terminated}, Truncated: {truncated}")
        print(f"     → Position: ({obs[0]:6.2f}, {obs[1]:6.2f})")
        print(f"     → Distance to goal: {obs[6]:6.2f}")
        
        total_reward += reward
        
        if terminated or truncated:
            print("     → Episode finished!")
            obs, info = env.reset()
            break
        
        time.sleep(0.5)  # Slow down for observation
    
    print(f"\n Test completed!")
    print(f"   Total reward: {total_reward:.2f}")
    
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()