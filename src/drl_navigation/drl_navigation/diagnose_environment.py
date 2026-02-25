#!/usr/bin/env python3

import rclpy
from drl_navigation.envs.robot_env import RobotGazeboEnv

def diagnose_environment():
    rclpy.init()
    
    try:
        print(" Diagnosing RobotGazeboEnv...")
        env = RobotGazeboEnv()
        
        print("\n Environment Properties:")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test if it's continuous or discrete
        if hasattr(env.action_space, 'n'):
            print("   Action type: Discrete")
        else:
            print("   Action type: Continuous")
            print(f"   Action low: {env.action_space.low}")
            print(f"   Action high: {env.action_space.high}")
        
        # Test reset
        print("\n Testing reset...")
        obs, info = env.reset()
        print(f"   Reset successful!")
        print(f"   Observation type: {type(obs)}")
        print(f"   Observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
        print(f"   Observation dtype: {obs.dtype if hasattr(obs, 'dtype') else type(obs[0])}")
        
        # Test step
        print("\n Testing step...")
        action = env.action_space.sample()
        print(f"   Sample action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step successful!")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        print(f"   Info keys: {list(info.keys()) if info else 'None'}")
        
        print("\n Environment diagnosis completed!")
        
    except Exception as e:
        print(f" Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    diagnose_environment()