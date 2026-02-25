#!/usr/bin/env python3
# test_training_setup.py

import sys
import os

print(" Testing training setup...")

# Test basic imports
try:
    import rclpy
    print(" rclpy imported successfully")
except ImportError as e:
    print(f" rclpy import failed: {e}")
    sys.exit(1)

try:
    import stable_baselines3
    print(" stable_baselines3 imported successfully")
except ImportError as e:
    print(f" stable_baselines3 import failed: {e}")
    sys.exit(1)

try:
    import gymnasium
    print(" gymnasium imported successfully")
except ImportError as e:
    print(f" gymnasium import failed: {e}")
    sys.exit(1)

# Test environment import
try:
    from drl_navigation.envs.robot_env import RobotGazeboEnv
    print(" RobotGazeboEnv imported successfully")
except ImportError as e:
    print(f" RobotGazeboEnv import failed: {e}")
    print(" Make sure your workspace is built and sourced")
    sys.exit(1)

print("\n All imports successful! Ready for training.")
