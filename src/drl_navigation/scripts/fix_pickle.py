#!/usr/bin/env python3
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from robot_nav_env2 import RobotNavigationEnv

rclpy.init()
env = DummyVecEnv([lambda: Monitor(RobotNavigationEnv())])
print("Loading old zip…")
model = SAC.load("trained_robot_model", env=env)   # compat load
print("Re-saving with current libraries…")
model.save("trained_robot_model_fixed")
print("Done – trained_robot_model_fixed.zip created")
rclpy.shutdown()