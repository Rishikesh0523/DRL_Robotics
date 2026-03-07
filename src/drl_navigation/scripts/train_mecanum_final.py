#!/usr/bin/env python3
"""
Mecanum-wheel robot → goal with LiDAR  (SB3 + ROS 2)
----------------------------------------------------
- 12-D obs : goal unit vector + heading error + 10 lidar sectors
- 3-D act  : vx, vy, w  →  inverse-kinematics  →  4 wheel speeds (rad/s)
- reward   : (last_dist - dist) * 500  - 0.02
- topic    : /wheel_cmd  (std_msgs/Float64MultiArray)
"""
import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist          # still imported (not used)
from std_msgs.msg import Float64MultiArray   # 4 wheel commands
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import threading
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch


# ------------------------------------------------------------------------------
#  ROS 2 ENV
# ------------------------------------------------------------------------------
class MecanumNavEnv(Node, gym.Env):
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, 'mecanum_nav_env')

        # 3-D holonomic action  [vx, vy, w]
        self.action_space = spaces.Box(
            low=np.array([-0.4, -0.3, -0.5], dtype=np.float32),
            high=np.array([0.4, 0.3, 0.5], dtype=np.float32))

        # 12-D observation  [goal_unit_x, goal_unit_y, heading_err, 10 lidar]
        self.lidar_sectors = 10
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -math.pi] + [0.0] * self.lidar_sectors, dtype=np.float32),
            high=np.array([1.0, 1.0, math.pi] + [10.0] * self.lidar_sectors, dtype=np.float32))

        # ---------------- ROS ----------------
        self.wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_cmd', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/lidar', self.lidar_cb, 10)

        # --------------- state ---------------
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.goal = np.array([2.8, 2.8])
        self.lidar_ranges = None

        # -------------- episode --------------
        self.last_dist = np.inf
        self.step_cnt = 0
        self.max_steps = 150
        self.history = []

        # start ROS spinner
        threading.Thread(target=lambda: rclpy.spin(self), daemon=True).start()
        self.wait_until_lidar()

    # --------------------------------------------------------------------------
    #  ROS helpers
    # --------------------------------------------------------------------------
    def wait_until_lidar(self):
        print(' waiting for lidar ...')
        while self.lidar_ranges is None:
            time.sleep(0.1)
        print(' sensors ready')

    def odom_cb(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        self.history.append((self.pos[0], self.pos[1]))
        if len(self.history) > 20:
            self.history.pop(0)

    def lidar_cb(self, msg):
        self.lidar_ranges = list(msg.ranges)

    # --------------------------------------------------------------------------
    #  observation
    # --------------------------------------------------------------------------
    def obs(self):
        disp = self.goal - self.pos
        disp_norm = disp / (np.linalg.norm(disp) + 1e-8)
        angle_err = math.atan2(disp[1], disp[0]) - self.yaw
        angle_err = math.atan2(math.sin(angle_err), math.cos(angle_err))

        if self.lidar_ranges is None:
            lidar = [10.0] * self.lidar_sectors
        else:
            n = len(self.lidar_ranges)
            sector = n // self.lidar_sectors
            lidar = []
            for i in range(self.lidar_sectors):
                rng = self.lidar_ranges[i * sector:(i + 1) * sector]
                valid = [v for v in rng if 0.1 < v < 10.0]
                lidar.append(min(valid) if valid else 10.0)
        return np.concatenate([disp_norm, [angle_err], lidar]).astype(np.float32)

    # --------------------------------------------------------------------------
    #  reward
    # --------------------------------------------------------------------------
    def reward(self):
        d = np.linalg.norm(self.goal - self.pos)
        r = (self.last_dist - d) * 500.0 - 0.02   # big progress, tiny time cost
        self.last_dist = d
        return r

    def is_done(self):
        return (d := np.linalg.norm(self.goal - self.pos)) < 0.3 or self.step_cnt >= self.max_steps

    # --------------------------------------------------------------------------
    #  mecanum inverse-kinematics  →  4 wheel speeds (rad/s)
    # --------------------------------------------------------------------------
    def send_cmd(self, act):
        vx, vy, w = float(act[0]), float(act[1]), float(act[2])
        B = 0.15          # half track-width  (m)  ← measure your robot
        R = 0.05          # wheel radius      (m)  ← measure your wheel

        # tangential speeds (m/s)
        fl_mps = vx + vy + w * B
        fr_mps = vx - vy - w * B
        rl_mps = vx - vy + w * B
        rr_mps = vx + vy - w * B

        cmd = Float64MultiArray()
        cmd.data = [fl_mps / R, fr_mps / R, rl_mps / R, rr_mps / R]
        self.wheel_pub.publish(cmd)

    def stop(self):
        self.wheel_pub.publish(Float64MultiArray())

    # --------------------------------------------------------------------------
    #  gym interface
    # --------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.step_cnt = 0
        self.last_dist = np.linalg.norm(self.goal - self.pos)
        self.history.clear()
        self.stop()
        return self.obs(), {}

    def step(self, action):
        self.step_cnt += 1
        self.send_cmd(action)
        time.sleep(0.05)          # 20 Hz
        return self.obs(), self.reward(), self.is_done(), False, {}

    def close(self):
        self.stop()
        self.destroy_node()
        rclpy.shutdown()


# ------------------------------------------------------------------------------
#  TRAIN
# ------------------------------------------------------------------------------
def main():
    rclpy.init()
    set_random_seed(42)

    env = MecanumNavEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,               # no entropy bonus at start
        verbose=1,
        tensorboard_log="./tb_mecanum")

    # ---- clamp initial exploration ----
    with torch.no_grad():
        model.policy.log_std.param.data *= 0.1   # small initial std

    print("  START TRAINING")
    model.learn(total_timesteps=30_000, tb_log_name="phase1")
    print(" SAVE MODEL")
    model.save("mecanum_phase1")

    env.close()


if __name__ == '__main__':
    main()