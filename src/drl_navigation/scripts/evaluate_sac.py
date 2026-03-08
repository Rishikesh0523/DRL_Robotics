#!/usr/bin/env python3
"""
Quick evaluation script for trained PPO / SAC model
Reports success-rate, avg return, avg length + CSV log
"""

import argparse
import csv
import os
import time
from typing import List

import numpy as np
import rclpy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_nav_env2 import RobotNavigationEnv  # your env file


# ---------- helpers ----------
def make_env():
    """Wrapped env for evaluation."""
    def _init():
        env = RobotNavigationEnv()
        return env
    return _init


def evaluate(model_path: str, n_episodes: int, deterministic: bool = True) -> List[dict]:
    """Run episodes and collect stats."""
    rclpy.init()
    vec_env = DummyVecEnv([make_env()])

    # auto-detect algorithm
    if "sac" in os.path.basename(model_path).lower():
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    rows = []
    for ep in range(1, n_episodes + 1):
        obs = vec_env.reset()
        done, ep_len, ep_ret = False, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)
            ep_ret += reward.item()
            ep_len += 1
        # success = within 0.3 m (same condition as env)
        info = _info[0] if isinstance(_info, list) else _info
        success = info.get("is_success", False) if isinstance(info, dict) else (ep_ret > 90)
        rows.append(
            {
                "episode": ep,
                "length": ep_len,
                "return": ep_ret,
                "success": bool(success),
            }
        )
        print(f"Ep {ep:3d} │ len {ep_len:3d} │ ret {ep_ret:7.2f} │ {'✓' if success else '✗'}")

    vec_env.close()
    rclpy.shutdown()
    return rows


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .zip model")
    parser.add_argument("--ep", type=int, default=30, help="number of episodes")
    parser.add_argument("--csv", help="optional CSV output file")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Evaluating {args.model}  ({args.ep} episodes)\n")
    rows = evaluate(args.model, args.ep, deterministic=args.deterministic)

    # ---------- summary ----------
    successes = [r["success"] for r in rows]
    returns   = [r["return"]  for r in rows]
    lengths   = [r["length"]  for r in rows]

    print("\n========== RESULTS ==========")
    print(f"Success rate : {np.mean(successes)*100:5.1f} %")
    print(f"Avg return   : {np.mean(returns):7.2f}")
    print(f"Avg length   : {np.mean(lengths):7.1f}")
    print("=============================\n")

    # ---------- CSV ----------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "length", "return", "success"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Raw data saved → {args.csv}")


if __name__ == "__main__":
    main()