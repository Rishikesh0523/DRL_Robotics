# Robot Gazebo Simulation

A Gazebo simulation environment for testing and developing Deep Reinforcement Learning (DRL) navigation algorithms for the RoboCon robot, optimized for **ROS 2 Jazzy** on **Ubuntu 24.04**.

---

## Overview

This package provides a complete simulation environment for:

- **Testing** RoboCon robot navigation algorithms
- **Developing** DRL-based navigation solutions
- **Simulating** robot behavior in custom environments
- **Bridging** ROS 2 and Gazebo topics for seamless integration

---

## Prerequisites

### System Requirements

- **Ubuntu 24.04**
- **ROS 2 Jazzy Jalisco**
- **Gazebo Harmonic** (recommended) or **Gazebo Fortress**

### Required Dependencies

```bash
# ROS 2 Jazzy
sudo apt update
sudo apt install ros-jazzy-desktop

# Gazebo Sim for ROS 2 Jazzy
sudo apt install ros-jazzy-ros-gz

# Additional ROS 2 tools
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep2
sudo apt install python3-pip

# Optional: Additional useful tools
sudo apt install git wget curl build-essential
```

---

## Installation

### 1. Create Workspace Directory

```bash
mkdir -p ~/gz_ws/src
cd ~/gz_ws/src
```

### 2. Clone the Repository

```bash
git clone https://github.com/fuseai-fellowship/DRL-for-Mobile-Robot-Navigation.git
```

### 3. Initialize and Update rosdep

```bash
sudo rosdep init
rosdep update
```

### 4. Install Dependencies

```bash
cd ~/gz_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5. Build the Workspace

```bash
colcon build
```

### 6. Source the Workspace

```bash
source ~/gz_ws/install/setup.bash
```

> **Tip:** Add this to your `.bashrc` for permanent sourcing:
> ```bash
> echo "source ~/gz_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## Usage

### Launch the Simulation

```bash
ros2 launch ros_gz_sim_demos robocon_robot_test.launch.py
```

#### Alternative Launch Methods

```bash
# Launch with specific Gazebo version
ros2 launch ros_gz_sim_demos robocon_robot_test.launch.py gz_version:=harmonic

# Launch with verbose output
ros2 launch ros_gz_sim_demos gazebo_launch.xml --verbose
```

---

## What This Launches

- **Gazebo Sim** with a custom world environment
- **RoboCon robot model** with sensors and controllers
- **ROS-Gazebo bridge** for topic communication
- **Essential topic bridges** for DRL navigation

---

## Available Topics (ROS 2 Jazzy)

### Subscribed Topics

- `/cmd_vel` (`geometry_msgs/msg/Twist`) — Velocity commands for robot control

### Published Topics

- `/scan` (`sensor_msgs/msg/LaserScan`) — Laser scan data from robot
- `/odom` (`nav_msgs/msg/Odometry`) — Robot odometry information
- `/camera/image_raw` (`sensor_msgs/msg/Image`) — Camera feed
- `/imu` (`sensor_msgs/msg/Imu`) — IMU data (if configured)

---
