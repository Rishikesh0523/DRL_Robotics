#!/usr/bin/env python3
"""
LiDAR Launch File for ROS 2 Jazzy Hardware DRL
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'lidar_min_range',
            default_value='0.1',
            description='Minimum LiDAR range in meters'
        ),
        DeclareLaunchArgument(
            'lidar_max_range',
            default_value='10.0',
            description='Maximum LiDAR range in meters'
        ),
        DeclareLaunchArgument(
            'collision_threshold', 
            default_value='0.4',
            description='Collision detection threshold in meters'
        ),
        
        LogInfo(msg=' Starting LiDAR-Enabled DRL Environment...'),
        
        # FIX: Changed from 'hardware_drl_env' to 'drl_env'
        Node(
            package='ros_gz_hard_demos',
            executable='drl_env',
            name='drl_env',
            output='screen',
            parameters=[{
                'lidar_min_range': LaunchConfiguration('lidar_min_range'),
                'lidar_max_range': LaunchConfiguration('lidar_max_range'),
                'collision_threshold': LaunchConfiguration('collision_threshold'),
                'use_lidar': True,
            }]
        ),
        
        LogInfo(msg=' LiDAR DRL Environment launched successfully!'),
        LogInfo(msg=' Listening to: /scan (LiDAR), /odom, /hardware/camera/processed'),
    ])