#!/usr/bin/env python3
"""
LiDAR System with RViz Visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, LogInfo

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='🚀 Starting LiDAR System with RViz...'),
        
        # LiDAR Driver
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_driver',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'frame_id': 'laser',
            }]
        ),
        
        # LiDAR Processor
        Node(
            package='ros_gz_hard_demos',
            executable='simple_lidar_processor',
            name='simple_lidar_processor',
            output='screen'
        ),
        
        # RViz2 with pre-saved configuration
        ExecuteProcess(
            cmd=['rviz2', '-d', 'src/ros_gz_hard_demos/config/lidar.rviz'],
            output='screen'
        ),
        
        LogInfo(msg='✅ LiDAR + RViz System Ready!'),
    ])