#!/usr/bin/env python3
"""
Final Camera + LiDAR System with RViz Visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, LogInfo
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='🚀 Starting Final Camera + LiDAR System with RViz...'),
        
        # Robust Camera Publisher (use logitech_c270_publisher for no JPEG corruption)
        Node(
            package='ros_gz_hard_demos',
            executable='improved_camera_publisher',  # Use this for no JPEG corruption
            name='camera_publisher',
            output='screen',
            parameters=[{
                'camera_index': 0,
                'frame_rate': 15,
                'width': 640,
                'height': 480,
            }]
        ),
        
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
        
        # RViz2 with pre-configured setup
        ExecuteProcess(
            cmd=['rviz2', '-d', PathJoinSubstitution([
                FindPackageShare('ros_gz_hard_demos'), 
                'config', 
                'camera_lidar.rviz'
            ])],
            output='screen'
        ),
        
        LogInfo(msg='✅ Camera + LiDAR + RViz System Ready!'),
        LogInfo(msg='📷 Camera: /hardware/camera/processed'),
        LogInfo(msg='📡 LiDAR: /scan (raw), /drl/processed_lidar (processed)'),
        LogInfo(msg='🚨 Collision: /drl/collision_warning'),
        LogInfo(msg='🖥️  RViz2: Pre-configured visualization'),
    ])