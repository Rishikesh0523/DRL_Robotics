#!/usr/bin/env python3
"""
Fixed Camera + LiDAR with RViz - Corrected
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, LogInfo
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='Starting Fixed Camera + LiDAR + RViz'),
        
        # Camera Node
        Node(
            package='ros_gz_hard_demos',
            executable='improved_camera_publisher',
            name='camera_publisher',
            output='screen',
            parameters=[{
                'camera_index': 2,
                'frame_rate': 30,
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
                'angle_compensate': True,
                'scan_mode': 'Standard'
            }]
        ),
        
        # Static Transform (CRITICAL for RViz)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='laser_to_base',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser']
        ),
        
        # RViz2 with fixed config
        ExecuteProcess(
            cmd=['rviz2', '-d', PathJoinSubstitution([
                FindPackageShare('ros_gz_hard_demos'), 
                'config', 
                '/home/Downloads/gz_ws/src/ros_gz_hard_demos/config/working_camera_lidar.rviz'  
            ])],
            output='screen'
        ),
    ])