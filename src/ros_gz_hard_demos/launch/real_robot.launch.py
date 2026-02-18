# ~/Downloads/gz_ws/src/ros_gz_hard_demos/launch/real_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Jazzy Camera Publisher
        Node(
            package='ros_gz_hard_demos',
            executable='jazzy_camera_publisher',
            name='jazzy_camera',
            output='screen',
            parameters=[{
                'device': '/dev/video2',
            }]
        ),
        
        # Hardware DRL Environment
        Node(
            package='ros_gz_hard_demos',
            executable='hardware_drl_env',
            name='hardware_drl_env',
            output='screen',
            parameters=[{
                'real_world_safety': True,
                'max_linear_speed': 0.3,
                'max_angular_speed': 0.8,
            }]
        ),
    ])