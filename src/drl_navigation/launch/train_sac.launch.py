import sys
import os

# CRITICAL: Adds the script's installation directory to the Python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drl_navigation',
            executable='train_sac_robot.py',
            name='drl_trainer_node',
            output='screen',
            parameters=[
                {'use_sim_time': True} # This is the critical line
            ],
            arguments=[]
        ),
    ])