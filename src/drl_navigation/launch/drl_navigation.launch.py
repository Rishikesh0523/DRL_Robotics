import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to your simulation launch file
    sim_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim_demos'),
        'launch',
       # 'diff_drive.launch.py'  # Adjust based on your actual launch file
          'gazebo_launch.xml'
    )
    
    # Path to wrapper script
    drl_wrapper_script = os.path.join(
        get_package_share_directory('drl_navigation'),
        'launch',
        'drl_wrapper.sh'
    )
    
    return LaunchDescription([
        # Launch the simulation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch_file)
        ),
        
        # Launch DRL navigation node using wrapper
        ExecuteProcess(
            cmd=[drl_wrapper_script],
            output='screen',
            shell=True
        )
    ])