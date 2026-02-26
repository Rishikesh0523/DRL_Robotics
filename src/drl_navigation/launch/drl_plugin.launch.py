import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_drl_navigation = get_package_share_directory('drl_navigation')
    pkg_ros_gz_sim_demos = get_package_share_directory('ros_gz_sim_demos')

    # Use plugin-enabled SDF
    world_path = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'env_1.sdf')
    # world_path = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'env_2.sdf')
    # world_path = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'env_3.sdf')
    # world_path = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'env_4.sdf')
    # world_path = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'env_5.sdf')

    # Launch Gazebo with plugin-enabled world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_path}'  # -v 4 for verbose plugin loading
        }.items()
    )

    # Static transform (still needed)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'chassis', 'base_footprint'],
        output='screen'
    )

    # DRL Training
    drl_training = ExecuteProcess(
        cmd=['python3', os.path.join(pkg_drl_navigation, 'scripts', 'train_drl_robot.py')],
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        static_tf,
        # drl_training  # Start manually for testing
    ])