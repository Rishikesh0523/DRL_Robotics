#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Paths using PathJoinSubstitution
    urdf_path = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim_demos'), 
        'urdf/robot/mech_mobile.urdf.xacro'
    ])
    
    world_launch_path = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim_demos'),
        'launch/world.launch.py'
    ])
    
    # 1. Launch Gazebo world
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([world_launch_path]),
        launch_arguments={'on_exit_shutdown': 'true'}.items()
    )
    
    # 2. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                FindExecutable(name='xacro'), ' ', urdf_path
            ]),
            'use_sim_time': True
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # 3. Static TF transforms
    lidar_frame_fix = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_fix',
        arguments=['0', '0', '0', '-0.707', '0', '0', '0.707', 
                   'base_lidar_link', 'ros_gz_sim_demos/base_link/lidar_sensor']
    )
    
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    
    odom_to_base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base_link',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    )
    
    # 4. RViz (optional - comment out if not needed)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        parameters=[{'use_sim_time': True}]
    )
    
    # 5. Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'ros_gz_sim_demos',
            '-x', '2', '-y', '2', '-z', '0.0', '-Y', '0.0'
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # 6. ROS-Gazebo bridge
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        '/joint_states@sensor_msgs/msg/JointState]gz.msgs.Model',
        '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/lidar/points@sensor_msgs/msg/PointCloud2]gz.msgs.PointCloudPacked',
        '/rgbd/depth_image@sensor_msgs/msg/Image]gz.msgs.Image',
        '/rgbd/points@sensor_msgs/msg/PointCloud2]gz.msgs.PointCloudPacked'
    ]
    
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add actions
    ld.add_action(world_launch)
    ld.add_action(robot_state_publisher)
    ld.add_action(lidar_frame_fix)
    ld.add_action(map_to_odom)
    ld.add_action(odom_to_base_link)
    ld.add_action(rviz)
    ld.add_action(spawn_robot)
    ld.add_action(bridge)
    
    return ld