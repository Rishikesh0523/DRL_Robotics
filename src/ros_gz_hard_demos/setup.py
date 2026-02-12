from setuptools import setup
import os
from glob import glob

package_name = 'ros_gz_hard_demos'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name, 
        f'{package_name}.nodes', 
        f'{package_name}.utils'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sandeep',
    maintainer_email='sandeep@pc',
    description='Hardware integration for DRL navigation with ROS 2 Jazzy',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'improved_camera_publisher = ros_gz_hard_demos.nodes.improved_camera_publisher:main',
            'simple_lidar_processor = ros_gz_hard_demos.nodes.simple_lidar_processor:main',
            'optimized_camera_publisher = ros_gz_hard_demos.nodes.jazzy_camera_publisher:main',
            'jazzy_camera_publisher = ros_gz_hard_demos.nodes.jazzy_camera_publisher:main',
            # FIX: Change 'hardware_drl_env' to 'drl_env' to match your actual filename
            'drl_env = ros_gz_hard_demos.nodes.drl_env:main',
        ],
    },
)