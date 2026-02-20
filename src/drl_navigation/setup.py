from setuptools import setup
import os
from glob import glob

package_name = 'drl_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    
    # --- CRITICAL: Explicitly list the files for installation ---
    (os.path.join('lib', package_name), [
        'scripts/train_sac_robot.py', 
        'scripts/robot_nav_env2.py'
    ]),
    # -----------------------------------------------------------
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sandeep',
    maintainer_email='079bel078.sandeep@pcampus.edu.np',
    description='DRL-based autonomous navigation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drl_navigation_node = drl_navigation.drl_navigation_node:main',
            'train_drl = drl_navigation.train_drl:main',  # updated path
        ],
    },
)