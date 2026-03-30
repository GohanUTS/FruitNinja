#!/usr/bin/env python3
"""
movie.launch.py

Brings up the UR3e robot driver + MoveIt 2 against a REAL robot via Polyscope.

Prerequisites on the UR pendant (Polyscope):
  1. Install the "External Control" URCap
  2. Add an External Control node to your program
  3. Set the Host IP to THIS machine's IP (e.g. 192.168.56.1)
  4. Hit PLAY on the pendant — then run this launch file

Usage:
  ros2 launch fruitninja movie.launch.py robot_ip:=<YOUR_ROBOT_IP>

Then separately run the movement node:
  ros2 run fruitninja movement
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ------------------------------------------------------------------ args
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.56.101',  # UR3e Polyscope IP
        description='IP address of the UR robot (Polyscope)',
    )

    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Launch RViz for visualisation alongside the real robot',
    )

    robot_ip    = LaunchConfiguration('robot_ip')
    launch_rviz = LaunchConfiguration('launch_rviz')

    # --------------------------------------------------------- UR robot driver
    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch',
                'ur_control.launch.py',
            ])
        ]),
        launch_arguments={
            'ur_type':                  'ur3e',
            'robot_ip':                 robot_ip,
            'use_fake_hardware':        'false',
            'launch_rviz':              'false',
            'initial_joint_controller': 'scaled_joint_trajectory_controller',
        }.items(),
    )

    # ------------------------------------------------------------ MoveIt 2
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py',
            ])
        ]),
        launch_arguments={
            'ur_type':           'ur3e',
            'use_fake_hardware': 'false',
            'launch_rviz':       launch_rviz,
        }.items(),
    )

    return LaunchDescription([
        robot_ip_arg,
        launch_rviz_arg,
        ur_driver,
        moveit,
    ])