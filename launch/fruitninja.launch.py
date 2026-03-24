#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    ur_robot_driver_dir = get_package_share_directory('ur_robot_driver')
    ur_moveit_config_dir = get_package_share_directory('ur_moveit_config')

    robot_ip = '192.168.56.101'
    ur_type = 'ur3e'

    # -------------------------------------------------------
    # 1. UR Robot Driver
    # -------------------------------------------------------
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_robot_driver_dir, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'launch_rviz': 'false',
            'description_package': 'fruitninja',
            'description_file': 'ur3e_workcell.urdf.xacro',
            'use_fake_hardware': 'true',
            'initial_joint_controller': 'joint_trajectory_controller',
        }.items()
    )

    # -------------------------------------------------------
    # 2. MoveIt with RViz (delayed 5s)
    # -------------------------------------------------------
    ur_moveit = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_moveit_config_dir, 'launch', 'ur_moveit.launch.py')
                ),
                launch_arguments={
                    'ur_type': ur_type,
                    'robot_ip': robot_ip,
                    'launch_rviz': 'true',
                    'description_package': 'fruitninja',
                    'description_file': 'ur3e_workcell.urdf.xacro',
                    'use_sim_time': 'false',
                }.items()
            )
        ]
    )

    # -------------------------------------------------------
    # 3. Planning Scene (delayed 12s)
    # -------------------------------------------------------
    planning_scene = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='fruitninja',
                executable='planning_scene',
                name='planning_scene_setup',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        ur_control,
        ur_moveit,
        planning_scene,
    ])