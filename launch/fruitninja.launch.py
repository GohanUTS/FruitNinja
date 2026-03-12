#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    ur_robot_driver_dir = get_package_share_directory('ur_robot_driver')
    ur_moveit_config_dir = get_package_share_directory('ur_moveit_config')

    robot_ip = '192.168.56.101'
    ur_type = 'ur3e'
    urdf_file = '/home/dinesh/ros2_ws/src/ur3e_camera_workcell.urdf.xacro'

    # -------------------------------------------------------
    # 1. UR Robot Driver (no RViz - MoveIt will handle that)
    # -------------------------------------------------------
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_robot_driver_dir, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'launch_rviz': 'false',
            'urdf_file': urdf_file,
        }.items()
    )

    # -------------------------------------------------------
    # 2. MoveIt with RViz (delayed 5s to let driver start up)
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
                }.items()
            )
        ]
    )

    # -------------------------------------------------------
    # 3. Planning Scene (delayed 10s to let MoveIt start up)
    # -------------------------------------------------------
    planning_scene = TimerAction(
        period=10.0,
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
