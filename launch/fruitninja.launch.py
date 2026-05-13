#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    ur_robot_driver_dir = get_package_share_directory('ur_robot_driver')
    ur_moveit_config_dir = get_package_share_directory('ur_moveit_config')

    ur_type = 'ur3e'

    # ── Launch arguments (overridable from GUI or CLI) ─────────────────────────
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.56.101',
        description='IP of the UR3e (simulator default: 192.168.56.101)',
    )
    fake_hw_arg = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value='true',
        description='true = simulation, false = real robot',
    )
    reverse_port_arg = DeclareLaunchArgument(
        'reverse_port',
        default_value='50001',
        description='Reverse port for UR External Control program (real robot)',
    )
    safety_arg = DeclareLaunchArgument(
        'enable_safety_node',
        default_value='true',
        description='Run the standalone hand-detection safety_node alongside the GUI',
    )

    robot_ip       = LaunchConfiguration('robot_ip')
    use_fake_hw    = LaunchConfiguration('use_fake_hardware')
    reverse_port   = LaunchConfiguration('reverse_port')
    enable_safety  = LaunchConfiguration('enable_safety_node')

    # ── 1. UR Robot Driver ─────────────────────────────────────────────────────
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_robot_driver_dir, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type':                  ur_type,
            'robot_ip':                 robot_ip,
            'launch_rviz':              'false',
            'description_package':      'fruitninja',
            'description_file':         'ur3e_workcell.urdf.xacro',
            'use_fake_hardware':        use_fake_hw,
            'initial_joint_controller': 'joint_trajectory_controller',
            'reverse_port':             reverse_port,
        }.items()
    )

    # ── 2. MoveIt + RViz (delayed 5s) ─────────────────────────────────────────
    ur_moveit = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_moveit_config_dir, 'launch', 'ur_moveit.launch.py')
                ),
                launch_arguments={
                    'ur_type':             ur_type,
                    'robot_ip':            robot_ip,
                    'launch_rviz':         'true',
                    'description_package': 'fruitninja',
                    'description_file':    'ur3e_workcell.urdf.xacro',
                    'use_sim_time':        'true',
                }.items()
            )
        ]
    )

    # ── 3. Planning Scene (delayed 12s) ───────────────────────────────────────
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

    # ── 4. Safety hand-detection node (delayed 14s, optional) ─────────────────
    # Subscribes to /camera/image_raw/compressed (published by the GUI),
    # runs MediaPipe Hands, and pauses the UR Dashboard when a hand enters
    # the cutting zone.  The GUI also runs MediaPipe locally for visual
    # feedback; this node provides the redundant out-of-process backup.
    safety_node = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='fruitninja',
                executable='safety_node',
                name='fruitninja_safety',
                output='screen',
                parameters=[{'robot_ip': robot_ip}],
                condition=IfCondition(enable_safety),
            )
        ]
    )

    return LaunchDescription([
        robot_ip_arg,
        fake_hw_arg,
        reverse_port_arg,
        safety_arg,
        ur_control,
        ur_moveit,
        planning_scene,
        safety_node,
    ])
