# FruitNinja
Fruit cutting chef


cd /home/dinesh/ros2_ws
colcon build --packages-select fruitninja
source install/setup.bash
ros2 launch fruitninja fruitninja.launch.py


terminal 1:
docker pull universalrobots/ursim_e-series
ros2 run ur_client_library start_ursim.sh -m ur3e

terminal 2: 
ros2 launch fruitninja fruitninja.launch.py


cd /home/dinesh/ros2_ws && source install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  initial_joint_controller:=joint_trajectory_controller \
  launch_rviz:=false



cd /home/dinesh/ros2_ws && source install/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur3e \
  launch_rviz:=true



cd /home/dinesh/ros2_ws && source install/setup.bash
ros2 run fruitninja planning_scene





TO CONNECT TO UR3e

Terminal 1 — UR Driver
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.0.194 launch_rviz:=false initial_joint_controller:=scaled_joint_trajectory_controller


Terminal 2 — MoveIt
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e description_package:=fruitninja description_file:=ur3e_workcell.urdf.xacro launch_servo:=false launch_rviz:=true

Terminal 3 — Planning Scene
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja planning_scene

Terminal 4 — GUI
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja real_gui_points

Terminal 5 — Fix Controller (once after Terminal 1 is fully connected)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['scaled_joint_trajectory_controller'], deactivate_controllers: ['joint_trajectory_controller'], strictness: 2}"

