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
