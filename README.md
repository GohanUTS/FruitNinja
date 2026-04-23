# FruitNinja — UR3e Fruit Cutting Robot

Autonomous and operator-guided fruit cutting system using a Universal Robots UR3e arm,
ROS 2 Humble, and MoveIt 2. The operator selects grid cells on a cutting board via a GUI,
and the robot executes approach → dip → recover cut motions at each cell. A camera feed
with manual grid overlay detects red fruit in real time and can drive the arm automatically.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        startup_gui.py                        │
│   Step-by-step launcher — starts each process in order      │
└──────────┬──────────────────────────────────────────────────┘
           │ launches (via QProcess)
           ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│  ur_robot_driver     │    │  ur_moveit_config                │
│  (Step 1)            │    │  (Step 2)                        │
│                      │    │                                  │
│  Connects to UR3e    │    │  MoveIt motion planner + RViz    │
│  via RTDE protocol   │    │  ur_moveit.launch.py             │
│  Publishes:          │    │  Provides:                       │
│  /joint_states       │    │  /move_action  (action server)   │
└──────────────────────┘    └──────────────────────────────────┘
           │                           │
           │    ┌──────────────────────┘
           ▼    ▼
┌──────────────────────────────────────────────────────────────┐
│                     real_gui_points.py                        │
│                  Main Operator GUI (Step 4)                   │
│                                                              │
│  JointStateNode  → subscribes /joint_states                  │
│  MoverNode       → sends goals to /move_action               │
│  CameraWorker    → captures frames (webcam / RealSense)      │
│  MainWindow      → PyQt5 UI, grid selection, cut control     │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│  planning_scene.py   │    │  grid_mover.py                   │
│  (Step 3)            │    │  (imported by real_gui_points)   │
│                      │    │                                  │
│  Publishes workcell  │    │  Lookup table: cell name (A1–N4) │
│  collision objects   │    │  → 6 joint angles in degrees     │
│  to MoveIt           │    │                                  │
└──────────────────────┘    └──────────────────────────────────┘
```

---

## Package Modules

| File | Purpose |
|------|---------|
| `startup_gui.py` | Launcher GUI — runs each ROS process as an embedded QProcess with live output tabs |
| `real_gui_points.py` | Main operator GUI — grid cell selection, camera feed, cut execution |
| `planning_scene.py` | Publishes the workcell URDF collision objects (trolley, walls) to MoveIt |
| `grid_mover.py` | Maps grid cell names (A1–N4, 14 columns × 4 rows) to pre-computed joint angles |
| `colour_detection.py` | OpenCV HSV colour detection + blue-marker grid overlay (standalone util) |
| `movement.py` | Low-level MoveIt motion helpers used by older GUI scripts |
| `gui.py` | Legacy simulation GUI |

---

## How the Cut Motion Works

Each cell cut is a 3-step MoveIt sequence executed by `MoverNode`:

```
1. Approach  →  move_to(degrees)           arm arrives at cell pose above board
2. Dip       →  move_to(dip_degrees)       shoulder +30°, elbow -30° → end-effector drives down
3. Recover   →  move_to(degrees)           arm lifts back to approach pose
```

The next cell in the queue starts after recover completes. All moves run at 30% velocity and
acceleration (`max_velocity_scaling_factor = 0.3`).

**Cut Detected Fruit** button: builds the queue from camera detections and runs the same loop.

---

## Camera & Grid Detection

1. Press **Start Camera** and select a source (Webcam / RealSense D435i / Fisheye / Osmo DJI Pocket 3).
2. Press **Define Grid** and click 4 corner points on the camera image (TL → TR → BR → BL in any order — they are auto-sorted).
3. A 14×4 bilinear perspective grid is drawn on the live feed.
4. Only **red** objects whose centroid falls inside the grid are detected.
5. Each detection reports an **origin cell** (centroid) and **covered cells** (full bounding box).
6. Press **Cut Detected Fruit** to send those cells to the robot.

Detection uses OpenCV HSV masking (`H: 0–10 and 170–180, S ≥ 100, V ≥ 80`).

---

## ROS 2 Topics and Services Used

| Topic / Service / Action | Direction | Used by |
|--------------------------|-----------|---------|
| `/joint_states` | subscribe | `JointStateNode` — live angle display |
| `/move_action` | action client | `MoverNode` — sends MoveGroup goals |
| `/planning_scene` | publish | `planning_scene.py` — workcell collision objects |
| `/controller_manager/switch_controller` | service call | Step 5 — activates scaled controller |

---

## Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- MoveIt 2 (`ros-humble-moveit`)
- UR ROS 2 Driver (`ros-humble-ur`)
- PyQt5 (`python3-pyqt5`)
- OpenCV (`python3-opencv`)
- NumPy
- (Optional) `pyrealsense2` for RealSense D435i

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select fruitninja
source install/setup.bash
```

---

## Running — Real Robot

The recommended way is to use the **Startup GUI** which manages all processes in one window:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja startup_gui
```

Then press **Start** on each step **in order**, waiting for each to be ready before continuing:

| Step | Command run internally | When it's ready |
|------|----------------------|-----------------|
| Step 1 — UR Driver | `ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=<IP> launch_rviz:=false` | Log shows `"Robot connected to reverse interface. Ready to receive control commands."` |
| Step 2 — MoveIt | `ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e description_package:=fruitninja description_file:=ur3e_workcell.urdf.xacro launch_servo:=false launch_rviz:=true` | RViz opens and robot model appears |
| Step 3 — Planning Scene | `ros2 run fruitninja planning_scene` | Runs immediately, publishes once |
| Step 4 — Main GUI | `ros2 run fruitninja real_gui_points` | GUI window opens |
| Step 5 — Fix Controller | `ros2 service call /controller_manager/switch_controller ...` | One-shot, completes on its own — run after Step 1 is fully connected |

**Default robot IP:** `192.168.0.194`  
**Laptop IP on the robot network:** `192.168.0.101` (ethernet interface `enp4s0`)

To change the robot IP: edit the **IP field** in the Startup GUI and press **Apply** before starting any steps.

---

## Running — Simulator (URSim)

Toggle **Switch to Sim** in the Startup GUI title bar. This:
- Locks the IP to `192.168.56.101`
- Prepends **Step 0** to start the URSim Docker container

Then open the Polyscope virtual teach pendant at:
```
http://192.168.56.101:6080/vnc.html
```
Press **Play** on the **External Control** program in Polyscope, then continue with Steps 1–5.

Or manually:

```bash
# Terminal 1 — URSim
ros2 run ur_client_library start_ursim.sh -m ur3e

# Terminal 2 — UR Driver (sim IP)
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e robot_ip:=192.168.56.101 launch_rviz:=false

# Terminal 3 — MoveIt
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur3e \
  description_package:=fruitninja \
  description_file:=ur3e_workcell.urdf.xacro \
  launch_servo:=false launch_rviz:=true

# Terminal 4 — Planning Scene
ros2 run fruitninja planning_scene

# Terminal 5 — Main GUI
ros2 run fruitninja real_gui_points

# Terminal 6 — Fix Controller (after Terminal 2 is ready)
ros2 service call /controller_manager/switch_controller \
  controller_manager_msgs/srv/SwitchController \
  "{activate_controllers: ['scaled_joint_trajectory_controller'], \
    deactivate_controllers: ['joint_trajectory_controller'], strictness: 2}"
```

---

## Emergency Stop

- Press the red **E-STOP** button in the GUI, or
- Press **Spacebar** at any time

This sets a cancel flag on the MoverNode — the robot finishes its current waypoint then stops. No further cells in the queue are executed.

---

## Grid Layout

The cutting board is a **14-column × 4-row** grid:

```
Columns:  A  B  C  D  E  F  G  H  I  J  K  L  M  N   (left → right, ~73 cm)
Rows:     1  2  3  4                                    (far → near,  ~22 cm)
```

Cell `A1` is far-left, `N4` is near-right. The robot sweeps left-to-right when multiple cells are selected.
