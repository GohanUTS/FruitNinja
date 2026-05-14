# FruitNinja — UR3e Fruit Cutting Robot

Autonomous and operator-guided fruit cutting system using a Universal Robots UR3e arm,
ROS 2 Humble, and MoveIt 2. The operator selects grid cells on a cutting board via a GUI,
and the robot moves through fixed calibrated joint poses for each cell. A camera feed
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
┌─────────────────────────────────────────────────────────────┐
│  ur_robot_driver  (Step 1)                                   │
│                                                              │
│  Connects to UR3e via RTDE protocol                          │
│  Publishes: /joint_states                                    │
│  Once ready → auto-switches to scaled_joint_trajectory_      │
│               controller via /controller_manager service     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│           ur_moveit_config  (Step 2 — MoveIt)                │
│  MoveIt motion planner + RViz  (ur_moveit.launch.py)         │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  planning_scene.py   │
│  (Step 4)            │
│  Publishes workcell  │
│  collision objects   │
│  to MoveIt           │
└──────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                     real_gui_points.py                        │
│                  Main Operator GUI (Step 5)                   │
│                                                              │
│  JointStateNode    → subscribes /joint_states                │
│  MoverNode         → sends timed joint trajectories          │
│  CameraBridgeNode  → publishes /camera/image_raw/compressed  │
│                      and /safety/zone; subscribes            │
│                      /safety/hand_detected                   │
│  CameraWorker      → captures frames (webcam / RealSense)    │
│  MainWindow        → PyQt5 UI, grid selection, cut control   │
└─────────────────────────────────────────────────────────────┘
           │ (publishes topics)
           ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│  safety_node.py      │    │  grid_mover.py                   │
│  (Step 6)            │    │  (imported by real_gui_points)   │
│                      │    │                                  │
│  Subscribes camera   │    │  Fixed grid: cell name (A1–N4)  │
│  + zone topics       │    │  → tool pose + joint angles      │
│  MediaPipe hand det. │    │  + fractional UV interpolation   │
│  CancelGoal service  │    │                                  │
│  + Dashboard TCP     │    │                                  │
└──────────────────────┘    └──────────────────────────────────┘
```

---

## Package Modules

| File | Purpose |
|------|---------|
| `startup_gui.py` | Launcher GUI — runs each ROS process as an embedded QProcess with live output tabs |
| `real_gui_points.py` | Main operator GUI — grid cell selection, camera feed, cut execution, safety bridge |
| `safety_node.py` | Standalone hand-detection interlock — subscribes to camera + zone topics, pauses robot via Dashboard TCP (port 29999) |
| `planning_scene.py` | Publishes the workcell URDF collision objects (trolley, walls) to MoveIt |
| `grid_mover.py` | Maps grid cell names (A1–N4, 14 columns × 4 rows) to calibrated tool positions and interpolated joint angles |
| `colour_detection.py` | Standalone util: detects Apple/Lettuce/Banana/Orange via HSV + auto-locates grid from 4 blue corner markers |
| `movement.py` | Low-level MoveIt motion helpers used by older GUI scripts |
| `gui.py` | Legacy simulation GUI |

---

## How the Cut Motion Works

Each cell cut is a 3-step joint-space trajectory sequence executed by `MoverNode`:

```
1. Approach  →  move_to(approach_degrees)  shoulder lift is 11° above the grid pose
2. Cut       →  move_to(grid_degrees)      arm moves to the fixed calibrated pose
3. Recover   →  move_to(approach_degrees)  arm lifts back above the grid pose
```

The next cell in the queue starts after recover completes. The GUI sends timed
joint trajectories directly to the scaled controller; it does not send Cartesian
position goals or IK targets.

Per-joint tolerances: path `8°`, goal `2°`, time `5 s` — prevents small real-hardware
overshoot from failing the sequence.

**Cut Detected Fruit** button: cuts each detected fruit at its camera centroid with a
first cut followed by a 90° wrist-3 cross-cut.

---

## Camera & Grid Detection

1. Press **▶ Start Camera** and select a source (Webcam / RealSense D435i / Other).
2. Press **✛ Define Grid** and click 4 corner points on the camera image (any order — auto-sorted to TL/TR/BR/BL).
3. A 14×4 bilinear perspective grid is drawn on the live feed.
4. Only **red** objects whose centroid falls inside the grid are detected.
5. Each detection reports an **origin cell** (centroid) and **covered cells** (full bounding box).
6. Press **✂ Cut Detected Fruit** to execute.

Detection uses OpenCV HSV masking (`H: 0–8 and 172–180, S ≥ 170, V ≥ 90`).
The high saturation floor rejects skin, wood, and walls which cluster at S 30–130.

---

## Safety Interlock

The safety system has two independent layers:

| Layer | Where | How |
|-------|-------|-----|
| **Local** | `real_gui_points.py` | MediaPipe Hands runs on the GUI machine every 3rd frame; trips when palm centre or ≥ 5 landmarks are inside the locked grid polygon |
| **Standalone** | `safety_node.py` (Step 6) | Subscribes to `/camera/image_raw/compressed` and `/safety/zone`; runs its own MediaPipe instance; cancels trajectory goals and pauses the UR Dashboard Server |

Camera frames are published to `/camera/image_raw/compressed` only once the operator has locked the grid (4 corner points defined) — the safety node activates at that same moment when the zone polygon is published.

When a hand is detected:
1. **Local layer**: active trajectory goal is cancelled via `cancel_goal_async()`, Dashboard Server receives `pause`
2. **Standalone layer**: trajectory is cancelled via the `CancelGoal` service, Dashboard Server receives `pause`
3. New motion is blocked until the zone has been clear for 2 consecutive detection frames
4. Any interrupted sequence can be automatically resumed

---

## ROS 2 Topics and Services

| Topic / Service / Action | Direction | Used by |
|--------------------------|-----------|---------|
| `/joint_states` | subscribe | `JointStateNode` — live angle display |
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | action client | `MoverNode` — sends timed joint trajectories |
| `/camera/image_raw/compressed` | publish | `CameraBridgeNode` — forwards frames to `safety_node` |
| `/safety/zone` | publish | `CameraBridgeNode` — locked grid polygon `[x0,y0,…,x3,y3]` |
| `/safety/hand_detected` | subscribe | `CameraBridgeNode` — hand-in-zone events from `safety_node` |
| `/planning_scene` | publish | `planning_scene.py` — workcell collision objects |
| `/controller_manager/switch_controller` | service call | Step 2 — activates `scaled_joint_trajectory_controller` |

---

## Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- MoveIt 2 (`ros-humble-moveit`)
- UR ROS 2 Driver (`ros-humble-ur`)
- PyQt5 (`python3-pyqt5`)
- OpenCV (`python3-opencv`)
- NumPy < 2
- MediaPipe 0.10.5 (`pip install mediapipe==0.10.5`)
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

Press **Start** on each step **in order**, waiting for each to be ready before continuing:

| Step | Command run internally | When it's ready |
|------|----------------------|-----------------|
| Step 1 — UR Driver | `ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=<IP> launch_rviz:=false` | Log shows `"Ready to receive control commands"` — controller switch fires automatically 2 s later |
| Step 2 — MoveIt | `ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e …` | RViz opens and robot model appears |
| Step 3 — Planning Scene | `ros2 run fruitninja planning_scene` | Runs and publishes collision objects |
| Step 4 — Main GUI | `ros2 run fruitninja real_gui_points --robot-ip <IP>` | GUI window opens |
| Step 5 — Safety Node | `ros2 run fruitninja safety_node --robot-ip <IP>` | Logs `"Safety node ready"` — interlock activates once grid is locked in GUI |

**Default robot IP:** `192.168.0.194` (used by `real_gui_points` and the Startup GUI)  
**Note:** `safety_node` has a separate built-in default of `192.168.0.197` — always pass `--robot-ip` explicitly when running it manually.  
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
Press **Play** on the **External Control** program in Polyscope, then continue with Steps 1–6.

Or manually:

```bash
# Terminal 1 — URSim
ros2 run ur_client_library start_ursim.sh -m ur3e

# Terminal 2 — UR Driver (sim IP)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e robot_ip:=192.168.56.101 launch_rviz:=false

# Terminal 3 — Fix Controller (after Terminal 2 is ready)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 service call /controller_manager/switch_controller \
  controller_manager_msgs/srv/SwitchController \
  "{activate_controllers: ['scaled_joint_trajectory_controller'], \
    deactivate_controllers: ['joint_trajectory_controller'], \
    strictness: 1, activate_asap: true, timeout: {sec: 5, nanosec: 0}}"

# Terminal 4 — MoveIt
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur3e \
  description_package:=fruitninja \
  description_file:=ur3e_workcell.urdf.xacro \
  launch_servo:=false launch_rviz:=true

# Terminal 5 — Planning Scene
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja planning_scene

# Terminal 6 — Main GUI
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja real_gui_points --robot-ip 192.168.56.101

# Terminal 7 — Safety Node
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja safety_node --robot-ip 192.168.56.101
```

---

## Emergency Stop

- Press the red **⚠ E-STOP** button in the GUI, or press **Spacebar** at any time.

This immediately cancels the active trajectory goal via `cancel_goal_async()`, sends a `stop`
command to the UR Dashboard Server, and blocks new movement for a 2-second cooldown.

After the cooldown: press **↺ Reset (Home)** to return to the safe upright pose before resuming.

---

## Grid Layout

The cutting board is a **14-column × 4-row** grid:

```
Columns:  A  B  C  D  E  F  G  H  I  J  K  L  M  N   (left → right, ~73 cm)
Rows:     1  2  3  4                                    (far → near,  ~22 cm)
```

Cell `A1` is far-left, `N4` is near-right. The robot sweeps left-to-right when multiple cells are selected.
