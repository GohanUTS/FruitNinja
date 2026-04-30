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
│  MoverNode       → sends timed joint trajectories            │
│  CameraWorker    → captures frames (webcam / RealSense)      │
│  MainWindow      → PyQt5 UI, grid selection, cut control     │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│  planning_scene.py   │    │  grid_mover.py                   │
│  (Step 3)            │    │  (imported by real_gui_points)   │
│                      │    │                                  │
│  Publishes workcell  │    │  Fixed grid: cell name (A1–N4)  │
│  collision objects   │    │  → tool pose + joint angles      │
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
| `grid_mover.py` | Maps grid cell names (A1–N4, 14 columns × 4 rows) to calibrated tool positions and interpolated joint angles |
| `colour_detection.py` | OpenCV HSV colour detection + blue-marker grid overlay (standalone util) |
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
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | action client | `MoverNode` — sends timed joint trajectories |
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
i want the logic for the 
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

## AI Vision Pipeline

### Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python pyrealsense2 numpy
```

### Step 1 — Collect Dataset

Point the RealSense (or webcam if no RealSense) at the cutting board with fruit on it:

```bash
cd ~/ros2_ws/src/FruitNinja
python vision/collect_dataset.py
```

Controls:
- **S** — save current frame to `vision/dataset/raw/`
- **Q** — quit

Target: **200–300 images** under your actual lab lighting. Vary fruit position, rotation, and quantity across captures.

### Step 2 — Annotate on Roboflow

1. Go to [roboflow.com](https://roboflow.com) and create a free account
2. Create a new project → **Instance Segmentation**
3. Upload all images from `vision/dataset/raw/`
4. Draw segmentation masks around each fruit (use the polygon tool)
5. Apply augmentations: **Hue ±15°, Brightness ±30%, Mosaic, Horizontal Flip**
6. Export as **YOLOv8 format** → download and extract into:
   ```
   vision/dataset/fruitninja_roboflow/
     images/
       train/
       val/
     labels/
       train/
       val/
   ```

### Step 3 — Fine-tune YOLOv8

```bash
cd ~/ros2_ws/src/FruitNinja

# Download the base model (first run only)
# yolo automatically downloads yolov8m-seg.pt on first use

yolo task=segment mode=train \
     model=yolov8m-seg.pt \
     data=vision/fruitninja.yaml \
     epochs=100 imgsz=640 batch=16 \
     project=vision/runs name=fruitninja_seg
```

Trained weights will be saved to:
```
vision/runs/fruitninja_seg/weights/best.pt
```

Validate the model:
```bash
yolo task=segment mode=val \
     model=vision/runs/fruitninja_seg/weights/best.pt \
     data=vision/fruitninja.yaml
```

### Step 4 — Run the Vision Node

Once weights are in place, start Step 6 from the Startup GUI, or manually:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja vision_node
```

Topics published:
- `/fruit_target` — 3D centroid of detected fruit in `base_link` frame
- `/estop` — `Bool(True)` if a hand or person is detected
- `/fruit_detections_debug` — annotated camera image (view with `rqt_image_view`)

---

## RL Trajectory Planner

### Install Dependencies

```bash
pip install mujoco gymnasium stable-baselines3 torch numpy scipy tensorboard
```

### Step 1 — Convert URDF to MuJoCo XML (one-time)

```bash
cd ~/ros2_ws/src/FruitNinja
python -m mujoco.scripts.compile urdf/ur3e_workcell.urdf.xacro ur3e_workcell.xml
```

This generates `ur3e_workcell.xml` which the RL environment loads for simulation.

### Step 2 — Behavioural Cloning Pre-training

Pre-trains a policy network to imitate the existing joint-angle lookup table in `grid_mover.py`.
This warm-starts the SAC actor and significantly cuts training time.

```bash
cd ~/ros2_ws/src/FruitNinja/rl/pretraining
python behavioural_clone.py
```

Takes ~1 minute on CPU. Saves `bc_policy.pth` in the same folder.

The script:
1. Reads all 56 cells (14 cols × 4 rows) from `grid_mover.py`
2. For each cell, computes the joint-angle delta from the home pose
3. Trains a 3-layer MLP (256 hidden units) with MSE loss for 50 epochs

### Step 3 — SAC Training

```bash
cd ~/ros2_ws/src/FruitNinja/rl

# With BC warm-start (recommended)
python train_sac.py --bc_weights pretraining/bc_policy.pth

# Without warm-start (fresh training)
python train_sac.py
```

Training parameters:
- **2 million timesteps** — takes hours on GPU, days on CPU
- **4 parallel environments** for faster experience collection
- Domain randomisation applied every episode (friction, EE mass, latency)
- Checkpoints saved every 50,000 steps to `models/checkpoints/`
- Best model saved to `models/best/`

Monitor training with TensorBoard:
```bash
tensorboard --logdir ~/ros2_ws/src/FruitNinja/rl/tb_logs
# Open http://localhost:6006 in a browser
```

Final model saved as `sac_fruitninja.zip` in `rl/`.

### Step 4 — Evaluate the Trained Policy

```bash
cd ~/ros2_ws/src/FruitNinja/rl
python train_sac.py --eval
```

Runs 10 evaluation episodes and prints mean distance to target per episode.
Gate: policy must reach `< 5 mm` from target consistently before deployment.

### Step 5 — Validate in URSim Before Real Robot

**Never run the RL node on the physical arm without URSim validation first.**

1. In the Startup GUI, toggle **Switch to Sim**
2. Start Steps 0–5 (URSim + full ROS stack)
3. Copy `rl/sac_fruitninja.zip` to the working directory
4. Start Step 7 — RL Mover (defaults to **10% velocity scaling**)
5. Monitor the terminal output — deviation from target is logged every 50 ms

Deployment gate — all three must pass across 3 full A1→N4 grid sweeps:
- Max joint deviation vs `/joint_states` stays **< 0.1 rad**
- No unintended E-STOP triggers
- All 56 cells reached within the position tolerance

Only after passing all three gates should you run the RL mover on the physical UR3e.

### Step 6 — Run the RL Mover Node

Place `sac_fruitninja.zip` at `~/ros2_ws/src/FruitNinja/rl/sac_fruitninja.zip`, then start Step 7 from the Startup GUI, or manually:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja rl_mover_node
```

The node:
- Subscribes to `/fruit_target` (from the vision node) to get the 3D cut target
- Subscribes to `/estop` — immediately halts on any safety trigger
- Publishes joint trajectories at **20 Hz** to `/scaled_joint_trajectory_controller/joint_trajectory`
- Uses TF2 forward kinematics (`tool0 → base_link`) for real end-effector position in the observation

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
