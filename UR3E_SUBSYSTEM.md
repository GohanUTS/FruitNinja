# UR3e Subsystem — real_gui_points.py

Operator guide for running the FruitNinja UR3e cutting cell using the main GUI.

---

## Prerequisites

All six startup steps must be running before the GUI will move the robot.
Use the Startup GUI (recommended) or the manual terminal commands below.

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja startup_gui
```

| Step | What it does | Ready when |
|------|-------------|------------|
| Step 1 — UR Driver | Connects to the UR3e via RTDE | Log: `"Robot connected to reverse interface. Ready to receive control commands."` |
| Step 2 — Fix Controller | One-shot: activates `scaled_joint_trajectory_controller` | Completes automatically — run once after Step 1 is ready |
| Step 3 — MoveIt | Starts motion planner + RViz | RViz window opens, robot model visible |
| Step 4 — Planning Scene | Publishes workcell collision objects | Runs immediately |
| Step 5 — Main GUI | Opens the operator GUI | GUI window appears |
| Step 6 — Safety Node | Hand-detection interlock | Logs `"Safety node ready"` |

> **Step 2 must run before Steps 3–5.** If `scaled_joint_trajectory_controller` is not active, trajectory commands are silently ignored and the robot will not move.

---

## Running the Main GUI Standalone

If the rest of the stack is already up:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run fruitninja real_gui_points --robot-ip 192.168.0.194
```

`--robot-ip` sets the UR Dashboard Server address used by the safety interlock. Defaults to `192.168.0.194`.

---

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  FRUITNINJA  UR3e operator console                              │
├───────────────────────────────┬─────────────────────────────────┤
│  Live Joint States            │  [▶ Start Camera ▼] [Source]   │
│  Base (pan)    ±  0.00°       │  [✛ Define Grid] [✕ Clear Grid]│
│  Shoulder      ±  0.00°       │  [✂ Cut Detected Fruit]        │
│  Elbow         ±  0.00°       │                                 │
│  Wrist 1       ±  0.00°       │  ┌─────────────────────────┐   │
│  Wrist 2       ±  0.00°       │  │   Live Camera Feed      │   │
│  Wrist 3       ±  0.00°       │  │   (grid overlay drawn   │   │
│                               │  │    once grid locked)    │   │
│  Grid — Toggle Cells          │  └─────────────────────────┘   │
│  ┌──┬──┬──┬──┬──┬──┬──┐      │                                 │
│  │A1│B1│C1│D1│E1│..│N1│  Row1│  Detection info                 │
│  │A2│B2│C2│D2│E2│..│N2│  Row2│                                 │
│  │A3│B3│C3│D3│E3│..│N3│  Row3│                                 │
│  │A4│B4│C4│D4│E4│..│N4│  Row4│                                 │
│  └──┴──┴──┴──┴──┴──┴──┘      │                                 │
├───────────────────────────────┴─────────────────────────────────┤
│  [▶ Move to Selected]  [⬡ Trace Grid]  [✕ Clear]               │
│  [⚠ E-STOP  SPACE]     [↺ Reset (Home)]                        │
├─────────────────────────────────────────────────────────────────┤
│  🛡 Safety banner                                               │
│  Status bar                                                     │
│  Log output                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Internal Classes

| Class | Thread | Role |
|-------|--------|------|
| `JointStateNode` | Background spin | Subscribes `/joint_states`, updates angle display |
| `MoverNode` | Per-move thread | Builds and sends timed joint trajectories to `scaled_joint_trajectory_controller` |
| `CameraBridgeNode` | Background spin | Publishes `/camera/image_raw/compressed` + `/safety/zone`; subscribes `/safety/hand_detected` |
| `CameraWorker` | Background thread | Captures raw frames from webcam or RealSense |
| `UR3eDashboard` | Main thread | TCP connection to UR Dashboard Server (port 29999) for pause/resume |
| `MainWindow` | Qt main thread | Renders UI, runs MediaPipe hand detection, drives cut sequences |

---

## Operating Modes

### Manual Mode — fixed grid cells

1. Click any cell(s) on the 14×4 grid. Selected cells highlight blue.
2. Cells are sorted left-to-right (A→N) automatically.
3. Press **▶ Move to Selected**.

The robot runs the approach → cut → recover sequence for each cell in order.

### Trace Grid

Press **⬡ Trace Grid** to visit the four corners in order: `A1 → N1 → N4 → A4`.
Useful for verifying calibration after setup.

### Autonomous Mode — camera-detected fruit

1. Press **▶ Start Camera** and select a source from the dropdown.
2. Press **✛ Define Grid** and click four corners of the cutting board on the camera image (any order — auto-sorted to TL/TR/BR/BL).
3. The 14×4 grid overlay appears on the feed. Detected red fruit is annotated.
4. Press **✂ Cut Detected Fruit**.

Each detected fruit gets a **first cut** at its centroid, then a **90° wrist-3 cross-cut** for a clean slice.

---

## Cut Motion Detail

For each cell or detected fruit, `MoverNode` sends two trajectory points:

```
Point 0 (t = 0.5 s) — current joint state, zero velocity   [start hold]
Point 1 (t = 0.5 + move_duration) — target joint state, zero velocity
```

`move_duration` is calculated from the largest joint delta divided by `10 °/s`, with a minimum of `4 s`.

Per-joint tolerances applied to every goal:

| Tolerance | Value |
|-----------|-------|
| Path tolerance | 8 ° |
| Goal tolerance | 2 ° |
| Goal time tolerance | 5 s |

Home position (`↺ Reset`): `[0°, −90°, 0°, −90°, 0°, 0°]`

Approach offset: shoulder lift `+11°` above the calibrated cut pose.

---

## Camera Sources

| Dropdown option | Device opened |
|----------------|--------------|
| Webcam | `/dev/video0` |
| RealSense D435i | `pyrealsense2` pipeline (640×480, 30 fps) |
| Fisheye | `/dev/video2` |
| Osmo DJI Pocket 3 | `/dev/video3` |

`pyrealsense2` is optional — if not installed the RealSense option is hidden.

---

## Red Fruit Detection

Detection runs every 2nd frame inside the locked grid polygon only.

**HSV thresholds:**

| Channel | Range 1 | Range 2 |
|---------|---------|---------|
| Hue | 0 – 8 | 172 – 180 |
| Saturation | 170 – 255 | 170 – 255 |
| Value | 90 – 255 | 90 – 255 |

Additional filters: minimum contour area `400 px²`, solidity `≥ 0.70`, bounding box `≥ 12×12 px`.

Each detection stores:
- `origin` — grid cell the centroid lands in (e.g. `G2`)
- `grid_xy` — fractional grid coordinates used for continuous interpolated targeting
- `covered` — all cells touched by the bounding box
- `area_px`, `shape_angle_deg` — contour metrics

---

## Safety Interlock

The GUI runs a **local** MediaPipe Hands check (every 3rd frame) in addition to `safety_node`.

**Trip condition:** palm centre inside the locked grid polygon, OR ≥ 5 hand landmarks inside AND ≥ 2 palm landmarks inside.

**On detection:**
1. Active trajectory goal cancelled immediately (`cancel_goal_async`)
2. `pause` sent to UR Dashboard Server
3. Safety banner turns red: `⚠ SAFETY: HAND IN ZONE — ROBOT PAUSED`
4. New motion blocked until zone has been clear for **2 consecutive detection frames**

**On zone clear:**
- `play` sent to Dashboard Server
- If a cut sequence was interrupted, it resumes automatically from the interrupted cell

**Safety banner states:**

| State | Banner colour | Text |
|-------|--------------|------|
| No grid defined | Grey | `🛡 Safety: standby — define grid to activate interlock` |
| Grid locked, monitoring | Green | `🛡 Safety: active — monitoring cutting zone` |
| Hand detected | Red | `⚠ SAFETY: HAND IN ZONE — ROBOT PAUSED` |
| Zone clear, resuming | Green | `🛡 Safety: zone clear — resuming motion` |
| MediaPipe not installed | Amber | `⚠ Safety: mediapipe not installed — interlock disabled` |

---

## E-STOP

Press the red **⚠ E-STOP** button or **Spacebar** at any time.

- Active trajectory goal cancelled via `cancel_goal_async()`
- `stop` sent to UR Dashboard Server (stronger than `pause` — clears program state)
- New movement blocked for **2 seconds** cooldown
- Status bar shows `⚠ EMERGENCY STOP — wait 2 s before moving`

After cooldown: press **↺ Reset (Home)** before resuming any cut sequence.

---

## ROS Topics Published / Subscribed

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Subscribe | Live joint angles |
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Action client | Cut trajectories |
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | Publish | Camera frames → `safety_node` |
| `/safety/zone` | `std_msgs/Float32MultiArray` | Publish | Locked grid polygon `[x0,y0,x1,y1,x2,y2,x3,y3]` |
| `/safety/hand_detected` | `std_msgs/Bool` | Subscribe | Hand-in-zone events from `safety_node` |

---

## Troubleshooting

**Robot doesn't move — no error shown**
Run Step 2 (Fix Controller). If `scaled_joint_trajectory_controller` is not active, trajectory goals are silently dropped.

```bash
ros2 control list_controllers
# scaled_joint_trajectory_controller should show [active]
```

**`No live joint state yet` error on first move**
`/joint_states` has not arrived. Confirm Step 1 (UR Driver) is connected and the robot is powered on. Check with:
```bash
ros2 topic hz /joint_states
```

**Velocity limit error on move start**
The trajectory header stamp was not set — this is fixed in the current code. Ensure the built package is sourced:
```bash
source ~/ros2_ws/install/setup.bash
```

**`goal_time_tolerance` error after E-STOP**
Wait the full 2-second cooldown before pressing Reset or Move. The cooldown prevents a second trajectory from being sent while the controller is still finishing the cancelled one.

**Safety interlock fires immediately / false trigger**
The palm-centre check requires the hand body to be inside the grid, not just fingertips. If it trips too easily, increase `tolerance_px` on `safety_node`:
```bash
ros2 run fruitninja safety_node --robot-ip 192.168.0.194 --ros-args -p tolerance_px:=20
```

**Camera feed shows black / no image**
Check the selected source matches the connected device:
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

**`mediapipe not installed` banner**
Install to the system Python (the same interpreter used by `ros2 run`):
```bash
/usr/bin/pip3 install mediapipe==0.10.5 "numpy<2" "opencv-python-headless>=4.5,<4.9"
```
