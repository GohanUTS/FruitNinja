# FruitNinja — Movement & Position System

## Overview

The robot is a **UR3e** arm controlled through ROS 2. The main GUI sends fixed
joint trajectories directly to the active scaled trajectory controller. MoveIt
is still launched for robot state, RViz, and planning scene visibility.

The cutting board is a **14 × 4 grid** of named cells (columns A–N, rows 1–4).
Every movement targets a specific cell.

---

## System Startup Order

Managed by `startup_gui.py`. Steps must be launched in this order:

| Step | Process | What it does |
|------|---------|-------------|
| 1 | `ur_robot_driver` | RTDE connection to the physical UR3e (or URSim) |
| 2 | `ur_moveit.launch` | MoveIt planner + RViz |
| 3 | `planning_scene` | Publishes collision objects to MoveIt |
| 4 | `real_gui_points` | Main operator GUI |
| 5 | `switch_controller` | One-shot: activates `scaled_joint_trajectory_controller` |

---

## Coordinate System

All positions are expressed relative to **`base_link`** (the robot's base frame)
unless stated otherwise.

- **+X** → robot's right (from its perspective)
- **+Y** → away from the robot, toward the cutting board
- **+Z** → upward

The cutting board corners are the fixed UR teach-pendant calibration points from
the four photos:

| Corner | col | row | x (m)   | y (m)   | z (m)  |
|--------|-----|-----|---------|---------|--------|
| A1     | 0   | 0   | −0.25322 | −0.30822 | −0.37047 |
| A4     | 0   | 3   | −0.25440 | −0.43063 | −0.36707 |
| N1     | 13  | 0   | +0.24152 | −0.31635 | −0.36815 |
| N4     | 13  | 3   | +0.24024 | −0.44040 | −0.36745 |

Z is interpolated across the four measured corner heights.

---

## Grid Cell Lookup — `grid_mover.py`

**File:** `fruitninja/grid_mover.py`

Converts a cell label (e.g. `"C2"`) to fixed joint angles using
**bilinear interpolation** across 4 measured corner joint poses. The matching
photographed tool positions are kept for logging/checking through
`cell_to_pose()`, but movement commands use `cell_to_joints_deg()`.

### Measured corner joint angles (degrees)

| Corner | pan    | lift    | elbow  | wrist1  | wrist2 | wrist3 |
|--------|--------|---------|--------|---------|--------|--------|
| A1     | 30.42  | −46.19 | 98.65  | −138.12 | −87.00 | 31.65  |
| A4     | 43.35  | −27.50 | 56.58  | −115.76 | −86.63 | 42.26  |
| N1     | 107.42 | −46.70 | 104.71 | −152.19 | −87.90 | 108.96 |
| N4     | 101.90 | −28.80 | 61.76  | −124.25 | −83.95 | 101.79 |

### Interpolation steps

1. Convert column letter → `col_idx` (0–13), row number → `row_idx` (0–3)
2. Compute `u = col_idx / 13` (0 = A, 1 = N)
3. Compute `v = row_idx / 3`  (0 = row1, 1 = row4)
4. Interpolate the top edge A1→N1 and bottom edge A4→N4
5. Blend between those two edges with `v`

```
cell_to_pose("C2")        →  (x, y, z) in metres
cell_to_joints_deg("C2")  →  [pan°, lift°, elbow°, w1°, w2°, w3°]
```

---

## Motion Execution — `real_gui_points.py`

**File:** `fruitninja/real_gui_points.py`

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `HOME_DEG` | `[0, -90, 0, 0, 0, 360]` | Safe upright home pose |
| `APPROACH_LIFT_DELTA` | `−11°` | Shoulder lift offset above the fixed cut pose |
| `TRAJECTORY_ACTION` | `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Active UR trajectory action |

### MoverNode

Sends fixed joint-space trajectories to the active UR scaled trajectory
controller. No Cartesian position goals, IK targets, or `compute_cartesian_path`
calls are used for grid movement.

```
MoverNode.move_to(degrees, done_cb, fail_cb)
  → FollowJointTrajectory.Goal
      point 1 = current live joint positions at +0.5 s
      point 2 = target joint positions after a conservative duration
```

The goal runs in a **background daemon thread** so the GUI never freezes.
A `_cancel_flag` allows emergency stop between waypoints.

### Angle wrapping

Before sending, joint angles are wrapped into `[−180, 180)` and resolved to the
nearest equivalent (avoids 350° → 10° spinning the long way):

```python
target = current + wrap(-180..180)(target - current)
```

### Three-step cut sequence

For each selected cell, `_move_next` chains three fixed joint-space MoveIt goals:

```
1. APPROACH  →  move_to(approach_degrees)
                  approach_degrees = grid_degrees with shoulder −11°

2. CUT       →  move_to(grid_degrees)
                  arm moves to the fixed calibrated cell joint pose

3. RECOVER   →  move_to(approach_degrees)
                  returns to the fixed shoulder-lifted approach pose
                  then calls _move_next(remaining)
```

Each step only starts after the previous one confirms success. On any failure
the sequence aborts and the status bar shows the error. E-STOP (`Space` or
button) sets `_cancel_flag` — the current waypoint completes but no new ones
are sent.

---

## Planning Scene (Collision Avoidance) — `planning_scene.py`

**File:** `fruitninja/planning_scene.py`

Published once at startup to `/planning_scene`. MoveIt uses these collision
objects to route the arm around obstacles.

### Objects published

| Object ID | Type | Frame | Description |
|-----------|------|-------|-------------|
| `trolley` | Mesh (`.dae`) | `world` | Full trolley body. Scale 0.01 (DAE in cm). Orientation `Rz(180°)·Rx(90°)`. Position `(0.199, −0.13, 0.0)`. |
| `floor` | Box 3m×3m×2cm | `world` | Slab at z=−0.01. Stops arm reaching under the table. |

> **Note:** In a previous version, `cutting_board_surface` and `back_wall`
> were also published. They have been removed from the current file.

### Mesh loading (`load_dae_mesh`)

Parses the COLLADA file manually (no external mesh loader):
- Reads `<source id="*positions*">` float array for vertex coordinates
- Handles both `<triangles>` and `<polylist>` elements (fan-triangulates polys)
- Applies `scale=0.01` to convert centimetre-authored DAE to metres

---

## Home / Reset Position

**Home pose:** `[0°, −90°, 0°, 0°, 0°, 0°]`

- Pan centered, shoulder pointing down, all other joints neutral
- Keeps the arm vertically above the base, clear of the board
- Triggered by the Reset button in the GUI or the `reset` CLI entry point

---

## Legacy Fixed-Sweep — `movement.py`

**File:** `fruitninja/movement.py` (not used by the main GUI)

An earlier motion mode that sweeps the shoulder_pan across N evenly-spaced
positions while the other joints stay fixed. Used for simple linear cuts before
the grid system was built.

```
HOVER_LIFT = −52°  (arm raised)
CUT_LIFT   = −41°  (blade touching board)
```

For each cut position: hover → lower to cut depth → raise back.

---

## Data Flow Summary

```
Operator clicks cell in GUI
         │
         ▼
cell label (e.g. "C2")
         │
         ▼
grid_mover.cell_to_joints_deg("C2")
  Bilinear interpolation of 4 measured joint poses
         │
         ▼
[pan°, lift°, elbow°, w1°, w2°, w3°]   ← fixed grid joint target
         │
         ├─── approach_degrees = target with shoulder −11°
         │
MoverNode.move_to(degrees)
  FollowJointTrajectory with current joints as first point
         │
         ▼
scaled_joint_trajectory_controller
  Executes timed joint trajectory without an instant first-point jump
         │
         ▼
UR3e executes trajectory via RTDE driver
```
