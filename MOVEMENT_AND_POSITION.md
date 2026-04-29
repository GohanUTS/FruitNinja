# FruitNinja — Movement & Position System

## Overview

The robot is a **UR3e** arm controlled through **ROS 2 + MoveIt**. All motion
goes through the MoveIt `/move_action` action server, which handles collision
avoidance, path planning and trajectory execution.

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

The cutting board corners (measured physically):

| Corner | col | row | x (m)   | y (m)   | z (m)  |
|--------|-----|-----|---------|---------|--------|
| A1     | 0   | 0   | +0.3124 | +0.4549 | 0.0875 |
| A4     | 0   | 3   | +0.3130 | +0.3493 | 0.0875 |
| N1     | 13  | 0   | −0.3124 | +0.4549 | 0.0875 |
| N4     | 13  | 3   | −0.3130 | +0.3493 | 0.0875 |

Board surface is at **z = 0.0875 m**.

---

## Grid Cell Lookup — `grid_mover.py`

**File:** `fruitninja/grid_mover.py`

Converts a cell label (e.g. `"C2"`) to 6 joint angles in degrees using
**bilinear interpolation in joint space** across 4 measured corner poses.

### Measured corner joint angles (degrees)

| Corner | pan    | lift    | elbow  | wrist1  | wrist2 | wrist3 |
|--------|--------|---------|--------|---------|--------|--------|
| A1     | −47.63 | −174.40 | −1.88  | −94.15  | 92.29  | 353.57 |
| A4     | −32.58 | −133.07 | −74.35 | −55.93  | 92.23  | 8.03   |
| N1     | −110.86| −171.95 | −8.95  | −91.13  | 90.68  | 290.33 |
| N4     | −115.72| −140.00 | −71.39 | −60.67  | 90.44  | 285.49 |

### Interpolation steps

1. Convert column letter → `col_idx` (0–13), row number → `row_idx` (0–3)
2. Compute `u = col_idx / 13` (0 = A, 1 = N) — then **mirror**: `u = 1 - u`
3. Compute `v = row_idx / 3`  (0 = row1, 1 = row4) — then **mirror**: `v = 1 - v`
4. Remap into board inset: `u ∈ [0.14, 0.86]`, `v ∈ [-0.035, 1.035]`
5. Bilinear interpolate each of the 6 joints independently

The mirroring and inset were calibrated to keep the arm on the board surface
and away from the trolley frame edges. The slightly wider `v` span gives the
four row positions more separation without changing the left/right column inset.

```
cell_to_joints_deg("C2")  →  [pan°, lift°, elbow°, w1°, w2°, w3°]
```

---

## Motion Execution — `real_gui_points.py`

**File:** `fruitninja/real_gui_points.py`

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `HOME_DEG` | `[0, -90, 0, 0, 0, 0]` | Safe upright home pose |
| `CUT_DIP_LIFT` | `+40°` | Shoulder delta for cutting dip |
| `CUT_DIP_ELBOW` | `−40°` | Elbow delta for cutting dip |
| `MOVE_GROUP` | `'ur_manipulator'` | MoveIt planning group |

### MoverNode

Sends joint-space goals to MoveIt's `/move_action` server.

```
MoverNode.move_to(degrees, done_cb, fail_cb)
  → _make_constraints(degrees)
      → JointConstraint × 6 joints
         position     = math.radians(deg)
         tolerance    = ±0.01 rad
         weight       = 1.0
  → MoveGroup.Goal
      group_name                    = 'ur_manipulator'
      num_planning_attempts         = 10
      allowed_planning_time         = 5.0 s
      max_velocity_scaling_factor   = 0.3  (30%)
      max_acceleration_scaling_factor = 0.3
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

For each selected cell, `_move_next` chains three MoveIt calls:

```
1. APPROACH  →  move_to(degrees)
                  end-effector arrives at cell position (lookup angles)

2. DIP       →  move_to(dip_degrees)
                  dip_degrees[1] += 40°  (shoulder)
                  dip_degrees[2] -= 40°  (elbow)
                  pushes end-effector downward into the board

3. RECOVER   →  move_to(degrees)
                  lifts back to approach position
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
  Bilinear interpolation of 4 measured corners
         │
         ▼
[pan°, lift°, elbow°, w1°, w2°, w3°]   ← approach angles
         │
         ├─── dip_degrees = approach ± (40°, −40°) on joints 1,2
         │
         ▼
MoverNode._make_constraints(degrees)
  JointConstraint × 6, tolerance ±0.01 rad
         │
         ▼
MoveGroup.Goal → /move_action (MoveIt)
  MoveIt plans collision-free trajectory
  Executes at 30% velocity/acceleration
         │
         ▼
UR3e executes trajectory via RTDE driver
```
