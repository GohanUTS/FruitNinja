#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e end-effector to any cell in a 7×4 grid (A1–G4).

Positions are computed from a fixed four-corner calibration measured on the
UR3e teach pendant.  The photographed corner readings are stored below as both
tool positions and joint positions; movement uses the bilinearly interpolated
joint positions.

Movement sends timed joint trajectories from the live joint state to the fixed
calibrated grid angles.  Tool positions are logged for checking, but no
Cartesian position goal / IK target is sent.

Coordinate mapping (base_link frame):
  X axis  width  :  A → N  (left → right,  +X direction)
  Y axis  depth  :  row 4 → row 1  (far → near robot,  +Y direction)
  Z              :  interpolated from measured corner heights

Usage (CLI):
  ros2 run fruitninja grid_mover --cell B2
  ros2 run fruitninja grid_mover --trace
"""

import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


REQUIRED_GRID_CORNERS = ('A1', 'G1', 'G4', 'A4')
LEGACY_GRID_CORNER_ALIASES = {
    'G1': 'N1',
    'G4': 'N4',
}

# Cells the GUI can optionally capture in addition to the 4 corners.
# Capturing the full row 1 line + the full column A line lets the Coons-patch
# interpolation honour those edges exactly, which removes the joint-space
# curvature error in middle cells.  Order matters for UI layout only.
EXTENDED_ROW1_CELLS = ('B1', 'C1', 'D1', 'E1', 'F1')
EXTENDED_COL_A_CELLS = ('A2', 'A3')
EXTENDED_CALIBRATION_CELLS = (
    REQUIRED_GRID_CORNERS + EXTENDED_ROW1_CELLS + EXTENDED_COL_A_CELLS
)

GRID_CALIBRATION_ENV = 'FRUITNINJA_GRID_CALIBRATION'
DEFAULT_GRID_CALIBRATION_PATH = '~/.fruitninja/grid_calibration.json'

# Named per-robot profiles.  Each robot can keep its own four-corner joint
# calibration on disk, and the active profile is remembered between launches.
CALIBRATION_PROFILES = ('robot1', 'robot2', 'robot3', 'robot4')
CALIBRATION_PROFILE_DIR = '~/.fruitninja/calibrations'
ACTIVE_PROFILE_MARKER = '~/.fruitninja/active_profile.txt'


def grid_calibration_path() -> str:
    return os.path.expanduser(
        os.environ.get(GRID_CALIBRATION_ENV, DEFAULT_GRID_CALIBRATION_PATH)
    )


def profile_calibration_path(profile_name: str) -> str:
    return os.path.expanduser(
        os.path.join(CALIBRATION_PROFILE_DIR, f'{profile_name}.json')
    )


def get_active_profile() -> str | None:
    p = os.path.expanduser(ACTIVE_PROFILE_MARKER)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            name = f.read().strip()
        return name if name in CALIBRATION_PROFILES else None
    except Exception:
        return None


def set_active_profile(profile_name: str | None) -> None:
    p = os.path.expanduser(ACTIVE_PROFILE_MARKER)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if profile_name is None:
        if os.path.exists(p):
            os.remove(p)
        return
    if profile_name not in CALIBRATION_PROFILES:
        raise ValueError(f'unknown profile {profile_name!r}; expected one of {CALIBRATION_PROFILES}')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(profile_name + '\n')


def list_saved_profiles() -> dict[str, bool]:
    """Return {profile_name: exists_on_disk} for each known profile."""
    return {
        name: os.path.exists(profile_calibration_path(name))
        for name in CALIBRATION_PROFILES
    }


# ── Grid layout ────────────────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
GRID_ROWS = ['1', '2', '3', '4']

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

TRAJECTORY_ACTIONS = [
    '/scaled_joint_trajectory_controller/follow_joint_trajectory',
    '/joint_trajectory_controller/follow_joint_trajectory',
]

START_HOLD_SEC = 0.5
MIN_MOVE_DURATION_SEC = 4.0
MAX_JOINT_SPEED_DEG_S = 10.0

# Direct controller tolerances.  These mirror real_gui_points.py; the old
# MoveIt trace path used tight 0.01 rad constraints and could abort on small
# real-robot settling error before the corner trace completed.
PATH_TOLERANCE_RAD = math.radians(8.0)
GOAL_TOLERANCE_RAD = math.radians(2.0)

# ── Fixed grid calibration — photographed UR teach pendant readings ───────────
#
# Corner assignment from the base joint in the photos:
#   Base  31.07° → A1
#   Base  43.97° → A4
#   Base -30.35° → G1
#   Base -42.83° → G4
#
# Tool positions are stored in metres in the UR/base_link frame.  RX/RY/RZ are
# the UR tool rotation-vector values in radians, kept here for traceability.

CORNER_TOOL_POSES_M = {
    'A1': (-0.24444, -0.30023, -0.34079),
    'A4': (-0.24502, -0.42324, -0.33927),
    'G1': ( 0.25091, -0.30122, -0.33918),
    'G4': ( 0.25719, -0.42685, -0.33850),
}

CORNER_TOOL_ROT_VEC_RAD = {
    'A1': (3.093, -0.047, -0.145),
    'A4': (3.118, -0.057,  0.017),
    'G1': (3.124, -0.060, -0.035),
    'G4': (3.137, -0.034, -0.179),
}

CORNER_JOINTS_DEG = {
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    'A1': ( 31.07, -46.60, 107.83, -157.25, -90.72, 123.03),
    'A4': ( 43.97, -30.44,  65.77, -125.94, -88.64, 136.30),
    'G1': (-30.35, -132.74, -103.43,  -34.39,  88.55, 241.85),
    'G4': (-42.83, -152.31,  -57.19,  294.72,  85.65, 228.51),
}

# Optional extra calibrated cells (poses + orientations) beyond the 4 corners,
# captured via FK during calibration.  Mirror the joint extras below.
EXTRA_CELL_POSES_M: dict = {}
EXTRA_CELL_ROTVEC_RAD: dict = {}

# Optional extra calibrated cells beyond the 4 corners.  Populated by
# apply_calibration() when a calibration file ships extra row/column anchors.
# When non-empty, the interpolation uses a Coons-patch surface that exactly
# honours every captured cell — this removes the joint-space curvature error
# that plagues 4-corner bilinear at middle cells.
EXTRA_CELL_JOINTS_DEG: dict = {}

# ── Per-robot calibration overrides ──────────────────────────────────────────

def _normalise_corner_map(corners: dict, expected_len: int, label: str) -> dict:
    normalised = dict(corners)
    for new_key, old_key in LEGACY_GRID_CORNER_ALIASES.items():
        if new_key not in normalised and old_key in normalised:
            normalised[new_key] = normalised[old_key]

    missing = [c for c in REQUIRED_GRID_CORNERS if c not in normalised]
    if missing:
        raise ValueError(f'{label}: missing corners {missing}')

    for c in REQUIRED_GRID_CORNERS:
        v = normalised[c]
        if len(v) != expected_len:
            raise ValueError(f'{label}: corner {c} expected {expected_len} values, got {len(v)}')
    return normalised


def _apply_extra_map(target: dict, source: dict | None, n: int, label: str) -> None:
    """Replace `target` extras dict from `source`, skipping corners, validating length."""
    if source is None:
        return
    target.clear()
    for cell, vals in source.items():
        cell = str(cell).upper()
        if cell in REQUIRED_GRID_CORNERS:
            continue
        if len(vals) != n:
            raise ValueError(f'{label}: cell {cell} expected {n} values, got {len(vals)}')
        target[cell] = tuple(float(v) for v in vals)


def apply_calibration(poses_m: dict | None = None,
                      rot_vec_rad: dict | None = None,
                      joints_deg: dict | None = None,
                      extra_joints_deg: dict | None = None,
                      extra_poses_m: dict | None = None,
                      extra_rot_vec_rad: dict | None = None) -> None:
    """Mutate the corner + extra-cell tables in place from a captured calibration."""
    if poses_m is not None:
        poses_m = _normalise_corner_map(poses_m, 3, 'corner_tool_poses_m')
        for c in REQUIRED_GRID_CORNERS:
            CORNER_TOOL_POSES_M[c] = tuple(float(x) for x in poses_m[c])
    if rot_vec_rad is not None:
        rot_vec_rad = _normalise_corner_map(rot_vec_rad, 3, 'corner_tool_rot_vec_rad')
        for c in REQUIRED_GRID_CORNERS:
            CORNER_TOOL_ROT_VEC_RAD[c] = tuple(float(x) for x in rot_vec_rad[c])
    if joints_deg is not None:
        joints_deg = _normalise_corner_map(joints_deg, 6, 'corner_joints_deg')
        for c in REQUIRED_GRID_CORNERS:
            CORNER_JOINTS_DEG[c] = tuple(float(x) for x in joints_deg[c])
    _apply_extra_map(EXTRA_CELL_JOINTS_DEG, extra_joints_deg, 6, 'extra_joints_deg')
    _apply_extra_map(EXTRA_CELL_POSES_M, extra_poses_m, 3, 'extra_poses_m')
    _apply_extra_map(EXTRA_CELL_ROTVEC_RAD, extra_rot_vec_rad, 3, 'extra_rot_vec_rad')


def calibrated_cell_joints_map() -> dict:
    """Return {cell_name: (6 floats)} for all calibrated cells (corners + extras)."""
    out = {c: tuple(CORNER_JOINTS_DEG[c]) for c in REQUIRED_GRID_CORNERS}
    out.update(EXTRA_CELL_JOINTS_DEG)
    return out


def calibrated_cell_poses_map() -> dict:
    """Return {cell_name: (x, y, z)} for all cells with a calibrated pose."""
    out = {c: tuple(CORNER_TOOL_POSES_M[c]) for c in REQUIRED_GRID_CORNERS}
    out.update(EXTRA_CELL_POSES_M)
    return out


def calibrated_cell_rotvec_map() -> dict:
    """Return {cell_name: (rx, ry, rz)} for all cells with a calibrated orientation."""
    out = {c: tuple(CORNER_TOOL_ROT_VEC_RAD[c]) for c in REQUIRED_GRID_CORNERS}
    out.update(EXTRA_CELL_ROTVEC_RAD)
    return out


def save_calibration(path: str | None = None,
                     frame_id: str = 'base_link',
                     eef_link: str = 'tool0',
                     profile_name: str | None = None) -> str:
    if profile_name is not None:
        target = profile_calibration_path(profile_name)
    elif path:
        target = os.path.expanduser(path)
    else:
        target = grid_calibration_path()
    all_joints = calibrated_cell_joints_map()
    all_poses = calibrated_cell_poses_map()
    all_rotvec = calibrated_cell_rotvec_map()
    data = {
        'frame_id': frame_id,
        'eef_link': eef_link,
        # Legacy key — kept for backward compatibility with older readers.
        'corner_joints_deg': {c: list(CORNER_JOINTS_DEG[c]) for c in REQUIRED_GRID_CORNERS},
        # New keys — full map of every captured cell (corners + extras).
        'calibrated_cells_joints_deg': {c: list(v) for c, v in all_joints.items()},
        'calibrated_cells_poses_m': {c: list(v) for c, v in all_poses.items()},
        'calibrated_cells_rotvec_rad': {c: list(v) for c, v in all_rotvec.items()},
    }
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    return target


def load_calibration(path: str | None = None,
                     profile_name: str | None = None) -> tuple[bool, str]:
    """Load + apply calibration from JSON. Returns (loaded, message)."""
    if profile_name is not None:
        target = profile_calibration_path(profile_name)
    elif path:
        target = os.path.expanduser(path)
    else:
        target = grid_calibration_path()
    if not os.path.exists(target):
        return False, f'no calibration file at {target}'
    try:
        with open(target, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def _split(full_map, legacy_corner_map):
            """Return (corner_dict_or_None, extras_dict_or_None) from a full cell map."""
            if not full_map:
                return legacy_corner_map, None
            corners = {c: full_map[c] for c in REQUIRED_GRID_CORNERS if c in full_map}
            corners = corners if len(corners) == len(REQUIRED_GRID_CORNERS) else legacy_corner_map
            extras = {c: v for c, v in full_map.items() if c not in REQUIRED_GRID_CORNERS}
            return corners, extras

        corner_joints, extra_joints = _split(
            data.get('calibrated_cells_joints_deg'), data.get('corner_joints_deg'))
        corner_poses, extra_poses = _split(
            data.get('calibrated_cells_poses_m'), data.get('corner_tool_poses_m'))
        corner_rotvec, extra_rotvec = _split(
            data.get('calibrated_cells_rotvec_rad'), data.get('corner_tool_rot_vec_rad'))

        apply_calibration(
            poses_m=corner_poses,
            rot_vec_rad=corner_rotvec,
            joints_deg=corner_joints,
            extra_joints_deg=extra_joints,
            extra_poses_m=extra_poses,
            extra_rot_vec_rad=extra_rotvec,
        )
        return True, target
    except Exception as e:
        return False, f'calibration load failed ({target}): {e}'


_active = get_active_profile()
if _active:
    load_calibration(profile_name=_active)
else:
    load_calibration()
del _active


# Backwards-compatible aliases for older scripts that imported the old A4 anchor
# and average grid spacing constants directly.
A4_X, A4_Y, A4_Z = CORNER_TOOL_POSES_M['A4']
COL_SPACING = (
    (CORNER_TOOL_POSES_M['G1'][0] - CORNER_TOOL_POSES_M['A1'][0]) +
    (CORNER_TOOL_POSES_M['G4'][0] - CORNER_TOOL_POSES_M['A4'][0])
) / (2 * (len(GRID_COLS) - 1))
ROW_SPACING = (
    (CORNER_TOOL_POSES_M['A1'][1] - CORNER_TOOL_POSES_M['A4'][1]) +
    (CORNER_TOOL_POSES_M['G1'][1] - CORNER_TOOL_POSES_M['G4'][1])
) / (2 * (len(GRID_ROWS) - 1))

# ── Cell → fixed-grid interpolation ────────────────────────────────────────────

def _cell_uv(cell: str) -> tuple[float, float]:
    """Return normalised grid coordinates: u=0..1 A→G, v=0..1 row1→row4."""
    cell = cell.strip().upper()
    if len(cell) != 2:
        raise ValueError(f"Cell must be 2 characters e.g. 'B3'. Got: '{cell}'")

    col_char, row_char = cell[0], cell[1]
    if col_char not in GRID_COLS:
        raise ValueError(f"Invalid column '{col_char}'. Valid: {GRID_COLS}")
    if row_char not in GRID_ROWS:
        raise ValueError(f"Invalid row '{row_char}'. Valid: {GRID_ROWS}")

    col_idx = GRID_COLS.index(col_char)   # 0 = A … 6 = G
    row_idx = GRID_ROWS.index(row_char)   # 0 = row1 … 3 = row4
    return col_idx / (len(GRID_COLS) - 1), row_idx / (len(GRID_ROWS) - 1)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bilinear_tuple(corners: dict[str, tuple[float, ...]],
                    u: float,
                    v: float) -> tuple[float, ...]:
    """
    Interpolate a tuple-valued corner table.

    Top edge is A1→G1, bottom edge is A4→G4, then v blends row1→row4.
    """
    n = len(corners['A1'])
    tl, tr, br, bl = REQUIRED_GRID_CORNERS
    return tuple(
        _lerp(
            _lerp(corners[tl][i], corners[tr][i], u),
            _lerp(corners[bl][i], corners[br][i], u),
            v,
        )
        for i in range(n)
    )


def _piecewise_linear(anchors: list[tuple[float, tuple[float, ...]]],
                      t: float) -> tuple[float, ...]:
    """
    1-D piecewise-linear interp through a list of (param, value-tuple) anchors.
    `anchors` is assumed sorted by param.  Clamps `t` to the anchor range.
    """
    if not anchors:
        raise ValueError('_piecewise_linear: no anchors')
    if t <= anchors[0][0]:
        return anchors[0][1]
    if t >= anchors[-1][0]:
        return anchors[-1][1]
    for (t0, v0), (t1, v1) in zip(anchors, anchors[1:]):
        if t0 <= t <= t1:
            span = t1 - t0
            if span <= 1e-12:
                return v0
            f = (t - t0) / span
            return tuple(a + (b - a) * f for a, b in zip(v0, v1))
    return anchors[-1][1]


def _edge_anchors(cells: dict[str, tuple[float, ...]],
                  axis: str,
                  fixed_letter: str) -> list[tuple[float, tuple[float, ...]]]:
    """
    Collect (param, value) anchors along one edge of the grid.

    axis='row'  → walk all columns at the row whose char == fixed_letter.
                  param = col_idx / (n_cols - 1).
    axis='col'  → walk all rows    at the column whose char == fixed_letter.
                  param = row_idx / (n_rows - 1).

    Only includes cells that exist in `cells`.
    """
    out: list[tuple[float, tuple[float, ...]]] = []
    if axis == 'row':
        n = len(GRID_COLS) - 1
        for i, col_char in enumerate(GRID_COLS):
            name = f'{col_char}{fixed_letter}'
            if name in cells:
                out.append((i / n, cells[name]))
    else:
        n = len(GRID_ROWS) - 1
        for i, row_char in enumerate(GRID_ROWS):
            name = f'{fixed_letter}{row_char}'
            if name in cells:
                out.append((i / n, cells[name]))
    return out


def _coons_tuple(cells: dict[str, tuple[float, ...]],
                 u: float,
                 v: float) -> tuple[float, ...]:
    """
    Transfinite (Coons-patch) interpolation over a tuple-valued cell map.

    Uses the four edge curves (row 1, row 4, col A, col G) plus a bilinear
    correction over the 4 corners.  Each edge is piecewise-linear through
    whatever calibration anchors are available on that edge.  When only the
    4 corners are anchored, this collapses exactly to the bilinear surface.

    Formula per component:
        C(u, v) = (1-v)·Row1(u) + v·Row4(u)
                + (1-u)·ColA(v) + u·ColG(v)
                - [(1-u)(1-v)·A1 + u(1-v)·G1 + (1-u)v·A4 + uv·G4]
    """
    row1 = _edge_anchors(cells, 'row', '1')
    row4 = _edge_anchors(cells, 'row', '4')
    col_a = _edge_anchors(cells, 'col', 'A')
    col_g = _edge_anchors(cells, 'col', 'G')

    r1 = _piecewise_linear(row1, u)
    r4 = _piecewise_linear(row4, u)
    ca = _piecewise_linear(col_a, v)
    cg = _piecewise_linear(col_g, v)
    A1, G1, A4, G4 = (cells['A1'], cells['G1'], cells['A4'], cells['G4'])

    def bil(i):
        return ((1 - u) * (1 - v) * A1[i]
                + u * (1 - v) * G1[i]
                + (1 - u) * v * A4[i]
                + u * v * G4[i])

    n = len(A1)
    return tuple(
        (1 - v) * r1[i] + v * r4[i]
        + (1 - u) * ca[i] + u * cg[i]
        - bil(i)
        for i in range(n)
    )


def _joints_interp_tuple(u: float, v: float) -> tuple[float, ...]:
    """
    Pick the right interpolation for the joint table.
    - With no extra calibrated cells: original 4-corner bilinear (unchanged behaviour).
    - With any extras: Coons patch over corners + extras.  This still returns
      the original corners exactly at u,v ∈ {0,1}, but bends to honour the
      extra anchors along row 1 / column A.
    """
    if not EXTRA_CELL_JOINTS_DEG:
        return _bilinear_tuple(CORNER_JOINTS_DEG, u, v)
    cells = calibrated_cell_joints_map()
    return _coons_tuple(cells, u, v)


def _pose_interp_tuple(u: float, v: float) -> tuple[float, ...]:
    """Pose interpolation: Coons patch when extra cell poses exist, else bilinear."""
    if not EXTRA_CELL_POSES_M:
        return _bilinear_tuple(CORNER_TOOL_POSES_M, u, v)
    return _coons_tuple(calibrated_cell_poses_map(), u, v)


def _rotvec_interp_tuple(u: float, v: float) -> tuple[float, ...]:
    """Orientation interpolation: Coons patch when extra rotvecs exist, else bilinear."""
    if not EXTRA_CELL_ROTVEC_RAD:
        return _bilinear_tuple(CORNER_TOOL_ROT_VEC_RAD, u, v)
    return _coons_tuple(calibrated_cell_rotvec_map(), u, v)


def cell_to_pose(cell: str) -> tuple[float, float, float]:
    """
    Return (x, y, z) in metres for the centre of a grid cell.
    Raises ValueError for invalid cell names.
    """
    u, v = _cell_uv(cell)
    return _pose_interp_tuple(u, v)


def grid_uv_to_pose(u: float, v: float) -> tuple[float, float, float]:
    """Return (x, y, z) for normalized grid coordinates u/v in [0, 1]."""
    u = min(1.0, max(0.0, float(u)))
    v = min(1.0, max(0.0, float(v)))
    return _pose_interp_tuple(u, v)


def cell_to_tool_rotvec_rad(cell: str) -> tuple[float, float, float]:
    """Return interpolated UR tool rotation-vector (RX, RY, RZ) in radians."""
    u, v = _cell_uv(cell)
    return _rotvec_interp_tuple(u, v)


def grid_uv_to_rotvec_rad(u: float, v: float) -> tuple[float, float, float]:
    """Return interpolated UR tool rotation-vector for normalized u/v in [0, 1]."""
    u = min(1.0, max(0.0, float(u)))
    v = min(1.0, max(0.0, float(v)))
    return _rotvec_interp_tuple(u, v)


def cell_to_joints_deg(cell: str) -> list[float]:
    """Return interpolated joint angles in degrees (Coons patch if extras exist)."""
    u, v = _cell_uv(cell)
    return list(_joints_interp_tuple(u, v))


def grid_uv_to_joints_deg(u: float, v: float) -> list[float]:
    """Return interpolated joint angles for normalized grid coordinates."""
    u = min(1.0, max(0.0, float(u)))
    v = min(1.0, max(0.0, float(v)))
    return list(_joints_interp_tuple(u, v))


# ── Direct trajectory construction ─────────────────────────────────────────────

def _wrap_deg(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _nearest_equivalent_target(degrees: list[float],
                               current_degrees: list[float]) -> list[float]:
    return [
        current + _wrap_deg(target - current)
        for target, current in zip(degrees, current_degrees)
    ]


def _duration_msg(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1_000_000_000)
    return Duration(sec=sec, nanosec=nanosec)


def _move_duration(current_degrees: list[float],
                   target_degrees: list[float]) -> float:
    max_delta = max(
        abs(target - current)
        for target, current in zip(target_degrees, current_degrees)
    )
    return max(MIN_MOVE_DURATION_SEC, max_delta / MAX_JOINT_SPEED_DEG_S)


def _make_trajectory_goal(node: Node,
                          current_degrees: list[float],
                          target_degrees: list[float]) -> FollowJointTrajectory.Goal:
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    goal.trajectory.header.stamp = node.get_clock().now().to_msg()

    start = JointTrajectoryPoint()
    start.positions = [math.radians(deg) for deg in current_degrees]
    start.velocities = [0.0] * len(JOINT_NAMES)
    start.time_from_start = _duration_msg(START_HOLD_SEC)

    target = JointTrajectoryPoint()
    target.positions = [math.radians(deg) for deg in target_degrees]
    target.velocities = [0.0] * len(JOINT_NAMES)
    target.time_from_start = _duration_msg(
        START_HOLD_SEC + _move_duration(current_degrees, target_degrees)
    )

    goal.trajectory.points = [start, target]
    goal.goal_time_tolerance = _duration_msg(5.0)

    for name in JOINT_NAMES:
        path_t = JointTolerance()
        path_t.name = name
        path_t.position = PATH_TOLERANCE_RAD
        goal.path_tolerance.append(path_t)

        goal_t = JointTolerance()
        goal_t.name = name
        goal_t.position = GOAL_TOLERANCE_RAD
        goal.goal_tolerance.append(goal_t)

    return goal


def _trajectory_error_name(code: int) -> str:
    names = {
        FollowJointTrajectory.Result.SUCCESSFUL: 'SUCCESSFUL',
        FollowJointTrajectory.Result.INVALID_GOAL: 'INVALID_GOAL',
        FollowJointTrajectory.Result.INVALID_JOINTS: 'INVALID_JOINTS',
        FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: 'OLD_HEADER_TIMESTAMP',
        FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: 'PATH_TOLERANCE_VIOLATED',
        FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: 'GOAL_TOLERANCE_VIOLATED',
    }
    return names.get(code, f'UNKNOWN_{code}')


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class GridMoverNode(Node):
    def __init__(self):
        super().__init__('fruitninja_grid_mover')
        self._clients = [
            (name, ActionClient(self, FollowJointTrajectory, name))
            for name in TRAJECTORY_ACTIONS
        ]
        self._current_joints_deg: list[float] | None = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState):
        joints = {
            name: math.degrees(pos)
            for name, pos in zip(msg.name, msg.position)
            if name in JOINT_NAMES
        }
        if all(name in joints for name in JOINT_NAMES):
            self._current_joints_deg = [joints[name] for name in JOINT_NAMES]

    def _wait_for_current_joints(self, timeout_sec: float = 5.0) -> list[float] | None:
        deadline = time.monotonic() + timeout_sec
        while self._current_joints_deg is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._current_joints_deg is None:
            self.get_logger().error(
                'No live joint state yet — check the UR driver and /joint_states'
            )
            return None
        return list(self._current_joints_deg)

    def _select_trajectory_client(self):
        for action_name, client in self._clients:
            if client.wait_for_server(timeout_sec=1.0):
                return action_name, client
        self.get_logger().error(
            'No FollowJointTrajectory action server available. '
            'Check that the UR controller is running and that either '
            'scaled_joint_trajectory_controller or joint_trajectory_controller is active.'
        )
        return None, None

    def _move_to_joints(self, degrees: list[float]) -> bool:
        current = self._wait_for_current_joints()
        if current is None:
            return False

        action_name, client = self._select_trajectory_client()
        if client is None:
            return False

        target = _nearest_equivalent_target(degrees, current)
        self.get_logger().info(f'Using trajectory controller: {action_name}')
        self.get_logger().info(
            f'Move duration: {_move_duration(current, target):.1f} s  '
            f'goal tolerance: {math.degrees(GOAL_TOLERANCE_RAD):.1f}°'
        )

        goal = _make_trajectory_goal(self, current, target)
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error('Trajectory goal send failed')
            return False
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected by controller')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result_msg = result_future.result()
        if result_msg is None:
            self.get_logger().error('Trajectory result unavailable')
            return False
        result = result_msg.result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True

        error_name = _trajectory_error_name(result.error_code)
        error_text = result.error_string or 'no controller error string'
        self.get_logger().error(
            f'Trajectory failed ({error_name}, code: {result.error_code}): {error_text}'
        )
        return False

    def move_to_cell(self, cell: str) -> bool:
        cell_name = cell.strip().upper()
        try:
            x, y, z = cell_to_pose(cell_name)
            target_joints = cell_to_joints_deg(cell_name)
        except ValueError as e:
            self.get_logger().error(str(e))
            return False

        self.get_logger().info(
            f'Grid move → {cell_name}  '
            f'x={x*1000:.1f} mm  y={y*1000:.1f} mm  z={z*1000:.1f} mm'
        )
        joints = ', '.join(f'{j:.2f}°' for j in target_joints)
        self.get_logger().info(f'Interpolated joints → [{joints}]')
        success = self._move_to_joints(target_joints)
        if success:
            self.get_logger().info(f'[OK] Reached {cell_name}')
        else:
            self.get_logger().error(f'[FAIL] {cell_name}')
        return success

    def trace_corners(self) -> bool:
        """Visit all 4 grid corners in order: A1 → G1 → G4 → A4."""
        for cell in REQUIRED_GRID_CORNERS:
            self.get_logger().info(f'Tracing corner {cell}')
            if not self.move_to_cell(cell):
                self.get_logger().error(f'Trace aborted at {cell}')
                return False
        return True


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='Move UR3e to a grid cell')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cell',  type=str,
                       help='Grid cell e.g. A1, B3, G4')
    group.add_argument('--trace', action='store_true',
                       help='Trace all 4 grid corners: A1 → G1 → G4 → A4')
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = GridMoverNode()
    if parsed.trace:
        success = node.trace_corners()
        print('=== Trace complete ===' if success else '=== Trace FAILED ===')
    else:
        success = node.move_to_cell(parsed.cell)
        print('=== Grid move complete ===' if success else '=== Grid move FAILED ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
