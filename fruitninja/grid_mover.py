#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e end-effector to any cell in a 14×4 grid (A1–N4).

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
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ── Grid layout ────────────────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
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
#   Base -30.35° → N1
#   Base -42.83° → N4
#
# Tool positions are stored in metres in the UR/base_link frame.  RX/RY/RZ are
# the UR tool rotation-vector values in radians, kept here for traceability.

CORNER_TOOL_POSES_M = {
    'A1': (-0.24444, -0.30023, -0.34079),
    'A4': (-0.24502, -0.42324, -0.33927),
    'N1': ( 0.25091, -0.30122, -0.33918),
    'N4': ( 0.25719, -0.42685, -0.33850),
}

CORNER_TOOL_ROT_VEC_RAD = {
    'A1': (3.093, -0.047, -0.145),
    'A4': (3.118, -0.057,  0.017),
    'N1': (3.124, -0.060, -0.035),
    'N4': (3.137, -0.034, -0.179),
}

CORNER_JOINTS_DEG = {
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    'A1': ( 31.07, -46.60, 107.83, -157.25, -90.72, 123.03),
    'A4': ( 43.97, -30.44,  65.77, -125.94, -88.64, 136.30),
    'N1': (-30.35, -132.74, -103.43,  -34.39,  88.55, 241.85),
    'N4': (-42.83, -152.31,  -57.19,  294.72,  85.65, 228.51),
}

# Backwards-compatible aliases for older scripts that imported the old A4 anchor
# and average grid spacing constants directly.
A4_X, A4_Y, A4_Z = CORNER_TOOL_POSES_M['A4']
COL_SPACING = (
    (CORNER_TOOL_POSES_M['N1'][0] - CORNER_TOOL_POSES_M['A1'][0]) +
    (CORNER_TOOL_POSES_M['N4'][0] - CORNER_TOOL_POSES_M['A4'][0])
) / (2 * (len(GRID_COLS) - 1))
ROW_SPACING = (
    (CORNER_TOOL_POSES_M['A1'][1] - CORNER_TOOL_POSES_M['A4'][1]) +
    (CORNER_TOOL_POSES_M['N1'][1] - CORNER_TOOL_POSES_M['N4'][1])
) / (2 * (len(GRID_ROWS) - 1))

# ── Cell → fixed-grid interpolation ────────────────────────────────────────────

def _cell_uv(cell: str) -> tuple[float, float]:
    """Return normalised grid coordinates: u=0..1 A→N, v=0..1 row1→row4."""
    cell = cell.strip().upper()
    if len(cell) != 2:
        raise ValueError(f"Cell must be 2 characters e.g. 'B3'. Got: '{cell}'")

    col_char, row_char = cell[0], cell[1]
    if col_char not in GRID_COLS:
        raise ValueError(f"Invalid column '{col_char}'. Valid: {GRID_COLS}")
    if row_char not in GRID_ROWS:
        raise ValueError(f"Invalid row '{row_char}'. Valid: {GRID_ROWS}")

    col_idx = GRID_COLS.index(col_char)   # 0 = A … 13 = N
    row_idx = GRID_ROWS.index(row_char)   # 0 = row1 … 3 = row4
    return col_idx / (len(GRID_COLS) - 1), row_idx / (len(GRID_ROWS) - 1)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bilinear_tuple(corners: dict[str, tuple[float, ...]],
                    u: float,
                    v: float) -> tuple[float, ...]:
    """
    Interpolate a tuple-valued corner table.

    Top edge is A1→N1, bottom edge is A4→N4, then v blends row1→row4.
    """
    n = len(corners['A1'])
    return tuple(
        _lerp(
            _lerp(corners['A1'][i], corners['N1'][i], u),
            _lerp(corners['A4'][i], corners['N4'][i], u),
            v,
        )
        for i in range(n)
    )


def cell_to_pose(cell: str) -> tuple[float, float, float]:
    """
    Return (x, y, z) in metres for the centre of a grid cell.
    Raises ValueError for invalid cell names.
    """
    u, v = _cell_uv(cell)
    return _bilinear_tuple(CORNER_TOOL_POSES_M, u, v)


def grid_uv_to_pose(u: float, v: float) -> tuple[float, float, float]:
    """Return (x, y, z) for normalized grid coordinates u/v in [0, 1]."""
    u = min(1.0, max(0.0, float(u)))
    v = min(1.0, max(0.0, float(v)))
    return _bilinear_tuple(CORNER_TOOL_POSES_M, u, v)


def cell_to_tool_rotvec_rad(cell: str) -> tuple[float, float, float]:
    """Return interpolated UR tool rotation-vector (RX, RY, RZ) in radians."""
    u, v = _cell_uv(cell)
    return _bilinear_tuple(CORNER_TOOL_ROT_VEC_RAD, u, v)


def cell_to_joints_deg(cell: str) -> list[float]:
    """Return interpolated corner-calibrated joint angles in degrees."""
    u, v = _cell_uv(cell)
    return list(_bilinear_tuple(CORNER_JOINTS_DEG, u, v))


def grid_uv_to_joints_deg(u: float, v: float) -> list[float]:
    """Return interpolated joint angles for normalized grid coordinates."""
    u = min(1.0, max(0.0, float(u)))
    v = min(1.0, max(0.0, float(v)))
    return list(_bilinear_tuple(CORNER_JOINTS_DEG, u, v))


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
        """Visit all 4 grid corners in order: A1 → N1 → N4 → A4."""
        for cell in ['A1', 'N1', 'N4', 'A4']:
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
                       help='Grid cell e.g. A1, B3, N4')
    group.add_argument('--trace', action='store_true',
                       help='Trace all 4 grid corners: A1 → N1 → N4 → A4')
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
