#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e end-effector to any cell in a 14×4 grid (A1–N4).

Positions are computed from a fixed four-corner calibration measured on the
UR3e teach pendant.  The photographed corner readings are stored below as both
tool positions and joint positions; movement uses the bilinearly interpolated
joint positions.

Movement uses fixed joint-angle goals from the calibrated grid.  Tool positions
are logged for checking, but no Cartesian position goal / IK target is sent.

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
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


# ── Grid layout ────────────────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
GRID_ROWS = ['1', '2', '3', '4']

MOVE_GROUP   = 'ur_manipulator'

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# ── Fixed grid calibration — photographed UR teach pendant readings ───────────
#
# Corner assignment from the base joint in the photos:
#   Base  30.42° → A1
#   Base  43.35° → A4
#   Base 107.42° → N1
#   Base 101.90° → N4
#
# Tool positions are stored in metres in the UR/base_link frame.  RX/RY/RZ are
# the UR tool rotation-vector values in radians, kept here for traceability.

CORNER_TOOL_POSES_M = {
    'A1': (-0.25322, -0.30822, -0.37047),
    'A4': (-0.25440, -0.43063, -0.36707),
    'N1': ( 0.24152, -0.31635, -0.36815),
    'N4': ( 0.24024, -0.44040, -0.36745),
}

CORNER_TOOL_ROT_VEC_RAD = {
    'A1': (2.193, 2.151,  0.095),
    'A4': (2.155, 2.203,  0.087),
    'N1': (2.187, 2.141, -0.001),
    'N4': (2.159, 2.176,  0.114),
}

CORNER_JOINTS_DEG = {
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    'A1': ( 30.42, -46.19,  98.65, -138.12, -87.00,  31.65),
    'A4': ( 43.35, -27.50,  56.58, -115.76, -86.63,  42.26),
    'N1': (107.42, -46.70, 104.71, -152.19, -87.90, 108.96),
    'N4': (101.90, -28.80,  61.76, -124.25, -83.95, 101.79),
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


# ── MoveGroup joint-goal construction ──────────────────────────────────────────

def _make_joint_goal(degrees: list[float]) -> Constraints:
    """Joint-only goal. This avoids Cartesian position goals and IK."""
    c = Constraints()
    for name, deg in zip(JOINT_NAMES, degrees):
        jc = JointConstraint()
        jc.joint_name      = name
        jc.position        = math.radians(deg)
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight          = 1.0
        c.joint_constraints.append(jc)
    return c


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class GridMoverNode(Node):
    def __init__(self):
        super().__init__('fruitninja_grid_mover')
        self._client = ActionClient(self, MoveGroup, '/move_action')

    def _move_to_joints(self, degrees: list[float]) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP
        goal.request.goal_constraints.append(_make_joint_goal(degrees))
        goal.request.num_planning_attempts           = 10
        goal.request.allowed_planning_time           = 5.0
        goal.request.max_velocity_scaling_factor     = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveIt')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            return True

        self.get_logger().error(f'Planning failed (code: {result.error_code.val})')
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
