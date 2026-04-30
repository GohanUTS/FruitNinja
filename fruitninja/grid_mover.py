#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e end-effector to any cell in a 14×4 grid (A1–N4).

Positions are computed as Cartesian offsets from the measured A4 corner
(read directly from the UR3e teach pendant, base_link frame).

Movement uses MoveGroup joint-space motion planning — not straight-line
Cartesian paths, and not explicit IK constraints.

Coordinate mapping (base_link frame):
  X axis  width  :  A → N  (left → right,  +X direction)
  Y axis  depth  :  row 4 → row 1  (far → near robot,  +Y direction)
  Z              :  fixed at A4 measured height

Usage (CLI):
  ros2 run fruitninja grid_mover --cell B2
  ros2 run fruitninja grid_mover --trace
"""

import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, BoundingVolume, MoveItErrorCodes,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion


# ── Grid layout ────────────────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
GRID_ROWS = ['1', '2', '3', '4']

MOVE_GROUP   = 'ur_manipulator'
END_EFFECTOR = 'tool0'
BASE_FRAME   = 'base_link'

# ── A4 anchor — read from UR3e teach pendant (metres, base_link frame) ─────────
#
#   Pendant reading at A4:
#     Tool X = −237.73 mm   Tool Y = −445.58 mm   Tool Z = −351.88 mm
#     Base=46.85°  Shoulder=−29.30°  Elbow=57.63°  W1=−119.30°  W2=−90°  W3=7.67°

A4_X = -0.23773   # m
A4_Y = -0.44558   # m
A4_Z = -0.35188   # m

# ── Grid spacing derived from physical board dimensions ────────────────────────
#   Width : 500 mm across 13 column steps  (A=col0 → N=col13, +X direction)
#   Depth : 160 mm across  3 row    steps  (row4=near A4 → row1=+Y toward robot)

COL_SPACING = 0.500 / 13   # ≈ 38.46 mm per column (+X)
ROW_SPACING = 0.160 / 3    # ≈ 53.33 mm per row    (+Y toward robot)

POSITION_TOL = 0.010   # 10 mm sphere tolerance


# ── Cell → Cartesian pose ──────────────────────────────────────────────────────

def cell_to_pose(cell: str) -> tuple[float, float, float]:
    """
    Return (x, y, z) in metres for the centre of a grid cell.
    Raises ValueError for invalid cell names.
    """
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

    x = A4_X + col_idx * COL_SPACING            # A4 is col0, steps go +X
    y = A4_Y + (3 - row_idx) * ROW_SPACING      # A4 is row4 (offset 0), row1 = +3 steps
    z = A4_Z

    return x, y, z


# ── MoveGroup goal construction ────────────────────────────────────────────────

def _make_pose_goal(x: float, y: float, z: float) -> Constraints:
    """Position-only goal — no orientation constraint so the planner has full freedom."""
    c = Constraints()

    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[POSITION_TOL])
    target = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(w=1.0))
    bv = BoundingVolume()
    bv.primitives.append(sphere)
    bv.primitive_poses.append(target)

    pc = PositionConstraint()
    pc.header.frame_id    = BASE_FRAME
    pc.link_name          = END_EFFECTOR
    pc.constraint_region  = bv
    pc.weight             = 1.0
    c.position_constraints.append(pc)

    return c


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class GridMoverNode(Node):
    def __init__(self):
        super().__init__('fruitninja_grid_mover')
        self._client = ActionClient(self, MoveGroup, '/move_action')

    def _move_to_xyz(self, x: float, y: float, z: float) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP
        goal.request.goal_constraints.append(_make_pose_goal(x, y, z))
        goal.request.num_planning_attempts           = 10
        goal.request.allowed_planning_time           = 10.0
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
        try:
            x, y, z = cell_to_pose(cell)
        except ValueError as e:
            self.get_logger().error(str(e))
            return False

        self.get_logger().info(
            f'Grid move → {cell.upper()}  '
            f'x={x*1000:.1f} mm  y={y*1000:.1f} mm  z={z*1000:.1f} mm'
        )
        success = self._move_to_xyz(x, y, z)
        if success:
            self.get_logger().info(f'[OK] Reached {cell.upper()}')
        else:
            self.get_logger().error(f'[FAIL] {cell.upper()}')
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
