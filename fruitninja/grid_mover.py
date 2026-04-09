#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e to any cell in a 14×4 grid (A1–N4).

The grid is defined by 4 measured corner positions:
  A1 (col=0, row=0) — far-left
  A4 (col=0, row=3) — near-left
  N1 (col=13, row=0) — far-right
  N4 (col=13, row=3) — near-right

All intermediate cells are computed via bilinear interpolation in joint space.

Usage (CLI):
  ros2 run fruitninja grid_mover --cell B2

Importable API:
  from fruitninja.grid_mover import cell_to_joints_deg, GridMoverNode
  node = GridMoverNode()
  node.move_to_cell('C3')
"""

import argparse
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


# ── Joint names ────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

MOVE_GROUP = 'ur_manipulator'

# ── Grid layout ────────────────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
GRID_ROWS = ['1', '2', '3', '4']

# Measured corner joint angles in degrees [pan, lift, elbow, w1, w2, w3]
# u=0 → column A,  u=1 → column N
# v=0 → row 1,     v=1 → row 4
CORNER_A1 = [-47.63,  -174.40,   -1.88,  -94.15,  92.29, 353.57]  # u=0, v=0
CORNER_A4 = [-32.58,  -133.07,  -74.35,  -55.93,  92.23,   8.03]  # u=0, v=1
CORNER_N1 = [-110.86, -171.95,   -8.95,  -91.13,  90.68, 290.33]  # u=1, v=0
CORNER_N4 = [-115.72, -140.00,  -71.39,  -60.67,  90.44, 285.49]  # u=1, v=1


# ── Interpolation ──────────────────────────────────────────────────────────────

def cell_to_joints_deg(cell: str) -> list:
    """
    Bilinear interpolation across the 4 measured corners.
    Returns a list of 6 joint angles in degrees.
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

    col_idx = GRID_COLS.index(col_char)   # 0 … 13
    row_idx = GRID_ROWS.index(row_char)   # 0 … 3

    u = col_idx / (len(GRID_COLS) - 1)   # 0.0 → 1.0  (A → N)
    v = row_idx / (len(GRID_ROWS) - 1)   # 0.0 → 1.0  (row1 → row4)

    return [
        (1 - u) * (1 - v) * CORNER_A1[i]
        + u       * (1 - v) * CORNER_N1[i]
        + (1 - u) * v       * CORNER_A4[i]
        + u       * v       * CORNER_N4[i]
        for i in range(6)
    ]


def _make_joint_goal(degrees: list) -> Constraints:
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

    def move_to_cell(self, cell: str) -> bool:
        """Move to the named grid cell. Returns True on success."""
        try:
            degrees = cell_to_joints_deg(cell)
        except ValueError as e:
            self.get_logger().error(str(e))
            return False

        labels = ['pan', 'lift', 'elbow', 'w1', 'w2', 'w3']
        info = '  '.join(f'{l}={d:+.1f}°' for l, d in zip(labels, degrees))
        self.get_logger().info(f'Grid move → {cell.upper()}  {info}')

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(_make_joint_goal(degrees))
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor     = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected for cell {cell}')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'[OK] Reached {cell.upper()}')
            return True

        self.get_logger().error(
            f'[FAIL] {cell.upper()} — error code: {result.error_code.val}'
        )
        return False


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='Move UR3e to a grid cell')
    parser.add_argument('--cell', type=str, required=True,
                        help='Grid cell e.g. A1, B3, N4')
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = GridMoverNode()
    success = node.move_to_cell(parsed.cell)
    print('=== Grid move complete ===' if success else '=== Grid move FAILED ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
