#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e to a named grid cell (e.g. "A1", "C3").

Grid layout matches the vision grid in colour_detection.py:
  Columns A–D → shoulder_pan  (left → right)
  Rows    1–4 → shoulder_lift (top  → bottom / high → low)

Usage (CLI):
  ros2 run fruitninja grid_mover --cell B2
  ros2 run fruitninja grid_mover --cell A1 --lift-row-min -55 --lift-row-max -41

Importable API:
  from fruitninja.grid_mover import GridMoverNode
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


# ── Joint names & fixed wrist configuration (same as movement.py) ─────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

MOVE_GROUP = 'ur_manipulator'

FIXED_JOINTS = {
    'elbow_joint':   math.radians(103),
    'wrist_1_joint': math.radians(-152),
    'wrist_2_joint': math.radians(-100),
    'wrist_3_joint': math.radians(110),
}

# ── Grid defaults ──────────────────────────────────────────────────────────────
#
# Columns A-D map to shoulder_pan across the cutting sweep range.
# Rows    1-4 map to shoulder_lift between hover height and cut height.
#
# Defaults match the existing cutting configuration:
#   pan  : centre=-13°  half_range=28°  →  -41° (col A) … +15° (col D)
#   lift : -52° (row 1, high/hover)     …  -41° (row 4, low/cut)

GRID_COLS = ['A', 'B', 'C', 'D']
GRID_ROWS = ['1', '2', '3', '4']

PAN_MIN_DEG  = -41.0   # column A
PAN_MAX_DEG  =  15.0   # column D
LIFT_MIN_DEG = -52.0   # row 1  (highest — hover)
LIFT_MAX_DEG = -41.0   # row 4  (lowest  — cut height)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def cell_to_joints(cell: str,
                   pan_min: float, pan_max: float,
                   lift_min: float, lift_max: float):
    """
    Convert a cell name like 'B3' to (pan_rad, lift_rad).
    Raises ValueError if the cell name is invalid.
    """
    cell = cell.strip().upper()
    if len(cell) != 2:
        raise ValueError(f"Cell must be 2 characters, e.g. 'B3'. Got: '{cell}'")

    col_char, row_char = cell[0], cell[1]
    if col_char not in GRID_COLS:
        raise ValueError(f"Invalid column '{col_char}'. Valid: {GRID_COLS}")
    if row_char not in GRID_ROWS:
        raise ValueError(f"Invalid row '{row_char}'. Valid: {GRID_ROWS}")

    col_idx = GRID_COLS.index(col_char)        # 0 … len-1
    row_idx = GRID_ROWS.index(row_char)

    n_cols = len(GRID_COLS)
    n_rows = len(GRID_ROWS)

    # Centre of each cell
    col_t = (col_idx + 0.5) / n_cols
    row_t = (row_idx + 0.5) / n_rows

    pan_rad  = math.radians(_lerp(pan_min,  pan_max,  col_t))
    lift_rad = math.radians(_lerp(lift_min, lift_max, row_t))
    return pan_rad, lift_rad


def _make_joint_goal(pan: float, lift: float) -> Constraints:
    c = Constraints()
    values = {
        'shoulder_pan_joint':  pan,
        'shoulder_lift_joint': lift,
        **FIXED_JOINTS,
    }
    for name in JOINT_NAMES:
        jc = JointConstraint()
        jc.joint_name      = name
        jc.position        = values[name]
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight          = 1.0
        c.joint_constraints.append(jc)
    return c


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class GridMoverNode(Node):
    def __init__(self,
                 pan_min:  float = PAN_MIN_DEG,
                 pan_max:  float = PAN_MAX_DEG,
                 lift_min: float = LIFT_MIN_DEG,
                 lift_max: float = LIFT_MAX_DEG):
        super().__init__('fruitninja_grid_mover')
        self._client   = ActionClient(self, MoveGroup, '/move_action')
        self._pan_min  = pan_min
        self._pan_max  = pan_max
        self._lift_min = lift_min
        self._lift_max = lift_max

    def move_to_cell(self, cell: str) -> bool:
        """Move the robot to the centre of the named grid cell. Returns True on success."""
        try:
            pan, lift = cell_to_joints(
                cell,
                self._pan_min, self._pan_max,
                self._lift_min, self._lift_max,
            )
        except ValueError as e:
            self.get_logger().error(str(e))
            return False

        self.get_logger().info(
            f'Grid move → cell {cell.upper()}  '
            f'(pan={math.degrees(pan):.1f}°  lift={math.degrees(lift):.1f}°)'
        )

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(_make_joint_goal(pan, lift))
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
            self.get_logger().info(f'[OK] Reached cell {cell.upper()}')
            return True

        self.get_logger().error(
            f'[FAIL] Cell {cell.upper()} — error code: {result.error_code.val}'
        )
        return False

    def print_grid(self):
        """Print the full grid with joint angles for reference."""
        print(f'\n{"Cell":>6}  {"Pan (°)":>9}  {"Lift (°)":>9}')
        print('-' * 30)
        for row in GRID_ROWS:
            for col in GRID_COLS:
                cell = col + row
                pan, lift = cell_to_joints(
                    cell,
                    self._pan_min, self._pan_max,
                    self._lift_min, self._lift_max,
                )
                print(f'  {cell:>4}  {math.degrees(pan):>+9.1f}  {math.degrees(lift):>+9.1f}')
        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='Move UR3e to a grid cell')
    parser.add_argument('--cell',          type=str,   required=True,
                        help='Grid cell, e.g. A1, B3, D4')
    parser.add_argument('--pan-min',       type=float, default=PAN_MIN_DEG,
                        help=f'Pan angle for column A (deg, default {PAN_MIN_DEG})')
    parser.add_argument('--pan-max',       type=float, default=PAN_MAX_DEG,
                        help=f'Pan angle for column D (deg, default {PAN_MAX_DEG})')
    parser.add_argument('--lift-row-min',  type=float, default=LIFT_MIN_DEG,
                        help=f'Lift angle for row 1 (deg, default {LIFT_MIN_DEG})')
    parser.add_argument('--lift-row-max',  type=float, default=LIFT_MAX_DEG,
                        help=f'Lift angle for row 4 (deg, default {LIFT_MAX_DEG})')
    parser.add_argument('--show-grid',     action='store_true',
                        help='Print the full grid layout and exit without moving')
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = GridMoverNode(
        pan_min  = parsed.pan_min,
        pan_max  = parsed.pan_max,
        lift_min = parsed.lift_row_min,
        lift_max = parsed.lift_row_max,
    )

    if parsed.show_grid:
        node.print_grid()
        node.destroy_node()
        rclpy.shutdown()
        return

    node.print_grid()
    success = node.move_to_cell(parsed.cell)
    print('=== Grid move complete ===' if success else '=== Grid move FAILED ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
