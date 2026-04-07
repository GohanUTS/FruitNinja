#!/usr/bin/env python3
"""
FruitNinja Grid Mover
=====================
Moves the UR3e to a named grid cell (e.g. "A1", "C3").

Grid layout matches the vision grid in colour_detection.py:
  Columns A–N → shoulder_pan  (left → right, 14 cols, 73 cm wide)
  Rows    1–4 → shoulder_lift (top  → bottom / high → low, 22 cm long)

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
    'wrist_1_joint': math.radians(-152),
    'wrist_2_joint': math.radians(-100),
    'wrist_3_joint': math.radians(110),
}

# ── Grid defaults ──────────────────────────────────────────────────────────────
#
# Columns A-N (14) map to shoulder_pan across the 73 cm wide cutting area.
# Rows    1-4  (4) map to depth (front-to-back) via coupled elbow+lift IK,
# keeping Z constant at cutting height (~0.765 m).
#
#   Row 1 — far end  (~54 cm from robot base)
#   Row 4 — near end (~33 cm from robot base)

GRID_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
GRID_ROWS = ['1', '2', '3', '4']

PAN_MIN_DEG  = -76.5   # column A — left edge of cutting zone
PAN_MAX_DEG  =  43.4   # column N — right edge of cutting zone

# Per-row (lift_deg, elbow_deg) pairs derived from IK at constant cutting height.
# Row 1 = far end (away from robot), Row 4 = near end (close to robot).
ROW_CONFIGS = {
    '1': (0.0,   30.0),   # far end — arm nearly extended
    '2': (-17.0, 60.0),   # mid-far
    '3': (-34.0, 90.0),   # mid-near
    '4': (-51.0, 120.0),  # near end (confirmed correct)
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def cell_to_joints(cell: str, pan_min: float, pan_max: float):
    """
    Convert a cell name like 'B3' to (pan_rad, lift_rad, elbow_rad).
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
    n_cols  = len(GRID_COLS)

    # Centre of each column cell
    col_t   = (col_idx + 0.5) / n_cols
    pan_rad = math.radians(_lerp(pan_min, pan_max, col_t))

    lift_deg, elbow_deg = ROW_CONFIGS[row_char]
    lift_rad  = math.radians(lift_deg)
    elbow_rad = math.radians(elbow_deg)

    return pan_rad, lift_rad, elbow_rad


def _make_joint_goal(pan: float, lift: float, elbow: float) -> Constraints:
    c = Constraints()
    values = {
        'shoulder_pan_joint':  pan,
        'shoulder_lift_joint': lift,
        'elbow_joint':         elbow,
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
                 pan_min: float = PAN_MIN_DEG,
                 pan_max: float = PAN_MAX_DEG):
        super().__init__('fruitninja_grid_mover')
        self._client  = ActionClient(self, MoveGroup, '/move_action')
        self._pan_min = pan_min
        self._pan_max = pan_max

    def move_to_cell(self, cell: str) -> bool:
        """Move the robot to the centre of the named grid cell. Returns True on success."""
        try:
            pan, lift, elbow = cell_to_joints(cell, self._pan_min, self._pan_max)
        except ValueError as e:
            self.get_logger().error(str(e))
            return False

        self.get_logger().info(
            f'Grid move → cell {cell.upper()}  '
            f'(pan={math.degrees(pan):.1f}°  lift={math.degrees(lift):.1f}°  '
            f'elbow={math.degrees(elbow):.1f}°)'
        )

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(_make_joint_goal(pan, lift, elbow))
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
        print(f'\n{"Cell":>6}  {"Pan (°)":>9}  {"Lift (°)":>9}  {"Elbow (°)":>10}')
        print('-' * 42)
        for row in GRID_ROWS:
            col = GRID_COLS[0]   # pan is same for all cols in a row, just show col A
            cell = col + row
            pan, lift, elbow = cell_to_joints(cell, self._pan_min, self._pan_max)
            lift_deg, elbow_deg = ROW_CONFIGS[row]
            print(f'  Row {row}  pan A={math.degrees(pan):>+6.1f}°  '
                  f'lift={lift_deg:>+6.1f}°  elbow={elbow_deg:>+6.1f}°')
        print(f'\n  Pan range: A={self._pan_min:+.1f}°  N={self._pan_max:+.1f}°\n')


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='Move UR3e to a grid cell')
    parser.add_argument('--cell',      type=str,   required=True,
                        help='Grid cell, e.g. A1, B3, N4')
    parser.add_argument('--pan-min',   type=float, default=PAN_MIN_DEG,
                        help=f'Pan angle for column A (deg, default {PAN_MIN_DEG})')
    parser.add_argument('--pan-max',   type=float, default=PAN_MAX_DEG,
                        help=f'Pan angle for column N (deg, default {PAN_MAX_DEG})')
    parser.add_argument('--show-grid', action='store_true',
                        help='Print the full grid layout and exit without moving')
    parsed, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)
    node = GridMoverNode(
        pan_min = parsed.pan_min,
        pan_max = parsed.pan_max,
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
