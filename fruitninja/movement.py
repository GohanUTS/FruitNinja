#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
import time


JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

MOVE_GROUP = 'ur_manipulator'

# ── Cutting configuration ──────────────────────────────────────────────────────
#
# Hover:  shoulder_lift = -52°  (arm raised clear of table)
# Cut:    shoulder_lift = -41°  (blade touching table)
#
# shoulder_pan sweeps from PAN_START to PAN_END across NUM_CUTS positions.
# All other joints remain fixed throughout.

HOVER_LIFT = math.radians(-52)
CUT_LIFT   = math.radians(-41)

PAN_START  = math.radians(15)
PAN_END    = math.radians(-41)
NUM_CUTS   = 5

FIXED_JOINTS = {
    'elbow_joint':   math.radians(103),
    'wrist_1_joint': math.radians(-152),
    'wrist_2_joint': math.radians(-100),
    'wrist_3_joint': math.radians(110),
}

READY_POSITION = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_joint_goal(pan, lift) -> Constraints:
    c = Constraints()
    values = {
        'shoulder_pan_joint':  pan,
        'shoulder_lift_joint': lift,
        **FIXED_JOINTS,
    }
    for name in JOINT_NAMES:
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = values[name]
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


def pan_positions() -> list:
    if NUM_CUTS == 1:
        return [(PAN_START + PAN_END) / 2.0]
    step = (PAN_END - PAN_START) / (NUM_CUTS - 1)
    return [PAN_START + i * step for i in range(NUM_CUTS)]


# ── Node ───────────────────────────────────────────────────────────────────────

class MovementNode(Node):
    def __init__(self):
        super().__init__('fruitninja_movement')
        self._client = ActionClient(self, MoveGroup, '/move_action')

    def move_to(self, pan: float, lift: float, label: str) -> bool:
        self.get_logger().info(
            f'Moving to: {label}  '
            f'(pan={math.degrees(pan):.1f}°  lift={math.degrees(lift):.1f}°)'
        )

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(make_joint_goal(pan, lift))
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected: {label}')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'[OK] {label}')
            return True
        self.get_logger().error(f'[FAIL] {label} — error: {result.error_code.val}')
        return False

    def move_to_ready(self) -> bool:
        self.get_logger().info('Moving to: Ready position')
        if not self._client.wait_for_server(timeout_sec=5.0):
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        c = Constraints()
        for name, pos in zip(JOINT_NAMES, READY_POSITION):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('[OK] Ready position')
            return True
        self.get_logger().error(f'[FAIL] Ready position — error: {result.error_code.val}')
        return False

    def perform_cuts(self) -> None:
        pans = pan_positions()
        for i, pan in enumerate(pans):
            label = f'Cut {i+1}/{NUM_CUTS}  pan={math.degrees(pan):.1f}°'
            self.get_logger().info(f'--- {label} ---')

            if not self.move_to(pan, HOVER_LIFT, f'Hover {i+1}'):
                self.get_logger().warn(f'Skipping {label}')
                continue
            time.sleep(0.3)

            if not self.move_to(pan, CUT_LIFT, f'Cut {i+1}'):
                self.get_logger().warn(f'Cut stroke failed for {label}')

            time.sleep(0.3)
            self.move_to(pan, HOVER_LIFT, f'Raise {i+1}')
            time.sleep(0.3)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MovementNode()

    print('=== FruitNinja Cutting Sequence ===')
    print(f'  pan range  : {math.degrees(PAN_START):.1f}° → {math.degrees(PAN_END):.1f}°')
    print(f'  hover lift : {math.degrees(HOVER_LIFT):.1f}°')
    print(f'  cut lift   : {math.degrees(CUT_LIFT):.1f}°')
    print(f'  num cuts   : {NUM_CUTS}')
    print()

    node.move_to_ready()
    time.sleep(0.5)

    node.perform_cuts()
    time.sleep(0.5)

    node.move_to_ready()

    print('=== Sequence complete ===')
    node.destroy_node()
    rclpy.shutdown()


def reset_main(args=None):
    """Move robot to ready position only — used by GUI reset button."""
    rclpy.init(args=args)
    node = MovementNode()
    print('=== FruitNinja Reset → Ready Position ===')
    node.move_to_ready()
    print('=== Done ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
