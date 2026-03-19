#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint, MoveItErrorCodes
)
import time


# UR3e joint names (in order)
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Joint configs [base, shoulder, elbow, wrist1, wrist2, wrist3] in radians
READY_POSITION = [0.0,  -1.57,  1.57, -1.57, -1.57,  0.0]
ABOVE_BLOCK    = [0.0,  -1.20,  1.40, -1.77, -1.57,  0.0]
CUT_DOWN       = [0.0,  -1.00,  1.60, -2.17, -1.57,  0.0]


def make_joint_goal(positions: list) -> Constraints:
    c = Constraints()
    for name, pos in zip(JOINT_NAMES, positions):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = pos
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


class MovementNode(Node):
    def __init__(self):
        super().__init__('fruitninja_movement')
        self._client = ActionClient(self, MoveGroup, '/move_action')

    def move_to(self, positions: list, label: str) -> bool:
        self.get_logger().info(f'Moving to: {label}')

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = 'ur_manipulator'
        goal.request.goal_constraints.append(make_joint_goal(positions))
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
        else:
            self.get_logger().error(f'[FAIL] {label} — error code: {result.error_code.val}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = MovementNode()

    print('=== FruitNinja Movement Sequence ===')

    steps = [
        (READY_POSITION, 'Ready position'),
        (ABOVE_BLOCK,    'Above chopping block'),
        (CUT_DOWN,       'Cutting stroke'),
        (ABOVE_BLOCK,    'Retract'),
        (READY_POSITION, 'Return to ready'),
    ]

    for positions, label in steps:
        node.move_to(positions, label)
        time.sleep(0.5)

    print('=== Sequence complete ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
