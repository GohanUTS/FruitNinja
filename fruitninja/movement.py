#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, JointConstraint, PositionConstraint,
    OrientationConstraint, BoundingVolume, MoveItErrorCodes,
)
from moveit_msgs.srv import GetCartesianPath
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from shape_msgs.msg import SolidPrimitive
import time


# ── Joint names ───────────────────────────────────────────────────────────────
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

READY_POSITION = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# ── Cutting zone ──────────────────────────────────────────────────────────────
# Table surface is at z=0.724m (top of trolley).
# Cutting zone is the far side of the table, within UR3e reach.
#
#   x: 0.40 → 0.70   (depth into table, away from robot)
#   y: -0.30 → 0.30  (width across table)
#
# Each cut is a straight stroke along Y at a fixed X position.
# num_cuts evenly-spaced X positions are used.
CUTTING_ZONE = {
    'x_start':  0.40,   # near edge of cutting zone
    'x_end':    0.70,   # far edge of cutting zone
    'y_start':  -0.30,  # left edge
    'y_end':     0.30,  # right edge
    'z_surface': 0.724, # table surface height
    'z_above':   0.15,  # hover height above surface
    'z_cut':     0.02,  # depth below surface the blade goes
    'num_cuts':  5,     # number of evenly-spaced cuts across X
}

# End-effector orientation — blade pointing down (~90° pitch)
BLADE_DOWN = Quaternion(x=0.0, y=0.7071, z=0.0, w=0.7071)

MOVE_GROUP   = 'ur_manipulator'
EEF_LINK     = 'tool0'
WORLD_FRAME  = 'world'
BASE_FRAME   = 'base_link'

# Robot base position in world frame (from urdf/ur3e_workcell.urdf.xacro)
ROBOT_BASE_X = 0.23
ROBOT_BASE_Y = -0.03
ROBOT_BASE_Z = 0.724


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def make_pose_goal(x, y, z, quat, frame=WORLD_FRAME) -> Constraints:
    c = Constraints()

    pc = PositionConstraint()
    pc.header.frame_id = frame
    pc.link_name = EEF_LINK
    pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    region = BoundingVolume()
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [0.01, 0.01, 0.01]   # 1cm tolerance box
    region.primitives.append(box)
    centre = Pose()
    centre.position = Point(x=x, y=y, z=z)
    centre.orientation.w = 1.0
    region.primitive_poses.append(centre)
    pc.constraint_region = region
    pc.weight = 1.0
    c.position_constraints.append(pc)

    oc = OrientationConstraint()
    oc.header.frame_id = frame
    oc.link_name = EEF_LINK
    oc.orientation = quat
    oc.absolute_x_axis_tolerance = 0.1
    oc.absolute_y_axis_tolerance = 0.1
    oc.absolute_z_axis_tolerance = 0.1
    oc.weight = 1.0
    c.orientation_constraints.append(oc)

    return c


def make_pose(x, y, z, quat) -> Pose:
    p = Pose()
    p.position = Point(x=x, y=y, z=z)
    p.orientation = quat
    return p


def world_to_base(x, y, z):
    """Convert world-frame coords to base_link-frame coords."""
    return (x - ROBOT_BASE_X,
            y - ROBOT_BASE_Y,
            z - ROBOT_BASE_Z)


def cut_x_positions(zone: dict) -> list:
    """Evenly-spaced X positions for each cut."""
    n = zone['num_cuts']
    x0, x1 = zone['x_start'], zone['x_end']
    if n == 1:
        return [(x0 + x1) / 2.0]
    step = (x1 - x0) / (n - 1)
    return [x0 + i * step for i in range(n)]


# ── Node ──────────────────────────────────────────────────────────────────────

class MovementNode(Node):
    def __init__(self):
        super().__init__('fruitninja_movement')
        self._move_client    = ActionClient(self, MoveGroup,         '/move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._cartesian_srv  = self.create_client(GetCartesianPath,  '/compute_cartesian_path')

    # ── Joint-space move ──────────────────────────────────────────────────────

    def move_to_joints(self, positions: list, label: str) -> bool:
        self.get_logger().info(f'Moving to joints: {label}')
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(make_joint_goal(positions))
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        return self._send_move_goal(goal, label)

    # ── Pose-space move ───────────────────────────────────────────────────────

    def move_to_pose(self, x, y, z, quat, label: str) -> bool:
        self.get_logger().info(f'Moving to pose: {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.goal_constraints.append(make_pose_goal(x, y, z, quat))
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        return self._send_move_goal(goal, label)

    def _send_move_goal(self, goal, label: str) -> bool:
        future = self._move_client.send_goal_async(goal)
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

    # ── Cartesian straight-line path ──────────────────────────────────────────

    def execute_cartesian_path(self, waypoints: list, label: str,
                               max_step: float = 0.005) -> bool:
        self.get_logger().info(f'Cartesian path: {label}')
        if not self._cartesian_srv.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('GetCartesianPath service not available')
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name      = MOVE_GROUP
        req.link_name       = EEF_LINK
        req.waypoints       = waypoints
        req.max_step        = max_step
        req.jump_threshold  = 0.0
        req.avoid_collisions = False

        srv_future = self._cartesian_srv.call_async(req)
        rclpy.spin_until_future_complete(self, srv_future)
        response = srv_future.result()

        if response.fraction < 0.50:
            self.get_logger().error(
                f'[FAIL] {label} — only {response.fraction*100:.1f}% planned')
            return False

        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('ExecuteTrajectory action server not available')
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution
        exec_future = self._execute_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self, exec_future)
        exec_handle = exec_future.result()
        if not exec_handle.accepted:
            self.get_logger().error(f'Execution rejected: {label}')
            return False

        res_future = exec_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'[OK] {label}')
            return True
        self.get_logger().error(f'[FAIL] {label} — error: {result.error_code.val}')
        return False

    # ── Cutting sequence ──────────────────────────────────────────────────────

    def perform_cuts(self, zone: dict = CUTTING_ZONE) -> None:
        """
        Execute evenly-spaced cuts across the cutting zone.

        Each cut is a straight stroke along Y at a fixed X position:
            hover above start → plunge down → stroke along Y → raise up

        Zone layout (top-down):
            x_start → x_end   : cuts spread across this depth
            y_start → y_end   : each cut strokes across this width
        """
        z_surf  = zone['z_surface']
        z_above = z_surf + zone['z_above']
        z_cut   = z_surf + zone['z_cut']
        y0      = zone['y_start']
        y1      = zone['y_end']
        xs      = cut_x_positions(zone)

        self.get_logger().info(
            f'Cutting zone: x=[{zone["x_start"]}→{zone["x_end"]}]  '
            f'y=[{y0}→{y1}]  z_cut={z_cut:.3f}  cuts={zone["num_cuts"]}')

        for i, x in enumerate(xs):
            label = f'Cut {i+1}/{zone["num_cuts"]} at x={x:.3f}'
            self.get_logger().info(f'--- {label} ---')

            # 1. Move to hover above cut start
            ok = self.move_to_pose(x, y0, z_above, BLADE_DOWN,
                                   f'Hover cut {i+1}')
            if not ok:
                self.get_logger().warn(f'Skipping {label}')
                continue
            time.sleep(0.3)

            # 2. Plunge → stroke along Y → raise (single Cartesian path)
            # Waypoints must be in base_link frame for GetCartesianPath
            waypoints = [
                make_pose(*world_to_base(x, y0, z_cut),   BLADE_DOWN),  # plunge
                make_pose(*world_to_base(x, y1, z_cut),   BLADE_DOWN),  # stroke along Y
                make_pose(*world_to_base(x, y1, z_above), BLADE_DOWN),  # raise
            ]
            self.execute_cartesian_path(waypoints, label)
            time.sleep(0.3)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MovementNode()

    print('=== FruitNinja Cutting Sequence ===')
    print(f'  x range  : {CUTTING_ZONE["x_start"]} → {CUTTING_ZONE["x_end"]} m')
    print(f'  y range  : {CUTTING_ZONE["y_start"]} → {CUTTING_ZONE["y_end"]} m')
    print(f'  z surface: {CUTTING_ZONE["z_surface"]} m')
    print(f'  num cuts : {CUTTING_ZONE["num_cuts"]}')
    print()

    node.move_to_joints(READY_POSITION, 'Ready position')
    time.sleep(0.5)

    node.perform_cuts(CUTTING_ZONE)
    time.sleep(0.5)

    node.move_to_joints(READY_POSITION, 'Return to ready')

    print('=== Sequence complete ===')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
