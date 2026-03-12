#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import time


class PlanningSceneSetup(Node):
    def __init__(self):
        super().__init__('planning_scene_setup')

        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )

        # Wait for publisher to be ready
        time.sleep(1.0)
        self.setup_scene()

    def setup_scene(self):
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        # -------------------------------------------------------
        # TABLE
        # Dimensions: 73cm x 71cm x 1m (height)
        # Represented as a flat box (5cm thick top surface)
        # -------------------------------------------------------
        table = CollisionObject()
        table.id = 'table'
        table.header.frame_id = 'world'

        table_shape = SolidPrimitive()
        table_shape.type = SolidPrimitive.BOX
        table_shape.dimensions = [0.73, 0.71, 0.05]  # length, width, thickness (m)

        table_pose = Pose()
        # Centre of the table surface at z = 1.0m (table height)
        # Table top surface sits at z=1.0, so box centre at z=0.975
        table_pose.position.x = 0.365   # half of 73cm
        table_pose.position.y = 0.355   # half of 71cm
        table_pose.position.z = 0.975   # just below 1m surface height
        table_pose.orientation.w = 1.0

        table.primitives.append(table_shape)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        # -------------------------------------------------------
        # TABLE LEG BODY (full table base as a box underneath)
        # -------------------------------------------------------
        table_body = CollisionObject()
        table_body.id = 'table_body'
        table_body.header.frame_id = 'world'

        body_shape = SolidPrimitive()
        body_shape.type = SolidPrimitive.BOX
        body_shape.dimensions = [0.73, 0.71, 0.95]  # full height of table legs/body

        body_pose = Pose()
        body_pose.position.x = 0.365
        body_pose.position.y = 0.355
        body_pose.position.z = 0.475   # centre of 0.95m tall body
        body_pose.orientation.w = 1.0

        table_body.primitives.append(body_shape)
        table_body.primitive_poses.append(body_pose)
        table_body.operation = CollisionObject.ADD

        # -------------------------------------------------------
        # UR3e BASE FOOTPRINT on table
        # Offset: 19cm x 39cm from one corner of the table
        # Base is approximately 15cm diameter - represented as box
        # -------------------------------------------------------
        robot_base = CollisionObject()
        robot_base.id = 'robot_base_mount'
        robot_base.header.frame_id = 'world'

        base_shape = SolidPrimitive()
        base_shape.type = SolidPrimitive.CYLINDER
        base_shape.dimensions = [0.05, 0.075]  # height=5cm, radius=7.5cm

        base_pose = Pose()
        # 19cm from x edge, 39cm from y edge of table
        base_pose.position.x = 0.19
        base_pose.position.y = 0.39
        base_pose.position.z = 1.025  # sitting on top of table surface
        base_pose.orientation.w = 1.0

        robot_base.primitives.append(base_shape)
        robot_base.primitive_poses.append(base_pose)
        robot_base.operation = CollisionObject.ADD

        # Add all objects to the scene
        planning_scene.world.collision_objects.append(table)
        planning_scene.world.collision_objects.append(table_body)
        planning_scene.world.collision_objects.append(robot_base)

        self.scene_pub.publish(planning_scene)
        self.get_logger().info('Planning scene published: table + robot base mount added.')


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneSetup()

    # Spin briefly to ensure message is sent
    rclpy.spin_once(node, timeout_sec=2.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
