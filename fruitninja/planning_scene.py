#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
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

        time.sleep(1.0)
        self.setup_scene()

    def setup_scene(self):
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        # -------------------------------------------------------
        # TABLE TOP (5cm thick surface at z=1.0m)
        # -------------------------------------------------------
        table = CollisionObject()
        table.id = 'table'
        table.header.frame_id = 'world'

        table_shape = SolidPrimitive()
        table_shape.type = SolidPrimitive.BOX
        table_shape.dimensions = [0.73, 0.71, 0.05]

        table_pose = Pose()
        table_pose.position.x = 0.365
        table_pose.position.y = 0.355
        table_pose.position.z = 0.975
        table_pose.orientation.w = 1.0

        table.primitives.append(table_shape)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        # -------------------------------------------------------
        # TABLE BODY (legs/base, 95cm tall)
        # -------------------------------------------------------
        table_body = CollisionObject()
        table_body.id = 'table_body'
        table_body.header.frame_id = 'world'

        body_shape = SolidPrimitive()
        body_shape.type = SolidPrimitive.BOX
        body_shape.dimensions = [0.73, 0.71, 0.95]

        body_pose = Pose()
        body_pose.position.x = 0.365
        body_pose.position.y = 0.355
        body_pose.position.z = 0.475
        body_pose.orientation.w = 1.0

        table_body.primitives.append(body_shape)
        table_body.primitive_poses.append(body_pose)
        table_body.operation = CollisionObject.ADD

        # -------------------------------------------------------
        # ROBOT BASE MOUNT (cylinder, centred on table)
        # -------------------------------------------------------
        robot_base = CollisionObject()
        robot_base.id = 'robot_base_mount'
        robot_base.header.frame_id = 'world'

        base_shape = SolidPrimitive()
        base_shape.type = SolidPrimitive.CYLINDER
        base_shape.dimensions = [0.05, 0.075]  # height, radius

        base_pose = Pose()
        base_pose.position.x = 0.365  # centre of table
        base_pose.position.y = 0.355  # centre of table
        base_pose.position.z = 1.025  # on top of table surface
        base_pose.orientation.w = 1.0

        robot_base.primitives.append(base_shape)
        robot_base.primitive_poses.append(base_pose)
        robot_base.operation = CollisionObject.ADD

        # -------------------------------------------------------
        # CHOPPING BLOCK (to the side, within UR3e reach)
        # -------------------------------------------------------
        chopping_block = CollisionObject()
        chopping_block.id = 'chopping_block'
        chopping_block.header.frame_id = 'world'

        block_shape = SolidPrimitive()
        block_shape.type = SolidPrimitive.BOX
        block_shape.dimensions = [0.30, 0.25, 0.08]  # 30cm x 25cm x 8cm

        block_pose = Pose()
        block_pose.position.x = 0.365       # same x as robot
        block_pose.position.y = 0.355 - 0.30  # 30cm in front of robot
        block_pose.position.z = 1.04        # on table surface
        block_pose.orientation.w = 1.0

        chopping_block.primitives.append(block_shape)
        chopping_block.primitive_poses.append(block_pose)
        chopping_block.operation = CollisionObject.ADD

        # Add all objects
        planning_scene.world.collision_objects.append(table)
        planning_scene.world.collision_objects.append(table_body)
        planning_scene.world.collision_objects.append(robot_base)
        planning_scene.world.collision_objects.append(chopping_block)

        self.scene_pub.publish(planning_scene)
        self.get_logger().info('Planning scene published: table + robot base + chopping block.')


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneSetup()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()