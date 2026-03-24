#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point
from ament_index_python.packages import get_package_share_directory
import time

_DAE_NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}


def load_dae_mesh(filepath: str, scale: float = 1.0) -> Mesh:
    """Parse a COLLADA (.dae) file into a shape_msgs/Mesh.

    scale=0.01 because UR3eTrolley.dae was authored in centimetres
    but exported with unit=meter.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    ros_mesh = Mesh()

    for geom in root.findall('.//c:geometry', _DAE_NS):
        mesh_el = geom.find('c:mesh', _DAE_NS)
        if mesh_el is None:
            continue

        # ── vertices ──────────────────────────────────────────────
        positions_src = None
        for src in mesh_el.findall('c:source', _DAE_NS):
            if 'positions' in src.get('id', '').lower():
                positions_src = src
                break
        if positions_src is None:
            continue

        fa = positions_src.find('c:float_array', _DAE_NS)
        if fa is None or not fa.text:
            continue

        raw = list(map(float, fa.text.split()))
        vertex_offset = len(ros_mesh.vertices)
        for i in range(0, len(raw) - 2, 3):
            ros_mesh.vertices.append(Point(
                x=raw[i]     * scale,
                y=raw[i + 1] * scale,
                z=raw[i + 2] * scale,
            ))

        # ── triangles ─────────────────────────────────────────────
        for tri_el in mesh_el.findall('c:triangles', _DAE_NS):
            stride = len(tri_el.findall('c:input', _DAE_NS))
            p_el = tri_el.find('c:p', _DAE_NS)
            if p_el is None or not p_el.text:
                continue
            indices = list(map(int, p_el.text.split()))
            for i in range(0, len(indices), stride * 3):
                t = MeshTriangle()
                t.vertex_indices = [
                    indices[i]              + vertex_offset,
                    indices[i + stride]     + vertex_offset,
                    indices[i + stride * 2] + vertex_offset,
                ]
                ros_mesh.triangles.append(t)

        for poly_el in mesh_el.findall('c:polylist', _DAE_NS):
            stride = len(poly_el.findall('c:input', _DAE_NS))
            p_el = poly_el.find('c:p', _DAE_NS)
            if p_el is None or not p_el.text:
                continue
            indices = list(map(int, p_el.text.split()))
            vcount_el = poly_el.find('c:vcount', _DAE_NS)
            counts = list(map(int, vcount_el.text.split())) if vcount_el is not None else []
            pos = 0
            for count in counts:
                verts = [indices[(pos + j) * stride] for j in range(count)]
                for j in range(1, count - 1):
                    t = MeshTriangle()
                    t.vertex_indices = [
                        verts[0]     + vertex_offset,
                        verts[j]     + vertex_offset,
                        verts[j + 1] + vertex_offset,
                    ]
                    ros_mesh.triangles.append(t)
                pos += count

    return ros_mesh


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

        pkg_share = get_package_share_directory('fruitninja')

        # -------------------------------------------------------
        # UR3e TROLLEY (mesh collision object)
        # DAE authored in cm, exported as meters → scale=0.01
        # Rotation (0.5, 0.5, 0.5, 0.5): stands mesh upright (Y-up→Z-up)
        # and rotates 90° CCW to face correct direction.
        # Position (0, 0, 0): world origin at floor level.
        # -------------------------------------------------------
        mesh_path = os.path.join(pkg_share, 'meshes', 'UR3eTrolley.dae')
        self.get_logger().info(f'Loading trolley mesh: {mesh_path}')

        trolley = CollisionObject()
        trolley.id = 'trolley'
        trolley.header.frame_id = 'world'

        trolley.meshes.append(load_dae_mesh(mesh_path, scale=0.01))

        mesh_pose = Pose()
        mesh_pose.position.x = 0.0
        mesh_pose.position.y = 0.0
        mesh_pose.position.z = 0.0
        # 90° around X (Y-up → Z-up) + 90° CCW around Z
        mesh_pose.orientation.x = 0.5
        mesh_pose.orientation.y = 0.5
        mesh_pose.orientation.z = 0.5
        mesh_pose.orientation.w = 0.5
        trolley.mesh_poses.append(mesh_pose)
        trolley.operation = CollisionObject.ADD

        planning_scene.world.collision_objects.append(trolley)

        self.scene_pub.publish(planning_scene)
        self.get_logger().info('Planning scene published: UR3e trolley mesh.')


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneSetup()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
