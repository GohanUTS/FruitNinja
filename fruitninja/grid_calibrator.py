#!/usr/bin/env python3
"""
Capture and manage per-robot FruitNinja grid calibration files.

A calibration file stores the four cutting-board corner TCP poses in the active
robot's base frame.  real_gui_points.py can then move by calibrated Cartesian
poses instead of using the old fixed joint table:

  export FRUITNINJA_USE_POSE_GRID=1
  export FRUITNINJA_GRID_CALIBRATION=~/.fruitninja/grid_calibration.json

Capture flow:
  1. Freedrive/jog the tool tip to A1 on the physical board.
  2. ros2 run fruitninja grid_calibrator --capture A1
  3. Repeat for N1, N4, A4.

For URSim smoke tests you can write a small reachable demo grid around the
current tool pose:

  ros2 run fruitninja grid_calibrator --write-sim-demo
"""

import argparse
import json
import math
import os
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

from fruitninja.grid_mover import (
    CORNER_TOOL_POSES_M,
    CORNER_TOOL_ROT_VEC_RAD,
    DEFAULT_GRID_CALIBRATION_PATH,
    GRID_CALIBRATION_ENV,
    REQUIRED_GRID_CORNERS,
    grid_calibration_path,
)


def _resolved_path(path: str | None) -> str:
    if path:
        return os.path.expanduser(path)
    return grid_calibration_path()


def _empty_calibration(frame_id: str, eef_link: str) -> dict:
    return {
        'frame_id': frame_id,
        'eef_link': eef_link,
        'corner_tool_poses_m': {},
        'corner_tool_rot_vec_rad': {},
    }


def _load(path: str, frame_id: str, eef_link: str) -> dict:
    if not os.path.exists(path):
        return _empty_calibration(frame_id, eef_link)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.setdefault('frame_id', frame_id)
    data.setdefault('eef_link', eef_link)
    data.setdefault('corner_tool_poses_m', {})
    data.setdefault('corner_tool_rot_vec_rad', {})
    return data


def _save(path: str, data: dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def _default_calibration(frame_id: str, eef_link: str) -> dict:
    return {
        'frame_id': frame_id,
        'eef_link': eef_link,
        'corner_tool_poses_m': {
            corner: [float(v) for v in CORNER_TOOL_POSES_M[corner]]
            for corner in REQUIRED_GRID_CORNERS
        },
        'corner_tool_rot_vec_rad': {
            corner: [float(v) for v in CORNER_TOOL_ROT_VEC_RAD[corner]]
            for corner in REQUIRED_GRID_CORNERS
        },
    }


def _demo_calibration(center_tf, frame_id: str, eef_link: str,
                      width_m: float, depth_m: float,
                      z_offset_m: float) -> dict:
    t = center_tf.transform.translation
    q = center_tf.transform.rotation
    cx, cy, cz = float(t.x), float(t.y), float(t.z) + float(z_offset_m)
    half_w = float(width_m) / 2.0
    half_d = float(depth_m) / 2.0
    rotvec = _quat_to_rotvec(q.x, q.y, q.z, q.w)
    return {
        'frame_id': frame_id,
        'eef_link': eef_link,
        'corner_tool_poses_m': {
            'A1': [cx - half_w, cy - half_d, cz],
            'N1': [cx + half_w, cy - half_d, cz],
            'N4': [cx + half_w, cy + half_d, cz],
            'A4': [cx - half_w, cy + half_d, cz],
        },
        'corner_tool_rot_vec_rad': {
            corner: list(rotvec)
            for corner in REQUIRED_GRID_CORNERS
        },
    }


def _quat_to_rotvec(x: float, y: float, z: float, w: float) -> list[float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    vec_norm = math.sqrt(x * x + y * y + z * z)
    if vec_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vec_norm, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    scale = angle / vec_norm
    return [x * scale, y * scale, z * scale]


class TfCaptureNode(Node):
    def __init__(self):
        super().__init__('fruitninja_grid_calibrator')
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def capture(self, frame_id: str, eef_link: str, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        last_error = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                return self.buffer.lookup_transform(
                    frame_id,
                    eef_link,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
            except Exception as e:
                last_error = e
        raise RuntimeError(f'Could not read TF {frame_id} -> {eef_link}: {last_error}')


def _print_summary(path: str, data: dict):
    print(f'Calibration: {path}')
    print(f'frame_id: {data.get("frame_id", "base_link")}')
    print(f'eef_link: {data.get("eef_link", "tool0")}')
    poses = data.get('corner_tool_poses_m', {})
    rots = data.get('corner_tool_rot_vec_rad', {})
    for corner in REQUIRED_GRID_CORNERS:
        pose = poses.get(corner)
        rot = rots.get(corner)
        if pose and rot:
            print(
                f'  {corner}: '
                f'xyz=({pose[0]:+.5f}, {pose[1]:+.5f}, {pose[2]:+.5f}) m  '
                f'rotvec=({rot[0]:+.3f}, {rot[1]:+.3f}, {rot[2]:+.3f}) rad'
            )
        else:
            print(f'  {corner}: missing')


def main(args=None):
    parser = argparse.ArgumentParser(description='Capture FruitNinja grid calibration')
    parser.add_argument('--path', default=None,
                        help=f'Calibration path (default: ${GRID_CALIBRATION_ENV} or {DEFAULT_GRID_CALIBRATION_PATH})')
    parser.add_argument('--frame-id', default='base_link',
                        help='Robot base frame used by MoveIt pose goals')
    parser.add_argument('--eef-link', default='tool0',
                        help='End-effector link to capture and command')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Seconds to wait for TF when capturing')
    parser.add_argument('--capture', choices=REQUIRED_GRID_CORNERS,
                        help='Capture the current end-effector pose as this board corner')
    parser.add_argument('--write-default', action='store_true',
                        help='Write the photographed built-in four-corner calibration')
    parser.add_argument('--write-sim-demo', action='store_true',
                        help='Write a small reachable demo grid around the current tool pose')
    parser.add_argument('--demo-width-m', type=float, default=0.18,
                        help='Width of the --write-sim-demo grid in metres')
    parser.add_argument('--demo-depth-m', type=float, default=0.08,
                        help='Depth of the --write-sim-demo grid in metres')
    parser.add_argument('--demo-z-offset-m', type=float, default=0.0,
                        help='Z offset applied to the current tool pose for --write-sim-demo')
    parser.add_argument('--show', action='store_true',
                        help='Print the calibration file contents')
    parsed, remaining = parser.parse_known_args(args)

    path = _resolved_path(parsed.path)
    if parsed.write_default:
        data = _default_calibration(parsed.frame_id, parsed.eef_link)
        _save(path, data)
        _print_summary(path, data)
        print('Wrote default calibration.')
        print('Enable it with: export FRUITNINJA_USE_POSE_GRID=1')
        return

    if parsed.write_sim_demo:
        rclpy.init(args=remaining)
        node = TfCaptureNode()
        try:
            tf = node.capture(parsed.frame_id, parsed.eef_link, parsed.timeout)
            data = _demo_calibration(
                tf,
                parsed.frame_id,
                parsed.eef_link,
                parsed.demo_width_m,
                parsed.demo_depth_m,
                parsed.demo_z_offset_m,
            )
            _save(path, data)
            _print_summary(path, data)
            print('Wrote sim demo calibration around the current tool pose.')
            print('Enable it with: export FRUITNINJA_USE_POSE_GRID=1')
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    if parsed.capture:
        rclpy.init(args=remaining)
        node = TfCaptureNode()
        try:
            data = _load(path, parsed.frame_id, parsed.eef_link)
            data['frame_id'] = parsed.frame_id
            data['eef_link'] = parsed.eef_link

            tf = node.capture(parsed.frame_id, parsed.eef_link, parsed.timeout)
            t = tf.transform.translation
            q = tf.transform.rotation
            data['corner_tool_poses_m'][parsed.capture] = [t.x, t.y, t.z]
            data['corner_tool_rot_vec_rad'][parsed.capture] = _quat_to_rotvec(
                q.x, q.y, q.z, q.w
            )
            _save(path, data)
            _print_summary(path, data)
            missing = [
                corner for corner in REQUIRED_GRID_CORNERS
                if corner not in data['corner_tool_poses_m']
                or corner not in data['corner_tool_rot_vec_rad']
            ]
            if missing:
                print('Still missing corners: ' + ', '.join(missing))
            else:
                print('All four corners captured.')
                print('Enable it with: export FRUITNINJA_USE_POSE_GRID=1')
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    data = _load(path, parsed.frame_id, parsed.eef_link)
    _print_summary(path, data)


if __name__ == '__main__':
    main()
