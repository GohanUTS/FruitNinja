#!/usr/bin/env python3
"""
safety_node.py — ROS 2 hand-detection safety interlock for the FruitNinja UR3e.

Subscribes to /camera/image_raw/compressed, runs MediaPipe Hands on each frame,
and pauses robot motion via ROS action cancellation plus the UR Dashboard Server
when a hand enters the cutting zone polygon.

The cutting zone is received dynamically on /safety/zone as a Float32MultiArray
[x0,y0, x1,y1, x2,y2, x3,y3] (TL, TR, BR, BL pixel coords in the camera frame).
real_gui_points.py publishes this whenever the operator locks the 4-corner grid.

Usage
-----
  ros2 run fruitninja safety_node --robot-ip 192.168.0.194

Parameters (ROS)
----------------
  robot_ip        (string, default '192.168.0.194')
  dashboard_port  (int,    default 29999)
  tolerance_px    (int,    default 0)    — pixels of extra margin outside zone
  zone_timeout_s  (double, default 1.0)  — clear stale zones if GUI stops publishing
  max_hands       (int,    default 2)
  detection_conf  (double, default 0.7)
  tracking_conf   (double, default 0.6)
"""

import sys
import socket
import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from action_msgs.srv import CancelGoal

import cv2
import numpy as np

try:
    import mediapipe.python.solutions.hands as _mp_hands
    import mediapipe.python.solutions.drawing_utils as _mp_drawing
    _HAS_MP = True
except Exception:
    _HAS_MP = False


# ── Dashboard TCP client ───────────────────────────────────────────────────────

TRAJECTORY_CANCEL_SERVICE = (
    '/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/cancel_goal'
)
HAND_ZONE_MIN_LANDMARKS = 5
HAND_ZONE_PALM_INDICES = (0, 5, 9, 13, 17)

class Dashboard:
    def __init__(self, ip: str, port: int = 29999):
        self._paused = False
        self._sock = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((ip, port))
            s.recv(1024)
            s.settimeout(1.0)
            self._sock = s
        except Exception as e:
            pass

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _send(self, cmd: str):
        if not self._sock:
            return
        try:
            self._sock.sendall((cmd + '\n').encode())
            self._sock.recv(1024)
        except Exception:
            pass

    def pause(self):
        if not self._paused:
            self._send('pause')
            self._paused = True

    def stop(self):
        self._send('stop')
        self._paused = False

    def resume(self):
        if self._paused:
            self._send('play')
            self._paused = False

    def close(self):
        if self._sock:
            try:
                self.resume()
                self._sock.close()
            except Exception:
                pass


# ── Safety node ────────────────────────────────────────────────────────────────

class SafetyNode(Node):
    def __init__(self, robot_ip: str):
        super().__init__('fruitninja_safety')

        # Parameters
        self.declare_parameter('robot_ip',       robot_ip)
        self.declare_parameter('dashboard_port', 29999)
        self.declare_parameter('tolerance_px',   0)
        self.declare_parameter('zone_timeout_s', 1.0)
        self.declare_parameter('max_hands',      2)
        self.declare_parameter('detection_conf', 0.7)
        self.declare_parameter('tracking_conf',  0.6)

        ip   = self.get_parameter('robot_ip').value
        port = self.get_parameter('dashboard_port').value
        self._tol = self.get_parameter('tolerance_px').value
        self._zone_timeout_s = self.get_parameter('zone_timeout_s').value
        self._last_stop_time = 0.0

        # Dashboard
        self._db = Dashboard(ip, port)
        if self._db.connected:
            self.get_logger().info(f'Dashboard connected to {ip}:{port}')
        else:
            self.get_logger().warn(f'Dashboard NOT connected to {ip}:{port} — robot pause disabled')

        # MediaPipe
        if _HAS_MP:
            self._hands = _mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.get_parameter('max_hands').value,
                min_detection_confidence=self.get_parameter('detection_conf').value,
                min_tracking_confidence=self.get_parameter('tracking_conf').value,
            )
            self.get_logger().info('MediaPipe Hands loaded')
        else:
            self._hands = None
            self.get_logger().error('mediapipe not installed — safety interlock INACTIVE')

        # Zone: TL, TR, BR, BL pixel coords  (None = not yet defined)
        self._zone_poly: np.ndarray | None = None
        self._zone_stamp = 0.0

        # State
        self._hand_in_zone = False

        # Subscribers
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self._on_image, 10)
        self.create_subscription(
            Float32MultiArray, '/safety/zone',
            self._on_zone, 10)

        # Publishers
        self._pub_status = self.create_publisher(Bool, '/safety/hand_detected', 10)
        self._cancel_client = self.create_client(CancelGoal, TRAJECTORY_CANCEL_SERVICE)

        self.get_logger().info('Safety node ready — waiting for zone definition and camera frames')

    # ── Zone update ───────────────────────────────────────────────────────────

    def _on_zone(self, msg: Float32MultiArray):
        """Receive [] to clear, or [x0,y0, x1,y1, x2,y2, x3,y3] corners."""
        d = msg.data
        if len(d) == 0:
            self._clear_zone('Zone cleared by GUI')
            return
        if len(d) != 8:
            self.get_logger().warn(f'Zone msg has {len(d)} values, expected 8 — ignored')
            return
        self._zone_poly = np.array(
            [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]],
            dtype=np.int32
        )
        self._zone_stamp = time.monotonic()
        self.get_logger().info(
            f'Zone updated: TL={d[0]:.0f},{d[1]:.0f}  TR={d[2]:.0f},{d[3]:.0f}  '
            f'BR={d[4]:.0f},{d[5]:.0f}  BL={d[6]:.0f},{d[7]:.0f}'
        )

    def _clear_zone(self, reason: str):
        self._zone_poly = None
        self._zone_stamp = 0.0
        if self._hand_in_zone:
            self._hand_in_zone = False
            self._pub_status.publish(Bool(data=False))
        self.get_logger().info(reason)

    def _cancel_active_trajectories(self):
        if not self._cancel_client.service_is_ready():
            self._cancel_client.wait_for_service(timeout_sec=0.05)
        if not self._cancel_client.service_is_ready():
            self.get_logger().warn(f'Trajectory cancel service not available: {TRAJECTORY_CANCEL_SERVICE}')
            return
        req = CancelGoal.Request()
        self._cancel_client.call_async(req)

    # ── Image callback ────────────────────────────────────────────────────────

    def _on_image(self, msg: CompressedImage):
        if self._hands is None or self._zone_poly is None:
            return
        if time.monotonic() - self._zone_stamp > self._zone_timeout_s:
            self._clear_zone('Safety zone expired — waiting for fresh locked grid')
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        # Expand zone by tolerance
        tol = self._tol
        poly_exp = self._zone_poly.copy()
        cx = int(poly_exp[:, 0].mean())
        cy = int(poly_exp[:, 1].mean())
        for pt in poly_exp:
            dx = pt[0] - cx
            dy = pt[1] - cy
            norm = max(1, (dx**2 + dy**2) ** 0.5)
            pt[0] = int(pt[0] + tol * dx / norm)
            pt[1] = int(pt[1] + tol * dy / norm)

        hand_detected = False
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                pts = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_lms.landmark
                ]
                inside = [
                    pt for pt in pts
                    if cv2.pointPolygonTest(poly_exp, pt, False) >= 0
                ]
                palm_pts = [pts[i] for i in HAND_ZONE_PALM_INDICES]
                palm_inside_count = sum(
                    1 for pt in palm_pts
                    if cv2.pointPolygonTest(poly_exp, pt, False) >= 0
                )
                palm_center = (
                    int(sum(p[0] for p in palm_pts) / len(palm_pts)),
                    int(sum(p[1] for p in palm_pts) / len(palm_pts)),
                )
                palm_inside = cv2.pointPolygonTest(poly_exp, palm_center, False) >= 0
                if palm_inside or (
                    len(inside) >= HAND_ZONE_MIN_LANDMARKS
                    and palm_inside_count >= 2
                ):
                    hand_detected = True
                if hand_detected:
                    break

        # Act on state change. While a hand remains in-zone, keep cancelling at
        # a limited rate so new goals from another client cannot continue motion.
        if hand_detected:
            now = time.monotonic()
            if now - self._last_stop_time > 0.5:
                self._cancel_active_trajectories()
                self._db.pause()
                self._last_stop_time = now
            if not self._hand_in_zone:
                self._hand_in_zone = True
                self.get_logger().warn('HAND IN GRID ZONE — trajectory cancelled and robot paused')
        elif not hand_detected and self._hand_in_zone:
            self._hand_in_zone = False
            self._db.resume()
            self.get_logger().info('Zone clear — robot resumed')

        self._pub_status.publish(Bool(data=self._hand_in_zone))

    def destroy_node(self):
        self._db.close()
        if self._hands:
            self._hands.close()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-ip', default='192.168.0.194')
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = SafetyNode(robot_ip=parsed.robot_ip)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
