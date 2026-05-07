#!/usr/bin/env python3
"""
safety_node.py — ROS 2 hand-detection safety interlock for the FruitNinja UR3e.

Subscribes to /camera/image_raw/compressed, runs MediaPipe Hands on each frame,
and pauses/resumes the robot via the UR Dashboard Server when a hand enters the
cutting zone polygon.

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
  tolerance_px    (int,    default 40)   — pixels of extra margin outside zone
  max_hands       (int,    default 2)
  detection_conf  (double, default 0.7)
  tracking_conf   (double, default 0.6)
"""

import sys
import socket
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32MultiArray

import cv2
import numpy as np

try:
    import mediapipe.python.solutions.hands as _mp_hands
    import mediapipe.python.solutions.drawing_utils as _mp_drawing
    _HAS_MP = True
except Exception:
    _HAS_MP = False


# ── Dashboard TCP client ───────────────────────────────────────────────────────

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
        self.declare_parameter('tolerance_px',   40)
        self.declare_parameter('max_hands',      2)
        self.declare_parameter('detection_conf', 0.7)
        self.declare_parameter('tracking_conf',  0.6)

        ip   = self.get_parameter('robot_ip').value
        port = self.get_parameter('dashboard_port').value
        self._tol = self.get_parameter('tolerance_px').value

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

        self.get_logger().info('Safety node ready — waiting for zone definition and camera frames')

    # ── Zone update ───────────────────────────────────────────────────────────

    def _on_zone(self, msg: Float32MultiArray):
        """Receive [x0,y0, x1,y1, x2,y2, x3,y3] — TL, TR, BR, BL corners."""
        d = msg.data
        if len(d) != 8:
            self.get_logger().warn(f'Zone msg has {len(d)} values, expected 8 — ignored')
            return
        self._zone_poly = np.array(
            [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]],
            dtype=np.int32
        )
        self.get_logger().info(
            f'Zone updated: TL={d[0]:.0f},{d[1]:.0f}  TR={d[2]:.0f},{d[3]:.0f}  '
            f'BR={d[4]:.0f},{d[5]:.0f}  BL={d[6]:.0f},{d[7]:.0f}'
        )

    # ── Image callback ────────────────────────────────────────────────────────

    def _on_image(self, msg: CompressedImage):
        if self._hands is None or self._zone_poly is None:
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
                for lm in hand_lms.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    if cv2.pointPolygonTest(poly_exp, (px, py), False) >= 0:
                        hand_detected = True
                        break
                if hand_detected:
                    break

        # Act on state change
        if hand_detected and not self._hand_in_zone:
            self._hand_in_zone = True
            self._db.pause()
            self.get_logger().warn('HAND IN ZONE — robot paused')
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
