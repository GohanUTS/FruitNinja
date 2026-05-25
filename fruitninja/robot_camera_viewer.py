#!/usr/bin/env python3
"""
robot_camera_viewer.py — Live UR3e joint visualiser.

Shows three simultaneous views:
  1. Camera feed with joint-angle text overlay (drawn with OpenCV)
  2. Animated joint-angle bars in a right-side panel (Qt)
  3. Two-pane FK skeleton diagram (front view + top view) computed from live joint angles

All three update at ~20 fps.  ROS2 joint_states are consumed in a background
thread; if /joint_states is not published the display shows 0 ° for all joints.
"""

import os, sys, math, threading, argparse
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QPushButton, QGroupBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QLinearGradient,
)

# ── ROS2 (optional) ───────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False

# ── RealSense (optional) ──────────────────────────────────────────────────────
# cv2.VideoCapture can't read the RealSense color stream directly — the depth/IR
# /dev/video* nodes show up but the camera needs the librealsense SDK to stream
# colour frames.  Loaded lazily so the viewer still works with a plain webcam if
# pyrealsense2 isn't installed.
try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    _HAS_RS = False

# ── UR3e joint names (order from /joint_states) ───────────────────────────────
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
SHORT_NAMES = ['Pan', 'Lift', 'Elbow', 'Wrist1', 'Wrist2', 'Wrist3']

# UR3e DH link lengths (metres)
_D1 = 0.15185
_A2 = -0.24355
_A3 = -0.2132
_D4 = 0.13105
_D5 = 0.08535
_D6 = 0.0921


# ── Forward kinematics ────────────────────────────────────────────────────────

def _dh(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,       sa,      ca,       d],
        [ 0,        0,       0,       1],
    ], dtype=float)


def ur3e_joint_positions(joints_deg: list) -> list:
    """
    Return list of 7 (x, y, z) positions — base + one per joint — in the
    robot base frame (metres).  Uses standard UR3e DH parameters.
    """
    j = [math.radians(d) for d in joints_deg]
    params = [
        (j[0],  _D1,   0,       math.pi / 2),
        (j[1],  0,    _A2,      0),
        (j[2],  0,    _A3,      0),
        (j[3],  _D4,   0,       math.pi / 2),
        (j[4],  _D5,   0,      -math.pi / 2),
        (j[5],  _D6,   0,       0),
    ]
    pts = [(0.0, 0.0, 0.0)]
    T = np.eye(4)
    for args in params:
        T = T @ _dh(*args)
        pts.append((T[0, 3], T[1, 3], T[2, 3]))
    return pts


# ── Skeleton painter widget ───────────────────────────────────────────────────

class SkeletonWidget(QWidget):
    """
    Draws a two-pane FK skeleton diagram updated by set_joints().
      Left pane  — FRONT view  (X right, Z up)
      Right pane — TOP view    (X right, Y down)
    """

    # Colours for each link segment (base→pan, pan→lift, …, wrist3→tcp)
    _SEG_COLOURS = [
        QColor('#00cc88'),
        QColor('#ffd166'),
        QColor('#ff6b6b'),
        QColor('#c678dd'),
        QColor('#61afef'),
        QColor('#e5c07b'),
    ]

    def __init__(self):
        super().__init__()
        self.setMinimumSize(340, 300)
        self._joints_deg = [0.0] * 6
        self._pts = ur3e_joint_positions(self._joints_deg)

    def set_joints(self, joints_deg: list):
        self._joints_deg = list(joints_deg)
        self._pts = ur3e_joint_positions(self._joints_deg)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # dark background
        p.fillRect(0, 0, w, h, QColor('#0b1117'))

        half = w // 2
        self._draw_view(p, self._pts, 0,    0, half, h, axis_h='x', axis_v='z', label='FRONT')
        self._draw_view(p, self._pts, half, 0, half, h, axis_h='x', axis_v='y', label='TOP')

        # divider
        p.setPen(QPen(QColor('#263946'), 1))
        p.drawLine(half, 0, half, h)
        p.end()

    def _draw_view(self, p, pts, ox, oy, pw, ph, axis_h, axis_v, label):
        margin = 28
        draw_w = pw - 2 * margin
        draw_h = ph - 2 * margin - 18  # leave room for label

        # find extents for autoscale
        xs = [getattr(_P(pt), axis_h) for pt in pts]
        ys = [getattr(_P(pt), axis_v) for pt in pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.05)
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        scale = min(draw_w, draw_h) / (span * 1.3)

        def to_screen(pt):
            px = getattr(_P(pt), axis_h)
            py = getattr(_P(pt), axis_v)
            sx = ox + margin + draw_w / 2 + (px - cx) * scale
            # Z goes up on screen (invert Y for screen coords)
            if axis_v == 'z':
                sy = oy + margin + 18 + draw_h / 2 - (py - cy) * scale
            else:
                sy = oy + margin + 18 + draw_h / 2 + (py - cy) * scale
            return int(sx), int(sy)

        # draw ground plane marker
        if axis_v == 'z':
            base_s = to_screen(pts[0])
            p.setPen(QPen(QColor('#263946'), 1, Qt.DashLine))
            p.drawLine(ox + margin, base_s[1], ox + pw - margin, base_s[1])

        # draw links
        for i in range(len(pts) - 1):
            colour = self._SEG_COLOURS[i % len(self._SEG_COLOURS)]
            pen = QPen(colour, 3)
            p.setPen(pen)
            x1, y1 = to_screen(pts[i])
            x2, y2 = to_screen(pts[i + 1])
            p.drawLine(x1, y1, x2, y2)

        # draw joint dots
        for i, pt in enumerate(pts):
            sx, sy = to_screen(pt)
            if i == 0:
                p.setBrush(QBrush(QColor('#ffffff')))
                p.setPen(QPen(QColor('#ffffff'), 1))
                p.drawRect(sx - 4, sy - 4, 8, 8)
            elif i == len(pts) - 1:
                p.setBrush(QBrush(QColor('#ff4444')))
                p.setPen(Qt.NoPen)
                p.drawEllipse(sx - 5, sy - 5, 10, 10)
            else:
                colour = self._SEG_COLOURS[(i - 1) % len(self._SEG_COLOURS)]
                p.setBrush(QBrush(colour))
                p.setPen(Qt.NoPen)
                p.drawEllipse(sx - 4, sy - 4, 8, 8)

        # label
        p.setPen(QPen(QColor('#607080')))
        p.setFont(QFont('monospace', 9))
        p.drawText(ox + margin, oy + 16, label)


class _P:
    """Tiny helper so _draw_view can address (x, y, z) by name."""
    __slots__ = ('x', 'y', 'z')
    def __init__(self, t):
        self.x, self.y, self.z = t


# ── Joint bar widget ──────────────────────────────────────────────────────────

class JointBarsWidget(QWidget):
    """Animated bars showing each joint's current angle in its ±180° range."""

    _COLOURS = [
        '#00cc88', '#ffd166', '#ff6b6b',
        '#c678dd', '#61afef', '#e5c07b',
    ]

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(180)
        self._joints_deg = [0.0] * 6

    def set_joints(self, joints_deg: list):
        self._joints_deg = list(joints_deg)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor('#0d1820'))

        row_h = h / 6
        bar_margin = 10
        bar_h = max(14, int(row_h * 0.28))
        label_y_off = int(row_h * 0.22)
        val_y_off   = int(row_h * 0.48)
        bar_y_off   = int(row_h * 0.58)

        for i, (name, deg) in enumerate(zip(SHORT_NAMES, self._joints_deg)):
            y = int(i * row_h)
            colour = QColor(self._COLOURS[i])

            p.setFont(QFont('monospace', 9, QFont.Bold))
            p.setPen(QPen(colour.lighter(130)))
            p.drawText(bar_margin, y + label_y_off, name)

            p.setFont(QFont('monospace', 11, QFont.Bold))
            p.setPen(QPen(QColor('#f0f4f4')))
            p.drawText(bar_margin, y + val_y_off, f'{deg:+7.1f}°')

            # bar background
            bx = bar_margin
            by = y + bar_y_off
            bw = w - 2 * bar_margin
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor('#1a2a36')))
            p.drawRoundedRect(bx, by, bw, bar_h, 3, 3)

            # filled portion (-180..+180 → 0..1)
            frac = (deg + 180.0) / 360.0
            frac = max(0.0, min(1.0, frac))
            fill_w = int(bw * frac)
            if fill_w > 0:
                grad = QLinearGradient(bx, 0, bx + bw, 0)
                grad.setColorAt(0.0, colour.darker(160))
                grad.setColorAt(frac, colour)
                grad.setColorAt(1.0, colour.darker(160))
                p.setBrush(QBrush(grad))
                p.drawRoundedRect(bx, by, fill_w, bar_h, 3, 3)

            # centre tick (0 °)
            cx = bx + bw // 2
            p.setPen(QPen(QColor('#405060'), 1))
            p.drawLine(cx, by, cx, by + bar_h)

        p.end()


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class JointListenerNode(Node if _HAS_ROS else object):
    def __init__(self, callback):
        if _HAS_ROS:
            super().__init__('robot_camera_viewer_js')
            self.create_subscription(JointState, '/joint_states', self._recv, 10)
        self._cb = callback

    def _recv(self, msg: 'JointState'):
        joints = {n: math.degrees(p) for n, p in zip(msg.name, msg.position)
                  if n in JOINT_NAMES}
        self._cb(joints)


# ── Main window ───────────────────────────────────────────────────────────────

class ViewerWindow(QMainWindow):

    _joints_sig = pyqtSignal(dict)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.setWindowTitle('FruitNinja — UR3e Joint Viewer')
        self.setMinimumSize(1100, 660)
        self.setStyleSheet('background:#0b1117; color:white;')

        self._joints_deg  = {n: 0.0 for n in JOINT_NAMES}
        self._cap         = None     # cv2.VideoCapture for v4l2 sources
        self._rs_pipe     = None     # pyrealsense2 pipeline (RealSense colour)
        self._cam_index   = camera_index
        self._ros_ok      = False
        self._frame_skip  = 0      # throttle FK + skeleton to every other frame

        self._build_ui()
        self._joints_sig.connect(self._on_joints)

        # 50 ms ≈ 20 fps render loop
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

        self._open_camera(camera_index)
        self._start_ros()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ── header bar ────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel('UR3e Joint Viewer')
        title.setStyleSheet('color:#00cc88; font-size:16px; font-weight:bold;')
        hdr.addWidget(title)

        hdr.addStretch()

        hdr.addWidget(QLabel('Camera:'))
        self._cam_combo = QComboBox()
        self._cam_combo.setStyleSheet(
            'background:#1a2a36; color:#cde; border:1px solid #2a3f50;'
            'border-radius:4px; padding:4px 8px;'
        )
        # Sentinel index for the RealSense colour stream — anything <0 means
        # "use librealsense" instead of cv2.VideoCapture.  -1 is the RealSense
        # D435i; remaining negative values are reserved for future cameras.
        if _HAS_RS:
            self._cam_combo.addItem('RealSense D435i', -1)
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._cam_combo.addItem(f'Camera {i} (/dev/video{i})', i)
                cap.release()
        self._cam_combo.currentIndexChanged.connect(
            lambda: self._open_camera(self._cam_combo.currentData())
        )
        hdr.addWidget(self._cam_combo)

        self._ros_label = QLabel('ROS2: —')
        self._ros_label.setStyleSheet(
            'color:#e0a000; font-size:11px; font-weight:bold;'
            'background:#1a1a00; border:1px solid #4a3a00; border-radius:4px; padding:4px 8px;'
        )
        hdr.addWidget(self._ros_label)

        outer.addLayout(hdr)

        # ── main body ─────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(8)

        # Camera feed
        self._cam_label = QLabel()
        self._cam_label.setStyleSheet('background:#000; border:1px solid #1c2f3a; border-radius:4px;')
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(620, 460)
        body.addWidget(self._cam_label, stretch=3)

        # Right panel: bars + skeleton
        right = QVBoxLayout()
        right.setSpacing(6)

        bars_box = QGroupBox('Joint Angles')
        bars_box.setStyleSheet(
            'QGroupBox{color:#c9d6de; font-weight:bold; border:1px solid #263946;'
            'border-radius:6px; margin-top:8px; background:#0d1820;}'
            'QGroupBox::title{subcontrol-origin:margin; left:8px; padding:0 4px;}'
        )
        bars_layout = QVBoxLayout(bars_box)
        bars_layout.setContentsMargins(4, 12, 4, 4)
        self._bars = JointBarsWidget()
        bars_layout.addWidget(self._bars)
        right.addWidget(bars_box, stretch=2)

        skel_box = QGroupBox('FK Skeleton')
        skel_box.setStyleSheet(
            'QGroupBox{color:#c9d6de; font-weight:bold; border:1px solid #263946;'
            'border-radius:6px; margin-top:8px; background:#0b1117;}'
            'QGroupBox::title{subcontrol-origin:margin; left:8px; padding:0 4px;}'
        )
        skel_layout = QVBoxLayout(skel_box)
        skel_layout.setContentsMargins(4, 12, 4, 4)
        self._skeleton = SkeletonWidget()
        skel_layout.addWidget(self._skeleton)
        right.addWidget(skel_box, stretch=3)

        body.addLayout(right, stretch=2)
        outer.addLayout(body, stretch=1)

        # ── status bar ────────────────────────────────────────────────────────
        self._status = QLabel('Starting…')
        self._status.setStyleSheet('color:#607080; font-size:10px;')
        outer.addWidget(self._status)

    # ── camera ────────────────────────────────────────────────────────────────

    def _open_camera(self, index):
        """Open the selected camera.

        index >= 0 → /dev/video{index} via cv2.VideoCapture (USB / webcam).
        index <  0 → RealSense colour stream via pyrealsense2.
        """
        # Tear down any previous source so we don't leak the device.
        self._close_camera()

        if index is None:
            self._cam_index = -2
            return

        if index < 0:
            # RealSense colour stream
            if not _HAS_RS:
                print('[viewer] pyrealsense2 not installed — pip install pyrealsense2')
                self._cam_index = -2
                return
            try:
                self._rs_pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self._rs_pipe.start(cfg)
                self._cam_index = index
            except Exception as e:
                print(f'[viewer] RealSense start failed: {e}')
                self._rs_pipe = None
                self._cam_index = -2
            return

        # USB / webcam
        self._cap = cv2.VideoCapture(index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cam_index = index
        else:
            self._cap.release()
            self._cap = None
            self._cam_index = -2

    def _close_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._rs_pipe is not None:
            try:
                self._rs_pipe.stop()
            except Exception:
                pass
            self._rs_pipe = None

    def _grab_frame(self):
        """Return the latest BGR frame or None if no source is active."""
        if self._rs_pipe is not None:
            try:
                frames = self._rs_pipe.wait_for_frames(200)
                cf = frames.get_color_frame()
                if cf:
                    return np.asanyarray(cf.get_data())
            except Exception:
                return None
            return None
        if self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                return frame
        return None

    # ── ROS2 ─────────────────────────────────────────────────────────────────

    def _start_ros(self):
        if not _HAS_ROS:
            self._set_ros_label(False)
            return
        try:
            rclpy.init(args=None)
        except Exception:
            try:
                rclpy.shutdown()
            except Exception:
                pass
            try:
                rclpy.init(args=None)
            except Exception:
                self._set_ros_label(False)
                return
        try:
            self._node = JointListenerNode(
                lambda joints: self._joints_sig.emit(joints)
            )
            exe = rclpy.executors.SingleThreadedExecutor()
            exe.add_node(self._node)
            threading.Thread(target=exe.spin, daemon=True).start()
            self._ros_ok = True
            self._set_ros_label(True)
        except Exception as e:
            self._set_ros_label(False)
            print(f'[viewer] ROS2 init failed: {e}')

    def _set_ros_label(self, ok: bool):
        if ok:
            self._ros_label.setText('ROS2: ✔ Connected')
            self._ros_label.setStyleSheet(
                'color:#00cc88; font-size:11px; font-weight:bold;'
                'background:#071a13; border:1px solid #136b4c; border-radius:4px; padding:4px 8px;'
            )
        else:
            self._ros_label.setText('ROS2: ✖ No /joint_states')
            self._ros_label.setStyleSheet(
                'color:#ff6060; font-size:11px; font-weight:bold;'
                'background:#1a0808; border:1px solid #6b1313; border-radius:4px; padding:4px 8px;'
            )

    # ── tick ──────────────────────────────────────────────────────────────────

    def _tick(self):
        frame = self._grab_frame()

        joints_list = [self._joints_deg.get(n, 0.0) for n in JOINT_NAMES]

        if frame is not None:
            frame = self._draw_overlay(frame, joints_list)
            h, w, ch = frame.shape
            qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
            self._cam_label.setPixmap(
                QPixmap.fromImage(qt_img).scaled(
                    self._cam_label.width(), self._cam_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )
        else:
            self._cam_label.setText('No camera')

        # Bars update every tick (cheap)
        self._bars.set_joints(joints_list)

        # Skeleton every other tick (still 10 fps, saves ~1 ms FK + repaint)
        self._frame_skip = (self._frame_skip + 1) % 2
        if self._frame_skip == 0:
            self._skeleton.set_joints(joints_list)

        self._status.setText(
            '  '.join(f'{s}: {d:+.1f}°' for s, d in zip(SHORT_NAMES, joints_list))
        )

    # ── OpenCV overlay ────────────────────────────────────────────────────────

    @staticmethod
    def _draw_overlay(frame: np.ndarray, joints_deg: list) -> np.ndarray:
        """Draw joint-angle text + a simple coloured bar overlay on the frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # semi-transparent dark panel top-left
        panel_w, panel_h = 220, 150
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 18, 24), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        colours_bgr = [
            (136, 204, 0),   # green
            (102, 209, 255), # yellow-ish
            (107, 107, 255), # red-ish
            (221, 120, 198), # purple
            (239, 175,  97), # blue
            (123, 192, 229), # tan
        ]

        for i, (name, deg, col) in enumerate(zip(SHORT_NAMES, joints_deg, colours_bgr)):
            y = 28 + i * 22
            cv2.putText(frame, f'{name:<7} {deg:+7.1f}',
                        (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # TCP position label bottom-left
        pts = ur3e_joint_positions(joints_deg)
        tcp = pts[-1]
        cv2.putText(frame,
                    f'TCP  X:{tcp[0]*1000:+6.0f}  Y:{tcp[1]*1000:+6.0f}  Z:{tcp[2]*1000:+6.0f} mm',
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 200), 1, cv2.LINE_AA)

        return frame

    # ── ROS2 callback ─────────────────────────────────────────────────────────

    def _on_joints(self, joints: dict):
        self._joints_deg.update(joints)
        if not self._ros_ok:
            self._ros_ok = True
            self._set_ros_label(True)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer.stop()
        self._close_camera()
        if _HAS_ROS:
            try:
                self._node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='FruitNinja UR3e joint viewer')
    parser.add_argument('--camera', type=int, default=-1,
                        help='Camera source.  -1 = RealSense D435i (default), '
                             '0..7 = /dev/video<N> via cv2.VideoCapture.')
    parsed, remaining = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + remaining)
    app.setStyle('Fusion')
    win = ViewerWindow(camera_index=parsed.camera)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
