#!/usr/bin/env python3
# Must be set before any Qt or cv2 import to avoid plugin path conflicts
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

"""
FruitNinja Control Panel
  - Live UR3e arm visualisation (2-D side view from /joint_states)
  - Buttons: Rebuild, Launch MoveIt/RViz, Start Cuts, Stop
  - OpenCV fruit-colour detection feed
  - Command log
"""

import sys
import subprocess
import threading
import math
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import cv2
import numpy as np
from fruitninja.colour_detection import detect_fruits

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit,
    QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont


# ── Constants ──────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# UR3e link lengths (mm) — used only for proportional drawing
L_UPPER_ARM = 243
L_FOREARM   = 213
L_WRIST     = 95


# ── ROS2 node (runs in background thread) ──────────────────────────────────────

class JointStateNode(Node):
    def __init__(self, callback):
        super().__init__('fruitninja_gui')
        self._cb = callback
        self.create_subscription(JointState, '/joint_states', self._recv, 10)

    def _recv(self, msg: JointState):
        joints = {n: p for n, p in zip(msg.name, msg.position) if n in JOINT_NAMES}
        self._cb(joints)


# ── 2-D arm visualisation ──────────────────────────────────────────────────────

class ArmWidget(QWidget):
    """Draws a simplified side-view stick figure of the UR3e."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 360)
        self._joints = {n: 0.0 for n in JOINT_NAMES}

    def set_joints(self, joints: dict):
        for name, val in joints.items():
            cur = self._joints.get(name, 0.0)
            # Reject spurious zero-resets from the hardware broadcaster:
            # real joint movement can't change more than ~0.1 rad between messages.
            if abs(cur) > 0.05 and abs(val - cur) > 0.5:
                continue
            self._joints[name] = val
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(25, 25, 25))

        # grid
        p.setPen(QPen(QColor(45, 45, 45), 1))
        for x in range(0, W, 40):
            p.drawLine(x, 0, x, H)
        for y in range(0, H, 40):
            p.drawLine(0, y, W, y)

        scale = min(W, H) / 750.0
        cx, cy = W // 2, H - 50

        pan   = self._joints.get('shoulder_pan_joint',  0.0)
        lift  = self._joints.get('shoulder_lift_joint', 0.0)
        elbow = self._joints.get('elbow_joint',         0.0)
        w1    = self._joints.get('wrist_1_joint',       0.0)

        # base plate
        p.setPen(QPen(QColor(180, 180, 180), 3))
        p.drawLine(cx - 35, cy, cx + 35, cy)

        def draw_link(x0, y0, length, angle, colour, thickness):
            x1 = x0 + int(length * scale * math.cos(angle))
            y1 = y0 - int(length * scale * math.sin(angle))
            p.setPen(QPen(QColor(*colour), thickness))
            p.drawLine(x0, y0, x1, y1)
            return x1, y1

        def draw_joint(x, y, r, colour):
            p.setBrush(QBrush(QColor(*colour)))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(x - r, y - r, r * 2, r * 2)

        # shoulder (base joint)
        draw_joint(cx, cy, 7, (255, 200, 50))

        # upper arm
        a = math.pi / 2 + lift
        ux, uy = draw_link(cx, cy, L_UPPER_ARM, a, (255, 140, 0), 5)
        draw_joint(ux, uy, 6, (100, 180, 255))

        # forearm
        a += elbow
        fx, fy = draw_link(ux, uy, L_FOREARM, a, (180, 180, 180), 4)
        draw_joint(fx, fy, 5, (100, 220, 100))

        # wrist
        a += w1
        ex, ey = draw_link(fx, fy, L_WRIST, a, (140, 140, 140), 3)
        draw_joint(ex, ey, 4, (255, 80, 80))

        # tool tip marker
        p.setPen(QPen(QColor(255, 50, 50), 2))
        p.setBrush(QBrush(QColor(255, 50, 50)))
        p.drawEllipse(ex - 4, ey - 4, 8, 8)

        # joint value readout (top-left)
        labels = ['Pan', 'Lift', 'Elbow', 'W1', 'W2', 'W3']
        values = [
            pan, lift, elbow,
            self._joints.get('wrist_1_joint', 0.0),
            self._joints.get('wrist_2_joint', 0.0),
            self._joints.get('wrist_3_joint', 0.0),
        ]
        p.setFont(QFont('Monospace', 8))
        for i, (lbl, val) in enumerate(zip(labels, values)):
            p.setPen(QPen(QColor(200, 200, 200), 1))
            p.drawText(6, 16 + i * 18, f'{lbl:6s}: {math.degrees(val):+7.1f}°')

        p.end()


# ── OpenCV camera widget ───────────────────────────────────────────────────────

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False


class CameraWidget(QLabel):
    """
    Grabs colour frames from either a webcam (cv2.VideoCapture) or an Intel
    RealSense D435i. Source is selected via switch_source('webcam'|'realsense').
    Emits detection_signal with a list of detection dicts on each frame.
    """

    detection_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setText('No camera connected')
        self.setStyleSheet('color: #888; background: #111; font-size: 13px;')
        self._cap      = None
        self._pipeline = None
        self._timer    = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, source: str = 'webcam'):
        self._start_source(source)

    def switch_source(self, source: str):
        """Stop current stream and restart with the chosen source."""
        self.stop()
        self._start_source(source)

    def _start_source(self, source: str):
        if source == 'realsense':
            if not _HAS_RS:
                self.setText('pyrealsense2 not installed')
                print('[camera] pyrealsense2 not available')
                return
            try:
                self._pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self._pipeline.start(cfg)
                self._timer.start(33)
                print('[camera] RealSense D435i started')
            except Exception as e:
                self._pipeline = None
                self.setText(f'RealSense error:\n{e}')
                print(f'[camera] RealSense error: {e}')
        else:
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self._timer.start(33)
                print('[camera] Webcam started')
            else:
                self._cap = None
                self.setText('No webcam found')

    def stop(self):
        self._timer.stop()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def _tick(self):
        frame = None
        if self._pipeline:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=100)
                cf = frames.get_color_frame()
                if cf:
                    frame = np.asanyarray(cf.get_data())
            except Exception:
                return
        elif self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                return

        if frame is None:
            return

        frame, detections = detect_fruits(frame)
        self.detection_signal.emit(detections)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )



# ── Detection log ─────────────────────────────────────────────────────────────

class DetectionLogWidget(QTextEdit):
    """Timestamped log of every fruit detection event."""
    _sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(140)
        self.setStyleSheet(
            'background:#0d0d1a; color:#00ddff;'
            'font-family:monospace; font-size:11px;'
        )
        self._sig.connect(self._append)

    def push(self, text: str):
        self._sig.emit(text)

    def _append(self, text: str):
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ── ROS log output ─────────────────────────────────────────────────────────────

class LogWidget(QTextEdit):
    _sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(130)
        self.setStyleSheet(
            'background:#0a0a0a; color:#00ee00;'
            'font-family:monospace; font-size:11px;'
        )
        self._sig.connect(self._append)

    def push(self, text: str):
        self._sig.emit(text)

    def _append(self, text: str):
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ── Grid selector widget ──────────────────────────────────────────────────────

GRID_COLS = ['A', 'B', 'C', 'D']
GRID_ROWS = ['1', '2', '3', '4']


class GridSelectorWidget(QWidget):
    """4×4 clickable grid matching the vision grid. Highlights the selected cell."""

    cell_selected = pyqtSignal(str)   # emits e.g. 'B3' on click

    _BTN_BASE  = 'background:#2a2a3a; color:#ccc; border:1px solid #555; border-radius:3px; font-size:12px; font-weight:bold;'
    _BTN_SEL   = 'background:#3a7aff; color:white; border:1px solid #88aaff; border-radius:3px; font-size:12px; font-weight:bold;'

    def __init__(self):
        super().__init__()
        self._selected: str | None = None
        self._buttons: dict[str, QPushButton] = {}

        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # Column headers (A B C D)
        for c, col in enumerate(GRID_COLS):
            hdr = QLabel(col)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#888; font-size:11px;')
            layout.addWidget(hdr, 0, c + 1)

        # Row headers + cell buttons
        for r, row in enumerate(GRID_ROWS):
            hdr = QLabel(row)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#888; font-size:11px;')
            layout.addWidget(hdr, r + 1, 0)

            for c, col in enumerate(GRID_COLS):
                cell = col + row
                btn = QPushButton(cell)
                btn.setFixedSize(46, 36)
                btn.setStyleSheet(self._BTN_BASE)
                btn.clicked.connect(lambda checked, ce=cell: self._on_click(ce))
                self._buttons[cell] = btn
                layout.addWidget(btn, r + 1, c + 1)

    def _on_click(self, cell: str):
        if self._selected:
            self._buttons[self._selected].setStyleSheet(self._BTN_BASE)
        self._selected = cell
        self._buttons[cell].setStyleSheet(self._BTN_SEL)
        self.cell_selected.emit(cell)

    def selected_cell(self) -> str | None:
        return self._selected


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _joints_sig = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja Control Panel')
        self.setMinimumSize(960, 720)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._procs: dict[str, subprocess.Popen] = {}
        self._last_log_time: dict[str, float] = {}   # throttle detection log
        self._build_ui()
        self._start_ros()
        self._cam.start()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── top row ───────────────────────────────────────────────────────────
        top = QHBoxLayout()

        arm_group = self._group('UR3e Live State')
        self._arm = ArmWidget()
        arm_group.layout().addWidget(self._arm)
        self._joints_sig.connect(self._arm.set_joints)
        top.addWidget(arm_group, stretch=2)

        ctrl_group = self._group('Controls')
        ctrl_group.setFixedWidth(260)
        cl = ctrl_group.layout()
        cl.setSpacing(8)

        # ── robot IP field ────────────────────────────────────────────────────
        ip_group = self._group('Robot IP')
        ip_layout = ip_group.layout()
        ip_layout.setContentsMargins(4, 4, 4, 4)
        self._ip_field = QLineEdit('192.168.0.150')
        self._ip_field.setStyleSheet(
            'background:#333; color:white; font-family:monospace; padding:4px;'
        )
        self._ip_field.setPlaceholderText('e.g. 192.168.1.100')
        ip_layout.addWidget(self._ip_field)
        cl.addWidget(ip_group)

        self._btn_build      = self._btn('🔨  Rebuild',          '#1e4a7a', self._rebuild)
        self._btn_launch_sim = self._btn('🖥   Launch Sim',       '#1e5c1e', self._launch_sim)
        self._btn_launch_real= self._btn('🤖  Connect Real UR3e','#5c3a1e', self._launch_real)
        self._btn_run        = self._btn('▶   Start Cuts',        '#6a1e1e', self._run)
        self._btn_move_point = self._btn('⊕   Move to Point',     '#1a4a5a', self._move_to_point)
        self._btn_stop       = self._btn('■   Stop All',          '#444444', self._stop)
        self._btn_reset      = self._btn('↺   Reset',             '#4a3a00', self._reset)
        self._btn_shutdown   = self._btn('⏻   Quit All',          '#5a1a1a', self._shutdown)

        self._status = QLabel('● Idle')
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet('color:#aaa; font-size:12px;')

        # ── cut parameters ────────────────────────────────────────────────────
        param_group = self._group('Cut Parameters')
        pl = param_group.layout()
        pl.setSpacing(6)

        SPIN_STYLE = 'background:#333; color:white; font-size:12px; padding:2px;'

        def _row(label, widget):
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(label)
            lbl.setFixedWidth(78)
            lbl.setStyleSheet('color:#ccc; font-size:11px;')
            widget.setFixedWidth(90)
            row.addWidget(lbl)
            row.addWidget(widget)
            row.addStretch()
            pl.addLayout(row)

        self._spin_cuts = QSpinBox()
        self._spin_cuts.setRange(1, 30)
        self._spin_cuts.setValue(5)
        self._spin_cuts.setStyleSheet(SPIN_STYLE)
        _row('Num cuts:', self._spin_cuts)

        self._spin_centre = QDoubleSpinBox()
        self._spin_centre.setRange(-90, 90)
        self._spin_centre.setValue(-13.0)
        self._spin_centre.setSuffix('°')
        self._spin_centre.setDecimals(1)
        self._spin_centre.setStyleSheet(SPIN_STYLE)
        _row('Centre pan:', self._spin_centre)

        self._spin_range = QDoubleSpinBox()
        self._spin_range.setRange(1, 90)
        self._spin_range.setValue(28.0)
        self._spin_range.setSuffix('°')
        self._spin_range.setDecimals(1)
        self._spin_range.setStyleSheet(SPIN_STYLE)
        _row('± Range:', self._spin_range)

        for w in (self._btn_build,
                  self._btn_launch_sim, self._btn_launch_real,
                  self._btn_run, self._btn_move_point, self._btn_stop,
                  self._btn_reset, self._btn_shutdown,
                  param_group):
            cl.addWidget(w)
        cl.addStretch()
        cl.addWidget(self._status)
        top.addWidget(ctrl_group)
        root.addLayout(top, stretch=3)

        # ── camera ────────────────────────────────────────────────────────────
        cam_group = self._group('OpenCV — Fruit Colour Detection')
        cam_header = QHBoxLayout()
        cam_source_label = QLabel('Camera:')
        cam_source_label.setStyleSheet('color:#aaa; font-size:12px;')
        self._cam_source_combo = QComboBox()
        self._cam_source_combo.addItems(['Webcam', 'RealSense D435i'])
        self._cam_source_combo.setStyleSheet(
            'background:#333; color:white; font-size:12px; padding:2px;'
        )
        self._cam_source_combo.setFixedWidth(150)
        if not _HAS_RS:
            self._cam_source_combo.model().item(1).setEnabled(False)
        cam_header.addWidget(cam_source_label)
        cam_header.addWidget(self._cam_source_combo)
        cam_header.addStretch()
        cam_group.layout().addLayout(cam_header)
        self._cam = CameraWidget()
        cam_group.layout().addWidget(self._cam)
        self._cam.detection_signal.connect(self._on_detections)
        self._cam_source_combo.currentTextChanged.connect(
            lambda t: self._cam.switch_source('realsense' if 'RealSense' in t else 'webcam')
        )
        root.addWidget(cam_group, stretch=2)

        # ── grid navigation ───────────────────────────────────────────────────
        grid_group = self._group('Grid Navigation — Select Target Cell')
        grid_inner = QHBoxLayout()
        self._grid_selector = GridSelectorWidget()
        self._grid_cell_label = QLabel('No cell selected')
        self._grid_cell_label.setAlignment(Qt.AlignCenter)
        self._grid_cell_label.setStyleSheet(
            'color:#aaa; font-family:monospace; font-size:12px;'
        )
        self._grid_selector.cell_selected.connect(
            lambda c: self._grid_cell_label.setText(f'Selected: {c}')
        )
        grid_inner.addWidget(self._grid_selector)
        grid_inner.addWidget(self._grid_cell_label)
        grid_inner.addStretch()
        grid_group.layout().addLayout(grid_inner)
        root.addWidget(grid_group)

        # ── detection info bar ────────────────────────────────────────────────
        self._detect_label = QLabel('No fruit detected')
        self._detect_label.setAlignment(Qt.AlignCenter)
        self._detect_label.setStyleSheet(
            'background:#111; color:#eee; font-family:monospace;'
            'font-size:13px; font-weight:bold; padding:6px;'
            'border:1px solid #333; border-radius:4px;'
        )
        self._detect_label.setMinimumHeight(36)
        root.addWidget(self._detect_label)

        # ── detection log ─────────────────────────────────────────────────────
        det_log_group = self._group('Detection Log')
        self._det_log = DetectionLogWidget()
        det_log_group.layout().addWidget(self._det_log)
        root.addWidget(det_log_group)

        # ── ROS log ───────────────────────────────────────────────────────────
        self._log = LogWidget()
        root.addWidget(self._log)

    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet('QGroupBox{color:white;font-weight:bold;'
                        'border:1px solid #444;border-radius:4px;margin-top:8px;}'
                        'QGroupBox::title{subcontrol-origin:margin;left:8px;}')
        g.setLayout(QVBoxLayout())
        return g

    @staticmethod
    def _btn(text: str, colour: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.setStyleSheet(f'''
            QPushButton{{background:{colour};color:white;border-radius:6px;
                         padding:10px;font-size:13px;font-weight:bold;}}
            QPushButton:hover{{background:{colour}cc;}}
            QPushButton:pressed{{background:{colour}88;}}
        ''')
        b.clicked.connect(slot)
        return b

    # ── button slots ──────────────────────────────────────────────────────────

    _ROS = 'source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash'

    def _rebuild(self):
        self._status_set('🔨 Building…', '#e0a000')
        self._shell(
            'build',
            f'{self._ROS} && cd ~/ros2_ws && colcon build --packages-select fruitninja',
            done_msg='✓ Built',
        )

    def _launch_sim(self):
        self._status_set('🖥 Launching sim…', '#3a7aff')
        self._shell(
            'launch',
            f'{self._ROS} && ros2 launch fruitninja fruitninja.launch.py'
            f' use_fake_hardware:=true robot_ip:=192.168.56.101',
            persistent=True,
        )
        self._status_set('🖥 Sim running', '#3a7aff')

    def _launch_real(self):
        # Kill any previous launch before starting real robot session
        if 'launch' in self._procs:
            try:
                self._procs.pop('launch').terminate()
            except Exception:
                pass
        subprocess.call(['pkill', '-f', 'ur_control.launch.py'])
        subprocess.call(['pkill', '-f', 'ur_ros2_control_node'])

        ip = self._ip_field.text().strip()
        self._status_set(f'🤖 Connecting {ip}…', '#e07000')
        # Use the full fruitninja launch so MoveIt + RViz start alongside
        # the real robot driver (use_fake_hardware:=false)
        self._shell(
            'launch',
            f'{self._ROS} && ros2 launch fruitninja fruitninja.launch.py'
            f' use_fake_hardware:=false'
            f' robot_ip:={ip}'
            f' reverse_port:=50001',
            persistent=True,
        )
        self._status_set(f' Real robot {ip}', '#e07000')

    def _run(self):
        n  = self._spin_cuts.value()
        c  = self._spin_centre.value()
        hr = self._spin_range.value()
        self._status_set('▶ Cutting…', '#ff5555')
        self._shell(
            'movement',
            f'{self._ROS} && ros2 run fruitninja movement'
            f' --num-cuts {n}'
            f' --pan-centre {c}'
            f' --pan-half-range {hr}',
            done_msg='✓ Sequence complete',
        )

    def _move_to_point(self):
        cell = self._grid_selector.selected_cell()
        if not cell:
            self._status_set('⊕ Select a cell first', '#e0a000')
            return
        self._status_set(f'⊕ Moving to {cell}…', '#1a9aaa')
        self._shell(
            'grid_move',
            f'{self._ROS} && ros2 run fruitninja grid_mover --cell {cell}',
            done_msg=f'✓ Reached {cell}',
        )

    def _stop(self):
        for proc in self._procs.values():
            try:
                proc.terminate()
            except Exception:
                pass
        self._procs.clear()
        self._status_set('■ Stopped', '#aaaaaa')

    def _reset(self):
        self._stop()
        self._status_set('↺ Resetting…', '#e0a000')
        self._shell(
            'reset',
            f'{self._ROS} && ros2 run fruitninja reset',
            done_msg='✓ Ready position restored',
        )

    def _shutdown(self):
        self._stop()
        subprocess.Popen(['bash', '-c', 'pkill -f "ros2 launch"'])
        subprocess.Popen(['bash', '-c', 'pkill -f "move_group"'])
        subprocess.Popen(['bash', '-c', 'pkill -f "rviz2"'])
        self.close()

    def _on_detections(self, detections: list):
        if not detections:
            self._detect_label.setText('No fruit detected')
            self._detect_label.setStyleSheet(
                'background:#111; color:#888; font-family:monospace;'
                'font-size:13px; font-weight:bold; padding:6px;'
                'border:1px solid #333; border-radius:4px;'
            )
            return

        parts = []
        now = time.time()
        for d in detections:
            label  = d['label']
            dist   = d['distance_mm']
            cell   = d.get('cell')
            colour = {'Apple': '#ff4444', 'Lettuce': '#44cc44',
                      'Banana': '#ffdd00', 'Orange': '#ff8800'}.get(label, '#ffffff')
            cell_part = f'  —  cell: <b style="color:#ffdd44">{cell}</b>' if cell else ''
            parts.append(
                f'<span style="color:{colour}">Detecting {label}</span>'
                f'  —  <b>{dist:.0f} mm</b>'
                f'{cell_part}'
            )
            # Log at most once per second per label
            key = f'{label}:{cell}'
            if now - self._last_log_time.get(key, 0) >= 1.0:
                self._last_log_time[key] = now
                ts       = datetime.now().strftime('%H:%M:%S')
                cell_str = f'cell {cell}' if cell else 'outside grid'
                self._det_log.push(
                    f'[{ts}]  {label:<8}  |  {dist:>6.0f} mm  |  {cell_str}'
                )
        self._detect_label.setText('    |    '.join(parts))
        self._detect_label.setStyleSheet(
            'background:#111; color:#eee; font-family:monospace;'
            'font-size:13px; font-weight:bold; padding:6px;'
            'border:1px solid #333; border-radius:4px;'
        )

    def _status_set(self, text: str, colour: str = 'white'):
        self._status.setText(text)
        self._status.setStyleSheet(f'color:{colour};font-size:12px;')

    def _shell(self, key: str, cmd: str,
               persistent: bool = False, done_msg: str = ''):
        def _run():
            proc = subprocess.Popen(
                ['bash', '-c', cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if persistent:
                self._procs[key] = proc
            for line in proc.stdout:
                self._log.push(line.rstrip())
            proc.wait()
            if done_msg:
                self._status_set(done_msg, '#00cc00')
                self._log.push(f'[{key}] {done_msg}')
        threading.Thread(target=_run, daemon=True).start()

    # ── ROS2 ──────────────────────────────────────────────────────────────────

    def _start_ros(self):
        try:
            rclpy.init(args=None)
            self._ros_node = JointStateNode(
                lambda joints: self._joints_sig.emit(joints)
            )
            threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True,
            ).start()
            self._log.push('[ros] Joint state subscriber started')
        except Exception as e:
            self._log.push(f'[ros] Warning: {e}')

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._cam.stop()
        self._stop()
        try:
            self._ros_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
