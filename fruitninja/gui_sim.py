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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import cv2
import numpy as np
from fruitninja.colour_detection import detect_fruits

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit,
    QSpinBox, QDoubleSpinBox,
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
        self._joints.update(joints)
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

class CameraWidget(QLabel):
    """Grabs frames from the first webcam, applies HSV fruit-colour detection."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setText('No camera connected')
        self.setStyleSheet('color: #888; background: #111; font-size: 13px;')
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._cap = cv2.VideoCapture(0)
        if self._cap.isOpened():
            self._timer.start(33)          # ~30 fps
        else:
            self.setText('Camera not found (index 0)')

    def stop(self):
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None

    def _tick(self):
        if not self._cap or not self._cap.isOpened():
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        frame = detect_fruits(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )



# ── Log output ────────────────────────────────────────────────────────────────

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


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _joints_sig = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja Control Panel')
        self.setMinimumSize(960, 720)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._procs: dict[str, subprocess.Popen] = {}
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
        ctrl_group.setFixedWidth(230)
        cl = ctrl_group.layout()
        cl.setSpacing(10)

        self._btn_build    = self._btn('🔨  Rebuild',       '#1e4a7a', self._rebuild)
        self._btn_launch   = self._btn('🚀  Launch MoveIt', '#1e5c1e', self._launch)
        self._btn_run      = self._btn('▶   Start Cuts',    '#6a1e1e', self._run)
        self._btn_stop     = self._btn('■   Stop All',      '#444444', self._stop)
        self._btn_reset    = self._btn('↺   Reset',         '#4a3a00', self._reset)
        self._btn_shutdown = self._btn('⏻   Quit All',      '#5a1a1a', self._shutdown)

        self._status = QLabel('● Idle')
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet('color:#aaa; font-size:12px;')

        # ── cut parameters ────────────────────────────────────────────────────
        param_group = self._group('Cut Parameters')
        pl = param_group.layout()
        pl.setSpacing(6)

        def _row(label, widget):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('color:#ccc; font-size:11px;')
            row.addWidget(lbl)
            row.addWidget(widget)
            pl.addLayout(row)

        self._spin_cuts = QSpinBox()
        self._spin_cuts.setRange(1, 30)
        self._spin_cuts.setValue(5)
        self._spin_cuts.setStyleSheet('background:#333; color:white;')
        _row('Num cuts:', self._spin_cuts)

        self._spin_centre = QDoubleSpinBox()
        self._spin_centre.setRange(-90, 90)
        self._spin_centre.setValue(-13.0)
        self._spin_centre.setSuffix('°')
        self._spin_centre.setStyleSheet('background:#333; color:white;')
        _row('Centre pan:', self._spin_centre)

        self._spin_range = QDoubleSpinBox()
        self._spin_range.setRange(1, 90)
        self._spin_range.setValue(28.0)
        self._spin_range.setSuffix('°')
        self._spin_range.setStyleSheet('background:#333; color:white;')
        _row('± Range:', self._spin_range)

        for w in (self._btn_build, self._btn_launch,
                  self._btn_run,   self._btn_stop,
                  self._btn_reset, self._btn_shutdown,
                  param_group):
            cl.addWidget(w)
        cl.addStretch()
        cl.addWidget(self._status)
        top.addWidget(ctrl_group)
        root.addLayout(top, stretch=3)

        # ── camera ────────────────────────────────────────────────────────────
        cam_group = self._group('OpenCV — Fruit Colour Detection')
        self._cam = CameraWidget()
        cam_group.layout().addWidget(self._cam)
        root.addWidget(cam_group, stretch=2)

        # ── log ───────────────────────────────────────────────────────────────
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

    def _launch(self):
        self._status_set('🚀 Launching…', '#3a7aff')
        self._shell(
            'launch',
            f'{self._ROS} && ros2 launch fruitninja fruitninja.launch.py',
            persistent=True,
        )
        self._status_set('🚀 MoveIt running', '#3a7aff')

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
