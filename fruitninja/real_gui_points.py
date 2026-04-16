#!/usr/bin/env python3
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import sys
import math
import threading
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes

from fruitninja.grid_mover import cell_to_joints_deg, GRID_COLS, GRID_ROWS

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit,
    QComboBox, QShortcut,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False

try:
    from fruitninja.colour_detection import detect_fruits
    _HAS_DETECTION = True
except ImportError:
    _HAS_DETECTION = False


# ── Camera helpers ────────────────────────────────────────────────────────────

def _find_dji_device_index() -> int:
    """
    Scan /sys/class/video4linux to find which /dev/videoN belongs to the
    DJI Osmo Pocket 3 (USB vendor ID 2ca3).  Returns -1 if not found.
    """
    import glob
    for node in sorted(glob.glob('/sys/class/video4linux/video*')):
        try:
            # Walk up the sysfs tree looking for the idVendor file
            path = os.path.realpath(node)
            parts = path.split('/')
            for i in range(len(parts), 0, -1):
                vendor_file = '/'.join(parts[:i]) + '/idVendor'
                if os.path.exists(vendor_file):
                    with open(vendor_file) as f:
                        if f.read().strip().lower() == '2ca3':   # DJI vendor ID
                            idx = int(os.path.basename(node).replace('video', ''))
                            return idx
                    break
        except Exception:
            continue
    return -1


# ── Camera widget ─────────────────────────────────────────────────────────────

class CameraWidget(QLabel):
    """Live camera feed. Supports webcam, fisheye, and RealSense D435i."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setText('Camera not started')
        self.setStyleSheet('color:#888; background:#111; font-size:13px;')
        self._cap      = None
        self._pipeline = None
        self._timer    = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, source: str = 'webcam'):
        self._start_source(source)

    def stop(self):
        self._timer.stop()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self.setText('Camera stopped')

    def _start_source(self, source: str):
        if source == 'realsense':
            if not _HAS_RS:
                self.setText('pyrealsense2 not installed')
                return
            try:
                self._pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self._pipeline.start(cfg)
                self._timer.start(33)
            except Exception as e:
                self._pipeline = None
                self.setText(f'RealSense error:\n{e}')
        elif source == 'fisheye':
            self._cap = cv2.VideoCapture(2)
            if self._cap.isOpened():
                self._timer.start(33)
            else:
                self._cap = None
                self.setText('No fisheye camera at index 2')
        elif source == 'dji':
            idx = _find_dji_device_index()
            if idx == -1:
                self.setText(
                    'DJI Osmo Pocket 3 not found.\n'
                    'Make sure it is in webcam/UVC mode:\n'
                    'On the camera: swipe down → USB → "Webcam" mode'
                )
                return
            self._cap = cv2.VideoCapture(idx)
            if self._cap.isOpened():
                self._timer.start(33)
            else:
                self._cap = None
                self.setText(f'DJI found at /dev/video{idx} but failed to open')
        else:
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self._timer.start(33)
            else:
                self._cap = None
                self.setText('No webcam found')

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

        if _HAS_DETECTION:
            frame, _ = detect_fruits(frame)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )


# ── Constants ──────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

JOINT_LABELS = ['Base (pan)', 'Shoulder (lift)', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

HOME_DEG = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0]

# Cutting motion: drive end-effector down close to the trolley surface
CUT_DIP_LIFT  =  30.0   # degrees added to shoulder_lift_joint (index 1)
CUT_DIP_ELBOW = -30.0   # degrees added to elbow_joint (index 2)

MOVE_GROUP = 'ur_manipulator'


# ── ROS2 nodes ─────────────────────────────────────────────────────────────────

class JointStateNode(Node):
    def __init__(self, joint_callback):
        super().__init__('real_gui_points_js')
        self._cb = joint_callback
        self.create_subscription(JointState, '/joint_states', self._recv, 10)

    def _recv(self, msg: JointState):
        joints = {n: math.degrees(p) for n, p in zip(msg.name, msg.position)
                  if n in JOINT_NAMES}
        self._cb(joints)


class MoverNode(Node):
    def __init__(self):
        super().__init__('real_gui_points_mover')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        self._cancel_flag = False

    def move_to(self, degrees: list, done_cb, fail_cb):
        self._cancel_flag = False
        threading.Thread(
            target=self._move_thread,
            args=(degrees, done_cb, fail_cb),
            daemon=True,
        ).start()

    def _move_thread(self, degrees, done_cb, fail_cb):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        try:
            if not self._client.wait_for_server(timeout_sec=5.0):
                fail_cb('MoveGroup not available — is MoveIt running?')
                return
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            goal = MoveGroup.Goal()
            goal.request.group_name = MOVE_GROUP
            goal.request.goal_constraints.append(self._make_constraints(degrees))
            goal.request.num_planning_attempts = 10
            goal.request.allowed_planning_time = 5.0
            goal.request.max_velocity_scaling_factor     = 0.3
            goal.request.max_acceleration_scaling_factor = 0.3

            future = self._client.send_goal_async(goal)
            executor.spin_until_future_complete(future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            goal_handle = future.result()
            if not goal_handle.accepted:
                fail_cb('Goal rejected by MoveIt')
                return

            result_future = goal_handle.get_result_async()
            executor.spin_until_future_complete(result_future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            result = result_future.result().result
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                done_cb()
            else:
                fail_cb(f'Move failed (error code: {result.error_code.val})')
        finally:
            executor.remove_node(self)

    def _make_constraints(self, degrees: list) -> Constraints:
        c = Constraints()
        for name, deg in zip(JOINT_NAMES, degrees):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = math.radians(deg)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        return c

    def cancel(self):
        self._cancel_flag = True


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _joints_sig = pyqtSignal(dict)
    _status_sig = pyqtSignal(str, str)
    _log_sig    = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('RealGuiPoints — UR3e Grid Control')
        self.setMinimumSize(820, 620)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._selected_cells = []   # ordered list, sorted left-to-right
        self._moving = False
        self._cell_btns = {}

        self._build_ui()
        self._start_ros()

        self._joints_sig.connect(self._update_joint_display)
        self._status_sig.connect(lambda t, c: self._set_status(t, c))
        self._log_sig.connect(self._log_widget.append)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QHBoxLayout(root_w)   # horizontal split: controls left, camera right
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── LEFT COLUMN ───────────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        # joint state display
        js_group = self._group('Live Joint States')
        js_layout = QGridLayout()
        js_layout.setSpacing(4)

        self._joint_labels = {}
        for col, header in enumerate(['Joint', 'Current Angle']):
            lbl = QLabel(header)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('color:#aaa; font-size:11px; font-weight:bold;')
            js_layout.addWidget(lbl, 0, col)

        for row, (jname, jlabel) in enumerate(zip(JOINT_NAMES, JOINT_LABELS)):
            name_lbl = QLabel(jlabel)
            name_lbl.setStyleSheet('color:#ccc; font-size:12px; padding:2px 8px;')
            js_layout.addWidget(name_lbl, row + 1, 0)

            val_lbl = QLabel('—')
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                'background:#2a2a3a; color:#00ddff; font-family:monospace;'
                'font-size:12px; border-radius:3px; padding:3px 10px;'
            )
            self._joint_labels[jname] = val_lbl
            js_layout.addWidget(val_lbl, row + 1, 1)

        js_group.layout().addLayout(js_layout)
        left.addWidget(js_group)

        # grid selector
        grid_group = self._group('Grid — Toggle Cells  (A1=far-left  N4=near-right  |  moves left→right)')
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(4, 4, 4, 4)

        for c, col in enumerate(GRID_COLS):
            hdr = QLabel(col)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#888; font-size:10px;')
            grid_layout.addWidget(hdr, 0, c + 1)

        for r, row in enumerate(GRID_ROWS):
            hdr = QLabel(row)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#888; font-size:10px;')
            grid_layout.addWidget(hdr, r + 1, 0)

            for c, col in enumerate(GRID_COLS):
                cell = col + row
                btn = QPushButton(cell)
                btn.setFixedSize(40, 30)
                btn.setCheckable(True)
                btn.setStyleSheet(self._cell_style_normal())
                btn.clicked.connect(lambda _checked, ce=cell: self._select_cell(ce))
                self._cell_btns[cell] = btn
                grid_layout.addWidget(btn, r + 1, c + 1)

        grid_group.layout().addLayout(grid_layout)
        left.addWidget(grid_group)

        # action buttons
        act_layout = QHBoxLayout()
        act_layout.setSpacing(8)
        self._btn_go    = self._action_btn('▶  Move to Selected', '#1a5c1a', self._go)
        self._btn_clear = self._action_btn('✕  Clear', '#3a3a3a', self._clear_selection)
        self._btn_stop  = self._action_btn('⚠  E-STOP  [SPACE]', '#cc0000', self._stop)
        self._btn_stop.setStyleSheet(
            'QPushButton{background:#cc0000;color:white;border:2px solid #ff4444;'
            'border-radius:6px;padding:12px;font-size:14px;font-weight:bold;}'
            'QPushButton:hover{background:#ee2222;}'
            'QPushButton:pressed{background:#991111;}'
        )
        self._btn_reset = self._action_btn('↺  Reset (Home)', '#4a3a00', self._reset)
        act_layout.addWidget(self._btn_go)
        act_layout.addWidget(self._btn_clear)
        act_layout.addWidget(self._btn_stop)
        act_layout.addWidget(self._btn_reset)
        left.addLayout(act_layout)

        # status
        self._status_label = QLabel('● Idle — select a cell and press Move')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            'background:#111; color:#aaa; font-size:13px; font-weight:bold;'
            'padding:8px; border:1px solid #333; border-radius:4px;'
        )
        left.addWidget(self._status_label)

        # log
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(90)
        self._log_widget.setStyleSheet(
            'background:#0a0a0a; color:#00ee00;'
            'font-family:monospace; font-size:11px;'
        )
        left.addWidget(self._log_widget)
        left.addStretch()

        root.addLayout(left, stretch=3)

        # ── RIGHT COLUMN — camera ─────────────────────────────────────────────
        cam_group = self._group('Camera — Live Feed')
        cam_outer = cam_group.layout()
        cam_outer.setContentsMargins(8, 8, 8, 8)
        cam_outer.setSpacing(6)

        cam_header = QHBoxLayout()

        cam_src_label = QLabel('Source:')
        cam_src_label.setStyleSheet('color:#ccc; font-size:12px;')
        cam_header.addWidget(cam_src_label)

        self._cam_source_combo = QComboBox()
        self._cam_source_combo.addItems(['Webcam', 'Fisheye (USB)', 'RealSense D435i', 'DJI Osmo Pocket 3'])
        self._cam_source_combo.setFixedWidth(150)
        self._cam_source_combo.setStyleSheet(
            'background:#2a2a3a; color:white; font-size:12px; padding:2px;'
            'border:1px solid #555; border-radius:4px;'
        )
        cam_header.addWidget(self._cam_source_combo)
        cam_header.addSpacing(8)

        self._btn_cam_start = QPushButton('▶  Start')
        self._btn_cam_start.setFixedWidth(80)
        self._btn_cam_start.setStyleSheet(self._cam_btn_style('#1a5c1a'))
        self._btn_cam_start.clicked.connect(self._cam_start)
        cam_header.addWidget(self._btn_cam_start)

        self._btn_cam_stop = QPushButton('■  Stop')
        self._btn_cam_stop.setFixedWidth(80)
        self._btn_cam_stop.setEnabled(False)
        self._btn_cam_stop.setStyleSheet(self._cam_btn_style('#5a1a1a'))
        self._btn_cam_stop.clicked.connect(self._cam_stop)
        cam_header.addWidget(self._btn_cam_stop)
        cam_header.addStretch()
        cam_outer.addLayout(cam_header)

        self._cam = CameraWidget()
        cam_outer.addWidget(self._cam)

        root.addWidget(cam_group, stretch=2)

        # spacebar E-STOP shortcut
        shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        shortcut.activated.connect(self._stop)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _group(title):
        g = QGroupBox(title)
        g.setStyleSheet(
            'QGroupBox{color:white;font-weight:bold;'
            'border:1px solid #444;border-radius:4px;margin-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}'
        )
        g.setLayout(QVBoxLayout())
        return g

    @staticmethod
    def _cell_style_normal():
        return (
            'QPushButton{background:#2a2a3a;color:#ccc;border:1px solid #555;'
            'border-radius:3px;font-size:10px;font-weight:bold;}'
            'QPushButton:checked{background:#3a7aff;color:white;border:1px solid #88aaff;}'
            'QPushButton:hover{background:#3a3a5a;}'
        )

    @staticmethod
    def _action_btn(text, colour, slot):
        b = QPushButton(text)
        b.setStyleSheet(
            f'QPushButton{{background:{colour};color:white;border-radius:6px;'
            f'padding:12px;font-size:13px;font-weight:bold;}}'
            f'QPushButton:hover{{background:{colour}cc;}}'
            f'QPushButton:pressed{{background:{colour}88;}}'
        )
        b.clicked.connect(slot)
        return b

    # ── slots ─────────────────────────────────────────────────────────────────

    def _cell_sort_key(self, cell: str):
        """Sort key: left-to-right (A→N), then top-to-bottom (1→4)."""
        return (GRID_COLS.index(cell[0]), GRID_ROWS.index(cell[1]))

    def _select_cell(self, cell: str):
        if cell in self._selected_cells:
            self._selected_cells.remove(cell)
            self._cell_btns[cell].setChecked(False)
        else:
            self._selected_cells.append(cell)
            self._cell_btns[cell].setChecked(True)
        self._selected_cells.sort(key=self._cell_sort_key)
        count = len(self._selected_cells)
        if count == 0:
            self._set_status('No cells selected', '#aaaaaa')
        else:
            seq = ' → '.join(self._selected_cells)
            self._set_status(f'{count} cell(s): {seq}', '#aaaaaa')

    def _clear_selection(self):
        for cell in list(self._selected_cells):
            self._cell_btns[cell].setChecked(False)
        self._selected_cells.clear()
        self._set_status('Selection cleared', '#aaaaaa')

    def _go(self):
        if not self._selected_cells:
            self._set_status('Toggle at least one cell first', '#e0a000')
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        self._moving = True
        queue = list(self._selected_cells)   # already sorted left-to-right
        self._log(f'Queue: {" → ".join(queue)}')
        self._move_next(queue)

    def _move_next(self, queue: list):
        if not queue or self._mover_node._cancel_flag:
            self._moving = False
            if not queue:
                self._status_sig.emit('All cells reached', '#00cc00')
                self._log_sig.emit('Sequence complete')
            return
        cell = queue[0]
        remaining = queue[1:]
        degrees = cell_to_joints_deg(cell)

        # Dip position: slightly lower the end-effector for cutting motion
        dip_degrees = list(degrees)
        dip_degrees[1] += CUT_DIP_LIFT
        dip_degrees[2] += CUT_DIP_ELBOW

        self._status_sig.emit(f'Moving to {cell}…  ({len(remaining)} remaining)', '#3a7aff')
        self._log(f'Moving to {cell}')

        def _on_arrive():
            self._log_sig.emit(f'Reached {cell} — cutting')
            self._status_sig.emit(f'Cutting at {cell}…', '#e0a000')
            self._mover_node.move_to(
                dip_degrees,
                done_cb=_on_dip,
                fail_cb=_on_fail,
            )

        def _on_dip():
            self._log_sig.emit(f'Cut at {cell} — recovering')
            self._mover_node.move_to(
                degrees,
                done_cb=lambda: self._move_next(remaining),
                fail_cb=_on_fail,
            )

        def _on_fail(msg):
            self._status_sig.emit(f'Failed at {cell}: {msg}', '#ff4444')
            self._log_sig.emit(f'FAIL at {cell}: {msg}')
            setattr(self, '_moving', False)

        self._mover_node.move_to(degrees, done_cb=_on_arrive, fail_cb=_on_fail)

    def _stop(self):
        self._mover_node.cancel()
        self._moving = False
        self._set_status('⚠ EMERGENCY STOP', '#ff4444')
        self._log('EMERGENCY STOP triggered')

    def _reset(self):
        if self._moving:
            self._set_status('Moving — press Stop first', '#e0a000')
            return
        self._moving = True
        self._set_status('Moving to Home…', '#e0a000')
        self._log('Resetting to home position')
        self._mover_node.move_to(
            HOME_DEG,
            done_cb=lambda: (
                self._status_sig.emit('Home position reached', '#00cc00'),
                setattr(self, '_moving', False),
            ),
            fail_cb=lambda msg: (
                self._status_sig.emit(f'Reset failed: {msg}', '#ff4444'),
                setattr(self, '_moving', False),
            ),
        )

    def _update_joint_display(self, joints: dict):
        for jname, lbl in self._joint_labels.items():
            val = joints.get(jname)
            if val is not None:
                lbl.setText(f'{val:+.2f}°')

    def _set_status(self, text: str, colour: str = '#aaaaaa'):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f'background:#111; color:{colour}; font-size:13px; font-weight:bold;'
            f'padding:8px; border:1px solid #333; border-radius:4px;'
        )

    def _log(self, text: str):
        self._log_sig.emit(text)

    # ── camera controls ───────────────────────────────────────────────────────

    @staticmethod
    def _cam_btn_style(colour: str) -> str:
        return (
            f'QPushButton{{background:{colour};color:white;border-radius:5px;'
            f'padding:7px 10px;font-size:12px;font-weight:bold;}}'
            f'QPushButton:hover{{background:{colour}cc;}}'
            f'QPushButton:disabled{{background:#2a2a2a;color:#555;}}'
        )

    def _cam_start(self):
        text = self._cam_source_combo.currentText()
        if 'RealSense' in text:
            source = 'realsense'
        elif 'Fisheye' in text:
            source = 'fisheye'
        elif 'DJI' in text:
            source = 'dji'
        else:
            source = 'webcam'
        self._cam.start(source)
        self._btn_cam_start.setEnabled(False)
        self._btn_cam_stop.setEnabled(True)

    def _cam_stop(self):
        self._cam.stop()
        self._btn_cam_start.setEnabled(True)
        self._btn_cam_stop.setEnabled(False)

    # ── ROS ───────────────────────────────────────────────────────────────────

    def _start_ros(self):
        try:
            rclpy.init(args=None)
            self._js_node = JointStateNode(
                lambda joints: self._joints_sig.emit(joints)
            )
            self._mover_node = MoverNode()
            threading.Thread(
                target=rclpy.spin, args=(self._js_node,), daemon=True,
            ).start()
            self._log('ROS2 nodes started')
        except Exception as e:
            self._log(f'ROS2 init failed: {e}')

    def closeEvent(self, event):
        self._cam.stop()
        try:
            self._js_node.destroy_node()
            self._mover_node.destroy_node()
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
