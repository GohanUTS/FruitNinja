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

from fruitninja.colour_detection import detect_fruits

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit,
    QShortcut, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QKeySequence, QImage, QPixmap


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


# ── Camera worker ─────────────────────────────────────────────────────────────

class CameraWorker(QObject):
    """Grabs frames from a webcam, runs detect_fruits(), emits the result."""
    frame_ready = pyqtSignal(object)   # numpy BGR array

    def __init__(self):
        super().__init__()
        self._running = False
        self._cap     = None
        self._thread  = None

    def start(self, cam_index: int = 0):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, args=(cam_index,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self, cam_index: int):
        self._cap = cv2.VideoCapture(cam_index)
        if not self._cap.isOpened():
            self._running = False
            return
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                annotated, _ = detect_fruits(frame)
                self.frame_ready.emit(annotated)
        self._cap.release()
        self._cap = None


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _joints_sig = pyqtSignal(dict)
    _status_sig = pyqtSignal(str, str)
    _log_sig    = pyqtSignal(str)
    _cam_sig    = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('RealGuiPoints — UR3e Grid Control')
        self.setMinimumSize(1200, 660)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._selected_cells = []
        self._moving = False
        self._cell_btns = {}
        self._camera = CameraWorker()

        self._build_ui()
        self._start_ros()

        self._joints_sig.connect(self._update_joint_display)
        self._status_sig.connect(lambda t, c: self._set_status(t, c))
        self._log_sig.connect(self._log_widget.append)
        self._cam_sig.connect(self._update_camera_frame)
        self._camera.frame_ready.connect(
            lambda f: self._cam_sig.emit(f)
        )

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        outer = QHBoxLayout(root_w)
        outer.setSpacing(10)
        outer.setContentsMargins(12, 12, 12, 12)

        # Left panel — all existing controls
        left_w = QWidget()
        root = QVBoxLayout(left_w)
        root.setSpacing(10)
        root.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left_w, stretch=3)

        # Right panel — camera
        outer.addWidget(self._build_camera_panel(), stretch=2)

        # ── joint state display ───────────────────────────────────────────────
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
        root.addWidget(js_group)

        # ── grid selector ─────────────────────────────────────────────────────
        grid_group = self._group('Grid — Toggle Cells  (A1=far-left  N4=near-right  |  moves left→right)')
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(4, 4, 4, 4)

        # column headers
        for c, col in enumerate(GRID_COLS):
            hdr = QLabel(col)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#888; font-size:10px;')
            grid_layout.addWidget(hdr, 0, c + 1)

        # row headers + cell buttons
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
        root.addWidget(grid_group)

        # ── action buttons ────────────────────────────────────────────────────
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
        root.addLayout(act_layout)

        # ── status ────────────────────────────────────────────────────────────
        self._status_label = QLabel('● Idle — select a cell and press Move')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            'background:#111; color:#aaa; font-size:13px; font-weight:bold;'
            'padding:8px; border:1px solid #333; border-radius:4px;'
        )
        root.addWidget(self._status_label)

        # ── log ───────────────────────────────────────────────────────────────
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(90)
        self._log_widget.setStyleSheet(
            'background:#0a0a0a; color:#00ee00;'
            'font-family:monospace; font-size:11px;'
        )
        root.addWidget(self._log_widget)

        # ── spacebar E-STOP shortcut ──────────────────────────────────────────
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

    # ── Camera panel ─────────────────────────────────────────────────────────

    def _build_camera_panel(self) -> QGroupBox:
        group = self._group('Camera Feed')
        layout = group.layout()

        # Video display
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(400, 300)
        self._cam_label.setStyleSheet(
            'background:#000; border:1px solid #333; border-radius:4px;'
        )
        self._cam_label.setText('Camera off')
        layout.addWidget(self._cam_label)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._cam_combo = QComboBox()
        self._cam_combo.addItem('Webcam',            0)
        self._cam_combo.addItem('RealSense D435i',   1)
        self._cam_combo.addItem('Fisheye',           2)
        self._cam_combo.setFixedWidth(160)
        self._cam_combo.setStyleSheet(
            'QComboBox{background:#2a2a3a;color:#ccc;border:1px solid #555;'
            'border-radius:3px;padding:4px 8px;font-size:12px;}'
            'QComboBox::drop-down{border:none;}'
            'QComboBox QAbstractItemView{background:#2a2a3a;color:#ccc;'
            'selection-background-color:#3a7aff;}'
        )
        ctrl.addWidget(self._cam_combo)

        self._btn_cam = QPushButton('▶  Start Camera')
        self._btn_cam.setStyleSheet(
            'QPushButton{background:#1a3a5c;color:white;border-radius:5px;'
            'padding:7px 12px;font-size:12px;font-weight:bold;}'
            'QPushButton:hover{background:#2a5a8ccc;}'
        )
        self._btn_cam.clicked.connect(self._toggle_camera)
        ctrl.addWidget(self._btn_cam)

        self._cam_status = QLabel('Off')
        self._cam_status.setStyleSheet('color:#666; font-size:11px; padding-left:4px;')
        ctrl.addWidget(self._cam_status)
        ctrl.addStretch()

        layout.addLayout(ctrl)
        return group

    def _toggle_camera(self):
        if self._camera._running:
            self._camera.stop()
            self._btn_cam.setText('▶  Start Camera')
            self._cam_status.setText('Off')
            self._cam_status.setStyleSheet('color:#666; font-size:11px; padding-left:4px;')
            self._cam_label.setPixmap(QPixmap())
            self._cam_label.setText('Camera off')
        else:
            idx  = self._cam_combo.currentData()
            name = self._cam_combo.currentText()
            self._camera.start(idx)
            self._btn_cam.setText('■  Stop Camera')
            self._cam_status.setText(f'Running — {name}')
            self._cam_status.setStyleSheet('color:#00cc88; font-size:11px; padding-left:4px;')

    def _update_camera_frame(self, frame: np.ndarray):
        """Convert BGR numpy frame to QPixmap and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.width(),
            self._cam_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._cam_label.setPixmap(pix)

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
        self._camera.stop()
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
