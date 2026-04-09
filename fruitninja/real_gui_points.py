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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal


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

        self._selected_cell = None
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
        root = QVBoxLayout(root_w)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

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
        grid_group = self._group('Grid — Select Target Cell  (A1=far-left  N4=near-right)')
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

        self._btn_go    = self._action_btn('▶  Move to Cell', '#1a5c1a', self._go)
        self._btn_stop  = self._action_btn('■  Stop',         '#5a1a1a', self._stop)
        self._btn_reset = self._action_btn('↺  Reset (Home)', '#4a3a00', self._reset)

        act_layout.addWidget(self._btn_go)
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

    def _select_cell(self, cell: str):
        if self._selected_cell:
            self._cell_btns[self._selected_cell].setChecked(False)
        self._selected_cell = cell
        self._cell_btns[cell].setChecked(True)
        degs = cell_to_joints_deg(cell)
        labels = ['pan', 'lift', 'elbow', 'w1', 'w2', 'w3']
        info = '  '.join(f'{l}={d:+.1f}°' for l, d in zip(labels, degs))
        self._set_status(f'Selected: {cell}  |  {info}', '#aaaaaa')

    def _go(self):
        if not self._selected_cell:
            self._set_status('Select a cell first', '#e0a000')
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        self._moving = True
        cell = self._selected_cell
        degrees = cell_to_joints_deg(cell)
        self._set_status(f'Moving to {cell}…', '#3a7aff')
        self._log(f'Moving to cell {cell}')
        self._mover_node.move_to(
            degrees,
            done_cb=lambda: (
                self._status_sig.emit(f'Reached {cell}', '#00cc00'),
                self._log_sig.emit(f'Reached {cell}'),
                setattr(self, '_moving', False),
            ),
            fail_cb=lambda msg: (
                self._status_sig.emit(f'Failed: {msg}', '#ff4444'),
                self._log_sig.emit(f'FAIL: {msg}'),
                setattr(self, '_moving', False),
            ),
        )

    def _stop(self):
        self._mover_node.cancel()
        self._moving = False
        self._set_status('Stopped', '#aaaaaa')
        self._log('Stop requested')

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
