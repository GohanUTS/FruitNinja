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
from moveit_msgs.msg import (
    Constraints, JointConstraint, MoveItErrorCodes,
    PlanningScene, CollisionObject,
)
from moveit_msgs.srv import GetPositionFK
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from fruitninja.grid_mover import cell_to_joints_deg, GRID_COLS, GRID_ROWS

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    _HAS_RS = False


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


# ── Fruit collision publisher ─────────────────────────────────────────────────

# Radius of each fruit collision sphere (m). Kept small so adjacent cells can
# still be reached; large enough to stop the dip from driving into the fruit.
FRUIT_RADIUS_M = 0.02
# Lift sphere centre this high above the cell's tool0 position so the normal
# approach pose does not collide with the obstacle.
FRUIT_Z_OFFSET_M = 0.08


class FruitSceneNode(Node):
    """
    Publishes collision spheres for detected fruits onto MoveIt's /planning_scene.
    Uses /compute_fk to resolve each cell's joint angles → Cartesian position.
    """

    def __init__(self):
        super().__init__('real_gui_fruit_scene')
        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._fk_cli    = self.create_client(GetPositionFK, '/compute_fk')
        self._active_ids = []   # currently-published fruit object ids

    def publish_for_cells(self, cells: list, done_cb=None, fail_cb=None):
        """Clear previous fruit objects, then publish one sphere per cell. Async."""
        threading.Thread(
            target=self._publish_thread,
            args=(cells, done_cb, fail_cb),
            daemon=True,
        ).start()

    def clear(self):
        threading.Thread(target=self._clear_thread, daemon=True).start()

    def _clear_thread(self):
        if not self._active_ids:
            return
        scene = PlanningScene()
        scene.is_diff = True
        for obj_id in self._active_ids:
            obj = CollisionObject()
            obj.id = obj_id
            obj.header.frame_id = 'world'
            obj.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(obj)
        self._scene_pub.publish(scene)
        self._active_ids = []

    def _publish_thread(self, cells, done_cb, fail_cb):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        try:
            # Drop old fruits first
            self._clear_thread()

            if not self._fk_cli.wait_for_service(timeout_sec=2.0):
                if fail_cb: fail_cb('compute_fk service not available')
                return

            scene = PlanningScene()
            scene.is_diff = True
            new_ids = []

            for cell in cells:
                degrees = cell_to_joints_deg(cell)
                req = GetPositionFK.Request()
                req.header.frame_id = 'world'
                req.fk_link_names = ['tool0']
                req.robot_state.joint_state.name     = JOINT_NAMES
                req.robot_state.joint_state.position = [
                    math.radians(d) for d in degrees
                ]
                future = self._fk_cli.call_async(req)
                executor.spin_until_future_complete(future, timeout_sec=2.0)
                resp = future.result()
                if resp is None or not resp.pose_stamped:
                    continue
                if resp.error_code.val != MoveItErrorCodes.SUCCESS:
                    continue

                ee_pose = resp.pose_stamped[0].pose
                obj = CollisionObject()
                obj.id = f'fruit_{cell}'
                obj.header.frame_id = 'world'
                sphere = SolidPrimitive()
                sphere.type = SolidPrimitive.SPHERE
                sphere.dimensions = [FRUIT_RADIUS_M]
                obj.primitives.append(sphere)

                pose = Pose()
                pose.position.x = ee_pose.position.x
                pose.position.y = ee_pose.position.y
                # Lift sphere centre slightly above cut-target so MoveIt plans
                # above it; simulates a fruit sitting on the trolley surface.
                pose.position.z = ee_pose.position.z + FRUIT_Z_OFFSET_M
                pose.orientation.w = 1.0
                obj.primitive_poses.append(pose)
                obj.operation = CollisionObject.ADD

                scene.world.collision_objects.append(obj)
                new_ids.append(obj.id)

            if scene.world.collision_objects:
                self._scene_pub.publish(scene)
                self._active_ids = new_ids
                if done_cb: done_cb(new_ids)
            else:
                if fail_cb: fail_cb('No fruit positions resolved via FK')
        finally:
            executor.remove_node(self)


# ── Camera worker ─────────────────────────────────────────────────────────────

class CameraWorker(QObject):
    """Grabs frames from a webcam or RealSense device and emits raw frames."""
    frame_ready = pyqtSignal(object)   # numpy BGR array
    stopped     = pyqtSignal(str)      # reason (empty if user-requested)

    def __init__(self):
        super().__init__()
        self._running = False
        self._cap     = None
        self._pipe    = None
        self._thread  = None

    def start(self, source: int):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, args=(source,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self, source: int):
        # source: 0=Webcam, 1=RealSense, 2=Fisheye, 3=DJI Osmo
        try:
            if source == 1:
                self._run_realsense()
            else:
                self._run_v4l2(source)
        finally:
            self._running = False

    def _run_realsense(self):
        if not _HAS_RS:
            self.stopped.emit('pyrealsense2 not installed — pip install pyrealsense2')
            return
        try:
            self._pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipe.start(cfg)
        except Exception as e:
            self._pipe = None
            self.stopped.emit(f'RealSense start failed: {e}')
            return
        try:
            while self._running:
                frames = self._pipe.wait_for_frames(1000)
                cf = frames.get_color_frame()
                if cf:
                    self.frame_ready.emit(np.asanyarray(cf.get_data()))
        except Exception as e:
            self.stopped.emit(f'RealSense error: {e}')
        finally:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
            self.stopped.emit('')

    def _run_v4l2(self, source: int):
        # Webcam=0, Fisheye=2 (try /dev/video2), DJI=3 (try /dev/video3)
        dev_index = {0: 0, 2: 2, 3: 3}.get(source, 0)
        self._cap = cv2.VideoCapture(dev_index)
        if not self._cap.isOpened():
            self._cap = None
            self.stopped.emit(f'Could not open /dev/video{dev_index}')
            return
        try:
            while self._running:
                ret, frame = self._cap.read()
                if ret:
                    self.frame_ready.emit(frame)
        finally:
            self._cap.release()
            self._cap = None
            self.stopped.emit('')


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

        # Manual grid state
        self._grid_pts       = []    # up to 4 (x, y) in frame coords
        self._selecting_grid = False
        self._last_frame_wh  = None  # (w, h) of last received frame

        # Latest fruit detections: list of {'label', 'origin', 'covered': [cells]}
        self._detected_fruits = []

        self._build_ui()
        self._start_ros()

        self._joints_sig.connect(self._update_joint_display)
        self._status_sig.connect(lambda t, c: self._set_status(t, c))
        self._log_sig.connect(self._log_widget.append)
        self._cam_sig.connect(self._update_camera_frame)
        self._camera.frame_ready.connect(
            lambda f: self._cam_sig.emit(f)
        )
        self._camera.stopped.connect(self._on_camera_stopped)
        self._cam_ui_on = False

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

    def _cut_detected_fruit(self):
        """Send detected fruit cells through the cut pipeline with collision spheres."""
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        if not self._detected_fruits:
            self._set_status('No fruit detected — aim camera at fruit first', '#e0a000')
            return
        if len(self._grid_pts) != 4:
            self._set_status('Define the grid before cutting detected fruit', '#e0a000')
            return

        # Queue: origin first, then remaining covered cells (no duplicates)
        queue = []
        seen  = set()
        for fruit in self._detected_fruits:
            ordered = [fruit['origin']] + [c for c in fruit['covered']
                                           if c != fruit['origin']]
            for c in ordered:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)

        if not queue:
            self._set_status('No valid cells from detection', '#e0a000')
            return

        self._moving = True
        self._log(f'Cut (detected): {" → ".join(queue)}')

        # Obstacle cells = any other detected fruit cells NOT in the current
        # cut queue. Only these get collision spheres — publishing them at the
        # target cells would make MoveIt refuse to plan there.
        queue_set = set(queue)
        obstacle_cells = []
        for fruit in self._detected_fruits:
            for c in fruit['covered']:
                if c not in queue_set and c not in obstacle_cells:
                    obstacle_cells.append(c)

        # Always clear any stale fruit spheres from a previous run first.
        self._fruit_scene_node.clear()

        def _start_move():
            self._status_sig.emit(
                f'Cutting {len(queue)} detected cell(s)…', '#3a7aff'
            )
            self._move_next_shallow(queue)

        if not obstacle_cells:
            self._log('No obstacle fruits outside queue — planning against static scene')
            _start_move()
            return

        def _on_scene_ready(ids):
            self._log_sig.emit(f'Obstacle collisions added: {", ".join(ids)}')
            _start_move()

        def _on_scene_fail(msg):
            self._log_sig.emit(f'Collision publish failed: {msg} — moving anyway')
            _start_move()

        self._fruit_scene_node.publish_for_cells(
            obstacle_cells, done_cb=_on_scene_ready, fail_cb=_on_scene_fail
        )

    def _move_next_shallow(self, queue: list):
        """Same as _move_next but with a smaller dip so we stay above fruits."""
        if not queue or self._mover_node._cancel_flag:
            self._moving = False
            try:
                self._fruit_scene_node.clear()
            except Exception:
                pass
            if not queue:
                self._status_sig.emit('All cells reached', '#00cc00')
                self._log_sig.emit('Detected-fruit sequence complete')
            return

        cell = queue[0]
        remaining = queue[1:]
        degrees = cell_to_joints_deg(cell)

        # Half the normal dip so end-effector approaches the fruit but does not
        # drive into it. MoveIt still has the collision sphere as a hard stop.
        dip_degrees = list(degrees)
        dip_degrees[1] += CUT_DIP_LIFT  * 0.5
        dip_degrees[2] += CUT_DIP_ELBOW * 0.5

        self._status_sig.emit(
            f'Approaching {cell}…  ({len(remaining)} remaining)', '#3a7aff'
        )
        self._log(f'Approaching {cell}')

        def _on_arrive():
            self._log_sig.emit(f'At {cell} — simulated cut (shallow)')
            self._status_sig.emit(f'Cutting at {cell}…', '#e0a000')
            self._mover_node.move_to(
                dip_degrees, done_cb=_on_dip, fail_cb=_on_fail,
            )

        def _on_dip():
            self._log_sig.emit(f'Cut at {cell} — recovering')
            self._mover_node.move_to(
                degrees,
                done_cb=lambda: self._move_next_shallow(remaining),
                fail_cb=_on_fail,
            )

        def _on_fail(msg):
            self._status_sig.emit(f'Failed at {cell}: {msg}', '#ff4444')
            self._log_sig.emit(f'FAIL at {cell}: {msg}')
            self._moving = False
            try:
                self._fruit_scene_node.clear()
            except Exception:
                pass

        self._mover_node.move_to(degrees, done_cb=_on_arrive, fail_cb=_on_fail)

    def _stop(self):
        self._mover_node.cancel()
        self._moving = False
        try:
            self._fruit_scene_node.clear()
        except Exception:
            pass
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

        # ── Controls row (camera select + start) ──────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._cam_combo = QComboBox()
        self._cam_combo.addItem('Webcam',            0)
        self._cam_combo.addItem('RealSense D435i',   1)
        self._cam_combo.addItem('Fisheye',           2)
        self._cam_combo.addItem('Osmo DJI Pocket 3', 3)
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

        # ── Grid selection row ────────────────────────────────────────────────
        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)

        self._btn_define_grid = QPushButton('✛  Define Grid')
        self._btn_define_grid.setStyleSheet(
            'QPushButton{background:#2a3a2a;color:#88cc88;border:1px solid #446644;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#3a4a3acc;}'
            'QPushButton:checked{background:#1a5c1a;color:white;border-color:#00cc44;}'
        )
        self._btn_define_grid.setCheckable(True)
        self._btn_define_grid.clicked.connect(self._toggle_grid_select)
        grid_row.addWidget(self._btn_define_grid)

        btn_clear_grid = QPushButton('✕  Clear Grid')
        btn_clear_grid.setStyleSheet(
            'QPushButton{background:#2a2a2a;color:#888;border:1px solid #444;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#3a2a2acc;}'
        )
        btn_clear_grid.clicked.connect(self._clear_grid)
        grid_row.addWidget(btn_clear_grid)

        self._btn_cut_detected = QPushButton('✂  Cut Detected Fruit')
        self._btn_cut_detected.setStyleSheet(
            'QPushButton{background:#3a1a5c;color:#d0b0ff;border:1px solid #6a44aa;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#4a2a7ccc;}'
            'QPushButton:disabled{background:#2a2a2a;color:#555;border-color:#333;}'
        )
        self._btn_cut_detected.clicked.connect(self._cut_detected_fruit)
        grid_row.addWidget(self._btn_cut_detected)

        self._grid_status = QLabel('No grid defined')
        self._grid_status.setStyleSheet('color:#666; font-size:11px; padding-left:6px;')
        grid_row.addWidget(self._grid_status)
        grid_row.addStretch()

        layout.addLayout(grid_row)

        # ── Video display ─────────────────────────────────────────────────────
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(400, 300)
        self._cam_label.setStyleSheet(
            'background:#000; border:1px solid #333; border-radius:4px;'
        )
        self._cam_label.setText('Camera off')
        self._cam_label.mousePressEvent = self._cam_label_clicked
        layout.addWidget(self._cam_label)

        # ── Detection info panel ──────────────────────────────────────────────
        self._detect_info = QTextEdit()
        self._detect_info.setReadOnly(True)
        self._detect_info.setMaximumHeight(110)
        self._detect_info.setPlaceholderText('Detections will appear here…')
        self._detect_info.setStyleSheet(
            'QTextEdit{background:#1a1a22;color:#ddd;border:1px solid #333;'
            'border-radius:4px;padding:6px;font-family:monospace;font-size:11px;}'
        )
        layout.addWidget(self._detect_info)

        return group

    def _toggle_camera(self):
        if self._cam_ui_on:
            self._camera.stop()
            self._set_cam_ui_off('Off', '#666')
        else:
            idx  = self._cam_combo.currentData()
            name = self._cam_combo.currentText()
            self._cam_ui_on = True
            self._btn_cam.setText('■  Stop Camera')
            self._cam_status.setText(f'Running — {name}')
            self._cam_status.setStyleSheet('color:#00cc88; font-size:11px; padding-left:4px;')
            self._camera.start(idx)

    def _set_cam_ui_off(self, status: str, colour: str):
        self._cam_ui_on = False
        self._btn_cam.setText('▶  Start Camera')
        self._cam_status.setText(status)
        self._cam_status.setStyleSheet(f'color:{colour}; font-size:11px; padding-left:4px;')
        self._cam_label.setPixmap(QPixmap())
        self._cam_label.setText('Camera off')

    def _on_camera_stopped(self, reason: str):
        if reason:
            self._set_cam_ui_off(reason, '#ff6666')
            self._log(f'Camera: {reason}')
        else:
            self._set_cam_ui_off('Off', '#666')

    def _update_camera_frame(self, frame: np.ndarray):
        """Convert BGR numpy frame to QPixmap and display it."""
        h, w = frame.shape[:2]
        self._last_frame_wh = (w, h)

        # Only run detection once the grid is fully defined
        if len(self._grid_pts) == 4:
            self._detect_in_grid(frame)

        # Draw grid overlay / in-progress dots on top
        self._draw_grid_overlay(frame)

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

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_in_grid(self, frame: np.ndarray):
        """Detect red objects inside the manual grid, report origin + covered cells."""
        tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in self._grid_pts]
        n_cols, n_rows = 14, 4

        src = np.float32([tl, tr, br, bl])
        dst = np.float32([[0, 0], [n_cols, 0], [n_cols, n_rows], [0, n_rows]])
        H = cv2.getPerspectiveTransform(src, dst)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        profiles = [
            ('Red',  (0, 80, 255), [
                (np.array([0,   100, 80]),   np.array([10,  255, 255])),
                (np.array([170, 100, 80]),   np.array([180, 255, 255])),
            ]),
        ]

        def cell_name(c, r):
            return f"{chr(ord('A') + c)}{r + 1}"

        info_lines = []
        fruits = []

        for label, bgr, ranges in profiles:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 12 or h < 12:
                    continue
                cx, cy = x + w // 2, y + h // 2

                # Origin cell from centroid
                uv = cv2.perspectiveTransform(
                    np.float32([[[cx, cy]]]), H
                )[0][0]
                col_i, row_i = int(uv[0]), int(uv[1])
                if not (0 <= col_i < n_cols and 0 <= row_i < n_rows):
                    continue
                origin_cell = cell_name(col_i, row_i)

                # All cells covered by bounding box (transform 4 corners)
                corners = np.float32([[
                    [x,     y],
                    [x + w, y],
                    [x + w, y + h],
                    [x,     y + h],
                ]])
                uvs = cv2.perspectiveTransform(corners, H)[0]
                cmin = max(0, int(np.floor(min(p[0] for p in uvs))))
                cmax = min(n_cols - 1, int(np.floor(max(p[0] for p in uvs))))
                rmin = max(0, int(np.floor(min(p[1] for p in uvs))))
                rmax = min(n_rows - 1, int(np.floor(max(p[1] for p in uvs))))
                covered = [cell_name(c, r)
                           for r in range(rmin, rmax + 1)
                           for c in range(cmin, cmax + 1)]

                # Draw bbox, origin marker, label
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                if len(covered) > 1:
                    tag = f'{label} origin:{origin_cell} spans:{",".join(covered)}'
                else:
                    tag = f'{label} [{origin_cell}]'
                cv2.putText(frame, tag, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                cv2.drawMarker(frame, (cx, cy), bgr,
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                info_lines.append(
                    f'{label}  origin: {origin_cell}  '
                    f'cells ({len(covered)}): {", ".join(covered)}'
                )
                fruits.append({'label': label,
                                'origin': origin_cell,
                                'covered': covered})

        self._detected_fruits = fruits
        if hasattr(self, '_detect_info'):
            if info_lines:
                self._detect_info.setPlainText('\n'.join(info_lines))
            else:
                self._detect_info.setPlainText('No fruit detected in grid')

    # ── Manual grid ──────────────────────────────────────────────────────────

    def _toggle_grid_select(self, checked: bool):
        self._selecting_grid = checked
        if checked:
            self._grid_pts = []
            self._grid_status.setText('Click point 1 of 4 on the camera feed')
            self._grid_status.setStyleSheet('color:#ffcc00; font-size:11px; padding-left:6px;')
            self._cam_label.setCursor(Qt.CrossCursor)
        else:
            self._cam_label.setCursor(Qt.ArrowCursor)
            if len(self._grid_pts) < 4:
                self._grid_status.setText('Selection cancelled')
                self._grid_status.setStyleSheet('color:#888; font-size:11px; padding-left:6px;')

    def _clear_grid(self):
        self._grid_pts = []
        self._selecting_grid = False
        self._btn_define_grid.setChecked(False)
        self._cam_label.setCursor(Qt.ArrowCursor)
        self._grid_status.setText('No grid defined')
        self._grid_status.setStyleSheet('color:#666; font-size:11px; padding-left:6px;')

    def _cam_label_clicked(self, event):
        if not self._selecting_grid:
            return
        if len(self._grid_pts) >= 4:
            return
        pt = self._label_to_frame(event.pos())
        if pt is None:
            return
        self._grid_pts.append(pt)
        n = len(self._grid_pts)
        if n < 4:
            self._grid_status.setText(f'Click point {n + 1} of 4')
        else:
            # Sort into TL, TR, BR, BL and lock the grid
            self._grid_pts = list(self._sort_corners(self._grid_pts))
            self._selecting_grid = False
            self._btn_define_grid.setChecked(False)
            self._cam_label.setCursor(Qt.ArrowCursor)
            self._grid_status.setText('Grid locked — 4 points set')
            self._grid_status.setStyleSheet('color:#00cc88; font-size:11px; padding-left:6px;')

    def _label_to_frame(self, pos):
        """Map a QLabel pixel position to frame pixel coordinates."""
        if self._last_frame_wh is None:
            return None
        fw, fh = self._last_frame_wh
        lw, lh = self._cam_label.width(), self._cam_label.height()
        scale  = min(lw / fw, lh / fh)
        ox     = (lw - fw * scale) / 2
        oy     = (lh - fh * scale) / 2
        fx     = (pos.x() - ox) / scale
        fy     = (pos.y() - oy) / scale
        if 0 <= fx < fw and 0 <= fy < fh:
            return (int(fx), int(fy))
        return None

    @staticmethod
    def _sort_corners(pts):
        """Sort 4 points into (TL, TR, BR, BL) order."""
        pts = np.array(pts, dtype=np.float32)
        s   = pts.sum(axis=1)
        d   = np.diff(pts, axis=1).ravel()
        tl  = pts[np.argmin(s)]
        br  = pts[np.argmax(s)]
        tr  = pts[np.argmin(d)]
        bl  = pts[np.argmax(d)]
        return [tuple(map(int, p)) for p in (tl, tr, br, bl)]

    def _draw_grid_overlay(self, frame):
        """Draw in-progress dots or the full static grid onto the frame."""
        pts = self._grid_pts

        # Draw dots for points selected so far
        colours = [(0, 200, 255), (0, 255, 150), (255, 200, 0), (255, 80, 80)]
        labels  = ['TL', 'TR', 'BR', 'BL']
        for i, pt in enumerate(pts):
            col = colours[i]
            cv2.circle(frame, pt, 8,  col,         -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 10, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, labels[i], (pt[0] + 12, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, labels[i], (pt[0] + 12, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        if len(pts) != 4:
            return

        tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in pts]
        n_cols, n_rows = 14, 4
        steps = 60

        def lerp(a, b, t):
            return a + (b - a) * t

        def ipt(p):
            return (int(round(p[0])), int(round(p[1])))

        grid_col = (255, 100, 0)  # blue (BGR)

        # Row lines
        for i in range(n_rows + 1):
            t     = i / n_rows
            left  = lerp(tl, bl, t)
            right = lerp(tr, br, t)
            prev  = ipt(lerp(left, right, 0))
            for j in range(1, steps + 1):
                cur = ipt(lerp(left, right, j / steps))
                cv2.line(frame, prev, cur, grid_col, 1, cv2.LINE_AA)
                prev = cur

        # Column lines
        for i in range(n_cols + 1):
            t   = i / n_cols
            top = lerp(tl, tr, t)
            bot = lerp(bl, br, t)
            prev = ipt(lerp(top, bot, 0))
            for j in range(1, steps + 1):
                cur = ipt(lerp(top, bot, j / steps))
                cv2.line(frame, prev, cur, grid_col, 1, cv2.LINE_AA)
                prev = cur

        # Cell labels
        for row in range(n_rows):
            for col in range(n_cols):
                u   = (col + 0.5) / n_cols
                v   = (row + 0.5) / n_rows
                top = lerp(tl, tr, u)
                bot = lerp(bl, br, u)
                ctr = ipt(lerp(top, bot, v))
                lbl = f"{chr(ord('A') + col)}{row + 1}"
                cv2.putText(frame, lbl, (ctr[0] - 8, ctr[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2)
                cv2.putText(frame, lbl, (ctr[0] - 8, ctr[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

        # Border quad
        quad = np.array([ipt(tl), ipt(tr), ipt(br), ipt(bl)], np.int32)
        cv2.polylines(frame, [quad], isClosed=True,
                      color=grid_col, thickness=2, lineType=cv2.LINE_AA)

    # ── ROS ───────────────────────────────────────────────────────────────────

    def _start_ros(self):
        try:
            rclpy.init(args=None)
            self._js_node = JointStateNode(
                lambda joints: self._joints_sig.emit(joints)
            )
            self._mover_node = MoverNode()
            self._fruit_scene_node = FruitSceneNode()
            threading.Thread(
                target=rclpy.spin, args=(self._js_node,), daemon=True,
            ).start()
            self._log('ROS2 nodes started')
        except Exception as e:
            self._log(f'ROS2 init failed: {e}')

    def closeEvent(self, event):
        self._camera.stop()
        try:
            self._fruit_scene_node.clear()
        except Exception:
            pass
        try:
            self._js_node.destroy_node()
            self._mover_node.destroy_node()
            self._fruit_scene_node.destroy_node()
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
