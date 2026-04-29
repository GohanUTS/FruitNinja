#!/usr/bin/env python3
"""
real_gui_points.py — Main operator GUI for the FruitNinja UR3e cutting robot.

HOW IT WORKS (overview)
========================
1. The GUI connects to a live UR3e robot via ROS 2 / MoveIt.
2. The operator selects grid cells (A1–N4) on the cutting board grid.
3. Pressing "Move to Selected" drives the robot through each cell in order,
   performing a three-step cut at each one: approach → dip → recover.
4. A camera feed (top-right) can run colour detection once the operator has
   manually defined the grid by clicking 4 corner points on the image.
   Only red and blue objects inside that grid are detected.
5. Detected fruits can be cut automatically using the "Cut Detected" button,
   which also publishes MoveIt collision spheres for any OTHER fruits so the
   planner routes around them.
6. The spacebar / E-STOP button cancels all motion immediately.

KEY CLASSES
===========
  JointStateNode   – Subscribes to /joint_states and feeds live angles to the UI.
  MoverNode        – Sends MoveGroup action goals to MoveIt (runs in background thread).
  CameraWorker     – Captures raw frames from a webcam or RealSense in a background thread.
  MainWindow       – PyQt5 main window that wires everything together.
"""
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
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, JointConstraint, MoveItErrorCodes,
    OrientationConstraint, PositionConstraint, BoundingVolume,
)
from moveit_msgs.srv import GetCartesianPath
from geometry_msgs.msg import Pose as _GPose, Quaternion as _GQuat
from shape_msgs.msg import SolidPrimitive as _SolidPrimitive

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
from PyQt5.QtGui import (
    QKeySequence, QImage, QPixmap, QPainter, QPen, QBrush,
    QColor, QFont, QLinearGradient, QRadialGradient,
)


# ── Constants ──────────────────────────────────────────────────────────────────
# JOINT_NAMES / JOINT_LABELS: the 6 UR3e joints in the order MoveIt expects them.
# HOME_DEG: the safe "rest" pose used by the Reset button.
# CUT_DIP_LIFT / CUT_DIP_ELBOW: how many degrees to shift the shoulder and elbow
#   joints to push the end-effector downward for the cutting stroke.
# MOVE_GROUP: the MoveIt planning group name for the UR arm.

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

JOINT_LABELS = ['Base (pan)', 'Shoulder (lift)', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

HOME_DEG = [0.0, -90.0, 0.0, 0.0, 0.0, 360.0]

# Cutting motion: deeper safety-limited dip after reaching the board cell.
CUT_DIP_LIFT  =  40.0   # degrees added to shoulder_lift_joint (index 1)
CUT_DIP_ELBOW = -40.0   # degrees added to elbow_joint (index 2)

MOVE_GROUP = 'ur_manipulator'

# ── Cartesian / IK constants ───────────────────────────────────────────────────
# Board corners in base_link frame (measured on physical robot)
_BOARD_A1 = (+0.3124, +0.4549, 0.0875)   # col 0,  row 0
_BOARD_A4 = (+0.3130, +0.3493, 0.0875)   # col 0,  row 3
_BOARD_N1 = (-0.3124, +0.4549, 0.0875)   # col 13, row 0
_BOARD_N4 = (-0.3130, +0.3493, 0.0875)   # col 13, row 3

IK_APPROACH_CLEARANCE = 0.10   # metres above board surface for approach
IK_DIP_DEPTH          = 0.010  # metres below board surface for cut stroke
IK_TILT_TOLERANCE_DEG = 5.0    # ±degrees allowed pitch/roll from straight-down
# "Straight down" in base_link: 90° rotation around Y axis (tool0 Z → base_link -Z)
IK_DOWN_QUAT = _GQuat(x=0.0, y=0.7071068, z=0.0, w=0.7071068)


def _board_position(col_idx: int, row_idx: int) -> tuple:
    """Bilinear interpolation of the 4 board corners → (x, y, z) in base_link."""
    u = col_idx / 13.0
    v = row_idx / 3.0
    x = (1-u)*(1-v)*_BOARD_A1[0] + u*(1-v)*_BOARD_N1[0] + (1-u)*v*_BOARD_A4[0] + u*v*_BOARD_N4[0]
    y = (1-u)*(1-v)*_BOARD_A1[1] + u*(1-v)*_BOARD_N1[1] + (1-u)*v*_BOARD_A4[1] + u*v*_BOARD_N4[1]
    z = (1-u)*(1-v)*_BOARD_A1[2] + u*(1-v)*_BOARD_N1[2] + (1-u)*v*_BOARD_A4[2] + u*v*_BOARD_N4[2]
    return (x, y, z)


def _wrap_deg(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# ── ROS2 nodes ─────────────────────────────────────────────────────────────────
# Three separate ROS 2 nodes are created so their executors don't block each other.
# They are all started once in MainWindow._start_ros() and run as daemon threads.

class JointStateNode(Node):
    """
    Listens to /joint_states and converts the 6 UR joint positions from
    radians to degrees, then fires joint_callback so the GUI can update
    the live angle display. Runs continuously in a background spin thread.
    """
    def __init__(self, joint_callback):
        super().__init__('real_gui_points_js')
        self._cb = joint_callback
        self.create_subscription(JointState, '/joint_states', self._recv, 10)

    def _recv(self, msg: JointState):
        joints = {n: math.degrees(p) for n, p in zip(msg.name, msg.position)
                  if n in JOINT_NAMES}
        self._cb(joints)


class MoverNode(Node):
    """
    Sends joint-space motion goals to MoveIt via the /move_action action server.

    move_to(degrees, done_cb, fail_cb, current_degrees=None)
      - Spawns a background thread so the GUI never freezes during planning.
      - Converts the target joint angles (degrees) into a MoveIt Constraints message.
      - Velocity and acceleration are capped at 30% for safety.
      - On success  → calls done_cb()
      - On failure  → calls fail_cb(reason_string)
      - cancel()    → sets _cancel_flag so the thread exits at the next wait point.
    """
    def __init__(self):
        super().__init__('real_gui_points_mover')
        self._client        = ActionClient(self, MoveGroup, '/move_action')
        self._exec_client   = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._cartesian_srv = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self._cancel_flag   = False

    def move_to(self, degrees: list, done_cb, fail_cb, current_degrees: list = None):
        self._cancel_flag = False
        target_degrees = self._nearest_equivalent_target(degrees, current_degrees)
        threading.Thread(
            target=self._move_thread,
            args=(target_degrees, done_cb, fail_cb),
            daemon=True,
        ).start()

    def _nearest_equivalent_target(self, degrees: list, current_degrees: list = None) -> list:
        if current_degrees is None:
            return [_wrap_deg(deg) for deg in degrees]
        return [
            current + _wrap_deg(target - current)
            for target, current in zip(degrees, current_degrees)
        ]

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
            goal.request.max_velocity_scaling_factor     = 0.1
            goal.request.max_acceleration_scaling_factor = 0.1

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

    # ── IK / Cartesian motion ──────────────────────────────────────────────────

    def move_to_pose_ik(self, x: float, y: float, z: float, done_cb, fail_cb):
        """Approach (x, y, z) using Cartesian IK: PositionConstraint + orientation cone."""
        self._cancel_flag = False
        threading.Thread(
            target=self._pose_ik_thread,
            args=(x, y, z, done_cb, fail_cb),
            daemon=True,
        ).start()

    def _pose_ik_thread(self, x, y, z, done_cb, fail_cb):
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
            goal.request.goal_constraints.append(self._make_ik_constraints(x, y, z))
            goal.request.path_constraints = self._make_ik_path_constraints()
            goal.request.num_planning_attempts = 20
            goal.request.allowed_planning_time = 10.0
            goal.request.max_velocity_scaling_factor     = 0.1
            goal.request.max_acceleration_scaling_factor = 0.1

            future = self._client.send_goal_async(goal)
            executor.spin_until_future_complete(future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            goal_handle = future.result()
            if not goal_handle.accepted:
                fail_cb('IK goal rejected by MoveIt')
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
                fail_cb(f'IK move failed (code: {result.error_code.val})')
        finally:
            executor.remove_node(self)

    def _make_ik_constraints(self, x: float, y: float, z: float) -> Constraints:
        """Build a Constraints with a 1cm position sphere + 5° downward orientation cone."""
        c = Constraints()

        # Position: 1 cm sphere around target in base_link
        pc = PositionConstraint()
        pc.header.frame_id = 'base_link'
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.link_name = 'tool0'
        bv = BoundingVolume()
        sp = _SolidPrimitive()
        sp.type = _SolidPrimitive.SPHERE
        sp.dimensions = [0.01]
        bv.primitives.append(sp)
        bv_pose = _GPose()
        bv_pose.position.x = x
        bv_pose.position.y = y
        bv_pose.position.z = z
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pc.constraint_region = bv
        pc.weight = 1.0
        c.position_constraints.append(pc)

        # Orientation: tool pointing straight down ± IK_TILT_TOLERANCE_DEG cone
        # Yaw (z-axis) is free (π) so the solver can use redundancy to avoid collisions.
        oc = OrientationConstraint()
        oc.header.frame_id = 'base_link'
        oc.header.stamp = self.get_clock().now().to_msg()
        oc.link_name = 'tool0'
        oc.orientation = IK_DOWN_QUAT
        tilt = math.radians(IK_TILT_TOLERANCE_DEG)
        oc.absolute_x_axis_tolerance = tilt
        oc.absolute_y_axis_tolerance = tilt
        oc.absolute_z_axis_tolerance = math.radians(45)  # limited yaw prevents wrist folding into lower arm
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        return c

    def _make_ik_path_constraints(self) -> Constraints:
        """
        Minimal path constraint: restrict only wrist_1 to the range seen across
        all 4 measured board corners (-94° to -56°) plus safety margin.
        This specifically prevents the C403A0 clamping stop (tool flange < 2.8 cm
        from lower arm) without over-constraining the planner on other joints.
        """
        c = Constraints()
        jc = JointConstraint()
        jc.joint_name      = 'wrist_1_joint'
        jc.position        = math.radians(-75.0)   # centre of measured range
        jc.tolerance_above = math.radians(45.0)    # → −30°
        jc.tolerance_below = math.radians(45.0)    # → −120°
        jc.weight          = 1.0
        c.joint_constraints.append(jc)
        return c

    def move_cartesian_z(self, x: float, y: float, z_target: float, done_cb, fail_cb):
        """Drop or raise the tool straight along Z via compute_cartesian_path."""
        self._cancel_flag = False
        threading.Thread(
            target=self._cartesian_z_thread,
            args=(x, y, z_target, done_cb, fail_cb),
            daemon=True,
        ).start()

    def _cartesian_z_thread(self, x, y, z_target, done_cb, fail_cb):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        try:
            if not self._cartesian_srv.wait_for_service(timeout_sec=5.0):
                fail_cb('compute_cartesian_path service unavailable')
                return
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            req = GetCartesianPath.Request()
            req.header.frame_id = 'base_link'
            req.header.stamp = self.get_clock().now().to_msg()
            req.group_name = MOVE_GROUP
            req.link_name = 'tool0'
            req.avoid_collisions = True
            req.max_step = 0.005       # 5 mm resolution
            req.jump_threshold = 0.0   # disable jump detection for short Z moves

            wp = _GPose()
            wp.position.x = x
            wp.position.y = y
            wp.position.z = z_target
            wp.orientation = IK_DOWN_QUAT
            req.waypoints = [wp]

            future = self._cartesian_srv.call_async(req)
            executor.spin_until_future_complete(future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            resp = future.result()
            if resp.fraction < 0.9:
                fail_cb(f'Cartesian path {resp.fraction * 100:.0f}% complete — collision likely')
                return

            if not self._exec_client.wait_for_server(timeout_sec=5.0):
                fail_cb('execute_trajectory server unavailable')
                return

            exec_goal = ExecuteTrajectory.Goal()
            exec_goal.trajectory = resp.solution

            exec_future = self._exec_client.send_goal_async(exec_goal)
            executor.spin_until_future_complete(exec_future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            exec_handle = exec_future.result()
            if not exec_handle.accepted:
                fail_cb('Trajectory execution rejected')
                return

            result_future = exec_handle.get_result_async()
            executor.spin_until_future_complete(result_future)
            if self._cancel_flag:
                fail_cb('Cancelled')
                return

            result = result_future.result().result
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                done_cb()
            else:
                fail_cb(f'Trajectory execution failed (code: {result.error_code.val})')
        finally:
            executor.remove_node(self)


# ── Camera worker ─────────────────────────────────────────────────────────────
# Runs cv2.VideoCapture in a daemon thread so the GUI never blocks on frame reads.
# Emits raw BGR numpy frames via frame_ready signal.
# Detection (red/blue inside the manual grid) is done in the main thread inside
# MainWindow._update_camera_frame so it has access to the current grid points.

class CameraWorker(QObject):
    """Captures raw frames from a webcam or RealSense and emits them for display."""
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


# ── Animated brand strip ─────────────────────────────────────────────────────

class FruitNinjaBrand(QWidget):
    """Animated 3-D style FruitNinja title plate for the operator console."""

    def __init__(self, subtitle: str):
        super().__init__()
        self._subtitle = subtitle
        self._phase = 0
        self.setMinimumHeight(90)
        self.setMaximumHeight(104)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(45)

    def _tick(self):
        self._phase = (self._phase + 3) % 360
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor('#101a22'))
        bg.setColorAt(0.52, QColor('#172630'))
        bg.setColorAt(1.0, QColor('#090f15'))
        p.setBrush(QBrush(bg))
        p.setPen(QPen(QColor('#334957'), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 8, 8)

        glow_x = int((self._phase / 360.0) * (w + 120)) - 60
        glow = QRadialGradient(glow_x, h // 2, max(90, h))
        glow.setColorAt(0.0, QColor(50, 225, 150, 78))
        glow.setColorAt(1.0, QColor(50, 225, 150, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(glow_x - h, -h // 2, h * 2, h * 2)

        title = 'FRUITNINJA'
        x, y = 22, 50
        p.setFont(QFont('Arial Black', 28, QFont.Black))
        for dx, dy, col in [
            (4, 5, QColor('#061016')),
            (3, 3, QColor('#173541')),
            (2, 2, QColor('#28596a')),
        ]:
            p.setPen(QPen(col))
            p.drawText(x + dx, y + dy, title)

        title_grad = QLinearGradient(x, 20, x + 340, 60)
        title_grad.setColorAt(0.0, QColor('#ff4b4b'))
        title_grad.setColorAt(0.45, QColor('#ffd166'))
        title_grad.setColorAt(1.0, QColor('#38d878'))
        p.setPen(QPen(QBrush(title_grad), 1))
        p.drawText(x, y, title)

        p.setFont(QFont('Arial', 10, QFont.DemiBold))
        p.setPen(QPen(QColor('#a9bbc7')))
        p.drawText(24, 75, self._subtitle)

        p.end()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _joints_sig = pyqtSignal(dict)
    _status_sig = pyqtSignal(str, str)
    _log_sig    = pyqtSignal(str)
    _cam_sig    = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja — UR3e Grid Control')
        self.setMinimumSize(1240, 740)
        self.setStyleSheet('background:#0b1117; color:white;')

        self._selected_cells = []
        self._moving = False
        self._cell_btns = {}
        self._camera = CameraWorker()
        self._current_joints_rad = {n: 0.0 for n in JOINT_NAMES}

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
        shell = QVBoxLayout(root_w)
        shell.setSpacing(10)
        shell.setContentsMargins(12, 12, 12, 12)
        shell.addWidget(FruitNinjaBrand('UR3e operator console  |  Select cells, cut fruit, monitor motion'))

        body_w = QWidget()
        shell.addWidget(body_w, stretch=1)

        outer = QHBoxLayout(body_w)
        outer.setSpacing(10)
        outer.setContentsMargins(0, 0, 0, 0)

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
            lbl.setStyleSheet('color:#a9b8c4; font-size:11px; font-weight:bold;')
            js_layout.addWidget(lbl, 0, col)

        for row, (jname, jlabel) in enumerate(zip(JOINT_NAMES, JOINT_LABELS)):
            name_lbl = QLabel(jlabel)
            name_lbl.setStyleSheet('color:#d1dde4; font-size:12px; padding:3px 8px;')
            js_layout.addWidget(name_lbl, row + 1, 0)

            val_lbl = QLabel('—')
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet(
                'background:#08131a; color:#62e6ff; font-family:monospace;'
                'font-size:12px; border:1px solid #2f5161; border-radius:5px; padding:4px 10px;'
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
            hdr.setStyleSheet('color:#91a5b2; font-size:10px; font-weight:bold;')
            grid_layout.addWidget(hdr, 0, c + 1)

        # row headers + cell buttons
        for r, row in enumerate(GRID_ROWS):
            hdr = QLabel(row)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet('color:#91a5b2; font-size:10px; font-weight:bold;')
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
        self._btn_ik    = self._action_btn('⊕  Inverse Movement', '#1c5f87', self._go_ik)
        self._btn_clear = self._action_btn('✕  Clear', '#3b4650', self._clear_selection)
        self._btn_stop  = self._action_btn('⚠  E-STOP  [SPACE]', '#c82020', self._stop)
        self._btn_stop.setStyleSheet(
            'QPushButton{background:#c82020;color:white;border:2px solid #ff5a5a;'
            'border-radius:6px;padding:12px;font-size:14px;font-weight:bold;}'
            'QPushButton:hover{background:#e62b2b;}'
            'QPushButton:pressed{background:#991111;}'
        )
        self._btn_reset = self._action_btn('↺  Reset (Home)', '#6a5012', self._reset)

        act_layout.addWidget(self._btn_go)
        act_layout.addWidget(self._btn_ik)
        act_layout.addWidget(self._btn_clear)
        act_layout.addWidget(self._btn_stop)
        act_layout.addWidget(self._btn_reset)
        root.addLayout(act_layout)

        # ── status ────────────────────────────────────────────────────────────
        self._status_label = QLabel('● Idle — select a cell and press Move')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            'background:#101a22; color:#a9b8c4; font-size:13px; font-weight:bold;'
            'padding:9px; border:1px solid #2f4654; border-radius:6px;'
        )
        root.addWidget(self._status_label)

        # ── log ───────────────────────────────────────────────────────────────
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(90)
        self._log_widget.setStyleSheet(
            'background:#071016; color:#79f7b2; border:1px solid #1c303b;'
            'border-radius:6px; font-family:monospace; font-size:11px; padding:6px;'
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
            'QGroupBox{color:#f5f9fa;font-weight:bold;'
            'border:1px solid #314652;border-radius:7px;margin-top:10px;'
            'background:#0f1820;}'
            'QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;}'
        )
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        g.setLayout(layout)
        return g

    @staticmethod
    def _cell_style_normal():
        return (
            'QPushButton{background:#14222b;color:#c9d6de;border:1px solid #2f4654;'
            'border-radius:4px;font-size:10px;font-weight:bold;}'
            'QPushButton:checked{background:#2f8bd8;color:white;border:1px solid #91cfff;}'
            'QPushButton:hover{background:#203442;}'
        )

    @staticmethod
    def _action_btn(text, colour, slot):
        b = QPushButton(text)
        b.setStyleSheet(
            f'QPushButton{{background:{colour};color:white;border-radius:6px;'
            f'padding:12px;font-size:13px;font-weight:bold;border:1px solid rgba(255,255,255,40);}}'
            f'QPushButton:hover{{background:{colour};border:1px solid rgba(255,255,255,110);}}'
            f'QPushButton:pressed{{background:{colour};padding-top:13px;padding-bottom:11px;}}'
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
        """
        Start the manual cut sequence for all selected grid cells.
        The cells are already sorted left-to-right (A→N) when toggled,
        so the robot always sweeps in one direction across the board.
        Kicks off _move_next which handles the per-cell approach→dip→recover loop.
        """
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
        """
        Recursive per-cell cut loop for the manual selection sequence.

        For each cell in the queue it chains three MoveIt calls:
          1. move_to(degrees)       → approach: arm arrives at the cell's pre-computed pose
          2. move_to(dip_degrees)   → dip:      shoulder+elbow shift drives end-effector down
          3. move_to(degrees)       → recover:  arm lifts back to approach pose
        After recover, _move_next is called again with remaining = queue[1:].
        When the queue is empty (or E-STOP fires) the loop exits.
        """
        if not queue or self._mover_node._cancel_flag:
            self._moving = False
            if not queue:
                self._status_sig.emit('All cells reached', '#00cc00')
                self._log_sig.emit('Sequence complete')
            return
        cell = queue[0]
        remaining = queue[1:]
        degrees = cell_to_joints_deg(cell)
        current_degrees = self._current_joint_degrees()

        # Dip position: slightly lower the end-effector for cutting motion
        dip_degrees = list(degrees)
        dip_degrees[1] += CUT_DIP_LIFT   # shoulder lifts more → end-effector moves down
        dip_degrees[2] += CUT_DIP_ELBOW  # elbow bends more   → reinforces the dip

        self._status_sig.emit(f'Moving to {cell}…  ({len(remaining)} remaining)', '#3a7aff')
        self._log(f'Moving to {cell}')

        def _on_arrive():
            self._log_sig.emit(f'Reached {cell} — cutting')
            self._status_sig.emit(f'Cutting at {cell}…', '#e0a000')
            self._mover_node.move_to(
                dip_degrees,
                done_cb=_on_dip,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_dip():
            self._log_sig.emit(f'Cut at {cell} — recovering')
            self._mover_node.move_to(
                degrees,
                done_cb=lambda: self._move_next(remaining),
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_fail(msg):
            self._status_sig.emit(f'Failed at {cell}: {msg}', '#ff4444')
            self._log_sig.emit(f'FAIL at {cell}: {msg}')
            setattr(self, '_moving', False)

        self._mover_node.move_to(
            degrees,
            done_cb=_on_arrive,
            fail_cb=_on_fail,
            current_degrees=current_degrees,
        )

    # ── Inverse Movement (Cartesian IK) ───────────────────────────────────────

    def _go_ik(self):
        """
        Start the IK cut sequence for all selected grid cells.

        Uses Cartesian IK instead of joint-space lookup:
          - APPROACH : MoveIt IK with PositionConstraint sphere + 5° orientation cone
          - DIP      : straight-line Z drop via compute_cartesian_path
          - RECOVER  : straight-line Z rise via compute_cartesian_path
        Cells are processed left-to-right (A→N), same order as the standard move.
        """
        if not self._selected_cells:
            self._set_status('Toggle at least one cell first', '#e0a000')
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        self._moving = True
        queue = list(self._selected_cells)
        self._log(f'IK Queue: {" → ".join(queue)}')
        self._move_next_ik(queue)

    def _move_next_ik(self, queue: list):
        """
        Recursive per-cell IK cut loop.

        For each cell:
          1. APPROACH  move_to_pose_ik(x, y, board_z + 0.10)
                         MoveIt plans to a 1cm position sphere pointing straight down.
                         5° pitch/roll tolerance + free yaw lets the solver pick the
                         best joint config to avoid collisions.
          2. DIP       move_cartesian_z(x, y, board_z - 0.01)
                         compute_cartesian_path straight down at 5mm resolution.
          3. RECOVER   move_cartesian_z(x, y, board_z + 0.10)
                         straight back up to approach height.
        Then recurses with queue[1:].
        """
        if not queue or self._mover_node._cancel_flag:
            self._moving = False
            if not queue:
                self._status_sig.emit('IK sequence complete', '#00cc00')
                self._log_sig.emit('IK sequence complete')
            return

        cell = queue[0]
        remaining = queue[1:]
        col_idx = GRID_COLS.index(cell[0])
        row_idx = GRID_ROWS.index(cell[1])
        bx, by, bz = _board_position(col_idx, row_idx)

        approach_z = bz + IK_APPROACH_CLEARANCE
        dip_z      = bz - IK_DIP_DEPTH

        self._status_sig.emit(f'IK → {cell}…  ({len(remaining)} remaining)', '#3a7aff')
        self._log(f'IK moving to {cell}  ({bx:.3f}, {by:.3f}, {bz:.4f})')

        def _on_arrive():
            self._log_sig.emit(f'IK reached {cell} — dipping')
            self._status_sig.emit(f'IK cutting at {cell}…', '#e0a000')
            self._mover_node.move_cartesian_z(bx, by, dip_z, done_cb=_on_dip, fail_cb=_on_fail)

        def _on_dip():
            self._log_sig.emit(f'IK cut at {cell} — recovering')
            self._mover_node.move_cartesian_z(bx, by, approach_z, done_cb=_on_recover, fail_cb=_on_fail)

        def _on_recover():
            self._move_next_ik(remaining)

        def _on_fail(msg):
            self._status_sig.emit(f'IK failed at {cell}: {msg}', '#ff4444')
            self._log_sig.emit(f'IK FAIL at {cell}: {msg}')
            setattr(self, '_moving', False)

        self._mover_node.move_to_pose_ik(bx, by, approach_z, done_cb=_on_arrive, fail_cb=_on_fail)

    def _current_joint_degrees(self) -> list:
        return [math.degrees(self._current_joints_rad[name]) for name in JOINT_NAMES]

    def _cut_detected_fruit(self):
        """
        Automated cut sequence driven by camera detection.
        Builds a queue of all detected fruit cells (origin first, no duplicates)
        and runs the standard approach→dip→recover loop through each one.
        Requires the manual grid to be defined (4 corner points clicked) before running.
        """
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
        self._status_sig.emit(f'Cutting {len(queue)} detected cell(s)…', '#3a7aff')
        self._move_next(queue)

    def _stop(self):
        """
        Emergency stop — triggered by the E-STOP button or the spacebar shortcut.

        Sets the cancel flag on MoverNode (the running move thread checks it and exits),
        clears any MoveIt collision spheres, and resets the moving flag.
        The robot completes its current waypoint before stopping (MoveIt has no
        mid-trajectory abort via this action client), but no further waypoints are sent.
        """
        self._mover_node.cancel()
        self._moving = False
        self._set_status('⚠ EMERGENCY STOP', '#ff4444')
        self._log('EMERGENCY STOP triggered')

    def _reset(self):
        """
        Drive the robot to the HOME_DEG pose ([0, -90, 0, 0, 0, 0] degrees).
        This is a safe upright posture that keeps the arm clear of the cutting board.
        Blocked while a move is in progress — press E-STOP first if needed.
        """
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
            current_degrees=self._current_joint_degrees(),
        )

    def _update_joint_display(self, joints: dict):
        for jname, lbl in self._joint_labels.items():
            val = joints.get(jname)
            if val is not None:
                lbl.setText(f'{val:+.2f}°')
                self._current_joints_rad[jname] = math.radians(val)

    def _set_status(self, text: str, colour: str = '#aaaaaa'):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f'background:#101a22; color:{colour}; font-size:13px; font-weight:bold;'
            f'padding:9px; border:1px solid #2f4654; border-radius:6px;'
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
            'QComboBox{background:#08131a;color:#c9d6de;border:1px solid #325262;'
            'border-radius:5px;padding:5px 8px;font-size:12px;}'
            'QComboBox::drop-down{border:none;}'
            'QComboBox QAbstractItemView{background:#101a22;color:#c9d6de;'
            'selection-background-color:#2f8bd8;}'
        )
        ctrl.addWidget(self._cam_combo)

        self._btn_cam = QPushButton('▶  Start Camera')
        self._btn_cam.setStyleSheet(
            'QPushButton{background:#1c5f87;color:white;border-radius:5px;'
            'padding:7px 12px;font-size:12px;font-weight:bold;}'
            'QPushButton:hover{background:#2b78a5;}'
        )
        self._btn_cam.clicked.connect(self._toggle_camera)
        ctrl.addWidget(self._btn_cam)

        self._cam_status = QLabel('Off')
        self._cam_status.setStyleSheet('color:#8195a3; font-size:11px; padding-left:4px;')
        ctrl.addWidget(self._cam_status)
        ctrl.addStretch()

        layout.addLayout(ctrl)

        # ── Grid selection row ────────────────────────────────────────────────
        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)

        self._btn_define_grid = QPushButton('✛  Define Grid')
        self._btn_define_grid.setStyleSheet(
            'QPushButton{background:#143221;color:#8ee6aa;border:1px solid #286947;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#1c4730;}'
            'QPushButton:checked{background:#1f7a42;color:white;border-color:#71e89f;}'
        )
        self._btn_define_grid.setCheckable(True)
        self._btn_define_grid.clicked.connect(self._toggle_grid_select)
        grid_row.addWidget(self._btn_define_grid)

        btn_clear_grid = QPushButton('✕  Clear Grid')
        btn_clear_grid.setStyleSheet(
            'QPushButton{background:#16222b;color:#a9b8c4;border:1px solid #314652;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#243442;}'
        )
        btn_clear_grid.clicked.connect(self._clear_grid)
        grid_row.addWidget(btn_clear_grid)

        self._btn_cut_detected = QPushButton('✂  Cut Detected Fruit')
        self._btn_cut_detected.setStyleSheet(
            'QPushButton{background:#58336b;color:#f0d8ff;border:1px solid #8d5fb0;'
            'border-radius:4px;padding:5px 10px;font-size:11px;font-weight:bold;}'
            'QPushButton:hover{background:#704186;}'
            'QPushButton:disabled{background:#172028;color:#52636f;border-color:#263946;}'
        )
        self._btn_cut_detected.clicked.connect(self._cut_detected_fruit)
        grid_row.addWidget(self._btn_cut_detected)

        self._grid_status = QLabel('No grid defined')
        self._grid_status.setStyleSheet('color:#8195a3; font-size:11px; padding-left:6px;')
        grid_row.addWidget(self._grid_status)
        grid_row.addStretch()

        layout.addLayout(grid_row)

        # ── Video display ─────────────────────────────────────────────────────
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setMinimumSize(400, 300)
        self._cam_label.setStyleSheet(
            'background:#03070a; color:#8195a3; border:1px solid #314652; border-radius:6px;'
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
            'QTextEdit{background:#071016;color:#d9e6ed;border:1px solid #1c303b;'
            'border-radius:6px;padding:7px;font-family:monospace;font-size:11px;}'
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
        """
        Detect red objects inside the manually defined camera grid.

        Method:
          1. Build a perspective transform H from the 4 clicked corner points
             that maps frame pixels → grid coordinates (0..14, 0..4).
          2. For each red contour:
             a. Skip if centroid maps outside the grid (0 ≤ col < 14, 0 ≤ row < 4).
             b. 'origin cell'  = the grid cell the centroid lands in.
             c. 'covered cells' = all cells touched by the bounding-box corners.
          3. Draws bounding box + label on the frame (annotations are visible in GUI).
          4. Stores results in self._detected_fruits for use by _cut_detected_fruit.

        Only red is detected (no green / yellow / orange) as requested.
        Nothing is detected until the grid is fully defined (4 points clicked).
        """
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
        """
        Enter or exit the 4-point grid selection mode.

        When checked=True: clears any existing points, switches cursor to crosshair,
        and waits for 4 clicks on the camera label.
        When checked=False (user unchecks before 4 points): cancels and discards points.
        The grid is NOT cleared — _clear_grid must be pressed explicitly for that.
        """
        self._selecting_grid = checked
        if checked:
            self._grid_pts = []
            self._grid_status.setText('Click point 1 of 4 on the camera feed')
            self._grid_status.setStyleSheet('color:#ffd166; font-size:11px; padding-left:6px;')
            self._cam_label.setCursor(Qt.CrossCursor)
        else:
            self._cam_label.setCursor(Qt.ArrowCursor)
            if len(self._grid_pts) < 4:
                self._grid_status.setText('Selection cancelled')
                self._grid_status.setStyleSheet('color:#8195a3; font-size:11px; padding-left:6px;')

    def _clear_grid(self):
        self._grid_pts = []
        self._selecting_grid = False
        self._btn_define_grid.setChecked(False)
        self._cam_label.setCursor(Qt.ArrowCursor)
        self._grid_status.setText('No grid defined')
        self._grid_status.setStyleSheet('color:#8195a3; font-size:11px; padding-left:6px;')

    def _cam_label_clicked(self, event):
        """
        Handle a mouse click on the camera label during grid selection.

        Converts the QLabel pixel coordinate to a frame pixel coordinate
        (accounting for KeepAspectRatio scaling and centering letterbox offsets),
        then appends the point to self._grid_pts.

        On the 4th click the 4 raw points are sorted into TL/TR/BR/BL order
        and the grid is locked — detection begins on the next frame.
        """
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
        """
        Convert a QLabel pixel position to the corresponding frame pixel coordinate.

        The camera image is displayed with Qt.KeepAspectRatio inside the label,
        which means there may be letterbox bars on the sides or top/bottom.
        We compute:
          scale = the uniform scale factor applied to fit the frame inside the label
          ox/oy = the letterbox offsets (pixels of empty space on each axis)
        Then invert the mapping: frame_px = (label_px - offset) / scale.
        Returns None if the click is outside the actual image area (on the bars).
        """
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
        """
        Sort 4 arbitrary points into (TL, TR, BR, BL) order.

        Uses the standard quadrilateral sorting trick:
          TL = point with smallest (x+y)   — closest to top-left origin
          BR = point with largest  (x+y)   — furthest from top-left
          TR = point with smallest (y-x)   — top-right has small y, large x
          BL = point with largest  (y-x)   — bottom-left has large y, small x
        Returns a list of integer (x, y) tuples in TL, TR, BR, BL order.
        """
        pts = np.array(pts, dtype=np.float32)
        s   = pts.sum(axis=1)
        d   = np.diff(pts, axis=1).ravel()
        tl  = pts[np.argmin(s)]
        br  = pts[np.argmax(s)]
        tr  = pts[np.argmin(d)]
        bl  = pts[np.argmax(d)]
        return [tuple(map(int, p)) for p in (tl, tr, br, bl)]

    def _draw_grid_overlay(self, frame):
        """
        Draw the manual grid onto the camera frame before it is displayed.

        During selection (< 4 points): draws a coloured dot + corner label (TL/TR/BR/BL)
        for each point clicked so far, giving the operator visual feedback.

        After selection (== 4 points): draws the full 14-column × 4-row bilinear grid
        using perspective-correct row and column lines, labels every cell centre (A1–N4),
        and outlines the quad border in blue.  The grid is drawn on every subsequent frame
        without moving because self._grid_pts is fixed after the 4th click.
        """
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
        """
        Initialise rclpy and create the two ROS 2 nodes.

        JointStateNode spins in a daemon thread so /joint_states callbacks
        arrive continuously without blocking Qt's event loop.

        MoverNode creates its own SingleThreadedExecutor inside its worker
        thread (see _move_thread), so it doesn't need a persistent spin here.
        """
        try:
            rclpy.init(args=None)
            self._js_node = JointStateNode(
                lambda joints: self._joints_sig.emit(joints)
            )
            self._mover_node = MoverNode()
            # Spin JointStateNode continuously so the angle display stays live
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
