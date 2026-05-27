#!/usr/bin/env python3
"""
real_gui_points.py — Main operator GUI for the FruitNinja UR3e cutting robot.

HOW IT WORKS (overview)
========================
1. The GUI connects to a live UR3e robot via ROS 2 / MoveIt.
2. The operator selects grid cells (A1–G4) on the cutting board grid.
3. Pressing "Move to Selected" drives the robot through each cell in order,
   using the fixed calibrated joint pose for that cell.
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
  MoverNode        – Sends timed joint trajectories to the scaled controller.
  CameraWorker     – Captures raw frames from a webcam or RealSense in a background thread.
  MainWindow       – PyQt5 main window that wires everything together.
"""
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import sys
import math
import threading
import time
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Float64
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from fruitninja.grid_mover import (
    cell_to_pose, cell_to_joints_deg, grid_uv_to_pose,
    grid_uv_to_joints_deg, grid_uv_to_rotvec_rad, GRID_COLS, GRID_ROWS,
    REQUIRED_GRID_CORNERS, EXTENDED_ROW1_CELLS, EXTENDED_COL_A_CELLS,
    EXTENDED_CALIBRATION_CELLS, apply_calibration, save_calibration,
    load_calibration, CALIBRATION_PROFILES,
    get_active_profile, set_active_profile, list_saved_profiles,
)
from fruitninja import theme as _theme

import argparse
import socket as _socket
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    _HAS_RS = False

try:
    import mediapipe.python.solutions.hands as _mp_hands
    import mediapipe.python.solutions.drawing_utils as _mp_drawing
    import mediapipe.python.solutions.drawing_styles as _mp_drawing_styles
    _HAS_MP = True
except Exception:
    _HAS_MP = False


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
# APPROACH_LIFT_DELTA: fixed shoulder lift offset from the calibrated cut pose.

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

JOINT_LABELS = ['Base (pan)', 'Shoulder (lift)', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

HOME_DEG = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]

TRAJECTORY_ACTION = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION = '/move_action'
MOVE_GROUP_NAME = 'ur_manipulator'

APPROACH_LIFT_DELTA = -11.0  # matches legacy hover(-52°) → cut(-41°) spacing

# Manual moves still use the home detour when the base swing is large. Trace
# Grid deliberately skips it so A→G corner checks do not reset to home.
SAFE_TRANSIT_BASE_DEG = 45.0

# Vestigial cell-trim constants.  Previously used to paper over the
# joint-space curvature error of the 4-corner bilinear interpolation.  With
# the new Coons-patch interpolation (which honours any extra row/column
# anchors captured by the operator) and the IK-based camera-cut path, these
# are no longer needed.  Kept at 0.0 in case a future tuning pass needs them.
CAMERA_TO_ROBOT_ROW_OFFSET_CELLS = 0.0
GLOBAL_GRID_ROW_TRIM_CELLS = 0.0
GLOBAL_GRID_COL_TRIM_CELLS = 0.0

# Raise all cut targets slightly. Shoulder lift uses the existing convention
# where more negative means higher; -0.5° is roughly the 3 mm over-depth seen
# on the board. The right-near corner gets a little extra because it was the
# protective-stop corner.
GLOBAL_CUT_LIFT_DEG = -0.5
RIGHT_NEAR_CELL = f'{GRID_COLS[-1]}{GRID_ROWS[-1]}'
CELL_CUT_LIFT_DEG = {
    RIGHT_NEAR_CELL: -0.5,
}

START_HOLD_SEC = 0.5
MIN_MOVE_DURATION_SEC = 4.0
MAX_JOINT_SPEED_DEG_S = 10.0
CROSS_CUT_WRIST_DELTA_DEG = 90.0
HAND_ZONE_MIN_LANDMARKS = 5
HAND_ZONE_PALM_INDICES = (0, 5, 9, 13, 17)
CLEAR_FRAMES_BEFORE_RESUME = 2

# Trajectory tolerances (per joint, in radians).
# Path tolerance applies during execution; goal tolerance applies at the end.
# Set generously so small UR controller overshoot doesn't fail the whole sequence.
PATH_TOLERANCE_RAD = math.radians(8.0)
GOAL_TOLERANCE_RAD = math.radians(2.0)

# Older real-robot commits that moved reliably used MoveIt MoveGroup joint
# constraints.  This is still joint-space movement, not a Cartesian pose / IK
# target.  Keep direct controller execution as a fallback when MoveIt is absent.
PREFER_MOVEIT_JOINT_GOALS = False

# When True, camera-detected fruit cuts solve IK against the bilinearly
# interpolated Cartesian pose of the corners instead of using the bilinearly
# interpolated joint angles.  This eliminates the joint-space curvature error
# that makes the middle cells drift when only 4 corners are calibrated.  Falls
# back to the joint-interpolation path if the /compute_ik service isn't up.
# Manual cell clicks and Trace Grid are not affected.
USE_IK_FOR_CAMERA_FRUITS = True
IK_TIMEOUT_SEC = 1.0
IK_BASE_FRAME = 'base_link'
IK_EEF_LINK = 'tool0'
IK_SERVICE_NAME = '/compute_ik'

# Compressed frame size published to /camera/image_raw/compressed for safety_node.
# 640x480 is the sweet spot for MediaPipe Hands — fast inference without losing
# the spatial resolution needed to resolve hand position against the grid polygon.
# Any camera source (webcam, RealSense, Other) is scaled to this before encoding.
SAFETY_FRAME_W = 640
SAFETY_FRAME_H = 480
SAFETY_JPEG_QUALITY = 60


def _trimmed_uv(col_idx: float, row_idx: float) -> tuple[float, float]:
    """Apply GLOBAL_GRID_*_TRIM_CELLS to a (col, row) and return normalised u, v."""
    n_cols = len(GRID_COLS)
    n_rows = len(GRID_ROWS)
    col_pos = float(col_idx) + GLOBAL_GRID_COL_TRIM_CELLS
    row_pos = float(row_idx) + GLOBAL_GRID_ROW_TRIM_CELLS
    col_pos = max(0.0, min(n_cols - 1, col_pos))
    row_pos = max(0.0, min(n_rows - 1, row_pos))
    return col_pos / (n_cols - 1), row_pos / (n_rows - 1)


def _board_position(col_idx: int, row_idx: int) -> tuple:
    """Return (x, y, z) for a grid cell by index, with global trim applied."""
    u, v = _trimmed_uv(col_idx, row_idx)
    return grid_uv_to_pose(u, v)


def _board_joints(col_idx: int, row_idx: int) -> list:
    """Return calibrated joint angles for the same grid cell, with global trim."""
    u, v = _trimmed_uv(col_idx, row_idx)
    return grid_uv_to_joints_deg(u, v)


def _cut_joints_for_cell(cell: str, joints_deg: list) -> list:
    out = list(joints_deg)
    out[1] += GLOBAL_CUT_LIFT_DEG + CELL_CUT_LIFT_DEG.get(cell, 0.0)
    return out


def _camera_grid_to_robot_target(grid_x: float,
                                 grid_y: float) -> tuple[float, float, float, float, str]:
    col_pos = min(
        len(GRID_COLS) - 1,
        max(0.0, grid_x - 0.5 + GLOBAL_GRID_COL_TRIM_CELLS),
    )
    row_pos = min(
        len(GRID_ROWS) - 1,
        max(0.0,
            grid_y - 0.5
            + CAMERA_TO_ROBOT_ROW_OFFSET_CELLS
            + GLOBAL_GRID_ROW_TRIM_CELLS),
    )
    u = col_pos / (len(GRID_COLS) - 1)
    v = row_pos / (len(GRID_ROWS) - 1)
    target_col = min(len(GRID_COLS) - 1, max(0, int(round(col_pos))))
    target_row = min(len(GRID_ROWS) - 1, max(0, int(round(row_pos))))
    target_cell = f'{GRID_COLS[target_col]}{GRID_ROWS[target_row]}'
    return col_pos, row_pos, u, v, target_cell


def _wrap_deg(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _rotvec_to_quat(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """UR tool rotation-vector (axis * angle, in radians) → quaternion (x, y, z, w)."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    half = angle * 0.5
    s = math.sin(half) / angle
    return (rx * s, ry * s, rz * s, math.cos(half))


# ── ROS2 nodes ─────────────────────────────────────────────────────────────────
# Three separate ROS 2 nodes are created so their executors don't block each other.
# They are all started once in MainWindow._start_ros() and run as daemon threads.

class JointStateNode(Node):
    """
    Listens to /joint_states and converts the 6 UR joint positions from
    radians to degrees, then fires joint_callback so the GUI can update
    the live angle display. Runs continuously in a background spin thread.
    """
    def __init__(self, joint_callback, program_callback, speed_callback):
        super().__init__('real_gui_points_js')
        self._joint_cb = joint_callback
        self._program_cb = program_callback
        self._speed_cb = speed_callback
        status_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(JointState, '/joint_states', self._recv, 10)
        self.create_subscription(
            Bool,
            '/io_and_status_controller/robot_program_running',
            self._recv_program_running,
            status_qos,
        )
        self.create_subscription(
            Float64,
            '/speed_scaling_state_broadcaster/speed_scaling',
            self._recv_speed_scaling,
            status_qos,
        )

    def _recv(self, msg: JointState):
        joints = {n: math.degrees(p) for n, p in zip(msg.name, msg.position)
                  if n in JOINT_NAMES}
        self._joint_cb(joints)

    def _recv_program_running(self, msg: Bool):
        self._program_cb(bool(msg.data))

    def _recv_speed_scaling(self, msg: Float64):
        self._speed_cb(float(msg.data))


class MoverNode(Node):
    """
    Sends fixed joint-space trajectories directly to the active UR controller.

    move_to(degrees, done_cb, fail_cb, current_degrees=None)
      - Spawns a background thread so the GUI never freezes during planning.
      - Sends the current joint state as the first trajectory point, then the
        target joint state with a conservative duration.
      - Does not send Cartesian position goals or compute_cartesian_path requests.
      - Does not use MoveIt/IK for execution, avoiding instant first-point jumps.
      - On success  → calls done_cb()
      - On failure  → calls fail_cb(reason_string)
      - cancel()    → sets _cancel_flag so the thread exits at the next wait point.
    """
    def __init__(self):
        super().__init__('real_gui_points_mover')
        self._client      = ActionClient(self, FollowJointTrajectory, TRAJECTORY_ACTION)
        self._moveit_client = ActionClient(self, MoveGroup, MOVE_GROUP_ACTION)
        self._ik_client   = self.create_client(GetPositionIK, IK_SERVICE_NAME)
        self._cancel_flag = False
        self._cancel_generation = 0
        self._goal_handle = None
        self._goal_lock   = threading.Lock()

    def move_to(self, degrees: list, done_cb, fail_cb, current_degrees: list = None):
        self._cancel_flag = False
        if current_degrees is None:
            fail_cb('No live joint state yet — wait for /joint_states before moving')
            return
        target_degrees = self._nearest_equivalent_target(degrees, current_degrees)
        generation = self._cancel_generation
        if PREFER_MOVEIT_JOINT_GOALS:
            threading.Thread(
                target=self._moveit_thread,
                args=(target_degrees, done_cb, fail_cb, generation),
                daemon=True,
            ).start()
            return
        threading.Thread(
            target=self._move_thread,
            args=(current_degrees, target_degrees, done_cb, fail_cb, generation),
            daemon=True,
        ).start()

    def _nearest_equivalent_target(self, degrees: list, current_degrees: list = None) -> list:
        if current_degrees is None:
            return [_wrap_deg(deg) for deg in degrees]
        return [
            current + _wrap_deg(target - current)
            for target, current in zip(degrees, current_degrees)
        ]

    @staticmethod
    def _duration_msg(seconds: float) -> Duration:
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1_000_000_000)
        return Duration(sec=sec, nanosec=nanosec)

    def _move_duration(self, current_degrees: list, target_degrees: list) -> float:
        max_delta = max(
            abs(target - current)
            for target, current in zip(target_degrees, current_degrees)
        )
        return max(MIN_MOVE_DURATION_SEC, max_delta / MAX_JOINT_SPEED_DEG_S)

    def _make_trajectory_goal(self, current_degrees: list,
                              target_degrees: list) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        start = JointTrajectoryPoint()
        start.positions = [math.radians(deg) for deg in current_degrees]
        start.velocities = [0.0] * len(JOINT_NAMES)
        start.time_from_start = self._duration_msg(START_HOLD_SEC)

        target = JointTrajectoryPoint()
        target.positions = [math.radians(deg) for deg in target_degrees]
        target.velocities = [0.0] * len(JOINT_NAMES)
        target.time_from_start = self._duration_msg(
            START_HOLD_SEC + self._move_duration(current_degrees, target_degrees)
        )

        goal.trajectory.points = [start, target]
        goal.goal_time_tolerance = self._duration_msg(5.0)

        # Per-joint tolerances — without these the controller falls back to its
        # configured defaults which are tight (~0.01 rad) and frequently fail
        # the sequence with GOAL_TOLERANCE_VIOLATED on the real hardware even
        # though the robot is physically capable of reaching the target.
        for name in JOINT_NAMES:
            path_t = JointTolerance()
            path_t.name = name
            path_t.position = PATH_TOLERANCE_RAD
            goal.path_tolerance.append(path_t)

            goal_t = JointTolerance()
            goal_t.name = name
            goal_t.position = GOAL_TOLERANCE_RAD
            goal.goal_tolerance.append(goal_t)
        return goal

    def _was_cancelled(self, generation: int) -> bool:
        return self._cancel_flag or generation != self._cancel_generation

    def _move_thread(self, current_degrees, target_degrees, done_cb, fail_cb, generation):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        goal_handle = None
        try:
            if not self._client.wait_for_server(timeout_sec=5.0):
                fail_cb(
                    f'No action server at {TRAJECTORY_ACTION}. '
                    'Run Startup Step 2 — Fix Controller, and confirm the UR '
                    'driver is fully connected / External Control is playing.'
                )
                return
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            goal = self._make_trajectory_goal(current_degrees, target_degrees)
            future = self._client.send_goal_async(goal)
            executor.spin_until_future_complete(future)
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            goal_handle = future.result()
            if not goal_handle.accepted:
                fail_cb('Trajectory rejected by scaled_joint_trajectory_controller')
                return
            with self._goal_lock:
                self._goal_handle = goal_handle
 
            result_future = goal_handle.get_result_async()
            executor.spin_until_future_complete(result_future)
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            result = result_future.result().result
            if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                done_cb()
            else:
                msg = result.error_string or 'no controller error string'
                fail_cb(f'Trajectory failed (code: {result.error_code}): {msg}')
        finally:
            with self._goal_lock:
                if self._goal_handle is goal_handle:
                    self._goal_handle = None
            executor.remove_node(self)

    def _make_moveit_goal(self, target_degrees: list) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP_NAME
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        for name, deg in zip(JOINT_NAMES, target_degrees):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = math.radians(deg)
            jc.tolerance_above = GOAL_TOLERANCE_RAD
            jc.tolerance_below = GOAL_TOLERANCE_RAD
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)
        return goal

    def _moveit_thread(self, target_degrees, done_cb, fail_cb, generation):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        goal_handle = None
        try:
            if not self._moveit_client.wait_for_server(timeout_sec=5.0):
                fail_cb(
                    f'No action server at {MOVE_GROUP_ACTION}. '
                    'Start Step 2 — MoveIt before sending grid moves.'
                )
                return
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            goal = self._make_moveit_goal(target_degrees)
            future = self._moveit_client.send_goal_async(goal)
            executor.spin_until_future_complete(future)
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            goal_handle = future.result()
            if goal_handle is None:
                fail_cb('MoveIt goal send failed')
                return
            if not goal_handle.accepted:
                fail_cb('MoveIt rejected joint goal')
                return
            with self._goal_lock:
                self._goal_handle = goal_handle

            result_future = goal_handle.get_result_async()
            executor.spin_until_future_complete(result_future)
            if self._was_cancelled(generation):
                fail_cb('Cancelled')
                return

            result = result_future.result().result
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                done_cb()
            else:
                fail_cb(f'MoveIt failed (code: {result.error_code.val})')
        finally:
            with self._goal_lock:
                if self._goal_handle is goal_handle:
                    self._goal_handle = None
            executor.remove_node(self)

    def solve_ik(self,
                 target_xyz: tuple,
                 target_rotvec: tuple,
                 seed_joints_deg: list,
                 timeout_sec: float = IK_TIMEOUT_SEC) -> list | None:
        """
        Ask MoveIt's /compute_ik service for joint angles that reach the given
        Cartesian pose in the IK_BASE_FRAME, seeded by seed_joints_deg.
        Returns a list of 6 degrees in JOINT_NAMES order, or None on any
        failure (service down, timeout, no IK solution, name mismatch).
        """
        if not self._ik_client.wait_for_service(timeout_sec=0.5):
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = MOVE_GROUP_NAME
        req.ik_request.ik_link_name = IK_EEF_LINK
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = self._duration_msg(timeout_sec)

        pose = PoseStamped()
        pose.header.frame_id = IK_BASE_FRAME
        pose.pose.position.x = float(target_xyz[0])
        pose.pose.position.y = float(target_xyz[1])
        pose.pose.position.z = float(target_xyz[2])
        qx, qy, qz, qw = _rotvec_to_quat(*target_rotvec)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        req.ik_request.pose_stamped = pose

        seed_state = JointState()
        seed_state.name = list(JOINT_NAMES)
        seed_state.position = [math.radians(d) for d in seed_joints_deg]
        req.ik_request.robot_state.joint_state = seed_state

        future = self._ik_client.call_async(req)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        try:
            executor.spin_until_future_complete(future, timeout_sec=timeout_sec + 0.5)
        finally:
            executor.remove_node(self)

        if not future.done():
            return None
        result = future.result()
        if result is None or result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        positions = dict(zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position,
        ))
        if not all(name in positions for name in JOINT_NAMES):
            return None
        return [math.degrees(positions[name]) for name in JOINT_NAMES]

    def cancel(self):
        self._cancel_flag = True
        self._cancel_generation += 1
        with self._goal_lock:
            goal_handle = self._goal_handle
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f'Failed to cancel active trajectory goal: {e}')

    def clear_cancel(self):
        self._cancel_flag = False


# ── Camera + safety bridge node ───────────────────────────────────────────────

class CameraBridgeNode(Node):
    """
    Publishes camera frames and grid zone so the standalone safety_node can
    subscribe without opening the camera itself.

    Topics published:
      /camera/image_raw/compressed  (sensor_msgs/CompressedImage)
      /safety/zone                  (std_msgs/Float32MultiArray)
                                    [x0,y0, x1,y1, x2,y2, x3,y3] TL→TR→BR→BL

    Topic subscribed:
      /safety/hand_detected         (std_msgs/Bool)
    """
    def __init__(self, hand_cb):
        super().__init__('fruitninja_camera_bridge')
        self._pub_img  = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        self._pub_zone = self.create_publisher(Float32MultiArray, '/safety/zone', 10)
        self.create_subscription(Bool, '/safety/hand_detected', hand_cb, 10)

    def publish_frame(self, frame):
        # Resize to fixed 640x480 before encoding so MediaPipe always receives
        # a consistent frame size regardless of camera source resolution.
        # INTER_AREA is fastest for downscaling and preserves edge sharpness.
        h, w = frame.shape[:2]
        if w != SAFETY_FRAME_W or h != SAFETY_FRAME_H:
            frame = cv2.resize(frame, (SAFETY_FRAME_W, SAFETY_FRAME_H),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, SAFETY_JPEG_QUALITY])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._pub_img.publish(msg)

    def publish_zone(self, pts):
        """pts: list of 4 (x, y) tuples in TL, TR, BR, BL order."""
        flat = [v for pt in pts for v in pt]
        msg = Float32MultiArray()
        msg.data = [float(v) for v in flat]
        self._pub_zone.publish(msg)

    def clear_zone(self):
        msg = Float32MultiArray()
        msg.data = []
        self._pub_zone.publish(msg)


# ── UR3e Dashboard safety interlock ───────────────────────────────────────────

class UR3eDashboard:
    """
    Thin TCP wrapper for the UR Dashboard Server (port 29999).
    Used by the safety interlock to pause/resume the robot when a hand
    is detected inside the cutting zone.

    Connection failure is silent: if the dashboard is unreachable the safety
    layer still blocks new trajectories via the _cancel_flag path.
    """
    def __init__(self, ip: str, port: int = 29999):
        self._paused = False
        self._sock   = None
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((ip, port))
            s.recv(1024)   # consume welcome banner
            s.settimeout(1.0)
            self._sock = s
            print(f'[Safety] Dashboard connected to {ip}:{port}')
        except Exception as e:
            print(f'[Safety] Dashboard connect failed ({ip}:{port}): {e}')

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _send(self, cmd: str):
        if self._sock is None:
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
            self._sock = None


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
        # Webcam=0, Other=2 (/dev/video2 — any non-standard USB camera)
        dev_index = {0: 0, 2: 2}.get(source, 0)
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
    _safety_sig = pyqtSignal(bool)
    _program_sig = pyqtSignal(bool)
    _speed_sig = pyqtSignal(float)

    def __init__(self, robot_ip: str = '192.168.0.194'):
        super().__init__()
        self.setWindowTitle('FruitNinja — UR3e Operator Console')
        self.setMinimumSize(1240, 740)
        self.setStyleSheet(_theme.root_qss())

        self._selected_cells = []
        self._moving = False
        self._cell_btns = {}
        self._camera = CameraWorker()
        self._current_joints_rad = {n: 0.0 for n in JOINT_NAMES}
        self._have_joint_state = False
        self._program_seen = False
        self._speed_seen = False
        self._robot_program_running = False
        self._speed_scaling = 0.0

        # Per-robot calibration capture state. Populated by the Capture buttons
        # and saved to ~/.fruitninja/calibrations/<robot>.json.
        self._pending_calibration: dict = {}

        # Manual grid state
        self._grid_pts       = []    # up to 4 (x, y) in frame coords
        self._selecting_grid = False
        self._last_frame_wh  = None  # (w, h) of last received frame

        # Per-frame throttle counter — MediaPipe Hands is ~30 ms/frame on CPU,
        # which alone exceeds the 33 ms budget at 30 fps and makes the feed
        # visibly lag.  Heavier work runs only every Nth frame so the display
        # stays smooth while detection still updates several times a second.
        self._frame_counter = 0
        self._last_zone_publish_frame = 0
        self._safety_clear_frames = 0
        self._resume_after_safety = None

        # Latest fruit detections: list of {'label', 'origin', 'covered': [cells]}
        self._detected_fruits = []

        # ── Safety interlock ──────────────────────────────────────────────────
        self._safety_paused = False
        self._dashboard = UR3eDashboard(robot_ip)
        if _HAS_MP:
            self._hands_mp = _mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
        else:
            self._hands_mp = None

        self._build_ui()
        self._joints_sig.connect(self._update_joint_display)
        self._status_sig.connect(lambda t, c: self._set_status(t, c))
        self._log_sig.connect(self._log_widget.append)
        self._cam_sig.connect(self._update_camera_frame)
        self._safety_sig.connect(self._handle_safety_node_detection)
        self._program_sig.connect(self._update_program_running)
        self._speed_sig.connect(self._update_speed_scaling)
        self._start_ros()
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

        # ── per-robot calibration ────────────────────────────────────────────
        root.addWidget(self._build_calibration_panel())

        # ── grid selector ─────────────────────────────────────────────────────
        grid_group = self._group(
            f'Grid — Toggle Cells  (A1=far-left  {RIGHT_NEAR_CELL}=near-right  |  moves left→right)'
        )
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
        self._btn_trace = self._action_btn('⬡  Trace Grid',        '#2a5a2a', self._trace_grid)
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
        act_layout.addWidget(self._btn_trace)
        act_layout.addWidget(self._btn_clear)
        act_layout.addWidget(self._btn_stop)
        act_layout.addWidget(self._btn_reset)
        root.addLayout(act_layout)

        # ── safety interlock banner ───────────────────────────────────────────
        self._safety_label = QLabel('🛡 Safety: standby — define grid to activate interlock')
        self._safety_label.setAlignment(Qt.AlignCenter)
        self._safety_label.setStyleSheet(
            'background:#101a22; color:#7a8fa0; font-size:12px; font-weight:bold;'
            'padding:6px; border:1px solid #2f4654; border-radius:6px;'
        )
        root.addWidget(self._safety_label)

        # ── status ────────────────────────────────────────────────────────────
        self._status_label = QLabel('● Idle — select a cell and press Move')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(_theme.status_label_qss(_theme.THEME['fg_muted']))
        root.addWidget(self._status_label)

        # ── log ───────────────────────────────────────────────────────────────
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(96)
        self._log_widget.setStyleSheet(_theme.log_widget_qss())
        root.addWidget(self._log_widget)

        # ── spacebar E-STOP shortcut ──────────────────────────────────────────
        shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        shortcut.activated.connect(self._stop)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _group(title):
        g = QGroupBox(title)
        g.setStyleSheet(_theme.panel_qss())
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 16, 12, 12)
        g.setLayout(layout)
        return g

    @staticmethod
    def _cell_style_normal():
        # Selector-grid cell toggle style with the centralised theme.
        return (
            f'QPushButton{{background:{_theme.THEME["bg_input"]};'
            f' color:{_theme.THEME["fg_primary"]};'
            f' border:1px solid {_theme.THEME["border"]};'
            f' border-radius:{_theme.THEME["radius_sm"]};'
            f' font-size:11px; font-weight:600;}}'
            f'QPushButton:checked{{background:{_theme.THEME["accent"]}; color:#ffffff;'
            f' border:1px solid {_theme.THEME["accent_hover"]};}}'
            f'QPushButton:hover{{background:{_theme.THEME["bg_hover"]};}}'
        )

    @staticmethod
    def _action_btn(text, colour, slot):
        b = QPushButton(text)
        # Map legacy hex colours to theme variants so existing call-sites keep
        # working without changing every caller.
        colour_to_qss = {
            '#1a5c1a': _theme.success_button_qss(),
            '#2a7a2a': _theme.success_button_qss(),
            '#0b6b6b': _theme.primary_button_qss(),
            '#0b5b8b': _theme.primary_button_qss(),
            '#3a7aff': _theme.primary_button_qss(),
            '#2a3a5c': _theme.ghost_button_qss(),
            '#3b4650': _theme.ghost_button_qss(),
            '#5c1a1a': _theme.danger_button_qss(),
            '#a83232': _theme.danger_button_qss(),
            '#ff4444': _theme.danger_button_qss(),
            '#a07020': _theme.warning_button_qss(),
            '#e0a000': _theme.warning_button_qss(),
        }
        b.setStyleSheet(colour_to_qss.get(colour, _theme.primary_button_qss()))
        b.clicked.connect(slot)
        return b

    # ── slots ─────────────────────────────────────────────────────────────────

    def _cell_sort_key(self, cell: str):
        """Sort key: left-to-right (A→G), then top-to-bottom (1→4)."""
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

    def _set_resume_motion(self, label: str, callback):
        self._resume_after_safety = (label, callback)

    def _clear_resume_motion(self):
        self._resume_after_safety = None

    def _resume_interrupted_motion(self):
        if self._resume_after_safety is None:
            return
        label, callback = self._resume_after_safety
        self._resume_after_safety = None
        self._moving = True
        self._mover_node.clear_cancel()
        self._log(f'SAFETY INTERLOCK: resuming {label}')
        QTimer.singleShot(500, callback)

    def _trace_grid(self):
        """Visit all 4 grid corners without routing through Home."""
        if self._safety_paused:
            self._set_status('Safety interlock active — remove hand from zone', '#ff4444')
            return
        if not self._robot_ready_to_move():
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        self._moving = True
        self._clear_resume_motion()
        corners = list(REQUIRED_GRID_CORNERS)
        self._log(f'Trace: {" → ".join(corners)}')
        self._move_next(corners, use_safe_transit=False)

    def _go(self):
        """
        Start the fixed-joint movement sequence for all selected grid cells,
        left-to-right (A→G).
        """
        if self._safety_paused:
            self._set_status('Safety interlock active — remove hand from zone', '#ff4444')
            return
        if not self._robot_ready_to_move():
            return
        if not self._selected_cells:
            self._set_status('Toggle at least one cell first', '#e0a000')
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        self._moving = True
        self._clear_resume_motion()
        queue = list(self._selected_cells)   # already sorted left-to-right
        self._log(f'Queue: {" → ".join(queue)}')
        self._move_next(queue)

    def _move_next(self, queue: list, use_safe_transit: bool = True):
        """
        Recursive per-cell cut loop for the manual selection sequence.

        For each cell in the queue it chains three fixed joint-space calls:
          1. move_to(approach_degrees)  → fixed shoulder lift above grid pose
          2. move_to(cut_degrees)       → calibrated fixed grid joint pose
          3. move_to(approach_degrees)  → recover above the grid joint pose
        After recover, _move_next is called again with remaining = queue[1:].
        When the queue is empty (or E-STOP fires) the loop exits.
        """
        if not queue or self._mover_node._cancel_flag:
            self._moving = False
            if not queue:
                self._clear_resume_motion()
                self._status_sig.emit('All cells reached', '#00cc00')
                self._log_sig.emit('Sequence complete')
            return
        cell = queue[0]
        remaining = queue[1:]
        self._set_resume_motion(
            f'manual sequence from {cell}',
            lambda q=list(queue), st=use_safe_transit: self._move_next(q, st),
        )
        col_idx = GRID_COLS.index(cell[0])
        row_idx = GRID_ROWS.index(cell[1])
        bx, by, bz = _board_position(col_idx, row_idx)
        raw_cut_degrees = _board_joints(col_idx, row_idx)
        cut_degrees = _cut_joints_for_cell(cell, raw_cut_degrees)
        approach_degrees = list(cut_degrees)
        approach_degrees[1] += APPROACH_LIFT_DELTA
        cut_lift = GLOBAL_CUT_LIFT_DEG + CELL_CUT_LIFT_DEG.get(cell, 0.0)

        self._status_sig.emit(f'Moving to {cell}…  ({len(remaining)} remaining)', '#3a7aff')
        approach_txt = ', '.join(f'{j:.2f}°' for j in approach_degrees)
        cut_txt = ', '.join(f'{j:.2f}°' for j in cut_degrees)
        self._log(
            f'Moving to {cell}: '
            f'tool=({bx*1000:.1f}, {by*1000:.1f}, {bz*1000:.1f}) mm  '
            f'cut_lift={cut_lift:+.1f}°  '
            f'approach=[{approach_txt}]  cut=[{cut_txt}]'
        )

        def _on_arrive():
            self._log_sig.emit(f'Reached {cell} — cutting')
            self._status_sig.emit(f'Cutting at {cell}…', '#e0a000')
            # Use the LIVE joint state, not approach_degrees: the previous
            # move may have been wrapped by _nearest_equivalent_target so the
            # robot now sits at an angle ±360° from approach_degrees.  Passing
            # the stale "commanded" target here causes the next wrap to be
            # computed against the wrong reference, which is what made the
            # arm spin or loop instead of moving smoothly.
            self._mover_node.move_to(
                cut_degrees,
                done_cb=_on_cut,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_cut():
            self._log_sig.emit(f'Cut at {cell} — recovering')
            self._mover_node.move_to(
                approach_degrees,
                done_cb=lambda st=use_safe_transit: self._move_next(remaining, st),
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_fail(msg):
            if 'Cancelled' in msg:
                return
            self._clear_resume_motion()
            self._status_sig.emit(f'Failed at {cell}: {msg}', '#ff4444')
            self._log_sig.emit(f'FAIL at {cell}: {msg}')
            setattr(self, '_moving', False)

        if use_safe_transit:
            self._move_with_safe_transit(
                approach_degrees,
                done_cb=_on_arrive,
                fail_cb=_on_fail,
                label=cell,
            )
        else:
            self._mover_node.move_to(
                approach_degrees,
                done_cb=_on_arrive,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

    def _current_joint_degrees(self) -> list | None:
        if not self._have_joint_state:
            return None
        return [math.degrees(self._current_joints_rad[name]) for name in JOINT_NAMES]

    def _move_with_safe_transit(self, target_degrees: list, done_cb, fail_cb, label: str = ''):
        current = self._current_joint_degrees()
        if current is None:
            fail_cb('No live joint state')
            return
        self._mover_node.move_to(
            target_degrees,
            done_cb=done_cb,
            fail_cb=fail_cb,
            current_degrees=current,
        )

    def _fruit_target(self, fruit: dict) -> dict:
        grid_x, grid_y = fruit['grid_xy']
        _col_pos, _row_pos, u_trim, v_trim, target_cell = _camera_grid_to_robot_target(
            grid_x, grid_y
        )

        # Joint-bilinear interp at the trimmed (u, v): used as the IK seed AND
        # as the fallback when /compute_ik is unavailable.  Keeping the trim on
        # the seed preserves the same arm configuration the operator is used to.
        bilinear_cut = _cut_joints_for_cell(
            target_cell, grid_uv_to_joints_deg(u_trim, v_trim)
        )

        cut = bilinear_cut
        ik_used = False
        if USE_IK_FOR_CAMERA_FRUITS:
            # The IK target is the pose at the RAW camera (u, v) — no trim/offset.
            # Camera homography already maps pixels → physical board coordinates,
            # so the bilinear corner-pose interp lands on the actual fruit position.
            n_cols, n_rows = len(GRID_COLS), len(GRID_ROWS)
            u_raw = min(1.0, max(0.0, (grid_x - 0.5) / (n_cols - 1)))
            v_raw = min(1.0, max(0.0, (grid_y - 0.5) / (n_rows - 1)))
            target_xyz = grid_uv_to_pose(u_raw, v_raw)
            target_rotvec = grid_uv_to_rotvec_rad(u_raw, v_raw)
            ik_solution = self._mover_node.solve_ik(
                target_xyz, target_rotvec, bilinear_cut
            )
            if ik_solution is not None:
                cut = _cut_joints_for_cell(target_cell, ik_solution)
                ik_used = True
            else:
                self._log(
                    f'IK fallback for {target_cell}: /compute_ik unavailable '
                    'or no solution — using joint-bilinear interp'
                )

        approach = list(cut)
        approach[1] += APPROACH_LIFT_DELTA

        cut_cross = list(cut)
        cut_cross[5] += CROSS_CUT_WRIST_DELTA_DEG
        approach_cross = list(cut_cross)
        approach_cross[1] += APPROACH_LIFT_DELTA

        x, y, z = grid_uv_to_pose(u_trim, v_trim)
        return {
            'cut': cut,
            'approach': approach,
            'cut_cross': cut_cross,
            'approach_cross': approach_cross,
            'tool': (x, y, z),
            'target_cell': target_cell,
            'cut_lift_deg': GLOBAL_CUT_LIFT_DEG + CELL_CUT_LIFT_DEG.get(target_cell, 0.0),
            'ik_used': ik_used,
        }

    def _cut_detected_fruit(self):
        """
        Automated cut sequence driven by camera detection.
        Cuts each detected fruit at its contour centroid, then repeats the cut
        with wrist 3 rotated 90 degrees for a cross-cut.
        Requires the manual grid to be defined (4 corner points clicked) before running.
        """
        if self._safety_paused:
            self._set_status('Safety interlock active — remove hand from zone', '#ff4444')
            return
        if not self._robot_ready_to_move():
            return
        if self._moving:
            self._set_status('Already moving — press Stop first', '#e0a000')
            return
        if not self._detected_fruits:
            self._set_status('No fruit detected — aim camera at fruit first', '#e0a000')
            return
        if len(self._grid_pts) != 4:
            self._set_status('Define the grid before cutting detected fruit', '#e0a000')
            return

        fruits = sorted(
            self._detected_fruits,
            key=lambda f: (f['grid_xy'][0], f['grid_xy'][1]),
        )
        if not fruits:
            self._set_status('No valid cells from detection', '#e0a000')
            return

        self._moving = True
        self._clear_resume_motion()
        self._log('Cut (detected): ' + ' → '.join(f["origin"] for f in fruits))
        self._status_sig.emit(f'Cutting {len(fruits)} detected fruit(s)…', '#3a7aff')
        self._cut_next_fruit(fruits)

    def _cut_next_fruit(self, fruits: list):
        if self._safety_paused or self._mover_node._cancel_flag:
            if not self._safety_paused:
                self._clear_resume_motion()
            self._moving = False
            return
        if not fruits:
            self._moving = False
            self._clear_resume_motion()
            self._status_sig.emit('Detected fruit cut complete', '#00cc00')
            self._log_sig.emit('Detected fruit sequence complete')
            return

        fruit = fruits[0]
        remaining = fruits[1:]
        self._set_resume_motion(
            f'detected fruit sequence from {fruit["origin"]}',
            lambda q=list(fruits): self._cut_next_fruit(q),
        )
        target = self._fruit_target(fruit)
        x, y, z = target['tool']
        label = fruit['origin']
        target_cell = target.get('target_cell', label)
        gx, gy = fruit['grid_xy']
        solver = 'IK' if target.get('ik_used') else 'joint-interp'
        self._log(
            f'Fruit {label}: centroid grid=({gx:.2f}, {gy:.2f})  '
            f'robot_target={target_cell}  solver={solver}  '
            f'tool=({x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f}) mm  '
            f'cut_lift={target["cut_lift_deg"]:+.1f}°  '
            f'cross wrist +{CROSS_CUT_WRIST_DELTA_DEG:.0f}°'
        )
        self._status_sig.emit(f'Cutting fruit at {label}…', '#3a7aff')

        def _safe_to_continue() -> bool:
            if self._safety_paused or self._mover_node._cancel_flag:
                self._moving = False
                return False
            return True

        def _on_first_approach():
            if not _safe_to_continue():
                return
            self._status_sig.emit(f'First cut at {label}…', '#e0a000')
            self._mover_node.move_to(
                target['cut'],
                done_cb=_on_first_cut,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_first_cut():
            if not _safe_to_continue():
                return
            self._log_sig.emit(f'First cut at {label} — lifting')
            self._mover_node.move_to(
                target['approach'],
                done_cb=_on_recovered,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_recovered():
            if not _safe_to_continue():
                return
            self._status_sig.emit(f'Rotating cutter for cross-cut at {label}…', '#e0a000')
            self._mover_node.move_to(
                target['approach_cross'],
                done_cb=_on_cross_ready,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_cross_ready():
            if not _safe_to_continue():
                return
            self._status_sig.emit(f'Cross-cut at {label}…', '#e0a000')
            self._mover_node.move_to(
                target['cut_cross'],
                done_cb=_on_cross_cut,
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_cross_cut():
            if not _safe_to_continue():
                return
            self._log_sig.emit(f'Cross-cut at {label} — lifting')
            self._mover_node.move_to(
                target['approach_cross'],
                done_cb=lambda: self._cut_next_fruit(remaining),
                fail_cb=_on_fail,
                current_degrees=self._current_joint_degrees(),
            )

        def _on_fail(msg):
            if 'Cancelled' in msg:
                return
            self._clear_resume_motion()
            self._status_sig.emit(f'Fruit cut failed at {label}: {msg}', '#ff4444')
            self._log_sig.emit(f'FAIL fruit {label}: {msg}')
            setattr(self, '_moving', False)

        self._move_with_safe_transit(
            target['approach'],
            done_cb=_on_first_approach,
            fail_cb=_on_fail,
            label=label,
        )

    def _stop(self):
        """
        Emergency stop — triggered by the E-STOP button or the spacebar shortcut.

        Sets the cancel flag on MoverNode (the running move thread checks it and exits).
        Keeps _moving=True for 2 seconds so the user cannot immediately send a new
        trajectory while the controller is still finishing the current one.
        """
        self._clear_resume_motion()
        self._mover_node.cancel()
        self._dashboard.stop()
        self._moving = True   # block new moves until cooldown expires
        self._set_status('⚠ EMERGENCY STOP — wait 2 s before moving', '#ff4444')
        self._log('EMERGENCY STOP triggered — trajectory cancelled and robot stopped')
        QTimer.singleShot(2000, self._estop_cooldown_done)

    def _estop_cooldown_done(self):
        self._moving = False
        self._set_status('⚠ Stopped — safe to continue', '#e0a000')

    def _reset(self):
        """
        Drive the robot to the HOME_DEG pose ([0, -90, 0, -90, 0, 0] degrees).
        This is a safe upright posture that keeps the arm clear of the cutting board.
        Blocked while a move is in progress — press E-STOP first if needed.
        """
        if self._safety_paused:
            self._set_status('Safety interlock active — remove hand from zone', '#ff4444')
            return
        if not self._robot_ready_to_move():
            return
        if self._moving:
            self._set_status('Moving — press Stop first', '#e0a000')
            return
        self._moving = True
        self._clear_resume_motion()
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
        if all(name in joints for name in JOINT_NAMES):
            self._have_joint_state = True

    def _update_program_running(self, running: bool):
        self._program_seen = True
        self._robot_program_running = running

    def _update_speed_scaling(self, speed_scaling: float):
        self._speed_seen = True
        self._speed_scaling = speed_scaling

    def _robot_ready_to_move(self) -> bool:
        if not self._have_joint_state:
            self._set_status('No live joint state — wait for UR Driver', '#e0a000')
            self._log('MOVE BLOCKED: no /joint_states yet')
            return False
        if not self._program_seen or not self._speed_seen:
            self._set_status('Waiting for UR driver status topics…', '#e0a000')
            self._log('MOVE BLOCKED: waiting for robot program / speed scaling status')
            return False
        if not self._robot_program_running:
            self._set_status('UR program stopped — press Play on pendant', '#ff4444')
            self._log('MOVE BLOCKED: /io_and_status_controller/robot_program_running is false')
            return False
        if self._speed_scaling <= 0.001:
            self._set_status('Robot speed is 0 — check pendant speed slider / pause', '#ff4444')
            self._log('MOVE BLOCKED: /speed_scaling_state_broadcaster/speed_scaling is 0')
            return False
        return True

    def _set_status(self, text: str, colour: str = None):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            _theme.status_label_qss(colour or _theme.THEME['fg_muted'])
        )

    def _log(self, text: str):
        self._log_sig.emit(text)

    # ── Robot calibration panel ─────────────────────────────────────────────

    def _build_calibration_panel(self) -> QGroupBox:
        group = self._group('Robot Calibration')
        layout = group.layout()

        info = QLabel(
            'Freedrive the UR3e to each cell, click its Capture button. '
            'The 4 corners (A1, G1, G4, A4) are required. Capturing the '
            'extra cells along Row 1 and Column A improves middle-cell '
            'accuracy via Coons-patch interpolation.'
        )
        info.setWordWrap(True)
        info.setStyleSheet(f'color:{_theme.THEME["fg_muted"]}; font-size:11px;')
        layout.addWidget(info)

        prof_row = QHBoxLayout()
        prof_row.setSpacing(8)
        prof_lbl = QLabel('Robot:')
        prof_lbl.setStyleSheet(f'color:{_theme.THEME["fg_primary"]}; font-size:12px; font-weight:600;')
        prof_row.addWidget(prof_lbl)

        self._profile_combo = QComboBox()
        for i, name in enumerate(CALIBRATION_PROFILES, start=1):
            self._profile_combo.addItem(f'Robot {i}', name)
        self._profile_combo.setStyleSheet(_theme.combo_qss())
        self._profile_combo.currentIndexChanged.connect(self._refresh_profile_labels)
        prof_row.addWidget(self._profile_combo, stretch=1)

        self._btn_calib_load = QPushButton('Load')
        self._btn_calib_load.setStyleSheet(_theme.ghost_button_qss())
        self._btn_calib_load.clicked.connect(self._load_selected_profile)
        prof_row.addWidget(self._btn_calib_load)
        layout.addLayout(prof_row)

        # ── Grid-of-cells capture pad ─────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 4, 0, 4)
        # Column header
        for col_i, col_char in enumerate(GRID_COLS):
            header = QLabel(col_char)
            header.setAlignment(Qt.AlignCenter)
            header.setStyleSheet(
                f'color:{_theme.THEME["fg_muted"]}; font-size:11px; font-weight:700;'
            )
            grid.addWidget(header, 0, col_i + 1)
        # Row header + cells
        self._calib_btns = {}
        self._calib_status_lbls = {}
        capturable = set(EXTENDED_CALIBRATION_CELLS)
        for row_i, row_char in enumerate(GRID_ROWS):
            row_lbl = QLabel(row_char)
            row_lbl.setAlignment(Qt.AlignCenter)
            row_lbl.setStyleSheet(
                f'color:{_theme.THEME["fg_muted"]}; font-size:11px; font-weight:700;'
            )
            grid.addWidget(row_lbl, row_i + 1, 0)
            for col_i, col_char in enumerate(GRID_COLS):
                cell = f'{col_char}{row_char}'
                btn = QPushButton(cell)
                btn.setMinimumHeight(34)
                if cell in capturable:
                    btn.setStyleSheet(_theme.ghost_button_qss())
                    btn.clicked.connect(
                        lambda _checked=False, cn=cell: self._capture_cell(cn)
                    )
                    btn.setToolTip(
                        f'Capture {cell}'
                        + ('  (required corner)' if cell in REQUIRED_GRID_CORNERS
                           else '  (optional, improves accuracy)')
                    )
                    self._calib_btns[cell] = btn
                    badge = QLabel('◌')
                    badge.setAlignment(Qt.AlignCenter)
                    badge.setStyleSheet(
                        f'color:{_theme.THEME["fg_dim"]}; font-size:14px;'
                    )
                    self._calib_status_lbls[cell] = badge
                else:
                    btn.setEnabled(False)
                    btn.setStyleSheet(_theme.cell_button_qss(False))
                    btn.setToolTip(f'{cell} — not part of the extended calibration set')
                grid.addWidget(btn, row_i + 1, col_i + 1)
        # Badges sit under each capturable button via tooltip — keep compact.
        layout.addLayout(grid)

        # Captured-count strip
        self._calib_progress = QLabel('Captured 0 / 4 corners  •  0 extra cells')
        self._calib_progress.setAlignment(Qt.AlignCenter)
        self._calib_progress.setStyleSheet(
            f'color:{_theme.THEME["fg_muted"]}; font-size:11px; padding:4px;'
        )
        layout.addWidget(self._calib_progress)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._btn_calib_save = QPushButton('Save & Use')
        self._btn_calib_save.setStyleSheet(_theme.success_button_qss())
        self._btn_calib_save.setEnabled(False)
        self._btn_calib_save.clicked.connect(self._save_calibration)
        actions.addWidget(self._btn_calib_save, stretch=1)

        self._btn_calib_reset = QPushButton('Reset Capture')
        self._btn_calib_reset.setStyleSheet(_theme.ghost_button_qss())
        self._btn_calib_reset.clicked.connect(self._reset_calibration)
        actions.addWidget(self._btn_calib_reset)
        layout.addLayout(actions)

        self._calib_status = QLabel('Calibration: not loaded — using built-in defaults')
        self._calib_status.setAlignment(Qt.AlignCenter)
        self._calib_status.setStyleSheet(_theme.status_label_qss(_theme.THEME['fg_muted']))
        layout.addWidget(self._calib_status)
        return group

    def _capture_cell(self, cell: str):
        if not self._have_joint_state:
            self._set_status(f'Capture {cell} blocked: no live joint state', '#e0a000')
            return
        joints_deg = [math.degrees(self._current_joints_rad[n]) for n in JOINT_NAMES]
        self._pending_calibration[cell] = {'joints_deg': joints_deg}
        if cell in self._calib_status_lbls:
            self._calib_status_lbls[cell].setText('✓')
            self._calib_status_lbls[cell].setStyleSheet(
                f'color:{_theme.THEME["success"]}; font-size:14px; font-weight:bold;'
            )
        # Tint the captured button so the user sees green-on-green progress.
        btn = self._calib_btns.get(cell)
        if btn is not None:
            btn.setStyleSheet(_theme.success_button_qss())
        self._log(
            f'Captured {cell}: joints=[{", ".join(f"{j:+.2f}" for j in joints_deg)}]'
        )
        self._refresh_calibration_progress()

    # Legacy alias — old internal callers expected this name.
    def _capture_corner(self, corner: str):
        self._capture_cell(corner)

    def _refresh_calibration_progress(self):
        corners_done = sum(
            1 for c in REQUIRED_GRID_CORNERS if c in self._pending_calibration
        )
        extras_total = (
            set(EXTENDED_CALIBRATION_CELLS) - set(REQUIRED_GRID_CORNERS)
        )
        extras_done = sum(1 for c in extras_total if c in self._pending_calibration)
        self._calib_progress.setText(
            f'Captured {corners_done} / 4 corners  •  '
            f'{extras_done} / {len(extras_total)} extra cells'
        )
        all_corners = corners_done == len(REQUIRED_GRID_CORNERS)
        self._btn_calib_save.setEnabled(all_corners)
        if all_corners:
            if extras_done == 0:
                self._set_status(
                    'Corners ready — capture extras for better accuracy, or Save & Use',
                    '#79f7b2',
                )
            else:
                self._set_status(
                    f'Corners + {extras_done} extras ready — click Save & Use',
                    '#79f7b2',
                )

    def _save_calibration(self):
        if not all(c in self._pending_calibration for c in REQUIRED_GRID_CORNERS):
            self._set_status('Capture all 4 corners before saving', '#e0a000')
            return
        corner_joints = {
            c: self._pending_calibration[c]['joints_deg']
            for c in REQUIRED_GRID_CORNERS
        }
        extra_joints = {
            c: self._pending_calibration[c]['joints_deg']
            for c in self._pending_calibration
            if c not in REQUIRED_GRID_CORNERS
        }
        profile = self._profile_combo.currentData()
        try:
            apply_calibration(joints_deg=corner_joints, extra_joints_deg=extra_joints)
            path = save_calibration(profile_name=profile)
            set_active_profile(profile)
        except Exception as e:
            self._set_status(f'Calibration save failed: {e}', '#ff4444')
            self._log(f'Calibration save failed: {e}')
            return
        self._mark_calibration_active(path, profile)
        self._refresh_profile_labels()
        self._set_status(
            f'Saved & active: {self._profile_combo.currentText()} '
            f'({len(corner_joints)} corners + {len(extra_joints)} extras)',
            '#00cc88',
        )
        self._log(
            f'Calibration saved to {path}: corners={list(corner_joints)}, '
            f'extras={list(extra_joints)} — profile {profile}'
        )

    def _load_selected_profile(self):
        profile = self._profile_combo.currentData()
        loaded, msg = load_calibration(profile_name=profile)
        if not loaded:
            self._set_status(
                f'{self._profile_combo.currentText()}: not saved yet — capture corners first',
                '#e0a000',
            )
            self._log(f'Load {profile} skipped: {msg}')
            return
        try:
            set_active_profile(profile)
        except Exception as e:
            self._log(f'Could not mark {profile} active: {e}')
        self._reset_calibration(silent=True)
        self._mark_calibration_active(msg, profile)
        self._set_status(
            f'Loaded {self._profile_combo.currentText()} — grid moves use saved corners',
            '#00cc88',
        )
        self._log(f'Loaded calibration from {msg}')

    def _refresh_profile_labels(self, *_args):
        saved = list_saved_profiles()
        for i, name in enumerate(CALIBRATION_PROFILES):
            base = f'Robot {i + 1}'
            label = f'{base}  saved' if saved[name] else f'{base}  empty'
            self._profile_combo.setItemText(i, label)

    def _mark_calibration_active(self, path: str, profile: str):
        label = dict(zip(
            CALIBRATION_PROFILES,
            [f'Robot {i+1}' for i in range(len(CALIBRATION_PROFILES))]
        )).get(profile, profile)
        self._calib_status.setText(f'Calibration active ({label}): {path}')
        self._calib_status.setStyleSheet(_theme.status_label_qss(_theme.THEME['success']))

    def _reset_calibration(self, silent: bool = False):
        self._pending_calibration = {}
        for cell, lbl in self._calib_status_lbls.items():
            lbl.setText('◌')
            lbl.setStyleSheet(
                f'color:{_theme.THEME["fg_dim"]}; font-size:14px;'
            )
        for cell, btn in self._calib_btns.items():
            btn.setStyleSheet(_theme.ghost_button_qss())
        self._btn_calib_save.setEnabled(False)
        if hasattr(self, '_calib_progress'):
            self._calib_progress.setText(
                f'Captured 0 / {len(REQUIRED_GRID_CORNERS)} corners  •  '
                f'0 / {len(EXTENDED_CALIBRATION_CELLS) - len(REQUIRED_GRID_CORNERS)} extra cells'
            )
        if not silent:
            self._set_status('Calibration capture cleared', '#a9b8c4')

    # ── Camera panel ─────────────────────────────────────────────────────────

    def _build_camera_panel(self) -> QGroupBox:
        group = self._group('Camera Feed')
        layout = group.layout()

        # ── Controls row (camera select + start) ──────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._cam_combo = QComboBox()
        self._cam_combo.addItem('Webcam',          0)
        self._cam_combo.addItem('RealSense D435i', 1)
        self._cam_combo.addItem('Other',           2)
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

        self._frame_counter += 1
        # MediaPipe Hands runs every 3rd frame (~10 Hz hand detection).
        # Red detection runs every 2nd frame (~15 Hz fruit detection).
        # All frames still display, draw the grid overlay, and publish.
        run_hands = (self._frame_counter % 3 == 0)
        run_detect = (self._frame_counter % 2 == 0)

        # Only run detection/safety once the grid is fully defined. The safety
        # node also gets frames only while this locked zone exists, so it cannot
        # keep using an old grid after the operator clears or redefines it.
        if len(self._grid_pts) == 4:
            if run_detect:
                self._detect_in_grid(frame)
            if run_hands:
                # Safety interlock — hand detection runs on every Nth frame
                if self._check_hand_in_zone(frame):
                    self._safety_clear_frames = 0
                    self._on_hand_detected()
                else:
                    if self._safety_paused:
                        self._safety_clear_frames += 1
                        if self._safety_clear_frames >= CLEAR_FRAMES_BEFORE_RESUME:
                            self._on_zone_clear()
            if self._frame_counter - self._last_zone_publish_frame >= 15:
                self._bridge_node.publish_zone(self._grid_pts)
                self._last_zone_publish_frame = self._frame_counter
        else:
            self._detected_fruits = []

        # Draw grid overlay / in-progress dots on top
        self._draw_grid_overlay(frame)

        # Publish frame to safety_node every 2nd frame, but only once the grid
        # is locked — the interlock is not active without a defined zone so
        # there is no point sending frames before the operator has set one up.
        if len(self._grid_pts) == 4 and self._frame_counter % 2 == 0:
            self._bridge_node.publish_frame(frame)

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

    # ── Safety interlock ─────────────────────────────────────────────────────

    def _check_hand_in_zone(self, frame: np.ndarray) -> bool:
        """
        Run MediaPipe Hands on frame and return True if any landmark falls
        inside the 4-corner cutting-zone polygon (the defined grid).
        Draws landmarks and zone highlight onto frame in-place.
        Returns False if MediaPipe isn't installed or grid isn't defined.
        """
        if self._hands_mp is None or len(self._grid_pts) != 4:
            return False

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands_mp.process(rgb)

        # Build a polygon from the 4 grid corners for point-in-polygon test
        poly = np.array(self._grid_pts, dtype=np.int32)  # TL, TR, BR, BL

        hand_in_zone = False
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                pts = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_lms.landmark
                ]
                inside = [
                    pt for pt in pts
                    if cv2.pointPolygonTest(poly, pt, False) >= 0
                ]
                palm_pts = [pts[i] for i in HAND_ZONE_PALM_INDICES]
                palm_inside_count = sum(
                    1 for pt in palm_pts
                    if cv2.pointPolygonTest(poly, pt, False) >= 0
                )
                palm_center = (
                    int(sum(p[0] for p in palm_pts) / len(palm_pts)),
                    int(sum(p[1] for p in palm_pts) / len(palm_pts)),
                )
                palm_inside = cv2.pointPolygonTest(poly, palm_center, False) >= 0

                # Only trip when the hand body is actually in the locked grid,
                # not when MediaPipe sees fingertips or a skeleton near it.
                if palm_inside or (
                    len(inside) >= HAND_ZONE_MIN_LANDMARKS
                    and palm_inside_count >= 2
                ):
                    hand_in_zone = True
                    _mp_drawing.draw_landmarks(
                        frame, hand_lms,
                        _mp_hands.HAND_CONNECTIONS,
                        _mp_drawing_styles.get_default_hand_landmarks_style(),
                        _mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    break

        return hand_in_zone

    def _on_hand_detected(self, source: str = 'local camera'):
        if self._safety_paused:
            return
        self._safety_paused = True
        self._safety_clear_frames = 0
        self._mover_node.cancel()          # abort any in-flight trajectory
        self._dashboard.pause()            # pause External Control / UR program
        self._moving = False
        self._safety_label.setText('⚠ SAFETY: HAND IN ZONE — ROBOT PAUSED')
        self._safety_label.setStyleSheet(
            'background:#5a0000; color:#ff6060; font-size:12px; font-weight:bold;'
            'padding:6px; border:2px solid #ff3030; border-radius:6px;'
        )
        self._log(f'SAFETY INTERLOCK: hand detected by {source} — trajectory cancelled and robot paused')

    def _on_zone_clear(self):
        if not self._safety_paused:
            return
        self._safety_paused = False
        self._safety_clear_frames = 0
        self._mover_node.clear_cancel()
        self._dashboard.resume()
        self._safety_label.setText('🛡 Safety: zone clear — resuming motion')
        self._safety_label.setStyleSheet(
            'background:#0a1f10; color:#4de87a; font-size:12px; font-weight:bold;'
            'padding:6px; border:1px solid #2a7a40; border-radius:6px;'
        )
        if self._resume_after_safety is not None:
            self._log('SAFETY INTERLOCK: zone clear — resuming interrupted motion')
            self._resume_interrupted_motion()
        else:
            self._log('SAFETY INTERLOCK: zone clear — ready')

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_in_grid(self, frame: np.ndarray):
        """
        Detect red objects inside the manually defined camera grid.

        Method:
          1. Build a perspective transform H from the 4 clicked corner points
             that maps frame pixels → grid coordinates (0..7, 0..4).
          2. For each red contour:
             a. Use contour moments for the fruit origin/centroid.
             b. Skip if centroid maps outside the grid (0 ≤ col < 7, 0 ≤ row < 4).
             c. 'origin cell'  = the grid cell the centroid lands in.
             d. 'covered cells' = all cells touched by the bounding-box corners.
          3. Draws bounding box + label on the frame (annotations are visible in GUI).
          4. Stores results in self._detected_fruits for use by _cut_detected_fruit.

        Only red is detected (no green / yellow / orange) as requested.
        Nothing is detected until the grid is fully defined (4 points clicked).
        """
        tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in self._grid_pts]
        n_cols, n_rows = len(GRID_COLS), len(GRID_ROWS)

        src = np.float32([tl, tr, br, bl])
        dst = np.float32([[0, 0], [n_cols, 0], [n_cols, n_rows], [0, n_rows]])
        H = cv2.getPerspectiveTransform(src, dst)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        # Restrict detection to inside the user-defined quad — anything outside
        # (face, walls, hands, etc.) is masked out at the source.
        quad_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(quad_mask,
                           np.array(self._grid_pts, dtype=np.int32), 255)

        # HSV gates tuned to reject skin: skin sits around H 0..20, S 30..130.
        # Real ripe-red fruit pegs S above ~170, so the high S floor here
        # rejects faces/hands/arms while still catching tomatoes/apples.
        profiles = [
            ('Red',  (0, 80, 255), [
                (np.array([0,   170, 90]),   np.array([8,   255, 255])),
                (np.array([172, 170, 90]),   np.array([180, 255, 255])),
            ]),
        ]

        # Minimum geometric thresholds — also help filter spurious blobs.
        MIN_AREA_PX     = 400    # ignore anything smaller than ~20×20 px
        MIN_SOLIDITY    = 0.70   # area / convex-hull area; skin patches are wispy

        def cell_name(c, r):
            return f"{GRID_COLS[c]}{GRID_ROWS[r]}"

        info_lines = []
        fruits = []

        for label, bgr, ranges in profiles:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask = cv2.bitwise_and(mask, quad_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA_PX:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                if hull_area > 0 and (area / hull_area) < MIN_SOLIDITY:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 12 or h < 12:
                    continue
                moments = cv2.moments(cnt)
                if abs(moments['m00']) < 1e-6:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                rect = cv2.minAreaRect(cnt)
                shape_angle = float(rect[2])

                # Origin cell from centroid
                uv = cv2.perspectiveTransform(
                    np.float32([[[cx, cy]]]), H
                )[0][0]
                grid_x, grid_y = float(uv[0]), float(uv[1])
                col_i, row_i = int(uv[0]), int(uv[1])
                if not (0 <= col_i < n_cols and 0 <= row_i < n_rows):
                    continue
                origin_cell = cell_name(col_i, row_i)
                _rp_col, _rp_row, _u, _v, robot_target = _camera_grid_to_robot_target(
                    grid_x, grid_y
                )

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
                    tag = f'{label} {origin_cell}->{robot_target} spans:{",".join(covered)}'
                else:
                    tag = f'{label} [{origin_cell}->{robot_target}]'
                cv2.putText(frame, tag, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                cv2.drawMarker(frame, (cx, cy), bgr,
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                info_lines.append(
                    f'{label}  origin: {origin_cell}  cut: {robot_target}  '
                    f'grid: ({grid_x:.2f}, {grid_y:.2f})  '
                    f'cells ({len(covered)}): {", ".join(covered)}'
                )
                fruits.append({
                    'label': label,
                    'origin': origin_cell,
                    'robot_target': robot_target,
                    'grid_xy': (grid_x, grid_y),
                    'frame_xy': (cx, cy),
                    'covered': covered,
                    'area_px': float(area),
                    'shape_angle_deg': shape_angle,
                })

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
            self._detected_fruits = []
            self._clear_resume_motion()
            if hasattr(self, '_bridge_node'):
                self._bridge_node.clear_zone()
            if self._safety_paused:
                self._on_zone_clear()
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
        self._detected_fruits = []
        self._clear_resume_motion()
        self._selecting_grid = False
        self._btn_define_grid.setChecked(False)
        self._cam_label.setCursor(Qt.ArrowCursor)
        self._grid_status.setText('No grid defined')
        self._grid_status.setStyleSheet('color:#8195a3; font-size:11px; padding-left:6px;')
        self._set_status('Grid cleared — safety standby', '#aaaaaa')
        if hasattr(self, '_bridge_node'):
            self._bridge_node.clear_zone()
        if hasattr(self, '_detect_info'):
            self._detect_info.setPlainText('')
        # If interlock was active, release it when grid is cleared
        if self._safety_paused:
            self._on_zone_clear()
        self._safety_label.setText('🛡 Safety: standby — define grid to activate interlock')
        self._safety_label.setStyleSheet(
            'background:#101a22; color:#7a8fa0; font-size:12px; font-weight:bold;'
            'padding:6px; border:1px solid #2f4654; border-radius:6px;'
        )

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
            # Publish zone to safety_node
            self._bridge_node.publish_zone(self._grid_pts)
            self._last_zone_publish_frame = self._frame_counter
            if self._hands_mp is not None:
                self._safety_label.setText('🛡 Safety: active — monitoring cutting zone')
                self._safety_label.setStyleSheet(
                    'background:#0a1f10; color:#4de87a; font-size:12px; font-weight:bold;'
                    'padding:6px; border:1px solid #2a7a40; border-radius:6px;'
                )
            else:
                self._safety_label.setText('⚠ Safety: mediapipe not installed — interlock disabled')
                self._safety_label.setStyleSheet(
                    'background:#1f1a0a; color:#e0a000; font-size:12px; font-weight:bold;'
                    'padding:6px; border:1px solid #7a5a00; border-radius:6px;'
                )

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

        After selection (== 4 points): draws the full 7-column × 4-row bilinear grid
        using perspective-correct row and column lines, labels every cell centre (A1–G4),
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
        n_cols, n_rows = len(GRID_COLS), len(GRID_ROWS)
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
                lbl = f"{GRID_COLS[col]}{GRID_ROWS[row]}"
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
                lambda joints: self._joints_sig.emit(joints),
                lambda running: self._program_sig.emit(running),
                lambda speed: self._speed_sig.emit(speed),
            )
            self._mover_node = MoverNode()
            self._bridge_node = CameraBridgeNode(self._on_safety_node_detection)
            # Spin JointStateNode + bridge node continuously in a shared executor
            exe = rclpy.executors.MultiThreadedExecutor()
            exe.add_node(self._js_node)
            exe.add_node(self._bridge_node)
            threading.Thread(target=exe.spin, daemon=True).start()
            self._bridge_node.clear_zone()
            self._log('ROS2 nodes started')

            self._refresh_profile_labels()
            active = get_active_profile()
            if active is not None:
                for i, name in enumerate(CALIBRATION_PROFILES):
                    if name == active:
                        self._profile_combo.setCurrentIndex(i)
                        break
                loaded, msg = load_calibration(profile_name=active)
                if loaded:
                    self._mark_calibration_active(msg, active)
                    self._log(f'Auto-loaded calibration: {msg}')
                else:
                    self._log(f'Active profile {active!r} not on disk: {msg}')
            else:
                self._log('No active calibration profile — using built-in defaults')
        except Exception as e:
            self._log(f'ROS2 init failed: {e}')

    def _on_safety_node_detection(self, msg: Bool):
        """Called from the ROS spin thread — route to main thread via signal."""
        self._safety_sig.emit(bool(msg.data))

    def _handle_safety_node_detection(self, hand_detected: bool):
        if hand_detected:
            self._safety_clear_frames = 0
            if len(self._grid_pts) == 4:
                self._on_hand_detected(source='safety node')
        elif self._safety_paused:
            self._safety_clear_frames += 1
            if self._safety_clear_frames >= CLEAR_FRAMES_BEFORE_RESUME:
                self._on_zone_clear()

    def closeEvent(self, event):
        self._camera.stop()
        self._dashboard.close()
        if self._hands_mp is not None:
            self._hands_mp.close()
        try:
            self._js_node.destroy_node()
            self._mover_node.destroy_node()
            self._bridge_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description='FruitNinja UR3e operator GUI')
    parser.add_argument('--robot-ip', default='192.168.0.194',
                        help='IP address of the UR3e controller (default: 192.168.0.194)')
    parsed, remaining = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + remaining)
    app.setStyle('Fusion')
    win = MainWindow(robot_ip=parsed.robot_ip)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
