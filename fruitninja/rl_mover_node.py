#!/usr/bin/env python3
"""
rl_mover_node.py — ROS 2 node for live SAC (Soft Actor-Critic) policy inference
on the FruitNinja UR3e cutting robot.

WHAT THIS NODE DOES
====================
RLMoverNode replaces the MoveIt-based MoverNode with a learned reinforcement-learning
policy (SAC, trained via Stable Baselines 3) that outputs joint-position deltas at
20 Hz.  Each inference tick reads the current joint state and end-effector pose from
TF2, builds a 15-dimensional observation vector, queries the SAC policy, clips the
resulting action for safety, and publishes a JointTrajectory command directly to the
scaled joint trajectory controller — bypassing MoveIt entirely.

OBSERVATION VECTOR (15-D)
--------------------------
  [0:6]   self.q       — current joint positions (rad), 6 joints
  [6:12]  self.q_dot   — current joint velocities (rad/s), 6 joints
  [12:15] p_ee         — end-effector position (m) from TF2 (tool0 in base_link)
  [15:18] rpy          — end-effector orientation (roll/pitch/yaw, rad) from TF2
  [18:21] self.target  — Cartesian target position (m) from /fruit_target

TOPICS SUBSCRIBED
------------------
  /joint_states           (sensor_msgs/JointState)   — live joint angles & velocities
  /fruit_target           (geometry_msgs/PointStamped) — 3-D target from vision_node
  /estop                  (std_msgs/Bool)             — safety kill-switch from vision_node

TOPICS PUBLISHED
-----------------
  /scaled_joint_trajectory_controller/joint_trajectory
                          (trajectory_msgs/JointTrajectory) — position setpoints at 20 Hz
  /rl_status              (std_msgs/String)            — human-readable telemetry string

CONNECTION TO VISION_NODE
==========================
vision_node publishes:
  • /fruit_target  (PointStamped) — 3-D centroid of the detected fruit to cut
  • /estop         (Bool)         — True when a safety condition is detected

When /fruit_target arrives, self.running becomes True and the inference loop starts
sending joint commands toward the target.  When /estop arrives with data=True, all
motion is halted immediately.

SAFE DEPLOYMENT GUIDE
======================
1. URSim validation first
   Run against URSim (PolyScope simulator) before touching the real robot:
     ros2 run fruitninja rl_mover_node --ros-args -p model_path:=sac_fruitninja -p velocity_scale:=0.05
   Observe /rl_status output and visually verify joint trajectories are smooth.

2. velocity_scale parameter (default 0.1)
   The raw SAC action (joint-position delta) is multiplied by velocity_scale before
   being sent to the controller.  At 0.1 (10%) the robot moves slowly enough for a
   human operator to intervene.  Do NOT set above 0.3 without extensive validation.
   Example:
     ros2 run fruitninja rl_mover_node --ros-args -p velocity_scale:=0.05

3. DEVIATION_THRESHOLD (0.1 rad ≈ 5.7°)
   Each inference cycle the raw delta is clipped to ±DEVIATION_THRESHOLD per joint.
   This caps the maximum single-step joint movement regardless of what the policy
   outputs, preventing runaway motion from a badly generalised policy.

4. E-STOP
   Publish `True` to /estop at any time to halt all motion:
     ros2 topic pub /estop std_msgs/Bool "{data: true}" -1
   After an E-STOP, re-publish /fruit_target to resume (if safe to do so).

5. model_path parameter
   The SAC model is loaded with `SAC.load(model_path)`.  The path may be absolute or
   relative to the working directory.  If the file is not found the node starts in a
   safe no-op state and logs an error.
"""

import rclpy
from rclpy.node import Node

import numpy as np
import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from stable_baselines3 import SAC
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

# ── Constants ──────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Maximum allowed single-step joint delta (radians).  Actions exceeding this are
# clipped before being sent to the controller.
DEVIATION_THRESHOLD = 0.1  # rad  (~5.7°)


# ── Node ───────────────────────────────────────────────────────────────────────

class RLMoverNode(Node):
    """
    ROS 2 node that drives the UR3e using a pre-trained SAC policy.

    The node runs at 20 Hz and on each tick:
      1. Reads the current joint state and EE pose.
      2. Builds an observation vector.
      3. Queries the SAC policy for a joint-position delta.
      4. Clips the delta for safety.
      5. Publishes a JointTrajectory command.
    """

    def __init__(self):
        super().__init__('rl_mover_node')

        # ── ROS parameters ────────────────────────────────────────────────────
        self.declare_parameter('velocity_scale', 0.1)
        self.declare_parameter('model_path', 'sac_fruitninja')

        self.velocity_scale = self.get_parameter('velocity_scale').get_parameter_value().double_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # ── Load SAC model ────────────────────────────────────────────────────
        if not _HAS_SB3:
            self.get_logger().error(
                'stable_baselines3 is not installed — '
                'install it with: pip install stable-baselines3'
            )
            self.model = None
        else:
            try:
                self.model = SAC.load(model_path)
                self.get_logger().info(f'SAC model loaded from "{model_path}"')
            except FileNotFoundError:
                self.get_logger().error(
                    f'SAC model file not found: "{model_path}". '
                    'Node will do nothing until a model is loaded.'
                )
                self.model = None

        # ── State ─────────────────────────────────────────────────────────────
        self.q       = np.zeros(6)   # joint positions (rad)
        self.q_dot   = np.zeros(6)   # joint velocities (rad/s)
        self.target  = np.zeros(3)   # Cartesian target (m)

        self.estop   = False
        self.running = False

        # ── TF2 ───────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(JointState,    '/joint_states',  self._js_cb,     10)
        self.create_subscription(PointStamped,  '/fruit_target',  self._target_cb, 10)
        self.create_subscription(Bool,          '/estop',         self._estop_cb,  10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10,
        )
        self.status_pub = self.create_publisher(String, '/rl_status', 10)

        # ── Inference timer (20 Hz) ───────────────────────────────────────────
        self.create_timer(0.05, self._inference_loop)

        # ── Startup log ───────────────────────────────────────────────────────
        model_status = 'LOADED' if self.model is not None else 'NOT LOADED (no-op)'
        self.get_logger().info(
            f'RLMoverNode started | '
            f'velocity_scale={self.velocity_scale} | '
            f'model_path="{model_path}" | '
            f'model={model_status}'
        )

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        """Update self.q and self.q_dot from the /joint_states message."""
        name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in name_to_idx:
                idx = name_to_idx[name]
                self.q[idx]     = pos
                self.q_dot[idx] = vel

    def _target_cb(self, msg: PointStamped):
        """Update the Cartesian target and start the inference loop."""
        self.target  = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.running = True
        self.get_logger().info(
            f'New target received: x={msg.point.x:.3f}  '
            f'y={msg.point.y:.3f}  z={msg.point.z:.3f}'
        )

    def _estop_cb(self, msg: Bool):
        """Handle an E-STOP signal from vision_node (or any other publisher)."""
        if msg.data:
            self.estop   = True
            self.running = False
            self.get_logger().warn('E-STOP received from vision node — all motion halted')

    # ── TF2 helper ───────────────────────────────────────────────────────────

    def _get_ee_pose_from_tf(self):
        """
        Look up the current end-effector pose (tool0 in base_link) from TF2.

        Returns:
            (p_ee, rpy) where p_ee is np.ndarray([x, y, z]) in metres and
            rpy is np.ndarray([roll, pitch, yaw]) in radians.
            On any TF2 exception returns (np.zeros(3), np.zeros(3)).
        """
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                'base_link',
                'tool0',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            t = tf_stamped.transform.translation
            p_ee = np.array([t.x, t.y, t.z])

            r = tf_stamped.transform.rotation
            q = [r.x, r.y, r.z, r.w]
            rpy = RLMoverNode.quaternion_to_rpy(q)

            return p_ee, rpy
        except Exception as exc:
            self.get_logger().debug(f'TF2 lookup failed: {exc}')
            return np.zeros(3), np.zeros(3)

    @staticmethod
    def quaternion_to_rpy(q):
        """
        Convert a quaternion [x, y, z, w] to roll-pitch-yaw (radians).

        Uses the standard ZYX convention:
          roll  = atan2(2*(w*x + y*z),  1 - 2*(x² + y²))
          pitch = asin(clip(2*(w*y - z*x), -1, 1))
          yaw   = atan2(2*(w*z + x*y),  1 - 2*(y² + z²))
        """
        x, y, z, w = q
        roll  = np.arctan2(2.0 * (w * x + y * z),
                           1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw   = np.arctan2(2.0 * (w * z + x * y),
                           1.0 - 2.0 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    # ── Inference loop ───────────────────────────────────────────────────────

    def _inference_loop(self):
        """
        20 Hz inference callback.

        Skipped when: E-STOP is active, running is False, or no model is loaded.
        On each tick:
          1. Build a 21-D observation vector.
          2. Query SAC policy (deterministic).
          3. Scale and clip the action for safety.
          4. Publish a single-point JointTrajectory 50 ms in the future.
          5. Publish a human-readable status string.
          6. Stop when distance to target drops below 5 mm.
        """
        if self.estop or not self.running or self.model is None:
            return

        # ── Observation ───────────────────────────────────────────────────────
        p_ee, rpy = self._get_ee_pose_from_tf()
        obs = np.concatenate([
            self.q,       # 6  joint positions (rad)
            self.q_dot,   # 6  joint velocities (rad/s)
            p_ee,         # 3  EE position (m)
            rpy,          # 3  EE orientation (rad)
            self.target,  # 3  target position (m)
        ]).astype(np.float32)  # total: 21-D

        # ── Policy query ──────────────────────────────────────────────────────
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        action *= self.velocity_scale

        # ── Safety clip ───────────────────────────────────────────────────────
        deviation = np.abs(action)
        if np.any(deviation > DEVIATION_THRESHOLD):
            self.get_logger().warn(
                f'Action clipped: max deviation {deviation.max():.4f} rad '
                f'(threshold {DEVIATION_THRESHOLD} rad)'
            )
            action = np.clip(action, -DEVIATION_THRESHOLD, DEVIATION_THRESHOLD)

        proposed = self.q + action

        # ── Publish trajectory ────────────────────────────────────────────────
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions  = proposed.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start.sec      = 0
        pt.time_from_start.nanosec  = 50_000_000   # 50 ms ahead

        msg.points = [pt]
        self.traj_pub.publish(msg)

        # ── Status telemetry ──────────────────────────────────────────────────
        dist = float(np.linalg.norm(p_ee - self.target))
        status_msg = String()
        status_msg.data = (
            f'dist={dist:.4f}m  '
            f'q={[f"{v:.2f}" for v in np.degrees(self.q)]}'
        )
        self.status_pub.publish(status_msg)

        # ── Target-reached check ──────────────────────────────────────────────
        if dist < 0.005:
            self.running = False
            self.get_logger().info(
                f'Target reached (dist={dist:.4f} m < 0.005 m) — stopping inference loop'
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RLMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
