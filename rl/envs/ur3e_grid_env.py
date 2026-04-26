#!/usr/bin/env python3
"""
UR3eGridEnv
===========
Gymnasium environment for the FruitNinja UR3e cutting-board task.

Observation space (dim=21):
  q[6]         — joint positions (rad)
  q_dot[6]     — joint velocities (rad/s)
  p_ee[3]      — end-effector position in robot base frame (m)
  tool_z[3]    — tool z-axis direction (R[:,2]) — used for orientation control
  p_target[3]  — current target cell centre in robot base frame (m)

Action space (dim=3):
  Cartesian delta (dx, dy, dz), clipped to [-1, 1].
  Scaled by EE_STEP_MAX (10 mm/step). IK maps to joint space.
  Null-space of position IK is used to drive tool z-axis toward [0,0,-1].

Board geometry — derived from measured corner positions in the robot base
frame.  The board is flat (z=0.0875 m) and symmetric in x.

  P_A1 = (+0.3124, -0.4549,  0.0875)  — far-left
  P_A4 = (+0.3130, -0.3493,  0.0875)  — near-left
  P_N1 = (-0.3124, -0.4549,  0.0875)  — far-right
  P_N4 = (-0.3130, -0.3493,  0.0875)  — near-right

Episode structure:
  Each episode visits n_targets (default 5) randomly selected, unique cells
  in sequence.  The arm advances to the next cell when it reaches within
  SUCCESS_DIST (1 mm) of the current target.  The episode ends when all
  targets are reached, the floor is violated, or max_steps is exhausted.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Measured board corners in robot base frame ─────────────────────────────────
_P_A1 = np.array([ 0.3124, -0.4549,  0.0875], dtype=np.float64)  # u=0, v=0
_P_A4 = np.array([ 0.3130, -0.3493,  0.0875], dtype=np.float64)  # u=0, v=1
_P_N1 = np.array([-0.3124, -0.4549,  0.0875], dtype=np.float64)  # u=1, v=0
_P_N4 = np.array([-0.3130, -0.3493,  0.0875], dtype=np.float64)  # u=1, v=1

NUM_COLS = 14
NUM_ROWS = 4


def cell_centre(col: int, row: int) -> np.ndarray:
    """
    Return the Cartesian centre of a grid cell in robot base frame (m).

    Uses bilinear interpolation of the 4 measured corner positions.

    Parameters
    ----------
    col : 0-based column index (0 = A … 13 = N)
    row : 0-based row index   (0 = row-1 … 3 = row-4)
    """
    u = col / (NUM_COLS - 1)
    v = row / (NUM_ROWS - 1)
    return (
        (1 - u) * (1 - v) * _P_A1
        +      u * (1 - v) * _P_N1
        + (1 - u) *      v * _P_A4
        +      u  *      v * _P_N4
    ).astype(np.float32)


# ── Robot configuration ────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Real robot home from /joint_states: shoulder lift = -π/2, others ≈ 0.
HOME_RAD = np.array([0.0, -math.pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# UR3e joint limits (radians). Q_MAX[1]=0 is a task constraint (keep shoulder
# below horizontal so the arm reaches down toward the table).
Q_MIN = np.array([-2*math.pi, -2*math.pi, -2*math.pi,
                  -2*math.pi, -2*math.pi, -2*math.pi], dtype=np.float32)
Q_MAX = np.array([ 2*math.pi,  0.0,        2*math.pi,
                   2*math.pi,  2*math.pi,  2*math.pi], dtype=np.float32)

# Coarse approach: up to 10 mm per step when far from target.
EE_STEP_MAX  = 0.010   # 10 mm
# Fine approach: up to 2 mm per step when within FINE_DIST of target.
EE_STEP_FINE = 0.002   # 2 mm
FINE_DIST    = 0.020   # switch to fine mode within 2 cm

# Damped least-squares IK damping factor.
IK_DAMPING = 0.05

# Number of Newton-Raphson iterations per IK solve.
IK_ITERS = 20

# Per-joint maximum speed (rad/s) from UR3e tech sheet.
# Base/Shoulder/Elbow: ±180°/s = ±π rad/s
# Wrist 1/2/3:         ±360°/s = ±2π rad/s
QDOT_MAX_PER_JOINT = np.array([
    math.pi, math.pi, math.pi,
    2 * math.pi, 2 * math.pi, 2 * math.pi,
], dtype=np.float32)

# Success threshold: arm must reach within 1 mm of target centre.
SUCCESS_DIST = 0.001   # 1 mm

# Null-space orientation gain: scales the secondary orientation task.
ORI_GAIN = 0.05

# Orientation penalty weight in reward.
ORI_PEN_WEIGHT = 0.15


# ── DH / forward-kinematics helpers ───────────────────────────────────────────

# UR3e DH parameters (Craig convention, metres)
# [a, d, alpha, theta_offset]
_UR3E_DH = [
    (0.0,      0.15185,  math.pi / 2,  0.0),
    (-0.24355, 0.0,      0.0,          0.0),
    (-0.2132,  0.0,      0.0,          0.0),
    (0.0,      0.13105,  math.pi / 2,  0.0),
    (0.0,      0.08535, -math.pi / 2,  0.0),
    (0.0,      0.0921,   0.0,          0.0),
]


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Return the 4×4 homogeneous transform for one DH link."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=np.float64)


def forward_kinematics(q: np.ndarray):
    """Return 4×4 EE transform for joint angles q (radians)."""
    T = np.eye(4, dtype=np.float64)
    for i, (a, d, alpha, offset) in enumerate(_UR3E_DH):
        T = T @ _dh_transform(a, d, alpha, q[i] + offset)
    return T


def jacobian(q: np.ndarray) -> np.ndarray:
    """
    Numerical Jacobian (6×6) via central differences.
    Rows 0–2: linear velocity; rows 3–5: angular velocity.
    """
    eps = 1e-6
    J  = np.zeros((6, 6), dtype=np.float64)
    T0 = forward_kinematics(q)
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        Tp = forward_kinematics(qp)
        Tm = forward_kinematics(qm)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * eps)
        S  = dR @ T0[:3, :3].T
        J[3, i] = S[2, 1]
        J[4, i] = S[0, 2]
        J[5, i] = S[1, 0]
    return J


# ── Environment ────────────────────────────────────────────────────────────────

class UR3eGridEnv(gym.Env):
    """
    Gym environment for the FruitNinja cutting-board grid task.

    The agent controls EE Cartesian displacement (IK-mapped to joints) to
    visit n_targets randomly chosen cells per episode in sequence.
    """

    metadata = {'render_modes': []}

    obs_dim    = 21   # q(6) + q_dot(6) + p_ee(3) + tool_z(3) + p_target(3)
    action_dim = 3    # Cartesian delta (dx, dy, dz)

    def __init__(self, dt: float = 0.02, max_steps: int = 1000, n_targets: int = 5,
                 fixed_target: np.ndarray = None):
        super().__init__()

        self.dt           = dt
        self.max_steps    = max_steps
        self.fixed_target = fixed_target.astype(np.float32) if fixed_target is not None else None
        self.n_targets    = 1 if fixed_target is not None else n_targets

        obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # State
        self._q              = HOME_RAD.copy()
        self._q_dot          = np.zeros(6, dtype=np.float32)
        self._prev_q_dot     = np.zeros(6, dtype=np.float32)
        self._target         = np.zeros(3, dtype=np.float32)
        self._target_queue   = []
        self._target_idx     = 0
        self._targets_reached = 0
        self._step_n         = 0

        self.np_random = np.random.default_rng()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Generate target queue: fixed point or n_targets unique random cells
        if self.fixed_target is not None:
            noise = self.np_random.uniform(-0.005, 0.005, size=3).astype(np.float32)
            self._target_queue = [self.fixed_target + noise]
        else:
            used = set()
            self._target_queue = []
            while len(self._target_queue) < self.n_targets:
                col = int(self.np_random.integers(0, NUM_COLS))
                row = int(self.np_random.integers(0, NUM_ROWS))
                if (col, row) not in used:
                    used.add((col, row))
                    noise = self.np_random.uniform(-0.005, 0.005, size=3).astype(np.float32)
                    self._target_queue.append(cell_centre(col, row) + noise)

        self._target_idx      = 0
        self._targets_reached = 0
        self._target          = self._target_queue[0]

        self._q          = HOME_RAD.copy().astype(np.float32)
        self._q_dot      = np.zeros(6, dtype=np.float32)
        self._prev_q_dot = np.zeros(6, dtype=np.float32)
        self._step_n     = 0

        return self._get_obs(), {}

    # ── Floor check: ALL link frames must have z >= 0 ─────────────────────────

    def _all_links_above_floor(self) -> bool:
        """Return True if every joint-frame origin has z >= 0 (robot base plane)."""
        T = np.eye(4, dtype=np.float64)
        for i, (a, d, alpha, offset) in enumerate(_UR3E_DH):
            T = T @ _dh_transform(a, d, alpha, float(self._q[i]) + offset)
            if T[2, 3] < 0.0:
                return False
        return True

    # ── IK with null-space orientation control ────────────────────────────────

    def _ik_step(self, dp: np.ndarray) -> np.ndarray:
        """
        Iterative damped least-squares IK (IK_ITERS Newton-Raphson steps)
        with null-space orientation control (tool z → [0,0,-1]).

        dp      : desired Cartesian EE displacement (3,) in metres.
        Returns : dq (6,) clamped to per-joint speed limits for this timestep.
        """
        q0  = self._q.astype(np.float64)
        T0  = forward_kinematics(q0)
        p_target = T0[:3, 3] + dp      # absolute EE position to reach

        desired_z = np.array([0.0, 0.0, -1.0])
        q = q0.copy()

        for _ in range(IK_ITERS):
            T      = forward_kinematics(q)
            J_full = jacobian(q)
            J_pos  = J_full[:3, :]    # 3×6
            J_ang  = J_full[3:, :]    # 3×6

            err_pos = p_target - T[:3, 3]
            if np.linalg.norm(err_pos) < 1e-6:
                break

            # Primary: position IK (damped pseudo-inverse)
            A        = J_pos @ J_pos.T + IK_DAMPING ** 2 * np.eye(3)
            J_pos_pi = J_pos.T @ np.linalg.solve(A, np.eye(3))   # 6×3
            dq_iter  = J_pos_pi @ err_pos                          # 6,

            # Secondary: orientation in null-space
            e_ori   = desired_z - T[:3, 2]
            N       = np.eye(6) - J_pos_pi @ J_pos
            dq_iter += N @ (ORI_GAIN * J_ang.T @ e_ori)

            q = np.clip(q + dq_iter, Q_MIN, Q_MAX)

        # Total joint change this timestep, clamped to speed limits
        dq_total = q - q0
        dq_max   = (QDOT_MAX_PER_JOINT * self.dt).astype(np.float64)
        dq_total = np.clip(dq_total, -dq_max, dq_max)
        return dq_total.astype(np.float32)

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Adaptive step size: fine control within FINE_DIST of target
        T_cur     = forward_kinematics(self._q.astype(np.float64))
        dist_cur  = float(np.linalg.norm(T_cur[:3, 3] - self._target))
        step_scale = EE_STEP_FINE if dist_cur < FINE_DIST else EE_STEP_MAX
        dp         = (action * step_scale).astype(np.float64)

        dq       = self._ik_step(dp)
        prev_q   = self._q.copy()
        self._q  = np.clip(self._q + dq, Q_MIN, Q_MAX).astype(np.float32)

        self._prev_q_dot = self._q_dot.copy()
        self._q_dot      = ((self._q - prev_q) / self.dt).astype(np.float32)
        self._step_n    += 1

        T      = forward_kinematics(self._q.astype(np.float64))
        p_ee   = T[:3, 3].astype(np.float32)
        tool_z = T[:3, 2]
        J      = jacobian(self._q.astype(np.float64))

        # Floor check: every link must stay above z=0
        below_floor = not self._all_links_above_floor()

        # Distance to current target
        dist    = float(np.linalg.norm(p_ee - self._target))
        reached = dist < SUCCESS_DIST

        # Advance to next target when within 1 mm
        all_done = False
        if reached:
            self._targets_reached += 1
            if self._target_idx + 1 < self.n_targets:
                self._target_idx += 1
                self._target = self._target_queue[self._target_idx]
            else:
                all_done = True   # all targets visited

        obs        = self._get_obs()
        reward     = self._compute_reward(p_ee, tool_z, J, reached, below_floor)
        terminated = all_done or below_floor
        truncated  = self._step_n >= self.max_steps

        info = {
            'distance_m':       dist,
            'below_floor':      below_floor,
            'targets_reached':  self._targets_reached,
            'target_idx':       self._target_idx,
            'step':             self._step_n,
        }
        return obs, reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        T      = forward_kinematics(self._q.astype(np.float64))
        p      = T[:3, 3].astype(np.float32)
        tool_z = T[:3, 2].astype(np.float32)   # tool z-axis direction
        obs = np.concatenate([
            self._q,          # 6
            self._q_dot,      # 6
            p,                # 3
            tool_z,           # 3
            self._target,     # 3
        ])
        return obs.astype(np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self, p_ee: np.ndarray, tool_z: np.ndarray,
                        J: np.ndarray, reached: bool, below_floor: bool) -> float:
        dist_sq = float(np.sum((p_ee - self._target) ** 2))
        dist    = math.sqrt(dist_sq)

        # Dual Gaussian: coarse signal (σ=5 cm) + fine signal (σ=5 mm).
        # Coarse pulls the arm in from far away; fine gives strong gradient
        # for the last centimetre of precision approach.
        proximity_reward = (
            0.5 * math.exp(-dist_sq / (2 * 0.05 ** 2))   # σ=5 cm
          + 0.5 * math.exp(-dist_sq / (2 * 0.005 ** 2))  # σ=5 mm
        )

        # Joint limit proximity penalty
        margin   = 0.1
        at_limit = np.any(
            (self._q > Q_MAX - margin) | (self._q < Q_MIN + margin)
        )
        collision_pen = 10.0 if at_limit else 0.0

        # Jerk penalty (sudden velocity changes)
        jerk     = float(np.sum((self._q_dot - self._prev_q_dot) ** 2))
        jerk_pen = 0.1 * jerk

        # Velocity smoothness penalty
        vel_pen = 0.02 * float(np.sum(self._q_dot ** 2))

        # Singularity penalty
        JJT = J @ J.T
        det = abs(float(np.linalg.det(JJT)))
        singularity_pen = min(0.5 / (det + 1e-6), 1.0)

        # Floor penalty: any link below z=0
        floor_pen = 50.0 if below_floor else 0.0

        # Orientation penalty: deviation of tool z-axis from [0,0,-1]
        desired_z = np.array([0.0, 0.0, -1.0])
        cos_a     = float(np.dot(tool_z, desired_z))
        cos_a     = max(-1.0, min(1.0, cos_a))
        ori_error = math.acos(cos_a)   # 0 (aligned) → π (inverted)
        ori_pen   = ORI_PEN_WEIGHT * ori_error

        # Completion bonus per target reached (1 mm threshold)
        completion_bonus = 10.0 if reached else 0.0

        # Small constant time penalty
        time_pen = 0.0002

        reward = (
            proximity_reward
            - collision_pen
            - jerk_pen
            - vel_pen
            - singularity_pen
            - floor_pen
            - ori_pen
            + completion_bonus
            - time_pen
        )
        return float(reward)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mat_to_rpy(R: np.ndarray):
        """ZYX Euler (roll, pitch, yaw) from 3×3 rotation matrix."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2( R[2, 1],  R[2, 2])
            pitch = math.atan2(-R[2, 0],  sy)
            yaw   = math.atan2( R[1, 0],  R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2],  R[1, 1])
            pitch = math.atan2(-R[2, 0],  sy)
            yaw   = 0.0
        return roll, pitch, yaw

    def render(self, mode='human'):
        pass

    def close(self):
        pass
