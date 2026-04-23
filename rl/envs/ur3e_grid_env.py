#!/usr/bin/env python3
"""
UR3eGridEnv
===========
Gymnasium environment for the FruitNinja UR3e cutting-board task.

Observation space (dim=21):
  q[6]         — joint positions (rad)
  q_dot[6]     — joint velocities (rad/s)
  p_ee[3]      — end-effector position in robot base frame (m)
  rpy[3]       — end-effector roll/pitch/yaw (rad)
  p_target[3]  — target cell centre in robot base frame (m)

Action space (dim=6):
  Joint velocity commands (rad/s), clipped to [-1, 1].

Board geometry — derived from FK of the 4 measured corner joint angles
in grid_mover.py.  The board is NOT axis-aligned and NOT flat in the
robot base frame, so targets are computed via bilinear interpolation of
the 4 measured Cartesian corners (matching grid_mover's joint-space
interpolation approach).

  P_A1 = ( 0.2697, -0.4846,  0.0968)  — far-left
  P_A4 = ( 0.3114, -0.3503,  0.1501)  — near-left
  P_N1 = (-0.3124, -0.4549,  0.0875)  — far-right
  P_N4 = (-0.3130, -0.3493,  0.1022)  — near-right

Home = board centre (u=0.5, v=0.5 bilinear interpolation of corner
joint angles from grid_mover.py).
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Measured board corners in robot base frame (from FK of grid_mover corners) ─
# u = col / 13  (0 → A, 1 → N)
# v = row / 3   (0 → row-1, 1 → row-4)

_P_A1 = np.array([ 0.2697, -0.4846,  0.0968], dtype=np.float64)  # u=0, v=0
_P_A4 = np.array([ 0.3114, -0.3503,  0.1501], dtype=np.float64)  # u=0, v=1
_P_N1 = np.array([-0.3124, -0.4549,  0.0875], dtype=np.float64)  # u=1, v=0
_P_N4 = np.array([-0.3130, -0.3493,  0.1022], dtype=np.float64)  # u=1, v=1

NUM_COLS = 14
NUM_ROWS = 4


def cell_centre(col: int, row: int) -> np.ndarray:
    """
    Return the Cartesian centre of a grid cell in robot base frame (m).

    Uses bilinear interpolation of the 4 measured corner positions, matching
    the same approach grid_mover uses in joint space.

    Parameters
    ----------
    col : 0-based column index (0 = A … 13 = N)
    row : 0-based row index   (0 = row-1 … 3 = row-4)
    """
    u = col / (NUM_COLS - 1)   # 0.0 → 1.0
    v = row / (NUM_ROWS - 1)   # 0.0 → 1.0
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

# Home = board centre joint angles (bilinear average of 4 measured corners,
# degrees → radians).  Keeps the arm in a realistic working configuration.
_HOME_DEG = [-76.697, -154.855, -39.142, -75.47, 91.41, 234.355]
HOME_RAD  = np.array([math.radians(d) for d in _HOME_DEG], dtype=np.float32)

# Realistic UR3e joint limits (radians) — tighter than ±2π to match
# the working envelope used by the real robot.
Q_MIN = np.array([-2*math.pi, -2*math.pi, -2*math.pi,
                  -2*math.pi, -2*math.pi, -2*math.pi], dtype=np.float32)
Q_MAX = np.array([ 2*math.pi,  0.0,        2*math.pi,
                   2*math.pi,  2*math.pi,  2*math.pi], dtype=np.float32)

# Joint velocity limits (rad/s)
QDOT_MAX = 1.0   # scalar; applied symmetrically

# ── DH / forward-kinematics helpers ───────────────────────────────────────────

# UR3e DH parameters (modified DH, metres)
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
    """
    Compute FK for UR3e.

    Parameters
    ----------
    q : array of 6 joint angles in radians

    Returns
    -------
    T : 4×4 homogeneous transform (end-effector in base frame)
    """
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
    J = np.zeros((6, 6), dtype=np.float64)
    T0 = forward_kinematics(q)
    p0 = T0[:3, 3]
    for i in range(6):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps
        Tp = forward_kinematics(qp)
        Tm = forward_kinematics(qm)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        # Angular velocity column via skew-symmetric part of dR/dtheta * R^T
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

    The agent controls joint velocities to move the end-effector to a
    randomly selected cell centre on the cutting board.
    """

    metadata = {'render_modes': []}

    # Observation and action dimensionality
    obs_dim    = 21   # q(6) + q_dot(6) + p_ee(3) + rpy(3) + p_target(3)
    action_dim = 6

    def __init__(self, dt: float = 0.02, max_steps: int = 500):
        super().__init__()

        self.dt        = dt
        self.max_steps = max_steps

        # Gym spaces
        obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # State
        self._q       = HOME_RAD.copy()
        self._q_dot   = np.zeros(6, dtype=np.float32)
        self._target  = np.zeros(3, dtype=np.float32)
        self._step_n  = 0
        self._prev_q_dot = np.zeros(6, dtype=np.float32)

        self.np_random = np.random.default_rng()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Randomise target cell and reset robot to home."""
        super().reset(seed=seed)

        # Random cell — bilinear interpolation gives true Cartesian centre
        col = int(self.np_random.integers(0, NUM_COLS))  # 0 … 13
        row = int(self.np_random.integers(0, NUM_ROWS))  # 0 … 3

        # +/- 5 mm noise
        noise = self.np_random.uniform(-0.005, 0.005, size=3).astype(np.float32)
        self._target = cell_centre(col, row) + noise

        # Reset robot state
        self._q      = HOME_RAD.copy().astype(np.float32)
        self._q_dot  = np.zeros(6, dtype=np.float32)
        self._prev_q_dot = np.zeros(6, dtype=np.float32)
        self._step_n = 0

        return self._get_obs(), {}

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        cmd_vel = action * QDOT_MAX  # scale to rad/s

        # Integrate joint positions
        self._q      = np.clip(
            self._q + cmd_vel * self.dt,
            Q_MIN, Q_MAX
        ).astype(np.float32)
        self._prev_q_dot = self._q_dot.copy()
        self._q_dot  = cmd_vel.astype(np.float32)
        self._step_n += 1

        obs    = self._get_obs()
        T      = forward_kinematics(self._q.astype(np.float64))
        p_ee   = T[:3, 3].astype(np.float32)
        J      = jacobian(self._q.astype(np.float64))
        reward = self._compute_reward(p_ee, J)

        dist       = float(np.linalg.norm(p_ee - self._target))
        terminated = dist < 0.005
        truncated  = self._step_n >= self.max_steps

        info = {
            'distance_m': dist,
            'col': None,
            'row': None,
            'step': self._step_n,
        }
        return obs, reward, terminated, truncated, info

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        T   = forward_kinematics(self._q.astype(np.float64))
        p   = T[:3, 3].astype(np.float32)
        rpy = np.array(self._mat_to_rpy(T[:3, :3]), dtype=np.float32)
        obs = np.concatenate([
            self._q,          # 6
            self._q_dot,      # 6
            p,                # 3
            rpy,              # 3
            self._target,     # 3
        ])
        return obs.astype(np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self, p_ee: np.ndarray, J: np.ndarray) -> float:
        # --- Gaussian proximity kernel ---
        dist_sq = float(np.sum((p_ee - self._target) ** 2))
        sigma   = 0.05   # ~5 cm half-width
        proximity_reward = math.exp(-dist_sq / (2 * sigma ** 2))

        # --- Collision penalty (joint limit proximity) ---
        margin        = 0.1   # rad
        at_limit      = np.any(
            (self._q > Q_MAX - margin) | (self._q < Q_MIN + margin)
        )
        collision_pen = 10.0 if at_limit else 0.0

        # --- Jerk penalty ---
        jerk    = float(np.sum((self._q_dot - self._prev_q_dot) ** 2))
        jerk_pen = 0.01 * jerk

        # --- Singularity penalty (capped to avoid dominating reward) ---
        JJT = J @ J.T
        det = abs(float(np.linalg.det(JJT)))
        singularity_pen = min(0.5 / (det + 1e-6), 1.0)

        # --- Task completion bonus ---
        dist           = math.sqrt(dist_sq)
        completion_bonus = 5.0 if dist < 0.005 else 0.0

        # --- Time penalty ---
        time_pen = 0.001

        reward = (
            proximity_reward
            - collision_pen
            - jerk_pen
            - singularity_pen
            + completion_bonus
            - time_pen
        )
        return float(reward)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mat_to_rpy(R: np.ndarray):
        """
        Convert a 3×3 rotation matrix to (roll, pitch, yaw) in radians
        using the ZYX Euler convention (same as tf.transformations).
        """
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
