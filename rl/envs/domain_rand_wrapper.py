#!/usr/bin/env python3
"""
DomainRandWrapper
=================
Gym wrapper that applies domain randomisation to UR3eGridEnv at each episode
reset, simulating real-world variation:

  1. Joint friction  — multiplicative noise ±20 % applied to joint velocity
                       damping (simulated by scaling the returned q_dot).
  2. End-effector mass — uniform in [0.1, 0.4] kg; shifts the EE position
                         downward by a gravity-induced deflection proportional
                         to the extra mass (simple linear model, ~0.5 mm/kg).
  3. Controller latency — integer delay 0–10 steps; observations are held
                          from N steps ago using a circular buffer.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DomainRandWrapper(gym.Wrapper):
    """
    Wraps UR3eGridEnv (or any compatible Gym env) and applies domain
    randomisation at each episode reset.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    friction_range : tuple(float, float)
        Multiplicative friction factor range (applied to q_dot portion of obs).
        Default: (0.8, 1.2)  — ±20 %.
    ee_mass_range : tuple(float, float)
        End-effector payload mass range in kg.
        Default: (0.1, 0.4) kg.
    max_latency_steps : int
        Maximum controller latency in simulation steps.
        Default: 10.
    """

    def __init__(
        self,
        env,
        friction_range=(0.8, 1.2),
        ee_mass_range=(0.1, 0.4),
        max_latency_steps=10,
    ):
        super().__init__(env)

        self._friction_range      = friction_range
        self._ee_mass_range       = ee_mass_range
        self._max_latency_steps   = max_latency_steps

        # Current randomisation parameters (set at each reset)
        self._friction_factor     = 1.0
        self._ee_mass             = 0.0
        self._latency_steps       = 0

        # Circular observation buffer for latency simulation
        obs_dim = env.observation_space.shape[0]
        self._obs_buffer          = np.zeros(
            (max_latency_steps + 1, obs_dim), dtype=np.float32
        )
        self._buf_ptr             = 0      # write pointer

        self.np_random = np.random.default_rng()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Sample new randomisation parameters, reset wrapped env."""
        rng = np.random.default_rng(seed)

        # Sample domain parameters
        self._friction_factor = float(rng.uniform(*self._friction_range))
        self._ee_mass         = float(rng.uniform(*self._ee_mass_range))
        self._latency_steps   = int(rng.integers(0, self._max_latency_steps + 1))

        # Reset the observation buffer
        obs_dim = self.observation_space.shape[0]
        self._obs_buffer = np.zeros(
            (self._max_latency_steps + 1, obs_dim), dtype=np.float32
        )
        self._buf_ptr = 0

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._apply_randomisation(obs)

        # Fill entire buffer with the initial observation so the agent sees
        # a valid (non-zero) observation even with maximum latency.
        for i in range(self._max_latency_steps + 1):
            self._obs_buffer[i] = obs
        self._buf_ptr = 0

        return self._delayed_obs(obs), info

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_randomisation(obs)

        # Write current obs into the circular buffer
        self._buf_ptr = (self._buf_ptr + 1) % (self._max_latency_steps + 1)
        self._obs_buffer[self._buf_ptr] = obs

        delayed = self._delayed_obs(obs)
        info['friction_factor'] = self._friction_factor
        info['ee_mass_kg']      = self._ee_mass
        info['latency_steps']   = self._latency_steps
        return delayed, reward, terminated, truncated, info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_randomisation(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply friction and ee_mass perturbations to the raw observation.

        Observation layout (from UR3eGridEnv):
          [0:6]   q       — joint positions
          [6:12]  q_dot   — joint velocities
          [12:15] p_ee    — end-effector position
          [15:18] rpy     — end-effector orientation
          [18:21] p_target
        """
        obs = obs.copy()

        # --- Joint friction: scale reported velocities ---
        obs[6:12] = obs[6:12] * self._friction_factor

        # --- EE mass: gravity-induced deflection (linear model) ---
        # deflection ≈ 0.5 mm per kg, downward (negative z in base frame)
        gravity_deflection = -0.0005 * self._ee_mass
        obs[14] += gravity_deflection   # p_ee z component

        return obs

    def _delayed_obs(self, current_obs: np.ndarray) -> np.ndarray:
        """Return the observation from `_latency_steps` ago."""
        if self._latency_steps == 0:
            return current_obs
        read_ptr = (
            self._buf_ptr - self._latency_steps
        ) % (self._max_latency_steps + 1)
        return self._obs_buffer[read_ptr].copy()
