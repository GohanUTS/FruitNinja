#!/usr/bin/env python3
"""
visualise_policy.py — watch the trained SAC policy drive the UR3e arm.

Usage:
  python visualise_policy.py                        # random targets
  python visualise_policy.py --cell A1              # specific grid cell
  python visualise_policy.py --episodes 5 --delay 0.02

Controls (matplotlib window):
  Close the window to stop.
"""

import argparse
import math
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Path setup ────────────────────────────────────────────────────────────────
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RL_DIR)
sys.path.insert(0, os.path.join(_RL_DIR, '..', 'fruitninja'))

from envs.ur3e_grid_env import (
    UR3eGridEnv, forward_kinematics, _dh_transform, _UR3E_DH,
    cell_centre, _P_A1, _P_A4, _P_N1, _P_N4, NUM_COLS, NUM_ROWS,
)
from envs.domain_rand_wrapper import DomainRandWrapper


# ── FK: positions of each joint frame origin ──────────────────────────────────

def joint_positions(q: np.ndarray) -> np.ndarray:
    """Return (7, 3) array: base origin + 6 joint-frame origins."""
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for i, (a, d, alpha, offset) in enumerate(_UR3E_DH):
        T = T @ _dh_transform(a, d, alpha, q[i] + offset)
        pts.append(T[:3, 3].copy())
    return np.array(pts)


# ── Board outline from measured corners ───────────────────────────────────────

def board_corners():
    """Return the 4 measured corners of the board (closed loop for plotting)."""
    return np.array([
        _P_A1, _P_N1, _P_N4, _P_A4, _P_A1,
    ])


# ── Cell name → target ────────────────────────────────────────────────────────

GRID_COLS = list('ABCDEFGHIJKLMN')
GRID_ROWS = ['1', '2', '3', '4']

def cell_to_target(cell: str) -> np.ndarray:
    col = GRID_COLS.index(cell[0].upper())
    row = GRID_ROWS.index(cell[1])
    return cell_centre(col, row)


# ── Visualiser ────────────────────────────────────────────────────────────────

def run(model, env_base: UR3eGridEnv, n_episodes: int, step_delay: float,
        fixed_cell: str = None):

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Fixed board
    corners = board_corners()
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color='#00b4d8', linewidth=1.5, label='Cutting board')

    # Arm segments (7 points = base + 6 joints)
    arm_line,  = ax.plot([], [], [], 'o-', color='#e0e0e0', linewidth=3,
                         markersize=6, label='UR3e arm')
    ee_trail,  = ax.plot([], [], [], '-', color='#06d6a0', linewidth=1,
                         alpha=0.6, label='EE path')
    target_pt  = ax.scatter([], [], [], color='#ef233c', s=120, zorder=5,
                            label='Target', marker='*')
    dist_text  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                           color='white', fontsize=10)

    ax.set_xlabel('X (m)', color='#aaaaaa')
    ax.set_ylabel('Y (m)', color='#aaaaaa')
    ax.set_zlabel('Z (m)', color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.1, 0.8)
    ax.set_zlim(-0.1, 0.8)
    ax.set_title('FruitNinja — SAC Policy Rollout', color='white', pad=10)
    ax.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='white',
              framealpha=0.7)
    ax.view_init(elev=25, azim=-60)

    trail_x, trail_y, trail_z = [], [], []

    for ep in range(1, n_episodes + 1):
        obs, _ = env_base.reset()

        # Override target if a fixed cell was requested
        if fixed_cell is not None:
            env_base._target = cell_to_target(fixed_cell)
            obs = env_base._get_obs()

        trail_x.clear(); trail_y.clear(); trail_z.clear()
        target = env_base._target

        ax.set_title(
            f'FruitNinja — SAC Policy  |  Episode {ep}/{n_episodes}  '
            f'|  Target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})',
            color='white', pad=10
        )
        target_pt._offsets3d = ([target[0]], [target[1]], [target[2]])

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_base.step(action)
            done = terminated or truncated
            step += 1

            # Joint positions for drawing
            pts = joint_positions(env_base._q.astype(np.float64))

            arm_line.set_data(pts[:, 0], pts[:, 1])
            arm_line.set_3d_properties(pts[:, 2])

            # EE trail
            trail_x.append(pts[-1, 0])
            trail_y.append(pts[-1, 1])
            trail_z.append(pts[-1, 2])
            ee_trail.set_data(trail_x, trail_y)
            ee_trail.set_3d_properties(trail_z)

            dist_mm = info['distance_m'] * 1000
            dist_text.set_text(
                f'Step: {step:4d}   Dist: {dist_mm:6.1f} mm   '
                f'Reward: {reward:+.3f}'
            )

            plt.pause(step_delay)

            if not plt.fignum_exists(fig.number):
                return   # window closed

        print(f'Episode {ep:3d} done — final dist: {info["distance_m"]*1000:.1f} mm')

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise SAC policy on UR3e')
    parser.add_argument('--model',    default='sac_fruitninja',
                        help='Path to model zip (without .zip)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to render')
    parser.add_argument('--delay',    type=float, default=0.03,
                        help='Pause between steps (seconds) — lower = faster')
    parser.add_argument('--cell',     type=str, default=None,
                        help='Fixed target cell, e.g. A1, G2, N4')
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except ImportError:
        print('stable-baselines3 not installed. Run: pip install stable-baselines3')
        sys.exit(1)

    print(f'Loading model from {args.model}.zip ...')
    model = SAC.load(args.model)

    env = UR3eGridEnv()
    print(f'Running {args.episodes} episode(s) — close the window to stop.')
    run(model, env, n_episodes=args.episodes, step_delay=args.delay,
        fixed_cell=args.cell)


if __name__ == '__main__':
    main()
