#!/usr/bin/env python3
"""
visualise_policy.py — watch the trained SAC policy drive the UR3e arm.

Usage:
  python visualise_policy.py --situation 1          # corners A1→N1→A4→N4
  python visualise_policy.py --situation 2          # 6 random grid points
  python visualise_policy.py --situation 3          # chop sequence
  python visualise_policy.py --situation 1 --delay 0.02

Controls (matplotlib window):
  Close the window to stop.
"""

import argparse
import math
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RL_DIR)
sys.path.insert(0, os.path.join(_RL_DIR, '..', 'fruitninja'))

from envs.ur3e_grid_env import (
    UR3eGridEnv, _dh_transform, _UR3E_DH,
    cell_centre, _P_A1, _P_A4, _P_N1, _P_N4, NUM_COLS, NUM_ROWS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_UPRIGHT_Q = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # real home


# ── FK: 7-point joint chain ───────────────────────────────────────────────────

def _joint_positions(q: np.ndarray) -> np.ndarray:
    """Return (7, 3) array: base origin + 6 joint-frame origins."""
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for i, (a, d, alpha, offset) in enumerate(_UR3E_DH):
        T = T @ _dh_transform(a, d, alpha, float(q[i]) + offset)
        pts.append(T[:3, 3].copy())
    return np.array(pts)


# ── Figure factory ────────────────────────────────────────────────────────────

def _make_figure():
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Board outline
    corners = np.array([_P_A1, _P_N1, _P_N4, _P_A4, _P_A1])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color='#00b4d8', linewidth=1.5, label='Cutting board')

    arm_line, = ax.plot([], [], [], 'o-', color='#e0e0e0', linewidth=3,
                        markersize=6, label='UR3e arm')
    ee_trail, = ax.plot([], [], [], '-', color='#06d6a0', linewidth=1,
                        alpha=0.6, label='EE path')
    tgt_pt    = ax.scatter([], [], [], color='#ef233c', s=120, zorder=5,
                           label='Target', marker='*')
    info_txt  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          color='white', fontsize=10)

    ax.set_xlabel('X (m)', color='#aaaaaa')
    ax.set_ylabel('Y (m)', color='#aaaaaa')
    ax.set_zlabel('Z (m)', color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.set_xlim(-0.5,  0.5)
    ax.set_ylim(-0.6,  0.2)
    ax.set_zlim(-0.1,  0.7)
    ax.set_title('FruitNinja — SAC Policy Rollout', color='white', pad=10)
    ax.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='white',
              framealpha=0.7)
    ax.view_init(elev=25, azim=-60)

    return fig, ax, arm_line, ee_trail, tgt_pt, info_txt


# ── Core step loop ────────────────────────────────────────────────────────────

def _step_to_target(model, env, target, fig, ax, arm_line, ee_trail,
                    tgt_pt, info_txt, trail, delay, label,
                    success_m=0.005, reset_arm=False):
    """
    Drive the arm to `target` using the policy.

    If reset_arm=True, env.reset() is called and then _q is overridden to
    the upright position (all zeros) before beginning the episode.

    Returns False if the matplotlib window was closed, True otherwise.
    """
    if reset_arm:
        obs, _ = env.reset()
        env._q    = _UPRIGHT_Q.copy()
        env._q_dot = np.zeros(6, dtype=np.float32)
        env._prev_q_dot = np.zeros(6, dtype=np.float32)
        env._step_n = 0

    env._target  = target.astype(np.float32)
    env._step_n  = 0
    obs = env._get_obs()

    trail[0].clear(); trail[1].clear(); trail[2].clear()

    tgt_pt._offsets3d = ([target[0]], [target[1]], [target[2]])
    ax.set_title(
        f'FruitNinja — SAC  |  {label}  |  '
        f'target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})',
        color='white', pad=10
    )

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        pts = _joint_positions(env._q.astype(np.float64))
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])

        trail[0].append(pts[-1, 0])
        trail[1].append(pts[-1, 1])
        trail[2].append(pts[-1, 2])
        ee_trail.set_data(trail[0], trail[1])
        ee_trail.set_3d_properties(trail[2])

        dist_mm = info['distance_m'] * 1000
        reached = info.get('targets_reached', 0)
        info_txt.set_text(
            f'Step: {env._step_n:4d}   Dist: {dist_mm:6.1f} mm   '
            f'Reached: {reached}   Reward: {reward:+.3f}'
        )

        plt.pause(delay)

        if not plt.fignum_exists(fig.number):
            return False

        if info['distance_m'] < success_m or truncated:
            print(f'  [{label}] done — dist {dist_mm:.1f} mm  '
                  f'({"reached" if info["distance_m"] < success_m else "truncated"})')
            return True


# ── Situation 1: corners A1 → N1 → A4 → N4 ──────────────────────────────────

def situation_1(model, env, delay):
    fig, ax, arm_line, ee_trail, tgt_pt, info_txt = _make_figure()
    trail = [[], [], []]

    waypoints = [
        (_P_A1.astype(np.float32), 'Sit-1  A1', True),
        (_P_N1.astype(np.float32), 'Sit-1  N1', False),
        (_P_A4.astype(np.float32), 'Sit-1  A4', False),
        (_P_N4.astype(np.float32), 'Sit-1  N4', False),
    ]

    for target, label, reset in waypoints:
        ok = _step_to_target(model, env, target, fig, ax, arm_line, ee_trail,
                             tgt_pt, info_txt, trail, delay, label,
                             reset_arm=reset)
        if not ok:
            return

    plt.show()


# ── Situation 2: 6 random grid points ────────────────────────────────────────

def situation_2(model, env, delay, n_points=6):
    fig, ax, arm_line, ee_trail, tgt_pt, info_txt = _make_figure()
    trail = [[], [], []]

    rng = np.random.default_rng()
    cols = rng.integers(0, NUM_COLS, size=n_points)
    rows = rng.integers(0, NUM_ROWS, size=n_points)

    for i, (col, row) in enumerate(zip(cols, rows)):
        target = cell_centre(int(col), int(row))
        label  = f'Sit-2  pt {i+1}/{n_points}  ({chr(65+col)}{row+1})'
        ok = _step_to_target(model, env, target, fig, ax, arm_line, ee_trail,
                             tgt_pt, info_txt, trail, delay, label,
                             reset_arm=(i == 0))
        if not ok:
            return

    plt.show()


# ── Situation 3: chop sequence ────────────────────────────────────────────────

def situation_3(model, env, delay, n_chops=4):
    """
    1. Move arm (from upright) to a random grid cell.
    2. Repeat n_chops times:
       a. Move to target_z − 1.5 mm (chop down).
       b. Move back to target_z (return).
    """
    fig, ax, arm_line, ee_trail, tgt_pt, info_txt = _make_figure()
    trail = [[], [], []]

    rng    = np.random.default_rng()
    col    = int(rng.integers(0, NUM_COLS))
    row    = int(rng.integers(0, NUM_ROWS))
    target = cell_centre(col, row)
    cell_name = f'{chr(65+col)}{row+1}'

    # ── phase 0: reach the cell ──
    ok = _step_to_target(model, env, target, fig, ax, arm_line, ee_trail,
                         tgt_pt, info_txt, trail, delay,
                         label=f'Sit-3  reach {cell_name}',
                         success_m=0.005, reset_arm=True)
    if not ok:
        return

    # ── chop cycles ──
    chop_target   = target.copy();  chop_target[2] -= 0.0015
    return_target = target.copy()

    for i in range(n_chops):
        # chop down
        ok = _step_to_target(model, env, chop_target, fig, ax, arm_line,
                             ee_trail, tgt_pt, info_txt, trail, delay,
                             label=f'Sit-3  chop {i+1}/{n_chops} ↓',
                             success_m=0.001, reset_arm=False)
        if not ok:
            return

        # return up
        ok = _step_to_target(model, env, return_target, fig, ax, arm_line,
                             ee_trail, tgt_pt, info_txt, trail, delay,
                             label=f'Sit-3  chop {i+1}/{n_chops} ↑',
                             success_m=0.001, reset_arm=False)
        if not ok:
            return

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise SAC policy on UR3e')
    parser.add_argument('--model',     default='sac_fruitninja',
                        help='Path to model zip (without .zip)')
    parser.add_argument('--situation', type=int, choices=[1, 2, 3], default=1,
                        help='Which demo situation to run (1, 2, or 3)')
    parser.add_argument('--delay',     type=float, default=0.03,
                        help='Pause between steps (seconds) — lower = faster')
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except ImportError:
        print('stable-baselines3 not installed. Run: pip install stable-baselines3')
        sys.exit(1)

    print(f'Loading model from {args.model}.zip ...')
    model = SAC.load(args.model)

    env = UR3eGridEnv()

    sit_fn = {1: situation_1, 2: situation_2, 3: situation_3}[args.situation]
    print(f'Running situation {args.situation} — close the window to stop.')
    sit_fn(model, env, delay=args.delay)


if __name__ == '__main__':
    main()
