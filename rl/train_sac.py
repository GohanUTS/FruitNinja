#!/usr/bin/env python3
"""
SAC Training Script — FruitNinja UR3e Grid Task
================================================
Usage:
  # Train from scratch (2M steps):
  python train_sac.py

  # Train with BC weight initialisation:
  python train_sac.py --bc_weights pretraining/bc_policy.pth

  # Evaluate a saved model (10 episodes):
  python train_sac.py --eval

Dependencies (install separately):
  pip install stable-baselines3[extra] gymnasium torch
"""

import argparse
import os
import sys

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the rl/ directory without a full colcon install.
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _RL_DIR)

from envs.ur3e_grid_env       import UR3eGridEnv
from envs.domain_rand_wrapper import DomainRandWrapper


# ── Hyperparameters ────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 500_000
N_ENVS            = 4
LEARNING_RATE     = 3e-4
BUFFER_SIZE       = 1_000_000
BATCH_SIZE        = 256
GAMMA             = 0.99
TAU               = 0.005
ENT_COEF          = 'auto'
TENSORBOARD_LOG   = './tb_logs/'
EVAL_FREQ         = 10_000     # evaluate every N steps (per env)
CHECKPOINT_FREQ   = 50_000     # save checkpoint every N steps
SAVE_PATH         = 'sac_fruitninja'
N_EVAL_EPISODES   = 10


# ── Environment factory ────────────────────────────────────────────────────────

def make_env(rank: int = 0, seed: int = 0):
    """Return a callable that creates a seeded, domain-randomised environment."""
    def _init():
        env = UR3eGridEnv()
        env = DomainRandWrapper(env)
        return env
    return _init


# ── BC weight loading helper ──────────────────────────────────────────────────

def _load_bc_weights(model, bc_weights_path: str):
    """
    Copy BC policy weights into the SAC actor network.

    BCPolicy architecture:  Linear(21,256) → ReLU → Linear(256,256) → ReLU
                            → Linear(256,6) → Tanh
    SAC actor (MlpPolicy): latent_pi network then mu / log_std heads.

    We match the first two hidden layers by parameter shape.
    """
    import torch

    sys.path.insert(
        0,
        os.path.join(_RL_DIR, 'pretraining')
    )
    from behavioural_clone import BCPolicy

    bc = BCPolicy()
    bc.load_state_dict(torch.load(bc_weights_path, map_location='cpu'))
    bc.eval()

    # SAC actor network (stable-baselines3 internals)
    actor = model.policy.actor
    bc_params  = list(bc.net.parameters())      # [W0,b0, W1,b1, W2,b2]
    sac_params = list(actor.latent_pi.parameters())

    copied = 0
    for sp, bp in zip(sac_params, bc_params):
        if sp.shape == bp.shape:
            sp.data.copy_(bp.data)
            copied += 1

    print(f'[BC→SAC] Copied {copied} parameter tensors from {bc_weights_path}')


# ── Training ──────────────────────────────────────────────────────────────────

def train(bc_weights: str = None):
    """
    Train a SAC agent on the FruitNinja grid task.

    Parameters
    ----------
    bc_weights : path to bc_policy.pth, or None to train from scratch
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util    import make_vec_env
    from stable_baselines3.common.callbacks   import (
        EvalCallback, CheckpointCallback, CallbackList
    )
    from stable_baselines3.common.vec_env     import SubprocVecEnv

    # --- Vectorised training environments ---
    vec_env = make_vec_env(
        make_env(),
        n_envs=N_ENVS,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    # --- Single evaluation environment ---
    eval_env = make_vec_env(
        make_env(rank=N_ENVS, seed=1000),
        n_envs=1,
    )

    # --- SAC model ---
    model = SAC(
        policy          = 'MlpPolicy',
        env             = vec_env,
        learning_rate   = LEARNING_RATE,
        buffer_size     = BUFFER_SIZE,
        batch_size      = BATCH_SIZE,
        gamma           = GAMMA,
        tau             = TAU,
        ent_coef        = ENT_COEF,
        tensorboard_log = TENSORBOARD_LOG,
        verbose         = 1,
    )

    # --- Optional BC weight initialisation ---
    if bc_weights is not None:
        _load_bc_weights(model, bc_weights)

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = './logs/best_model/',
        log_path             = './logs/',
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = './checkpoints/',
        name_prefix = 'sac_fruitninja',
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # --- Train ---
    print(f'[SAC] Starting training — {TOTAL_TIMESTEPS:,} timesteps')
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    # --- Save final model ---
    model.save(SAVE_PATH)
    print(f'[SAC] Model saved → {SAVE_PATH}.zip')

    vec_env.close()
    eval_env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_path: str = SAVE_PATH, n_episodes: int = N_EVAL_EPISODES):
    """
    Load a saved SAC model and run evaluation episodes.

    Parameters
    ----------
    model_path  : path to the .zip model (without extension)
    n_episodes  : number of evaluation episodes to run
    """
    from stable_baselines3 import SAC

    env   = DomainRandWrapper(UR3eGridEnv())
    model = SAC.load(model_path, env=env)
    print(f'[EVAL] Loaded model from {model_path}.zip')

    rewards   = []
    distances = []

    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        done    = False
        ep_r    = 0.0
        ep_dist = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done    = terminated or truncated
            ep_r   += reward
            ep_dist = info.get('distance_m', None)

        rewards.append(ep_r)
        if ep_dist is not None:
            distances.append(ep_dist)
        print(f'  Episode {ep:3d}: total_reward={ep_r:+.3f}  '
              f'final_dist={ep_dist*1000:.1f} mm' if ep_dist else
              f'  Episode {ep:3d}: total_reward={ep_r:+.3f}')

    print(f'\n[EVAL] Mean reward  : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}')
    if distances:
        print(f'[EVAL] Mean final dist: {np.mean(distances)*1000:.2f} mm')

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SAC training / evaluation for FruitNinja UR3e grid task'
    )
    parser.add_argument(
        '--bc_weights', type=str, default=None,
        help='Path to BC pre-trained weights (bc_policy.pth). '
             'If provided, initialises the SAC actor from these weights.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Run evaluation instead of training. '
             f'Loads {SAVE_PATH}.zip and runs {N_EVAL_EPISODES} episodes.'
    )
    parser.add_argument(
        '--model_path', type=str, default=SAVE_PATH,
        help=f'Path to saved model for --eval mode (default: {SAVE_PATH})'
    )
    args = parser.parse_args()

    if args.eval:
        evaluate(model_path=args.model_path)
    else:
        train(bc_weights=args.bc_weights)


if __name__ == '__main__':
    main()
