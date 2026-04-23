#!/usr/bin/env python3
"""
Behavioural Cloning Pre-training
=================================
BCPolicy: 3-layer MLP (obs_dim=21 → 256 → 256 → action_dim=6) with ReLU
hidden activations and Tanh output (maps to [-1, 1] action space).

build_dataset():
    Imports cell_to_joints_deg from fruitninja.grid_mover, iterates over all
    14×4 grid cells and records the joint-angle delta from home as the
    "expert action".  Observations are zero-padded (no FK required for BC).

train(epochs=50):
    Supervised regression with Adam (lr=1e-3) and MSELoss.

__main__:
    Run `python behavioural_clone.py` to train and save bc_policy.pth.
"""

import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Allow standalone execution (without ROS/colcon install) ───────────────────
# Inserts the fruitninja package directory so grid_mover can be imported
# directly without installing the ROS2 package.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../fruitninja')
)


# ── Constants ─────────────────────────────────────────────────────────────────

OBS_DIM    = 21
ACTION_DIM = 6
HIDDEN_DIM = 256

# Home joint angles in radians (all zeros — robot upright configuration)
HOME_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ── Neural network ────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    """
    Behavioural Cloning policy network.

    Architecture
    ------------
    Input  : obs_dim = 21
    Hidden : 256 → ReLU → 256 → ReLU
    Output : action_dim = 6, activated with Tanh (output range [-1, 1])
    """

    def __init__(
        self,
        obs_dim: int    = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int     = HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset():
    """
    Build a supervised dataset from the grid_mover joint-angle lookup table.

    For each cell (col × row) in the 14×4 board:
      - Call cell_to_joints_deg(cell) to get the target joint angles in degrees.
      - Convert to radians.
      - Compute delta from home (HOME_RAD).
      - Clip delta to [-π, π] so the Tanh output can represent it after
        dividing by π (normalisation applied inside train()).
      - Store: observation = zeros(21), action = delta_rad / π  (normalised)

    Returns
    -------
    states  : np.ndarray, shape (N, 21), dtype float32
    actions : np.ndarray, shape (N, 6),  dtype float32
    """
    # Import here so the path manipulation above is in effect
    from grid_mover import cell_to_joints_deg, GRID_COLS, GRID_ROWS

    home = np.array(HOME_RAD, dtype=np.float64)

    states_list  = []
    actions_list = []

    for col_char in GRID_COLS:
        for row_char in GRID_ROWS:
            cell = col_char + row_char
            deg  = cell_to_joints_deg(cell)          # list[6] degrees
            rad  = np.array([math.radians(d) for d in deg], dtype=np.float64)
            delta = rad - home                       # joint delta from home

            # Normalise to [-1, 1] via /pi so Tanh output is meaningful
            delta_norm = np.clip(delta / math.pi, -1.0, 1.0)

            states_list.append(np.zeros(OBS_DIM, dtype=np.float32))
            actions_list.append(delta_norm.astype(np.float32))

    states  = np.stack(states_list,  axis=0)   # (56, 21)
    actions = np.stack(actions_list, axis=0)   # (56, 6)

    return states, actions


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
          save_path: str = 'bc_policy.pth') -> BCPolicy:
    """
    Train BCPolicy via supervised regression on the grid-mover dataset.

    Parameters
    ----------
    epochs     : number of training epochs
    batch_size : mini-batch size
    lr         : Adam learning rate
    save_path  : where to save the trained weights (.pth)

    Returns
    -------
    policy : trained BCPolicy
    """
    print('[BC] Building dataset ...')
    states, actions = build_dataset()
    print(f'[BC] Dataset: {states.shape[0]} samples '
          f'(states {states.shape}, actions {actions.shape})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[BC] Training on {device}')

    policy   = BCPolicy().to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    s_tensor = torch.from_numpy(states).to(device)
    a_tensor = torch.from_numpy(actions).to(device)
    dataset  = TensorDataset(s_tensor, a_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          drop_last=False)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        policy.train()
        for s_batch, a_batch in loader:
            pred  = policy(s_batch)
            loss  = criterion(pred, a_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * s_batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f'[BC] Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}')

    torch.save(policy.state_dict(), save_path)
    print(f'[BC] Saved policy weights → {save_path}')
    return policy


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    trained_policy = train(epochs=50, save_path='bc_policy.pth')
    print('[BC] Behavioural cloning complete.')
