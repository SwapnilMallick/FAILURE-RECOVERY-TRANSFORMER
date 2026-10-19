"""
Ensemble of small feed-forward Q-networks, the DNN replacement for T_D.

Each member is QNet: Q(s, a) from the current state's standardized image
embedding (all views flattened) plus the 3-D action (scaled by
1/action_step), two GELU hidden layers, scalar output; no history, no
dropout. Members are trained on the same (state, action) -> Q pairs and
differ only in their random initial weights.

Uncertainty of a STATE: U(s) = max_i Q^i(s, a) - min_i Q^i(s, a) over the
members, averaged over a few random exploration-style actions a (uniform
direction, length uniform in the action_step ball -- or exactly action_step
with fixed_len, matching a fixed-step exploration). The members barely
react to the action, but Q(s, a) needs one; averaging removes most of the
dependence on which action happened to be drawn.

Used by fallback_explore.py (--td-model dnn) and probe_dnn_ensemble.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, action_scale: float):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a * self.action_scale], dim=-1)).squeeze(-1)


def train_dnn_ensemble(states: np.ndarray, actions: np.ndarray, labels: np.ndarray,
                       n_models: int, hidden: int, epochs: int, lr: float, seed: int,
                       action_step: float, log=None) -> List[QNet]:
    """Full-batch Adam on squared error; member i is initialised under
    torch.manual_seed(seed + 1000 * i)."""
    S = torch.tensor(np.asarray(states), dtype=torch.float32)
    A = torch.tensor(np.asarray(actions), dtype=torch.float32)
    Y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    models = []
    for i in range(n_models):
        torch.manual_seed(seed + 1000 * i)
        m = QNet(S.shape[1] + 3, hidden, 1.0 / action_step)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            loss = ((m(S, A) - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        models.append(m)
        if log:
            log(f"model {i} trained ({epochs} epochs, final MSE {loss.item():.2e})")
    return models


class DNNEnsemble:
    """Scores states with a trained list of QNets (see module docstring)."""

    def __init__(self, models: List[QNet], action_step: float, n_probe_actions: int, rng,
                 fixed_len: bool = False):
        self.models = models
        self.fixed_len = fixed_len          # probe actions exactly action_step long
        self.action_step = action_step
        self.n_probe = n_probe_actions
        self.rng = rng

    @torch.no_grad()
    def q_all(self, emb, act) -> np.ndarray:
        """Q of every member at one (state, action)."""
        s = torch.tensor(np.asarray(emb).reshape(1, -1), dtype=torch.float32)
        a = torch.tensor(np.asarray(act).reshape(1, 3), dtype=torch.float32)
        return np.array([float(m(s, a)) for m in self.models])

    def random_action(self) -> np.ndarray:
        v = self.rng.normal(size=3)
        length = self.action_step if self.fixed_len else self.action_step * self.rng.random() ** (1 / 3)
        return v * length / np.linalg.norm(v)

    def state_u(self, emb, actions: Optional[List[np.ndarray]] = None) -> Tuple[float, float]:
        """(mean Q, U) of a state: U = max - min of Q over members, both
        averaged over n_probe random actions (or the given actions)."""
        acts = actions if actions is not None else [self.random_action()
                                                    for _ in range(self.n_probe)]
        qs = np.stack([self.q_all(emb, a) for a in acts])        # (n_actions, n_models)
        return float(qs.mean()), float((qs.max(axis=1) - qs.min(axis=1)).mean())

    def state_dicts(self):
        return [m.state_dict() for m in self.models]
