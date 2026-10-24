"""
Memory-Consistent Neural Network (MCNN) BC for the pick-up task.

Sridhar et al., "Memory-Consistent Neural Networks for Imitation Learning",
ICLR 2024 (arXiv 2310.06171; code github.com/kaustubhsridhar/MCNN). The
policy is a mixture of the nearest memory's action and a bounded network:

    pi(x) = a_m * w + L * (1 - w) * tanh(f_theta(x)),   w = exp(-lambda * d(x, s_m))

(s_m, a_m) = the nearest memory to the embedding x (Euclidean distance d in
the standardized 3-view DINOv2 embedding space, brute force as in the repo's
update_data.py). At a memory the output IS its stored label; far from every
memory it is the network, squashed to [-L, L]. The loss is applied to pi(x),
so the network only learns what the memories do not cover.

Choices for this project (2026-10-23):
  * training data = the same BC-lift dataset as the MLP (on-trajectory +
    approach/descend corrective examples, bc_lift.build_bc_lift_dataset);
  * memories (budget mem_frac of the training examples, default 0.5): EVERY
    on-trajectory example (incl. later tau_k, fallback part included), the
    rest of the budget filled by neural gas over the corrective examples, each
    node snapped to its nearest real example (paper, Algorithm 1). Plain
    neural gas (Martinetz et al. 1993) here instead of the repo's neupy
    growing neural gas -- fixed node count, no extra dependency;
  * all 7 outputs inside the mixture, L = 1: the 6 motion numbers (already
    capped to [-1, 1] per component) and the gripper as a continuous value in
    [-1, 1] (-1 open, +1 close; executed by its sign); loss = squared error
    summed over the 7 outputs (no BCE);
  * lambda auto-calibrated per training run: memory weight 0.5 at the median
    nearest-memory distance of the non-memory training examples,
    lambda = ln 2 / median (the repo's lambda is in raw-observation units and
    does not transfer to 1152-D embeddings).
The validation split (10% of the corrective examples) is made BEFORE choosing
memories, so held-out examples are never memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from scripted_pick import ROT_STEP, STEP


@dataclass
class MCNNConfig:
    mem_frac: float = 0.5           # memory budget, fraction of the training examples
    lam: Optional[float] = None     # None = auto-calibrate (w = 0.5 at the median distance)
    L: float = 1.0                  # output bound far from the memories
    ng_epochs: int = 20             # neural-gas passes over the corrective examples


# --------------------------------------------------------------- neural gas
def neural_gas(X: np.ndarray, k: int, epochs: int, rng, eps=(0.5, 0.005), lam=None) -> np.ndarray:
    """Plain neural gas: k nodes; for each sample (shuffled, `epochs` passes)
    every node moves toward it by eps(t) * exp(-rank / lam(t)), rank = the
    node's distance rank; eps and lam decay exponentially (lam from k/2 to
    0.01). Returns the (k, dim) node positions."""
    X = np.asarray(X, np.float32)
    if k >= len(X):
        return X.copy()
    W = X[rng.choice(len(X), size=k, replace=False)].copy()
    lam_i, lam_f = (k / 2.0, 0.01) if lam is None else lam
    T = epochs * len(X)
    t = 0
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            frac = t / max(T - 1, 1)
            e = eps[0] * (eps[1] / eps[0]) ** frac
            la = lam_i * (lam_f / lam_i) ** frac
            diff = X[i] - W
            rank = np.argsort(np.argsort((diff ** 2).sum(1)))
            W += (e * np.exp(-rank / la))[:, None].astype(np.float32) * diff
            t += 1
    return W


def snap_to_examples(W: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Indices (into X) of k distinct real examples: each node snapped to its
    nearest example; duplicates are replaced by the examples farthest from
    the chosen set (greedy), so the budget is always met."""
    d = torch.cdist(torch.tensor(W), torch.tensor(X)).numpy()
    chosen = list(dict.fromkeys(int(j) for j in d.argmin(1)))
    if len(chosen) < k:
        dmin = torch.cdist(torch.tensor(X), torch.tensor(X[chosen])).min(1).values.numpy()
        while len(chosen) < k:
            j = int(dmin.argmax())
            chosen.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
    return np.array(chosen[:k])


def select_memories(X: np.ndarray, kind: np.ndarray, frac: float, epochs: int, rng) -> np.ndarray:
    """Indices of the memories among the training examples: all on-trajectory
    examples (kind 0) + neural gas over the corrective ones (kind 1) up to
    round(frac * N) in total."""
    on = np.flatnonzero(kind == 0)
    corr = np.flatnonzero(kind == 1)
    k_ng = max(int(round(frac * len(X))) - len(on), 0)
    if k_ng == 0 or len(corr) == 0:
        return on
    if k_ng >= len(corr):
        return np.concatenate([on, corr])
    W = neural_gas(X[corr], k_ng, epochs, rng)
    return np.concatenate([on, corr[snap_to_examples(W, X[corr], k_ng)]])


def nearest_memory(X, M) -> tuple:
    """(index, distance) of the nearest memory for each row of X."""
    d = torch.cdist(torch.as_tensor(np.asarray(X, np.float32)), torch.as_tensor(np.asarray(M, np.float32)))
    dist, idx = d.min(1)
    return idx.numpy(), dist.numpy()


# -------------------------------------------------------------------- model
class MCNNNet(nn.Module):
    """Backbone = the MLP-BC network (1152 -> 256 -> 256 -> 7)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 7))

    def forward(self, x, mem_y, dist, lam: float, L: float):
        w = torch.exp(-lam * dist).unsqueeze(-1)
        return mem_y * w + L * (1 - w) * torch.tanh(self.net(x))


class MCNNPolicy:
    def __init__(self, model: MCNNNet, M: np.ndarray, MY: np.ndarray, M_phase, lam: float, L: float):
        self.model, self.M, self.MY, self.M_phase, self.lam, self.L = model, M, MY, list(M_phase), lam, L
        self.M_t = torch.as_tensor(M, dtype=torch.float32)
        self.last_info = None

    @torch.no_grad()
    def act(self, emb):
        """-> (dpos m, drot rad, grip) for env.act; self.last_info = (memory
        weight, distance, phase of the nearest memory)."""
        x = torch.as_tensor(np.asarray(emb, np.float32).reshape(1, -1))
        d, j = torch.cdist(x, self.M_t)[0].min(0)
        out = self.model(x, torch.as_tensor(self.MY[int(j)][None]), d.reshape(1), self.lam, self.L)[0].numpy()
        self.last_info = (float(np.exp(-self.lam * float(d))), float(d), self.M_phase[int(j)])
        cap = lambda v: v if np.linalg.norm(v) <= 1 else v / np.linalg.norm(v)
        return cap(out[:3]) * STEP, cap(out[3:6]) * ROT_STEP, (1.0 if out[6] > 0 else -1.0)


def train_bc_lift_mcnn(data: Dict[str, np.ndarray], bccfg, mcfg: MCNNConfig, log=print) -> MCNNPolicy:
    """data = bc_lift.build_bc_lift_dataset / merge output (X, Y, kind, phase).
    Validation split -> memories -> lambda -> full-batch Adam on pi(x)."""
    rng = np.random.default_rng(bccfg.seed + 10_000_003)
    Xa, Ya, kind = data["X"].astype(np.float32), data["Y"].astype(np.float32), data["kind"]
    phase = np.asarray(data.get("phase", np.array(["?"] * len(Xa))))
    aug = np.flatnonzero(kind == 1)
    n_val = int(round(bccfg.val_frac * len(aug)))
    val = rng.choice(aug, size=n_val, replace=False) if n_val else np.array([], int)
    train = np.setdiff1d(np.arange(len(Xa)), val)

    mem = train[select_memories(Xa[train], kind[train], mcfg.mem_frac, mcfg.ng_epochs,
                                np.random.default_rng(bccfg.seed + 13_000_003))]
    M, MY, M_phase = Xa[mem], Ya[mem], phase[mem]
    idx_all, d_all = nearest_memory(Xa, M)
    non_mem = np.setdiff1d(train, mem)
    d_ref = d_all[non_mem] if len(non_mem) else torch.cdist(torch.tensor(M), torch.tensor(M)).fill_diagonal_(np.inf).min(1).values.numpy()
    lam = mcfg.lam if mcfg.lam is not None else float(np.log(2) / max(np.median(d_ref), 1e-6))
    ph, cnt = np.unique(M_phase, return_counts=True)
    log(f"MCNN memories: {len(mem)}/{len(train)} training examples ({100 * len(mem) / len(train):.0f}%; "
        f"{int((kind[mem] == 0).sum())} on-trajectory + {int((kind[mem] == 1).sum())} corrective by neural gas) | "
        f"by phase {dict(zip(ph.tolist(), cnt.tolist()))} | nearest-memory distance of non-memory examples: "
        f"median {np.median(d_ref):.1f}, 10-90% {np.percentile(d_ref, 10):.1f}-{np.percentile(d_ref, 90):.1f} | "
        f"lambda {lam:.4f} ({'auto: w = 0.5 at the median' if mcfg.lam is None else 'fixed'}), L {mcfg.L}")

    X, Y = torch.tensor(Xa), torch.tensor(Ya)
    MYn, D = torch.tensor(MY[idx_all]), torch.tensor(d_all)
    torch.manual_seed(bccfg.seed + 11_000_003)
    model = MCNNNet(X.shape[1], bccfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=bccfg.lr, weight_decay=bccfg.weight_decay)
    tr = torch.as_tensor(train)
    for _ in range(bccfg.epochs):
        out = model(X[tr], MYn[tr], D[tr], lam, mcfg.L)
        loss = ((out - Y[tr]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(X, MYn, D, lam, mcfg.L).numpy()

    def ang(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 0.2) & (nb > 0.2)
        c = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
        return np.degrees(np.arccos(np.clip(c, -1, 1))).mean() if ok.any() else float("nan")

    on = np.flatnonzero(kind == 0)
    tr_aug = np.setdiff1d(aug, val)
    grip_acc = ((out[:, 6] > 0) == (Ya[:, 6] > 0)).mean()
    w = np.exp(-lam * d_all)
    log(f"MCNN-BC trained ({bccfg.epochs} epochs, final loss {loss.item():.3e}) | translation angle error: "
        f"on-trajectory {ang(out[on, :3], Ya[on, :3]):.1f} deg, corrective train {ang(out[tr_aug, :3], Ya[tr_aug, :3]):.1f}"
        + (f", held-out {ang(out[val, :3], Ya[val, :3]):.1f} deg (memory weight {np.median(w[val]):.2f} median)"
           if len(val) else "")
        + f" | gripper accuracy {100 * grip_acc:.1f}%")
    return MCNNPolicy(model, M, MY, M_phase, lam, mcfg.L)
