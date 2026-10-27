"""
Behaviour-cloning (BC) policy that follows tau_1 to the cube.

Purpose: once the fallback loop has brought the arm close to tau_1, hand
control to this policy so it drives the arm along tau_1 to the cube; the
fallback path plus this rollout is then a new successful trajectory for D.

Model (BCNet)
-------------
    image embedding of the CURRENT state (V*D standardized DINOv2 features,
    the same input the DNN ensemble gets; no history)
      -> Linear(V*D, 256) -> GELU -> Linear(256, 256) -> GELU -> Linear(256, 3)
    The 3 outputs are a direction; the executed action is that direction
    rescaled to exactly action_step (1 cm), matching fixed-step exploration.

Training data (build_bc_dataset)
--------------------------------
Built from EVERY successful trajectory in D (tau_1 at first; later also the
tau_2's), since the fallback can merge into any of them. Per trajectory:
  * Waypoints W = that trajectory's states followed by the cube position.
  * Label of a state p whose nearest state ON THAT TRAJECTORY is i: the unit
    direction from p to W[i + lookahead] (default 2 steps ahead). On the
    trajectory this is its own direction smoothed over `lookahead` steps;
    off it, it steers back while still progressing toward the cube.
  * On-trajectory examples: every state (embedding already in its tokens),
    labelled as above.
  * Corrective examples (DART-style): for every state, n_aug points offset
    from it by aug_min..aug_max (0-2 cm) in a random 3-D direction. The arm
    is PLACED there -- IK seeded from that state's joint configuration, so
    its posture stays like the trajectory's -- and the point is kept
    only if IK converges, it is inside the workspace and the arm is
    collision-free there. Its images are rendered and embedded; nothing is
    executed. These teach the policy to recover from the small deviations
    its own imperfect actions cause (covariate shift / compounding error).
  * Labels use true gripper positions (as the scripted demo itself does);
    the policy only ever sees images.

Loss: squared error between the raw 3-D output and the unit target
direction (keeps the output near length 1, so gradients stay well behaved),
Adam with weight decay. A random val_frac of the corrective examples is held
out and the mean angle between prediction and target is reported.

Rollout (bc_rollout): render -> embed -> BC direction -> 1 cm action ->
IK + collision check (the same execution model as exploration) -> repeat,
until the gripper reaches the cube (task.reached) or max_steps. A blocked
action (IK failure / collision / out of bounds) is retried with a slightly
perturbed direction; if every retry fails the rollout ends "blocked".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from online_cost_transformer import fuse


@dataclass
class BCConfig:
    hidden: int = 256
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lookahead: int = 2               # label = direction to the tau_1 state this many steps ahead
    n_aug: int = 20                  # corrective examples per tau_1 state
    aug_min: float = 0.0             # their offset from that state (m)
    aug_max: float = 0.02
    aug_tries: int = 30              # placement attempts per corrective example
    val_frac: float = 0.1            # held-out share of the corrective examples
    n_retry: int = 5                 # rollout: perturbed retries of a blocked action
    retry_noise: float = 0.3         # rollout: direction-noise std for those retries
    seed: int = 0


class BCNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def waypoints(tau1, cube_pos) -> np.ndarray:
    return np.vstack([np.asarray(tau1.eefs), np.asarray(cube_pos)[None]])


def nearest_idx(p, tau1_pts: np.ndarray) -> int:
    return int(np.linalg.norm(tau1_pts - p, axis=1).argmin())


def target_direction(p, near: int, W: np.ndarray, lookahead: int) -> np.ndarray:
    """Unit direction from p to the waypoint `lookahead` steps past tau_1
    state `near` (clamped to the last waypoint, the cube)."""
    for k in range(lookahead, len(W)):                     # skip a goal we already sit on
        g = W[min(near + k, len(W) - 1)]
        v = g - p
        n = np.linalg.norm(v)
        if n > 1e-4 or near + k >= len(W) - 1:
            return v / max(n, 1e-9)
    return np.zeros(3)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_bc_dataset(task, encoder, trajs, cfg: BCConfig, log=print) -> Dict[str, np.ndarray]:
    """trajs: one Traj or a list (the successful trajectories in D; each needs
    tokens, eefs, qs and end_emb). Returns dict(X (N, V*D) embeddings, Y (N, 3)
    unit targets, P (N, 3) EEF positions, kind (N,) 0 = on a trajectory,
    1 = corrective, near (N,) nearest state index, src (N,) trajectory index)."""
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    ik, world = task.ik, task.world
    rng = np.random.default_rng(cfg.seed + 9_000_003)
    X, Y, P, kind, near, src = [], [], [], [], [], []
    n_fail = 0
    for k, tr in enumerate(trajs):
        T = np.asarray(tr.eefs)
        W = np.vstack([T, np.asarray(task.cube_pos)[None]])
        embs = [tok[:-3] for tok in tr.tokens] + [np.asarray(tr.end_emb).reshape(-1)]   # states 0..T
        for i in range(len(T)):                             # on-trajectory examples
            X.append(embs[i]); P.append(T[i]); kind.append(0); near.append(i); src.append(k)
            Y.append(target_direction(T[i], i, W, cfg.lookahead))
        for i in range(len(T)):                             # corrective examples
            q_seed = tr.qs[i]
            for _ in range(cfg.n_aug):
                for _try in range(cfg.aug_tries):
                    d = rng.normal(size=3)
                    d /= np.linalg.norm(d)
                    p = T[i] + d * rng.uniform(cfg.aug_min, cfg.aug_max)
                    if not world._in_bounds(p):
                        continue
                    q, ok = ik.solve_ik(p, q_init=q_seed)
                    if not ok:
                        continue
                    ik.set_q(q)
                    if world._forbidden_contact():
                        continue
                    pe = ik.eef_pos()
                    j = nearest_idx(pe, T)
                    X.append(encoder.embed(ik).reshape(-1)); P.append(pe); kind.append(1)
                    near.append(j); src.append(k)
                    Y.append(target_direction(pe, j, W, cfg.lookahead))
                    break
                else:
                    n_fail += 1
    task.reset_to_start()
    kind = np.array(kind)
    log(f"BC dataset from {len(trajs)} trajectory(ies) in D: {int((kind == 0).sum())} "
        f"on-trajectory + {int((kind == 1).sum())} corrective examples (offsets "
        f"{100 * cfg.aug_min:.1f}-{100 * cfg.aug_max:.1f} cm, lookahead {cfg.lookahead})"
        f"{f'; {n_fail} placements failed' if n_fail else ''}")
    return dict(X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32), P=np.asarray(P),
                kind=kind, near=np.array(near), src=np.array(src))


def merge_bc_datasets(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenates per-trajectory datasets (from build_bc_dataset); `src`
    becomes the trajectory's index in the list."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "Y", "P", "kind", "near")}
    out["src"] = np.concatenate([np.full(len(p["X"]), j) for j, p in enumerate(parts)])
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _angle_deg(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((pn * tgt).sum(1), -1.0, 1.0)))


def train_bc(data: Dict[str, np.ndarray], cfg: BCConfig, log=print) -> BCNet:
    rng = np.random.default_rng(cfg.seed + 10_000_003)
    aug = np.flatnonzero(data["kind"] == 1)
    n_val = int(round(cfg.val_frac * len(aug)))
    val = rng.choice(aug, size=n_val, replace=False) if n_val else np.array([], int)
    train = np.setdiff1d(np.arange(len(data["X"])), val)
    X = torch.tensor(data["X"]); Y = torch.tensor(data["Y"])
    torch.manual_seed(cfg.seed + 11_000_003)
    model = BCNet(X.shape[1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        loss = ((model(X[train]) - Y[train]) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy()
    ang = _angle_deg(pred, data["Y"])
    on = data["kind"] == 0
    tr_aug = np.setdiff1d(aug, val)
    log(f"BC trained ({cfg.epochs} epochs, final loss {loss.item():.2e}) | mean angle error: "
        f"on-tau_1 {ang[on].mean():.1f} deg, corrective train {ang[tr_aug].mean():.1f} deg"
        + (f", corrective held-out {ang[val].mean():.1f} deg" if len(val) else ""))
    return model


class BCPolicy:
    def __init__(self, model: BCNet, action_step: float):
        self.model = model
        self.action_step = action_step

    @torch.no_grad()
    def direction(self, emb) -> np.ndarray:
        out = self.model(torch.tensor(np.asarray(emb, np.float32).reshape(1, -1)))[0].numpy()
        return out / max(np.linalg.norm(out), 1e-9)

    def action(self, emb) -> np.ndarray:
        return self.direction(emb) * self.action_step


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def bc_rollout(policy: BCPolicy, task, encoder, start_q, tau1_pts: np.ndarray, cfg: BCConfig,
               max_steps: int, rng=None, start_emb=None) -> Dict:
    """Runs the policy from joint configuration start_q. Returns dict with
    success, end ("reached" | "max_steps" | "blocked"), eefs, qs, embs,
    actions, tokens ([emb, a] per step, like D's trajectories), dist (cm to
    the nearest tau_1 state per state), max_dev, progress (nearest tau_1
    index per state)."""
    ik, world = task.ik, task.world
    rng = rng or np.random.default_rng(cfg.seed + 12_000_003)
    task.reset_to_start()
    ik.set_q(start_q)
    q = np.asarray(start_q, float).copy()
    eef = ik.eef_pos()
    emb = start_emb if start_emb is not None else encoder.embed(ik)
    eefs, qs, embs, actions, tokens = [eef], [q.copy()], [emb], [], []
    end = "max_steps"
    for _ in range(max_steps):
        if task.reached(eef):
            end = "reached"
            break
        base = policy.direction(emb)
        moved = False
        for r in range(cfg.n_retry + 1):
            dirv = base if r == 0 else base + cfg.retry_noise * rng.normal(size=3)
            dirv /= np.linalg.norm(dirv)
            a = dirv * policy.action_step
            if not world._in_bounds(eef + a):
                continue
            q_to, ok = ik.solve_ik(eef + a, q_init=q)
            if not ok or world.collides(q, q_to):
                ik.set_q(q)
                continue
            moved = True
            break
        if not moved:
            ik.set_q(q)
            end = "blocked"
            break
        ik.set_q(q_to)
        tokens.append(fuse(emb.reshape(-1), a))
        actions.append(a)
        q, eef = q_to, ik.eef_pos()
        emb = encoder.embed(ik)
        eefs.append(eef); qs.append(q.copy()); embs.append(emb)
    else:
        if task.reached(eef):
            end = "reached"
    E = np.asarray(eefs)
    dist = 100 * np.linalg.norm(E[:, None, :] - tau1_pts[None], axis=-1).min(1)
    prog = np.linalg.norm(E[:, None, :] - tau1_pts[None], axis=-1).argmin(1)
    return dict(success=end == "reached", end=end, eefs=E, qs=np.asarray(qs), embs=embs,
                actions=np.asarray(actions), tokens=tokens, dist=dist,
                max_dev=float(dist.max()), mean_dev=float(dist.mean()), progress=prog)
