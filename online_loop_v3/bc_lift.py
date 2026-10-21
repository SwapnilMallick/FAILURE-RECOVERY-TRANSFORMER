"""
Behaviour-cloning policy for the pick-up task (v3, Stage B).

Same role as bc_policy.py (reaching): trained inside the run on every
successful trajectory in D, it takes over from the fallback path's end and
finishes the task -- here: approach, descend, grasp, lift.

Model (BCLiftNet): current state's image embedding (V*D, no history)
    -> Linear(V*D, 256) -> GELU -> Linear(256, 256) -> GELU -> Linear(256, 7)
  out[0:3] translation in action units (x 1 cm, capped at length 1),
  out[3:6] rotation in action units (x 0.1 rad, capped at length 1),
  out[6]   gripper logit (> 0 = close).

Labels (per trajectory in D, `lookahead` k):
  * on-trajectory state i: the trajectory's own motion over the next k
    actions, averaged per action and capped -- translation
    (eef[i+k] - eef[i]) / k, rotation rotvec(R[i+k] R[i]^T) / k -- so it is
    zero while grasping; gripper = the command of action i.
  * corrective (APPROACH phase only, decision 4): the arm is placed 0..aug_pos
    away and rotated 0..aug_rot about a random axis from approach state i
    (cube resting on the table, gripper open); label = one capped step toward
    the trajectory's state nearest (pose distance) + k, gripper open.
Loss: squared error on the 6 motion outputs + binary cross-entropy on the
gripper.

Rollout (bc_lift_rollout): embed -> BC -> env.act(...) until the cube is
lifted (env success) or max_steps; a blocked action (collision / outside the
workspace) is retried with a perturbed direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from lift_env import dist_to_poses, mat_to_rotvec, pose_dist, rotvec_to_mat
from scripted_pick import ROT_STEP, STEP, to_norm


@dataclass
class BCLiftConfig:
    hidden: int = 256
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lookahead: int = 2
    n_aug: int = 20                  # corrective examples per approach state
    aug_pos: float = 0.02            # max position offset (m)
    aug_rot: float = 0.2             # max rotation offset (rad, ~11 deg)
    aug_tries: int = 30
    val_frac: float = 0.1
    n_retry: int = 5
    retry_noise: float = 0.3
    seed: int = 0


class BCLiftNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 7))

    def forward(self, x):
        return self.net(x)


def _cap(v, m=1.0):
    n = np.linalg.norm(v)
    return v if n <= m else v * (m / n)


def _motion_label(p, Rm, p_t, R_t, k: int = 1) -> np.ndarray:
    """6 action-unit numbers: per-action translation / rotation toward
    (p_t, R_t) over k actions, each capped at one step."""
    return np.concatenate([_cap((np.asarray(p_t) - p) / k / STEP),
                           _cap(mat_to_rotvec(np.asarray(R_t) @ np.asarray(Rm).T) / k / ROT_STEP)])


def build_bc_lift_dataset(env, encoder, trajs, cfg: BCLiftConfig, log=print) -> Dict[str, np.ndarray]:
    """trajs: successful pick trajectories, each with eefs, rots, states,
    grip_cmd (command of the action leaving each state), phases (per action)
    and tokens (embedding of each state = tokens[i][:-7], last state = end_emb)."""
    rng = np.random.default_rng(cfg.seed + 9_000_003)
    X, Y, kind = [], [], []
    n_fail = 0
    for tr in trajs:
        E, Rs, T = np.asarray(tr.eefs), np.asarray(tr.rots), len(tr.tokens)
        embs = [tok[:-7] for tok in tr.tokens]
        for i in range(T):                                     # on-trajectory examples
            j = min(i + cfg.lookahead, T)
            y = _motion_label(E[i], Rs[i], E[j], Rs[j], max(j - i, 1))
            X.append(embs[i]); Y.append(np.concatenate([y, [tr.grip_cmd_act[i]]])); kind.append(0)
        appr = [i for i in range(T) if tr.phases[i] == "approach"]
        for i in appr:                                         # corrective, approach only
            for _ in range(cfg.n_aug):
                for _try in range(cfg.aug_tries):
                    env.set_state(tr.states[i])
                    d = rng.normal(size=3); d /= np.linalg.norm(d)
                    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                    p = E[i] + d * rng.uniform(0, cfg.aug_pos)
                    Rm = rotvec_to_mat(ax * rng.uniform(0, cfg.aug_rot)) @ Rs[i]
                    if not env.place_arm(p, Rm):
                        continue
                    pe, Re = env.eef_pose()
                    near = min(appr, key=lambda a_: pose_dist(pe, Re, E[a_], Rs[a_]))
                    j = min(near + cfg.lookahead, T)
                    y = _motion_label(pe, Re, E[j], Rs[j], 1)
                    X.append(encoder.embed(env).reshape(-1)); Y.append(np.concatenate([y, [-1.0]]))
                    kind.append(1)
                    break
                else:
                    n_fail += 1
    kind = np.array(kind)
    log(f"BC-lift dataset from {len(trajs)} trajectory(ies): {int((kind == 0).sum())} on-trajectory + "
        f"{int((kind == 1).sum())} approach-phase corrective examples (<= {100 * cfg.aug_pos:.0f} cm, "
        f"<= {np.degrees(cfg.aug_rot):.0f} deg){f'; {n_fail} placements failed' if n_fail else ''}")
    return dict(X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32), kind=kind)


def merge_bc_lift_datasets(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "Y", "kind")}


def train_bc_lift(data: Dict[str, np.ndarray], cfg: BCLiftConfig, log=print) -> BCLiftNet:
    rng = np.random.default_rng(cfg.seed + 10_000_003)
    aug = np.flatnonzero(data["kind"] == 1)
    n_val = int(round(cfg.val_frac * len(aug)))
    val = rng.choice(aug, size=n_val, replace=False) if n_val else np.array([], int)
    train = np.setdiff1d(np.arange(len(data["X"])), val)
    X, Y = torch.tensor(data["X"]), torch.tensor(data["Y"])
    torch.manual_seed(cfg.seed + 11_000_003)
    model = BCLiftNet(X.shape[1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(cfg.epochs):
        out = model(X[train])
        loss = ((out[:, :6] - Y[train, :6]) ** 2).sum(1).mean() + bce(out[:, 6], (Y[train, 6] + 1) / 2)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(X).numpy()
    Yn = data["Y"]

    def ang(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 0.2) & (nb > 0.2)
        c = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
        return np.degrees(np.arccos(np.clip(c, -1, 1))).mean() if ok.any() else float("nan")

    grip_acc = ((out[:, 6] > 0) == (Yn[:, 6] > 0)).mean()
    tr_aug = np.setdiff1d(aug, val)
    on = data["kind"] == 0
    log(f"BC-lift trained ({cfg.epochs} epochs, final loss {loss.item():.3e}) | translation angle error: "
        f"on-trajectory {ang(out[on, :3], Yn[on, :3]):.1f} deg, corrective train {ang(out[tr_aug, :3], Yn[tr_aug, :3]):.1f}"
        + (f", held-out {ang(out[val, :3], Yn[val, :3]):.1f} deg" if len(val) else "")
        + f" | gripper accuracy {100 * grip_acc:.1f}%")
    return model


class BCLiftPolicy:
    def __init__(self, model: BCLiftNet):
        self.model = model

    @torch.no_grad()
    def act(self, emb):
        """-> (dpos m, drot rad, grip) for env.act, and the normalized action."""
        out = self.model(torch.tensor(np.asarray(emb, np.float32).reshape(1, -1)))[0].numpy()
        t, r = _cap(out[:3]), _cap(out[3:6])
        g = 1.0 if out[6] > 0 else -1.0
        return t * STEP, r * ROT_STEP, g


def bc_lift_rollout(policy: BCLiftPolicy, env, encoder, start_state, start_grip, D_poses,
                    cfg: BCLiftConfig, max_steps: int, rng=None, start_emb=None) -> Dict:
    """Runs the policy from a full simulator state. D_poses = (P (N,3), R (N,3,3))
    of every state in D, for the distance trace. Returns dict(success, end,
    states, eefs, rots, embs, norm_actions, tokens, grip_cmd, dist (cm),
    max_dev, mean_dev)."""
    rng = rng or np.random.default_rng(cfg.seed + 12_000_003)
    env.set_state(start_state, last_grip=start_grip)
    emb = start_emb if start_emb is not None else encoder.embed(env)
    states, eefs, rots, embs = [env.get_state()], [env.eef_pos()], [env.eef_rot()], [emb]
    acts, tokens, grip_cmd = [], [], [start_grip]
    end = "max_steps"
    for _ in range(max_steps):
        if env.success():
            end = "reached"
            break
        dpos, drot, g = policy.act(emb)
        moved = False
        for r in range(cfg.n_retry + 1):
            dp = dpos if r == 0 else _cap(dpos / STEP + cfg.retry_noise * rng.normal(size=3)) * STEP
            s_before, lg = env.get_state(), env.last_grip
            out = env.act(dp, drot, g)
            if out["collided"] or out["out_of_bounds"]:
                env.set_state(s_before, last_grip=lg)
                continue
            moved = True
            break
        if not moved:
            end = "blocked"
            break
        a_n = to_norm(dp, drot, g)
        tokens.append(np.concatenate([emb.reshape(-1), a_n]).astype(np.float32))
        acts.append(a_n); grip_cmd.append(g)
        emb = encoder.embed(env)
        states.append(env.get_state()); eefs.append(env.eef_pos()); rots.append(env.eef_rot()); embs.append(emb)
    else:
        if env.success():
            end = "reached"
    P, Rr = D_poses                                     # pose distance (cm-equivalent) to D
    dist = np.array([100 * dist_to_poses(e, Rm, P, Rr).min() for e, Rm in zip(eefs, rots)])
    return dict(success=end == "reached", end=end, states=np.asarray(states), eefs=np.asarray(eefs),
                rots=np.asarray(rots), embs=embs, norm_actions=np.asarray(acts), tokens=tokens,
                grip_cmd=grip_cmd, dist=dist, max_dev=float(dist.max()), mean_dev=float(dist.mean()))
