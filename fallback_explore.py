"""
Falling back to a successful trajectory tau_1 with a Q-model T_D trained on
tau_1 and a Q-transformer T_explore (robosuite Lift/Panda cube-reaching
testbed, image-state tokens).

T_D (--td-model)
----------------
  * "dnn" (default): an ensemble of n_dnn (5) small feed-forward Q-networks
    (dnn_ensemble.py), Q(s, a) from the current state's image embedding and
    the action, no history. mu_TD of a STATE s is
        U(s) = max_i Q^i(s, a) - min_i Q^i(s, a),
    averaged over n_probe_actions random exploration-style actions, scored
    from s's own image. Everywhere below, "mu_TD" means this U: a candidate
    action's mu_TD is U of the state it leads to (rendered once per
    candidate), and T_explore's label for a step is -U of the state it leads
    into. Chosen after the band probes (probe_*.py): an ensemble's spread grew
    with distance from tau_1 where MC dropout's did not, and only when tau_1's
    actions vary -- hence the varied demo is the default too.
  * "transformer": the original single ImageCostTransformer with MC-dropout
    std as mu_TD, scored at (history, action) tokens. With --demo straight
    this reproduces the earlier runs.

Method (single fall-back iteration)
-----------------------------------
    T_D, T_explore <- freshly initialised (T_D: DNN ensemble or transformer)
    D              <- { tau_1 }        (tau_1 from a scripted reaching policy,
                                        varied or straight, see --demo)
    train T_D on tau_1
    s_0 <- initial state of tau_1 ;  s_1 <- a nearby perturbed start state
    start <- s_1 (empty history)
    repeat (one "round"):
        collect k m-step exploration trajectories from `start`; at every step
            a ~ uniform collision-free action (world.sample_free)
            accept a   iff   -mu_TD(h, a)  >  Q_exp(h, a) - mu_exp(h, a)
            (otherwise resample, up to max_retries; if nothing is accepted
             the trajectory ends early)
        train T_explore on the exploration buffer (all rounds so far); its
            label for every (h_t, a_t) is T_D's certainty there, -mu_TD(h_t, a_t)
        S  = every state visited this round (up to k*m, start state excluded)
        S' = the top keep_frac of S by mu_exp; i.e. alpha is set each round
             to the cutoff value of mu_exp that keeps that fraction
             (keep_frac = 1.0 for the DNN T_D, so S' = S; 0.5 for the transformer)
        s* = argmax_{s in S'}  -mu_TD(s)     (the state T_D is most certain about)
        chain  <- chain + the trajectory that reached s*, cut at s*
        stop   if  mu_TD(s*) ~ 0, i.e. mu_TD(s*) < mu_tol  (T_D is certain at s*:
               it looks like a tau_1 state)
        start <- s*  (with the history of the trajectory up to s*)
    tau_2 = chain = s_1 -> s*_1 -> s*_2 -> ... -> s*_last ; D <- D U {tau_2}
    There is no merging of similar states. EEF distance to tau_1 is logged
    as a diagnostic only; it never stops anything.

Interpretation choices (the method text leaves these open)
----------------------------------------------------------
  * Q-value (T_D only). Reward is 1 on the transition that reaches the cube
    (within eps_dist) and 0 otherwise, discount gamma. On tau_1 (length T)
    the label of token t = [emb(s_t), a_t] is the Monte-Carlo return
    gamma^(T-1-t). T_D is still trained on Q, but only its MC-dropout std
    (mu_TD) drives exploration; Q_TD is logged at s* for reference.
  * Acceptance rule. The update reads "mu_TD > Q_exp - mu_exp", but T_explore
    is trained on -mu_TD <= 0, so that would accept almost every action; it
    is implemented as certainty vs certainty, -mu_TD > Q_exp - mu_exp
    (confirmed with the author): take an action when T_D is more certain
    about it than T_explore's pessimistic estimate of T_D's certainty there.
  * T_explore labels. -mu_TD for every prefix of the trajectory (history
    included), from ONE batched MC-dropout pass over the whole sequence --
    the causal mask makes position t's output depend only on tokens[:t+1].
    T_D is frozen during the iteration, so labels are computed once, at
    collection time. Q_exp = mean of the n MC-dropout passes of T_explore.
  * Stop tolerance. "mu_TD ~ 0" is mu_TD(s*) < mu_tol; by default mu_tol is
    the MEAN mu_TD over tau_1's steps (s* must be at least as certain to T_D
    as tau_1 is on average; chosen with the author over the earlier max, which
    let almost every explored state pass). --mu-tol overrides it.
  * mu of a STATE. Models score (history, action) tokens, so a visited state
    s_j is scored at the token that led into it, (h_{j-1}, a_{j-1}) -- the
    same convention as before, now applied to every visited state, not only
    trajectory ends. mu = MC-dropout std; one batched pass per trajectory
    (mc_all) gives it for every position at once, for T_D and T_explore.
  * alpha is relative, not fixed (decided with the author): mu_exp shrinks
    every round as T_explore trains on a growing buffer, so any fixed alpha
    eventually leaves S' empty. Keeping the top keep_frac of each round's
    states by mu_exp means "more uncertain to T_explore than the rest of this
    round", and S' is never empty while S is not.
  * Fixed step length (fixed_step; default on for the DNN T_D): every
    exploration action, every varied-demo step and every random action the
    DNN uncertainty is averaged over is exactly action_step (1 cm) long, only
    its direction random -- so tau_1 and exploration use the same step size
    (with the straight demo's 1 cm steps too). Off, lengths are uniform in
    the action_step ball (mean 0.75 cm), as in the transformer runs.
  * DNN T_D: s* is chosen by U over ALL visited states (keep_frac = 1.0). In
    the first DNN runs the mu_exp filter removed the state closest to tau_1
    in 4-5 of 10 rounds -- once the single lowest-U state of the round --
    before U was looked at.
  * s* in the middle of a trajectory. tau_2 and the next round's history are
    that trajectory cut at s*; the rest of it is dropped. Every visited
    state's embedding is kept (Traj.embs) so the next round can start there.
  * T_D suffix augmentation (--td-suffix-aug, on by default): T_D is also
    trained on every suffix tau_1[i:] (labels unchanged), so it does not only
    know tau_1 states at the absolute positions they occupy in tau_1 --
    exploration histories start at s_1 / s*, not s_0.
  * Step sizes. tau_1 moves script_step (1 cm) per step, ~19 states for the
    ~0.22 m start-to-cube distance. Exploration actions are drawn uniformly
    from the ball of radius action_step, set to the same 1 cm, so T_D (which
    only sees tau_1's actions) is not queried on actions far larger than any
    it was trained on. gamma = 0.97 keeps Q at s_0 (gamma^(T-1) ~ 0.58)
    clear of zero over the longer horizon. The encoder's calibration random
    walk keeps the world's default 4 cm step so its standardization stats
    cover the same workspace region regardless of action_step.
  * s_1 is s_0's EEF displaced by a random direction with norm in
    [s1_min_dist, s1_max_dist], IK-reachable and collision-free.

Tokens are [ DINOv2 embeddings of agentview/sideview/frontview (V*D) , EEF
displacement action (3) ], exactly as in the rest of this project
(image_encoder.ImageEncoder + online_cost_transformer.ImageCostTransformer).
EEF xyz is never a model input; it is used only for the scripted policy, the
goal check, tau_2's labels and the diagnostics (distance to tau_1).

Usage:
    python fallback_explore.py                      # defaults
    python fallback_explore.py --k 8 --m 10 --max-rounds 10 --device mps
    python fallback_explore.py --smoke              # tiny, fast end-to-end check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from image_encoder import EncoderConfig, ImageEncoder
from dnn_ensemble import DNNEnsemble, train_dnn_ensemble
from plot_fallback_3d import PATH_COLOR, write_html
from online_cost_transformer import (
    ImageCostTransformer, causal_mask, collate, fuse, huber_loss, predict_last_mc,
)
from robosuite_env import WorldConfig
from task import RobosuiteTask, TaskConfig


@dataclass
class FallbackConfig:
    # --- transformer (field names shared with online_cost_transformer.Config,
    #     which ImageCostTransformer / predict_last_mc / collate read) ---
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1                 # also drives MC dropout (mu)
    max_len: int = 64
    mc_samples: int = 20
    huber_delta: float = 1.0
    lr: float = 1e-3
    batch_size: int = 32
    device: str = "cpu"
    token_fusion: str = "concat"

    # --- Q-values ---
    gamma: float = 0.97

    # --- T_D: "dnn" = ensemble of feed-forward Q-nets (dnn_ensemble.py),
    #     "transformer" = the single MC-dropout ImageCostTransformer ---
    td_model: str = "dnn"
    n_dnn: int = 5                       # ensemble members
    dnn_hidden: int = 256
    dnn_epochs: int = 400
    dnn_lr: float = 1e-3
    n_probe_actions: int = 3             # random actions U(s) is averaged over

    # --- tau_1 (scripted policy) + T_D ---
    demo: str = "varied"                 # "varied" (actions vary) | "straight"
    demo_noise: float = 1.0              # varied demo: direction-noise std
    script_step: float = 0.01            # m per scripted step (<= action_step)
    script_max_steps: int = 60
    td_epochs: int = 400
    td_suffix_aug: bool = True

    # --- s_1 ---
    s1_min_dist: float = 0.03            # s_1 lies 3-5 cm from s_0 (was 4-8 cm)
    s1_max_dist: float = 0.05

    # --- exploration ---
    action_step: float = 0.01            # exploration action bound (m), = world step_size
    fixed_step: Optional[bool] = None    # every exploration / varied-demo / DNN-probe action
                                         # exactly action_step long (random direction only);
                                         # None = True for dnn, False for transformer
    k: int = 8                           # trajectories per round
    m: int = 10                          # steps per trajectory
    max_retries: int = 10                # action resamples per step
    explore_epochs: int = 60             # T_explore epochs per round
    max_rounds: int = 10
    keep_frac: Optional[float] = None    # S' = top keep_frac of visited states by mu_exp;
                                         # None = 1.0 (all) for dnn, 0.5 for transformer
    mu_tol: Optional[float] = None       # stop when mu_TD(s*) < mu_tol;
                                         # None = mean mu_TD over tau_1

    seed: int = 0


@dataclass
class Traj:
    """tokens[t] = [emb(s_t), a_t]; eefs/qs hold the NEW states visited by this
    trajectory (eefs[0] = its start). `hist_len` leading tokens are the
    inherited history that led to the start state."""
    tokens: List[np.ndarray]
    eefs: List[np.ndarray]
    qs: List[np.ndarray]
    end_emb: np.ndarray
    embs: List[np.ndarray] = field(default_factory=list)   # embs[j] = emb(eefs[j])
    hist_len: int = 0
    reached: bool = False
    labels: Optional[np.ndarray] = None
    n_reject: int = 0
    n_sample_fail: int = 0
    stuck: bool = False                  # a step found no accepted action
    join_idx: int = -1                   # tau_2 only: index of the tau_1 state it joins
    # DNN T_D only: (Q, U) of the state each token leads into, parallel to tokens
    tok_qu: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return len(self.tokens) - self.hist_len


# --------------------------------------------------------------------------- #
# tau_1: scripted straight-line reaching policy
# --------------------------------------------------------------------------- #
def collect_scripted_trajectory(task: RobosuiteTask, encoder: ImageEncoder,
                                cfg: FallbackConfig) -> Traj:
    """Moves the EEF straight toward the cube in steps of cfg.script_step
    (IK + collision-checked, the same execution model the exploration uses)
    until task.reached()."""
    ik, world = task.ik, task.world
    assert cfg.script_step <= world.cfg.step_size + 1e-9, (
        "script_step should not exceed the exploration action bound")
    task.reset_to_start()
    q, eef = ik.get_q(), ik.eef_pos()
    emb = encoder.embed(ik)
    tokens, eefs, qs = [], [eef], [q]
    for _ in range(cfg.script_max_steps):
        if task.reached(eef):
            break
        v = task.cube_pos - eef
        d = float(np.linalg.norm(v))
        a = v / d * min(d, cfg.script_step)
        q_to, ok = ik.solve_ik(eef + a, q_init=q)
        if not ok:
            raise RuntimeError(f"scripted policy: IK failed at eef={eef.round(3)}")
        if world.collides(q, q_to):
            raise RuntimeError(f"scripted policy: collision at eef={eef.round(3)}")
        ik.set_q(q_to)
        tokens.append(fuse(emb.reshape(-1), a))
        q, eef = q_to, ik.eef_pos()
        emb = encoder.embed(ik)
        eefs.append(eef)
        qs.append(q)
    reached = task.reached(eef)
    if not reached:
        raise RuntimeError("scripted policy did not reach the cube")
    T = len(tokens)
    labels = cfg.gamma ** (T - 1 - np.arange(T, dtype=float))
    return Traj(tokens, eefs, qs, emb, reached=True, labels=labels)


def make_varied_demo(noise: float, seed: int, log, tries_per_step: int = 30,
                     max_demos: int = 20):
    """Returns demo_fn(task, encoder, cfg) -> Traj: a scripted reach whose
    actions vary step to step. Every step aims at the cube plus Gaussian
    direction noise (std `noise`, relative to the unit cube direction) and its
    length is drawn like an exploration action (uniform in the action_step
    ball); each step is IK-, workspace- and collision-checked. The demo must
    reach the cube within cfg.script_max_steps, else it restarts with fresh
    noise (up to max_demos times). Uses its own RNG stream (seed + 6_000_003)
    so it never shifts exploration's."""
    rng = np.random.default_rng(seed + 6_000_003)

    def demo(task, encoder, cfg):
        ik, world = task.ik, task.world
        for attempt in range(1, max_demos + 1):
            task.reset_to_start()
            q, eef = ik.get_q(), ik.eef_pos()
            emb = encoder.embed(ik)
            tokens, eefs, qs = [], [eef], [q]
            for _ in range(cfg.script_max_steps):
                if task.reached(eef):
                    break
                g = task.cube_pos - eef
                g /= np.linalg.norm(g)
                for _try in range(tries_per_step):
                    dirv = g + noise * rng.normal(size=3)
                    dirv /= np.linalg.norm(dirv)
                    a_ = dirv * (cfg.action_step if cfg.fixed_step
                                 else cfg.action_step * rng.random() ** (1 / 3))
                    if not world._in_bounds(eef + a_):
                        continue
                    q_to, ok = ik.solve_ik(eef + a_, q_init=q)
                    if ok and not world.collides(q, q_to):
                        break
                    ik.set_q(q)
                else:
                    break                               # stuck: restart the demo
                ik.set_q(q_to)
                tokens.append(fuse(emb.reshape(-1), a_))
                q, eef = q_to, ik.eef_pos()
                emb = encoder.embed(ik)
                eefs.append(eef)
                qs.append(q)
            if task.reached(eef):
                A = np.array([t[-3:] for t in tokens])
                U = A / np.linalg.norm(A, axis=1, keepdims=True)
                to_goal = np.array([task.cube_pos - e for e in eefs[:-1]])
                to_goal /= np.linalg.norm(to_goal, axis=1, keepdims=True)
                ang = np.degrees(np.arccos(np.clip((U * to_goal).sum(1), -1, 1)))
                cos = U @ U.T
                log(f"varied demo (noise {noise}, attempt {attempt}): {len(tokens)} steps | "
                    f"step length {100 * np.linalg.norm(A, axis=1).mean():.2f} cm mean "
                    f"({100 * np.linalg.norm(A, axis=1).min():.2f}-"
                    f"{100 * np.linalg.norm(A, axis=1).max():.2f}) | angle off the cube "
                    f"direction {ang.mean():.0f} deg mean, {ang.max():.0f} max | mean "
                    f"pairwise cosine between actions {cos[np.triu_indices(len(U), 1)].mean():.2f} "
                    f"(straight demo: 1.00)")
                T = len(tokens)
                return Traj(tokens, eefs, qs, emb, reached=True,
                            labels=cfg.gamma ** (T - 1 - np.arange(T, dtype=float)))
        raise RuntimeError(f"varied demo did not reach the cube in {max_demos} attempts; "
                           f"lower --demo-noise")

    return demo


# --------------------------------------------------------------------------- #
# s_1: a start state close to s_0
# --------------------------------------------------------------------------- #
def sample_nearby_start(task: RobosuiteTask, cfg: FallbackConfig, rng, tries: int = 200):
    ik, world = task.ik, task.world
    task.reset_to_start()
    q0, e0 = task.q_start.copy(), task.eef_start.copy()
    for _ in range(tries):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        target = e0 + d * rng.uniform(cfg.s1_min_dist, cfg.s1_max_dist)
        if not world._in_bounds(target) or task.reached(target):
            continue
        q1, ok = ik.solve_ik(target, q_init=q0)
        if not ok:
            ik.set_q(q0)
            continue
        if world.collides(q0, q1):
            ik.set_q(q0)
            continue
        ik.set_q(q1)
        return q1, ik.eef_pos()
    raise RuntimeError("could not find a collision-free s_1 near s_0")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mc_all(model, tokens: List[np.ndarray], cfg: FallbackConfig):
    """MC-dropout (mean, std) at EVERY position of one sequence (its last
    max_len tokens) in a single batched pass -- predict_last_mc for all
    prefixes at once, valid because the transformer is causal."""
    x = torch.tensor(np.array(tokens[-cfg.max_len:]), dtype=torch.float32, device=cfg.device)
    x = x.unsqueeze(0).repeat(cfg.mc_samples, 1, 1)
    model.train()                                    # dropout ON
    c = model(x, causal_mask(x.shape[1], cfg.device), None)
    model.eval()
    return c.mean(0).cpu().numpy(), c.std(0, unbiased=True).cpu().numpy()


def _window(tokens, labels, max_len):
    return np.asarray(tokens[-max_len:]), np.asarray(labels[-max_len:], float)


def train_q(model, seqs, epochs: int, cfg: FallbackConfig, rng) -> float:
    """Huber regression of every position's Q-label (causal transformer, so
    each position only sees its own history)."""
    if not seqs:
        return float("nan")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    order = np.arange(len(seqs))
    last = float("nan")
    for _ in range(epochs):
        model.train()
        rng.shuffle(order)
        total, nb = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            x, tgt, pad = collate([seqs[j] for j in order[i:i + cfg.batch_size]], cfg)
            c = model(x, causal_mask(x.shape[1], cfg.device), pad)
            loss = huber_loss(c, tgt, pad, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        last = total / max(nb, 1)
    model.eval()
    return last


def td_sequences(tau1: Traj, cfg: FallbackConfig):
    starts = range(len(tau1.tokens)) if cfg.td_suffix_aug else [0]
    return [_window(tau1.tokens[i:], tau1.labels[i:], cfg.max_len) for i in starts]


def explore_sequences(buffer: List[Traj], cfg: FallbackConfig):
    return [_window(t.tokens, t.labels, cfg.max_len) for t in buffer if t.n_new > 0]


# --------------------------------------------------------------------------- #
# Exploration
# --------------------------------------------------------------------------- #
def collect_exploration_trajectory(T_D, T_exp, task: RobosuiteTask, encoder: ImageEncoder,
                                   cfg: FallbackConfig, rng, start_q, start_emb,
                                   hist_tokens: List[np.ndarray], dnn=None,
                                   hist_qu: Optional[List[Tuple[float, float]]] = None) -> Traj:
    """One m-step trajectory from start_q. Candidate actions come from
    world.sample_free (uniform in the step ball, IK-reachable, collision-free);
    a candidate is taken iff -mu_TD > Q_exp - mu_exp. Labels for T_explore
    are -mu_TD at every position.

    With a DNN ensemble (`dnn`), mu_TD of a candidate is U(s') of the state it
    leads to, scored from s''s own image (rendered right after sample_free,
    which leaves the sim there); each token's (Q, U) is kept in tok_qu, the
    history's coming in as `hist_qu`."""
    ik, world = task.ik, task.world
    task.reset_to_start()
    ik.set_q(start_q)
    q_cur, emb = np.asarray(start_q, float).copy(), start_emb
    tokens = list(hist_tokens)
    tr = Traj(tokens, [ik.eef_pos()], [q_cur.copy()], emb, embs=[emb],
              hist_len=len(hist_tokens), tok_qu=list(hist_qu or []))

    for _ in range(cfg.m):
        accepted = None
        for _retry in range(cfg.max_retries):
            cand = world.sample_free(rng)
            if cand is None:                 # sample_free restored the sim to q_cur
                tr.n_sample_fail += 1
                continue
            a, _s_next, q_to = cand
            tok = fuse(emb.reshape(-1), a)
            if dnn is not None:
                emb_next = encoder.embed(ik)          # sim is at q_to (sample_free)
                q_d, mu_d = dnn.state_u(emb_next)
            else:
                emb_next, q_d = None, None
                _, mu_d = predict_last_mc(T_D, tokens + [tok], cfg)
            q_e, mu_e = predict_last_mc(T_exp, tokens + [tok], cfg)
            if -mu_d > q_e - mu_e:
                accepted = (tok, q_to, emb_next, (q_d, mu_d))
                break
            tr.n_reject += 1
            ik.set_q(q_cur)                  # undo the rejected candidate's move
        if accepted is None:
            tr.stuck = True
            ik.set_q(q_cur)
            break
        tok, q_cur, emb_next, qu = accepted
        ik.set_q(q_cur)
        tokens.append(tok)
        # same configuration, so the candidate's render is reused when we have it
        emb = emb_next if emb_next is not None else encoder.embed(ik)
        if dnn is not None:
            tr.tok_qu.append(qu)
        tr.eefs.append(ik.eef_pos())
        tr.embs.append(emb)
        tr.qs.append(q_cur.copy())
        if task.reached(tr.eefs[-1]):
            tr.reached = True
            break

    tr.end_emb = emb
    if tr.n_new > 0:
        # the caller trims history to max_len - m, so the whole sequence fits
        # in one window and every token gets a label
        assert len(tokens) <= cfg.max_len
        if dnn is not None:
            tr.labels = -np.array([u for _, u in tr.tok_qu], float)
        else:
            _, mu_all = mc_all(T_D, tokens, cfg)
            tr.labels = -mu_all.astype(float)
    return tr


def min_dist_to(points: List[np.ndarray], ref: np.ndarray) -> float:
    P = np.asarray(points)
    return float(np.min(np.linalg.norm(P[:, None, :] - ref[None, :, :], axis=-1)))


def build_tau2(chain_tokens, chain_eefs, chain_qs, end_emb, tau1: Traj, tau1_pts,
               cfg: FallbackConfig) -> Traj:
    """tau_2 = s_1 -> s*_1 -> ... -> s*_last, where T_D is certain at s*_last.
    It joins tau_1 at the nearest tau_1 state j (by EEF position; the
    distance is logged). Labelled so it is a valid D entry for a
    later iteration: tau_1 gives V(s_j) = gamma^(T-1-j) (=1 at the goal state
    j=T, see collect_scripted_trajectory), so the last tau_2 token gets
    gamma * V(s_j) = gamma^(T-j) (1 if j == T) and earlier tokens are
    discounted back from it."""
    T = len(tau1.tokens)
    j = int(np.argmin(np.linalg.norm(tau1_pts - chain_eefs[-1], axis=1)))
    q_last = cfg.gamma ** (T - j)
    n = len(chain_tokens)
    labels = q_last * cfg.gamma ** (n - 1 - np.arange(n, dtype=float))
    return Traj(list(chain_tokens), list(chain_eefs), list(chain_qs), end_emb,
                labels=labels, join_idx=j)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def save_plots(outdir, tau1: Traj, s1_eef, rounds_trajs, s_stars, cube,
               chain_eefs, made_tau2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("viridis", max(len(rounds_trajs), 2))
    for r, trajs in enumerate(rounds_trajs):
        for j, t in enumerate(trajs):
            P = np.asarray(t.eefs)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cmap(r), lw=0.8, alpha=0.6,
                    label=f"round {r + 1}" if j == 0 else None)
            ax.scatter(*P[-1], color=cmap(r), s=8)
    T = np.asarray(tau1.eefs)
    ax.plot(T[:, 0], T[:, 1], T[:, 2], "k-o", lw=2, ms=3, label=r"$\tau_1$")
    ax.scatter(*T[0], color="k", marker="s", s=50, label=r"$s_0$")
    ax.scatter(*s1_eef, color="tab:blue", marker="^", s=60, label=r"$s_1$")
    if s_stars:
        S = np.asarray(s_stars)
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], color="tab:orange", marker="*", s=120,
                   label=r"$s^*$ per round")
    C = np.asarray(chain_eefs)
    ax.plot(C[:, 0], C[:, 1], C[:, 2], color=PATH_COLOR, lw=2.6,
            label=r"fallback path $s_1 \to s^*$" + (r" ($\tau_2$)" if made_tau2 else " (unfinished)"))
    ax.scatter(*cube, color="tab:red", s=80, label="cube")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("EEF paths: exploration vs " + r"$\tau_1$")

    fig.tight_layout()
    path = os.path.join(outdir, "fallback.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(cfg: FallbackConfig, enc_cfg: EncoderConfig, outdir: str):
    if cfg.keep_frac is None:
        cfg.keep_frac = 1.0 if cfg.td_model == "dnn" else 0.5
    if cfg.fixed_step is None:
        cfg.fixed_step = cfg.td_model == "dnn"
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    calib_rng = np.random.default_rng(cfg.seed + 2_000_003)
    train_rng = np.random.default_rng(cfg.seed + 3_000_003)
    logf = open(os.path.join(outdir, "fallback.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    task = RobosuiteTask(TaskConfig(world=WorldConfig(step_size=cfg.action_step)),
                         seed=cfg.seed, enable_render=True)
    encoder = ImageEncoder(enc_cfg)
    # calibrate on the default-step random walk (see module docstring), then
    # restore the exploration action bound
    task.world.cfg.step_size = WorldConfig().step_size
    try:
        encoder.calibrate(task, calib_rng)
    finally:
        task.world.cfg.step_size = cfg.action_step
    task.world.cfg.fixed_step = bool(cfg.fixed_step)     # after calibration (random-length walk)
    encoder.save_stats(os.path.join(outdir, "encoder_stats.npz"))
    log(f"task: eef_start={task.eef_start.round(3)} cube={task.cube_pos.round(3)} "
        f"eps={task.cfg.eps_dist} | encoder {enc_cfg.model_name} x {list(enc_cfg.cameras)} "
        f"on {encoder.device}, calibrated on {enc_cfg.n_calib} states")
    log(f"config: {json.dumps(asdict(cfg))}")
    if cfg.fixed_step:
        log(f"step sizes: every action exactly {cfg.action_step} m (exploration, "
            f"{'varied demo' if cfg.demo == 'varied' else f'straight demo {cfg.script_step} m/step'}"
            f"{', DNN probe actions' if cfg.td_model == 'dnn' else ''}), gamma {cfg.gamma}")
    else:
        log(f"step sizes: demo {cfg.script_step} m/step, exploration actions <= "
            f"{task.world.cfg.step_size} m, gamma {cfg.gamma}")

    def new_model():
        return ImageCostTransformer(
            cfg, encoder.n_views, encoder.emb_dim, action_dim=3,
            action_scale=1.0 / task.world.cfg.step_size,
            token_fusion=cfg.token_fusion).to(cfg.device)

    torch.manual_seed(cfg.seed)
    T_D = new_model()
    torch.manual_seed(cfg.seed + 1)
    T_exp = new_model()

    # ---- tau_1 and T_D ----
    demo_fn = make_varied_demo(cfg.demo_noise, cfg.seed, log) if cfg.demo == "varied" else None
    tau1 = (demo_fn or collect_scripted_trajectory)(task, encoder, cfg)
    D = [tau1]
    tau1_pts = np.asarray(tau1.eefs)
    log(f"\ntau_1 ({cfg.demo} demo): {len(tau1.tokens)} steps, start {tau1.eefs[0].round(3)} -> "
        f"end {tau1.eefs[-1].round(3)} (dist to cube {task.dist_to_cube(tau1.eefs[-1]):.3f})")
    dnn = None
    if cfg.td_model == "dnn":
        log(f"T_D = ensemble of {cfg.n_dnn} feed-forward Q-nets (no history); mu_TD of a "
            f"state = U = max - min of their Q, mean over {cfg.n_probe_actions} random actions")
        models = train_dnn_ensemble(
            np.stack([tok[:-3] for tok in tau1.tokens]), np.stack([tok[-3:] for tok in tau1.tokens]),
            tau1.labels, cfg.n_dnn, cfg.dnn_hidden, cfg.dnn_epochs, cfg.dnn_lr, cfg.seed,
            cfg.action_step, log)
        dnn = DNNEnsemble(models, cfg.action_step, cfg.n_probe_actions,
                          np.random.default_rng(cfg.seed + 8_000_003),
                          fixed_len=bool(cfg.fixed_step))
        # every tau_1 state (s_0 .. s_T) scored exactly like an explored state
        qu_tau1 = [dnn.state_u(e) for e in
                   [tok[:-3] for tok in tau1.tokens] + [tau1.end_emb.reshape(-1)]]
        mu_all_states = np.array([u for _, u in qu_tau1])
        # stored per token t (= state t+1), the convention plot_fallback_3d reads
        tau1_q_td = [q for q, _ in qu_tau1[1:]]
        mu_tau1 = mu_all_states[1:]
        mu_ref = mu_all_states
    else:
        loss_d = train_q(T_D, td_sequences(tau1, cfg), cfg.td_epochs, cfg, train_rng)
        q_fit = [predict_last_mc(T_D, tau1.tokens[:t + 1], cfg) for t in range(len(tau1.tokens))]
        log(f"T_D trained ({cfg.td_epochs} epochs, final loss {loss_d:.2e}); "
            f"fit on tau_1 (label / Q_TD +- mu):")
        log("   " + "  ".join(f"{l:.3f}/{q:.3f}+-{s:.3f}" for l, (q, s) in zip(tau1.labels, q_fit)))
        tau1_q_td = [q for q, _ in q_fit]
        mu_tau1 = np.array([sd for _, sd in q_fit])
        mu_ref = mu_tau1
    mu_tol = cfg.mu_tol if cfg.mu_tol is not None else float(mu_ref.mean())
    log(f"stop when mu_TD(s*) < mu_tol = {mu_tol:.4f}"
        f"{'' if cfg.mu_tol is not None else '  (mean mu_TD over tau_1)'}  | mu_TD on "
        f"tau_1: min {mu_ref.min():.4f} mean {mu_ref.mean():.4f} max {mu_ref.max():.4f}")

    # ---- s_1 ----
    q1, s1_eef = sample_nearby_start(task, cfg, rng)
    emb1 = encoder.embed(task.ik)
    log(f"s_1 = {s1_eef.round(3)}  (|s_1 - s_0| = {np.linalg.norm(s1_eef - task.eef_start):.3f} m, "
        f"dist to tau_1 = {min_dist_to([s1_eef], tau1_pts):.3f} m)")

    start_q, start_emb, start_hist, start_eef = q1, emb1, [], s1_eef
    start_hist_qu: List[Tuple[float, float]] = []
    # the full, untruncated s_1 -> s*_1 -> s*_2 -> ... chain (becomes tau_2)
    chain_tokens: List[np.ndarray] = []
    chain_eefs: List[np.ndarray] = [s1_eef]
    chain_qs: List[np.ndarray] = [q1]
    explore_buffer: List[Traj] = []
    rounds_trajs, s_stars, rows = [], [], []
    state_scores = []          # per round: every visited state's scores (summary.json)
    tau2: Optional[Traj] = None
    stop_reason = "max_rounds"

    hdr = (f"{'rnd':>3} {'steps':>5} {'stuck':>5} {'rej':>5} {'sfail':>5} {'reach':>5} "
           f"{'loss_exp':>9} {'Q_TD*':>7} {'mu_TD*':>7} {'Q_exp*':>7} {'mu_exp*':>7} "
           f"{'d(s*,t1)':>8} {'dmin,t1':>8} {'sec':>6}")
    log("\n" + hdr)
    for rnd in range(1, cfg.max_rounds + 1):
        t_r = time.time()
        # max_len keeps the positional table valid: inherited history + m new steps
        keep_n = cfg.max_len - cfg.m
        hist = start_hist[-keep_n:] if keep_n > 0 else []
        hist_qu = start_hist_qu[-keep_n:] if keep_n > 0 else []
        trajs = [collect_exploration_trajectory(T_D, T_exp, task, encoder, cfg, rng,
                                                start_q, start_emb, hist, dnn, hist_qu)
                 for _ in range(cfg.k)]
        explore_buffer.extend(trajs)
        loss_e = train_q(T_exp, explore_sequences(explore_buffer, cfg),
                         cfg.explore_epochs, cfg, train_rng)

        # ---- s*: S = all visited states, S' = top keep_frac by mu_exp, s* = min mu_TD in S' ----
        # state j (1..n_new) of trajectory t is scored at the token that led
        # into it, position hist_len + j - 1 (see module docstring)
        S = []                                  # (traj, j, q_td, mu_td, q_exp, mu_exp)
        for t in trajs:
            if t.n_new == 0:
                continue
            if dnn is not None:                 # U(s_j) was scored when s_j was reached
                q_td_all = [q for q, _ in t.tok_qu]
                mu_td_all = [u for _, u in t.tok_qu]
            else:
                q_td_all, mu_td_all = mc_all(T_D, t.tokens, cfg)
            q_e_all, mu_e_all = mc_all(T_exp, t.tokens, cfg)
            for j in range(1, t.n_new + 1):
                p = t.hist_len + j - 1
                S.append((t, j, float(q_td_all[p]), float(mu_td_all[p]),
                          float(q_e_all[p]), float(mu_e_all[p])))
        if not S:
            stop_reason = "stuck"
            log(f"{rnd:>3}  every trajectory was stuck at its first step -- stopping")
            break
        mu_e_S = np.array([c[5] for c in S])
        n_keep = max(1, int(np.ceil(cfg.keep_frac * len(S))))
        S_prime = sorted(S, key=lambda c: -c[5])[:n_keep]
        alpha = S_prime[-1][5]                  # this round's effective threshold
        log(f"    |S| = {len(S)}  mu_exp over S: min {mu_e_S.min():.4f} median "
            f"{np.median(mu_e_S):.4f} max {mu_e_S.max():.4f}  ->  |S'| = {len(S_prime)} "
            f"(top {cfg.keep_frac:.0%}, alpha = {alpha:.4f})")
        best, j_star, q_d, mu_d, q_e, mu_e = min(S_prime, key=lambda c: c[3])
        s_star = best.eefs[j_star]
        i_star = next(i for i, t in enumerate(trajs) if t is best)
        log(f"    s* = traj {i_star} state {j_star}/{best.n_new}: mu_TD "
            f"{mu_d:.4f} (S' range {min(c[3] for c in S_prime):.4f}-"
            f"{max(c[3] for c in S_prime):.4f}), {min_dist_to([s_star], tau1_pts):.3f} m "
            f"from tau_1; closest state in S' is "
            f"{min(min_dist_to([c[0].eefs[c[1]]], tau1_pts) for c in S_prime):.3f} m")
        s_stars.append(s_star)
        rounds_trajs.append(trajs)
        kept = {id(c) for c in S_prime}
        state_scores.append([dict(
            traj=next(i for i, tt in enumerate(trajs) if tt is c[0]), step=c[1],
            q_td=c[2], mu_td=c[3], q_exp=c[4], mu_exp=c[5], in_S_prime=id(c) in kept,
            dist_tau1=min_dist_to([c[0].eefs[c[1]]], tau1_pts)) for c in S])
        cut = best.hist_len + j_star            # tokens up to (and leading into) s*
        chain_tokens.extend(best.tokens[best.hist_len:cut])
        chain_eefs.extend(best.eefs[1:j_star + 1])
        chain_qs.extend(best.qs[1:j_star + 1])

        new_pts = [p for t in trajs for p in t.eefs[1:]]
        row = dict(
            round=rnd, start=start_eef.round(4).tolist(),
            steps=int(sum(t.n_new for t in trajs)),
            stuck=int(sum(t.stuck for t in trajs)),
            rejects=int(sum(t.n_reject for t in trajs)),
            sample_fails=int(sum(t.n_sample_fail for t in trajs)),
            reached=int(sum(t.reached for t in trajs)),
            n_S=len(S), n_S_prime=len(S_prime), alpha=alpha, sstar_step=j_star,
            loss_exp=loss_e, q_td_sstar=q_d, mu_td_sstar=mu_d,
            q_exp_sstar=q_e, mu_exp_sstar=mu_e,
            sstar=s_star.round(4).tolist(),
            sstar_dist_tau1=min_dist_to([s_star], tau1_pts),
            min_dist_tau1=min_dist_to(new_pts, tau1_pts) if new_pts else float("nan"),
            sec=time.time() - t_r,
        )
        rows.append(row)
        log(f"{rnd:>3} {row['steps']:>5} {row['stuck']:>5} {row['rejects']:>5} "
            f"{row['sample_fails']:>5} {row['reached']:>5} {loss_e:>9.2e} {q_d:>7.3f} "
            f"{mu_d:>7.4f} {q_e:>7.3f} {mu_e:>7.3f} {row['sstar_dist_tau1']:>8.3f} "
            f"{row['min_dist_tau1']:>8.3f} {row['sec']:>6.1f}")

        if mu_d < mu_tol:
            stop_reason = "mu_td_certain"
            tau2 = build_tau2(chain_tokens, chain_eefs, chain_qs, best.embs[j_star],
                              tau1, tau1_pts, cfg)
            D.append(tau2)
            log(f"T_D certain at s*: mu_TD = {mu_d:.4f} < {mu_tol:.4f} -> tau_2 = s_1 -> "
                f"{rnd} s* ({len(tau2.tokens)} steps) added to D; joins tau_1 at state "
                f"{tau2.join_idx}/{len(tau1.tokens)}, {row['sstar_dist_tau1']:.3f} m away")
            break
        start_q, start_emb = best.qs[j_star], best.embs[j_star]
        start_hist, start_eef = best.tokens[:cut], s_star
        start_hist_qu = best.tok_qu[:cut]

    # ---- outputs ----
    with open(os.path.join(outdir, "rounds.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = dict(
        stop_reason=stop_reason, fell_back=tau2 is not None, rounds=len(rows),
        mu_tol=mu_tol, success_buffer_size=len(D),
        tau1_len=len(tau1.tokens), s0=task.eef_start.tolist(), s1=s1_eef.tolist(),
        cube=task.cube_pos.tolist(), wall_sec=time.time() - t0, config=asdict(cfg),
        tau1_eefs=np.asarray(tau1.eefs).tolist(),
        # T_D on its own demo; entry t is scored at token t, i.e. state t+1
        tau1_q_td=list(tau1_q_td), tau1_mu_td=mu_tau1.tolist(),
        s_stars=[s.tolist() for s in s_stars],
        chain_eefs=np.asarray(chain_eefs).tolist(),     # fallback path s_1 -> ... -> last s*
        state_scores=state_scores,
        tau2=None if tau2 is None else dict(
            eefs=np.asarray(tau2.eefs).tolist(), qs=np.asarray(tau2.qs).tolist(),
            actions=[t[-3:].tolist() for t in tau2.tokens], labels=tau2.labels.tolist(),
            join_idx=tau2.join_idx),
        exploration=[[np.asarray(t.eefs).tolist() for t in trajs] for trajs in rounds_trajs],
    )
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if tau2 is not None:
        np.savez(os.path.join(outdir, "tau2.npz"), tokens=np.asarray(tau2.tokens),
                 labels=tau2.labels, eefs=np.asarray(tau2.eefs), qs=np.asarray(tau2.qs),
                 join_idx=tau2.join_idx)
    if dnn is not None:
        torch.save(dnn.state_dicts(), os.path.join(outdir, "T_D_dnn.pt"))
    else:
        torch.save(T_D.state_dict(), os.path.join(outdir, "T_D.pt"))
    torch.save(T_exp.state_dict(), os.path.join(outdir, "T_explore.pt"))
    plot = (save_plots(outdir, tau1, s1_eef, rounds_trajs, s_stars, task.cube_pos,
                       chain_eefs, tau2 is not None) if rows else None)
    log(f"\nstop={stop_reason} rounds={len(rows)} tau_2={'yes' if tau2 else 'no'} | "
        f"|D|={len(D)} | {time.time() - t0:.0f}s total")
    html = write_html(outdir, summary) if rows else None
    log(f"outputs -> {outdir}/ (fallback.log, rounds.csv, summary.json, "
        f"{'T_D_dnn.pt' if dnn is not None else 'T_D.pt'}, T_explore.pt"
        f"{', tau2.npz' if tau2 else ''}{', ' + os.path.basename(plot) if plot else ''}"
        f"{', ' + os.path.basename(html) if html else ''})")
    task.close()
    logf.close()
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    d = FallbackConfig()
    p.add_argument("--k", type=int, default=d.k)
    p.add_argument("--m", type=int, default=d.m)
    p.add_argument("--max-rounds", type=int, default=d.max_rounds)
    p.add_argument("--max-retries", type=int, default=d.max_retries)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--keep-frac", type=float, default=None,
                   help="S' = top fraction of this round's visited states by mu_explore; "
                        "default 1.0 (all states) with --td-model dnn, 0.5 with transformer")
    p.add_argument("--mu-tol", type=float, default=None,
                   help="stop when mu_TD(s*) < this (default: mean mu_TD over tau_1)")
    p.add_argument("--td-model", choices=["dnn", "transformer"], default=d.td_model,
                   help="T_D: ensemble of feed-forward Q-nets (default) or the single "
                        "MC-dropout transformer")
    p.add_argument("--n-dnn", type=int, default=d.n_dnn, help="DNN ensemble members (>= 2)")
    p.add_argument("--dnn-epochs", type=int, default=d.dnn_epochs)
    p.add_argument("--n-probe-actions", type=int, default=d.n_probe_actions,
                   help="random actions the DNN uncertainty of a state is averaged over")
    p.add_argument("--demo", choices=["varied", "straight"], default=d.demo,
                   help="tau_1: scripted reach with varied actions (default) or the straight one")
    p.add_argument("--demo-noise", type=float, default=d.demo_noise,
                   help="varied demo: direction-noise std (larger = more varied)")
    p.add_argument("--td-epochs", type=int, default=d.td_epochs,
                   help="transformer T_D epochs (--td-model transformer)")
    p.add_argument("--explore-epochs", type=int, default=d.explore_epochs)
    p.add_argument("--mc-samples", type=int, default=d.mc_samples)
    p.add_argument("--script-step", type=float, default=d.script_step,
                   help="demo (tau_1) step length, m")
    p.add_argument("--action-step", type=float, default=d.action_step,
                   help="exploration action bound, m (keep >= --script-step)")
    p.add_argument("--fixed-step", action=argparse.BooleanOptionalAction, default=None,
                   help="every exploration / varied-demo action exactly --action-step long "
                        "(default: on with --td-model dnn, off with transformer)")
    p.add_argument("--s1-min-dist", type=float, default=d.s1_min_dist)
    p.add_argument("--s1-max-dist", type=float, default=d.s1_max_dist)
    p.add_argument("--no-td-suffix-aug", action="store_true")
    p.add_argument("--token-fusion", default=d.token_fusion, choices=["sum", "concat", "wide_sum"])
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default="cpu", help="transformer device: cpu | mps | cuda")
    p.add_argument("--encoder", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    p.add_argument("--encoder-device", default="auto")
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default="fallback_results")
    p.add_argument("--smoke", action="store_true", help="tiny fast run for an end-to-end check")
    a = p.parse_args()
    if a.keep_frac is None:
        a.keep_frac = 1.0 if a.td_model == "dnn" else 0.5
    if not 0.0 < a.keep_frac <= 1.0:
        p.error("--keep-frac must be in (0, 1]")
    if a.n_dnn < 2:
        p.error("--n-dnn must be at least 2")

    cfg = FallbackConfig(
        k=a.k, m=a.m, max_rounds=a.max_rounds, max_retries=a.max_retries, gamma=a.gamma,
        keep_frac=a.keep_frac, mu_tol=a.mu_tol, td_epochs=a.td_epochs, explore_epochs=a.explore_epochs,
        mc_samples=a.mc_samples, script_step=a.script_step, action_step=a.action_step, s1_min_dist=a.s1_min_dist,
        s1_max_dist=a.s1_max_dist, td_suffix_aug=not a.no_td_suffix_aug,
        token_fusion=a.token_fusion, seed=a.seed, device=a.device,
        td_model=a.td_model, n_dnn=a.n_dnn, dnn_epochs=a.dnn_epochs,
        n_probe_actions=a.n_probe_actions, demo=a.demo, demo_noise=a.demo_noise,
        fixed_step=a.fixed_step)
    enc_cfg = EncoderConfig(model_name=a.encoder, device=a.encoder_device, n_calib=a.n_calib)
    if a.smoke:
        cfg.k, cfg.m, cfg.max_rounds, cfg.td_epochs, cfg.explore_epochs = 3, 4, 2, 60, 10
        cfg.mc_samples = 8
        enc_cfg.n_calib = 20
    return cfg, enc_cfg, a.outdir


if __name__ == "__main__":
    cfg, enc_cfg, outdir = parse_args()
    main(cfg, enc_cfg, outdir)
