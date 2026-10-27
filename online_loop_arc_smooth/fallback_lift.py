"""
Online fallback loop for the PICK-UP task (v3 Stage B; v4: descend-phase BC
corrective examples, BC block reasons + gripper commands logged; v5: the
fallback only translates -- orientation held at s_1's (absolute target), gripper
open, probe actions for U translation-only (--explore-rot none, default);
position / orientation distance and the U-distance correlation logged; v6:
tau_1 holds the start orientation until the hover point, then turns to the
cube's yaw -- so D's approach region matches the fallback's orientation;
arc: tau_k gets real phases -- fallback part approach, BC part from its
gripper commands and heights (bc_action_phases) -- and cluster support;
arc_smooth: each successful tau_k is smoothed + arc-length resampled (smooth.py)
before it joins D -- fallback (~2 cm) and BC (~0.5 cm) parts separately, the
descent pinned at the column entry, pauses / hover collapsed, grasp + lift kept) -- the lift counterpart
of fallback_explore.py (reaching), on the physics environment lift_env.py.

Same method, same models, same outputs as the reaching loop; what changes is
everything that touches the robot:
  * actions are 7-D CLOSED-LOOP actions [dpos 1 cm, drot 0.1 rad, gripper]
    executed with LiftEnv.act until reached; model inputs use the normalized
    form [dpos / 1 cm, drot / 0.1 rad, grip] (scripted_pick.to_norm);
  * tau_1 = the scripted pick (scripted_pick.scripted_pick_cl) with a varied
    approach -- random translation AND random 3-D rotation noise
    (--demo-noise / --demo-rot-noise, --rot-mode xyz|yaw); descend, grasp and
    lift stay clean;
  * exploration: random 1 cm direction + random 0.1 rad rotation (3-D axis,
    or vertical only with --rot-mode yaw) + gripper OPEN (BC does the grasp);
    a candidate is executed, scored and -- if rejected or colliding / leaving
    the workspace -- undone by restoring the full simulator state;
  * distance to D = gripper position difference + orientation difference
    (lift_env.ROT_W: 10 deg ~ 1 cm); the cube is not part of it;
  * s_1: the arm placed by 6-D IK at s_0's orientation, in a random
    horizontal direction from s_0, s1-min..max-dist (pose distance) from D,
    cube resting on the table, gripper open;
  * BC = bc_lift.py (7-D, gripper head, approach-phase corrective examples);
  * success = robosuite's Lift check (cube lifted); tau_k = fallback path +
    BC rollout, added to D (true returns) only if the cube was lifted.
Unchanged: DNN ensemble T_D (10 members, U = max - min Q, history-free),
T_explore transformer (cold start per iteration, trained on -U), acceptance
-U(s') > Q_exp - mu_exp, s* = lowest U over all visited states, optional
keep-best, stop U(path end) < mean U over D, n_iters online loop, outputs
(rounds.csv, summary.json, 3D pages, replay.gif, iterations.csv, D.npz).

Usage (from the project root):
    python online_loop_v6/fallback_lift.py --seed 10 --no-keep-best --outdir pick_results/online/seed10
    python online_loop_v6/fallback_lift.py --smoke --seed 10 --outdir /tmp/lift_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
import torch

from shortcut import ShortcutConfig, shortcut_lift_traj
from smooth import SmoothConfig, smooth_lift_traj
from learning_metrics import iteration_metrics
from bc_lift import BCLiftConfig, BCLiftPolicy, bc_lift_rollout, build_bc_lift_dataset, merge_bc_lift_datasets, train_bc_lift
from dnn_ensemble import DNNEnsemble, train_dnn_ensemble
from fallback_explore import (
    S1_ATTEMPTS, FallbackConfig, Traj, explore_sequences, mc_all, train_q, write_iteration_outputs,
)
from image_encoder import EncoderConfig, ImageEncoder
from lift_env import LiftEnv, ROT_W, dist_to_poses, in_workspace, rot_angle
from online_cost_transformer import ImageCostTransformer, fuse, predict_last_mc
from plot_fallback_3d import write_html
from replay_gif import write_replay_gif
from scripted_pick import ROT_STEP, STEP, scripted_pick_cl, to_norm


@dataclass
class LiftConfig(FallbackConfig):
    rot_mode: str = "xyz"                # demo (tau_1) / calibration rotation axis: random 3-D | vertical
    explore_rot: str = "none"            # v5: fallback actions -- "none" = translation only, orientation held
                                         # at s_1's (absolute target), gripper open; "xyz"/"yaw" = v4 behaviour
    demo_noise: float = 1.0              # tau_1 approach: translation direction noise
    demo_rot_noise: float = 1.0          # tau_1 approach: rotation axis noise (ignored with demo_hold_orient)
    smooth: bool = True                  # arc_smooth: smooth + arc-length resample the fallback and BC parts
                                         # of each successful tau_k before D (grasp + lift kept)
    shortcut: bool = False               # arc_shortcutting's greedy shortcutting (off here; used only if not smooth)
    demo_hold_orient: bool = True        # v6: tau_1 keeps the start orientation until 3 cm from the hover
                                         # point, then turns to the cube's yaw (no approach rotation noise)
    calib_restart: int = 25              # encoder calibration random walk restarts


@dataclass
class LiftTraj(Traj):
    rots: List[np.ndarray] = field(default_factory=list)       # gripper orientation per state
    grip_cmd_act: List[float] = field(default_factory=list)    # gripper command of each action
    phases: List[str] = field(default_factory=list)            # per action

    @property
    def states(self):
        """Full simulator states, one per EEF entry (stored in Traj.qs)."""
        return self.qs


def sample_action(rng, rot_mode: str) -> np.ndarray:
    """Random exploration action, normalized: unit translation (1 cm), unit
    rotation (0.1 rad) about a random 3-D axis (or +-vertical), gripper open."""
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    if rot_mode == "xyz":
        r = rng.normal(size=3); r /= np.linalg.norm(r)
    else:
        r = np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
    return np.concatenate([d, r, [-1.0]])


def sample_explore_action(rng, cfg) -> np.ndarray:
    """Fallback exploration / ensemble probe action (normalized). v5 default
    explore_rot "none": unit translation (1 cm) in a random direction, zero
    rotation, gripper open; otherwise sample_action with that rotation mode."""
    if cfg.explore_rot == "none":
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        return np.concatenate([d, np.zeros(3), [-1.0]])
    return sample_action(rng, cfg.explore_rot)


def calibrate_encoder(env: LiftEnv, encoder: ImageEncoder, rng, rot_mode: str, n: int, restart: int):
    """Fit the encoder's standardization on a random walk of closed-loop
    exploration actions from the reset state (the reaching encoder's
    calibrate() needs the teleporting world)."""
    env.reset()
    s0 = env.get_state()
    raw, since = [encoder.embed_raw(env)], 0
    while len(raw) < n:
        if since >= restart:
            env.set_state(s0); since = 0
        a = sample_action(rng, rot_mode)
        s_b, lg = env.get_state(), env.last_grip
        out = env.act(a[:3] * STEP, a[3:6] * ROT_STEP, -1.0)
        if out["collided"] or out["out_of_bounds"]:
            env.set_state(s_b, last_grip=lg); env.set_state(s0); since = 0
            continue
        raw.append(encoder.embed_raw(env)); since += 1
    arr = np.stack(raw)
    encoder.mean, encoder.std = arr.mean(0), np.maximum(arr.std(0), 1e-6)
    env.set_state(s0)


def make_tau1(env: LiftEnv, encoder: ImageEncoder, cfg: LiftConfig, log) -> LiftTraj:
    r = scripted_pick_cl(env, rng=np.random.default_rng(cfg.seed + 6_000_003),
                         approach_noise=cfg.demo_noise, rot_noise=cfg.demo_rot_noise, rot_mode=cfg.rot_mode,
                         hold_orientation=cfg.demo_hold_orient)
    if not r["success"]:
        raise RuntimeError("scripted pick did not lift the cube")
    embs = []
    for s in r["states"]:
        env.set_state(s)
        embs.append(encoder.embed(env).reshape(-1))
    T = len(r["norm_actions"])
    tokens = [fuse(embs[i], r["norm_actions"][i]) for i in range(T)]
    labels = cfg.gamma ** (T - 1 - np.arange(T, dtype=float))
    log(f"tau_1 (scripted pick, approach noise {cfg.demo_noise}, "
        + ("orientation held until the hover point" if cfg.demo_hold_orient
           else f"rotation noise {cfg.demo_rot_noise} [{cfg.rot_mode}]")
        + f"): {T} actions | phases "
        f"{ {p: int((r['phase'] == p).sum()) for p in ('approach', 'descend', 'grasp', 'lift')} } | "
        f"cube lifted {100 * r['cube_lift']:.1f} cm")
    return LiftTraj(tokens, list(r["eef"]), list(r["states"]),
                    embs[-1], embs=embs, reached=True, labels=labels, rots=list(r["rot"]),
                    grip_cmd_act=[float(a[6]) for a in r["norm_actions"]], phases=list(r["phase"]))


def sample_s1(env: LiftEnv, cfg: LiftConfig, rng, D_P, D_R, tries: int = 400):
    """Arm placed (6-D IK, s_0's orientation) in a random horizontal direction
    from s_0, with pose distance to D in [s1_min, s1_max]; cube untouched."""
    env.reset()
    s_reset = env.get_state()
    p0, R0 = env.eef_pose()
    for _ in range(tries):
        th = rng.uniform(0, 2 * np.pi)
        p = p0 + np.array([np.cos(th), np.sin(th), 0.0]) * rng.uniform(cfg.s1_min_dist, 3 * cfg.s1_max_dist)
        if not in_workspace(p) or not cfg.s1_min_dist <= dist_to_poses(p, R0, D_P, D_R).min() <= cfg.s1_max_dist:
            continue
        env.set_state(s_reset)
        if not env.place_arm(p, R0):
            continue
        pe, Re = env.eef_pose()
        dD = dist_to_poses(pe, Re, D_P, D_R).min()
        if cfg.s1_min_dist <= dD <= cfg.s1_max_dist:
            return env.get_state(), pe, Re, float(dD)
    raise RuntimeError("no valid s_1")


_LM_COLS = ("D_before", "bc_ran", "bc_end", "lifted", "fallback_actions", "bc_actions", "total_actions",
            "return_episode", "return_bc", "s1_dist_D_cm", "end_dist_D_cm", "fallback_delta_cm", "fallback_closer")


def _lm_cols(lm):
    """learning_metrics columns for iterations.csv (position distances renamed
    *_pos_* so they do not clash with the pose-distance columns)."""
    ren = {"s1_dist_D_cm": "s1_pos_dist_D_cm", "end_dist_D_cm": "end_pos_dist_D_cm"}
    return {ren.get(k, k): (None if lm is None else lm[k]) for k in _LM_COLS}


def bc_action_phases(eefs, grip_cmds, cube, hover: float = 0.10) -> List[str]:
    """Phase of each BC action (arc: tau_k used to be labelled "approach"
    throughout, so its grasp / lift states got open-gripper corrective
    examples). Action i goes from state i to i+1:
      the FINAL run of close commands (the grasp that lifted the cube):
        "grasp" until the gripper first rises > 3 mm in one action, "lift" from then on;
      earlier close commands (attempts that were reopened): "grasp";
      open commands: "descend" if the gripper is within 3 cm (horizontally) of
        the cube and below hover + 1 cm above it, else "approach".
    Only approach / descend states get corrective examples (bc_lift)."""
    E, c, G = np.asarray(eefs, float), np.asarray(cube, float), list(grip_cmds)
    n = len(G)
    k = n
    while k > 0 and G[k - 1] > 0:                      # start of the final close run
        k -= 1
    out, lifting = [], False
    for i, g in enumerate(G):
        if g > 0:
            if i >= k and (lifting or E[i + 1, 2] - E[i, 2] > 0.003):
                lifting = True
                out.append("lift")
            else:
                out.append("grasp")
        elif np.linalg.norm(E[i, :2] - c[:2]) < 0.03 and E[i, 2] < c[2] + hover + 0.01:
            out.append("descend")
        else:
            out.append("approach")
    return out


def collect_lift_trajectory(T_exp, dnn, env, encoder, cfg, rng, start_state, start_emb, hist, hist_qu, R_hold=None):
    env.set_state(start_state, last_grip=-1.0)
    tokens = list(hist)
    tr = LiftTraj(tokens, [env.eef_pos()], [env.get_state()], start_emb, embs=[start_emb],
                  hist_len=len(hist), tok_qu=list(hist_qu), rots=[env.eef_rot()])
    emb = start_emb
    for _ in range(cfg.m):
        accepted = None
        for _retry in range(cfg.max_retries):
            a = sample_explore_action(rng, cfg)
            s_b, lg = env.get_state(), env.last_grip
            out = env.act(a[:3] * STEP, a[3:6] * ROT_STEP, -1.0, R_target=R_hold)
            if out["collided"] or out["out_of_bounds"]:
                env.set_state(s_b, last_grip=lg); tr.n_sample_fail += 1
                continue
            tok = fuse(emb.reshape(-1), a)
            emb_next = encoder.embed(env)
            q_d, u = dnn.state_u(emb_next)
            q_e, mu_e = predict_last_mc(T_exp, tokens + [tok], cfg)
            if -u > q_e - mu_e:
                accepted = (tok, emb_next, (q_d, u))
                break
            tr.n_reject += 1
            env.set_state(s_b, last_grip=lg)
        if accepted is None:
            tr.stuck = True
            break
        tok, emb, qu = accepted
        tokens.append(tok); tr.tok_qu.append(qu)
        tr.eefs.append(env.eef_pos()); tr.rots.append(env.eef_rot()); tr.qs.append(env.get_state())
        tr.embs.append(emb); tr.grip_cmd_act.append(-1.0); tr.phases.append("approach")
    tr.end_emb = emb
    if tr.n_new > 0:
        tr.labels = -np.array([u for _, u in tr.tok_qu], float)
    return tr


def main(cfg: LiftConfig, enc_cfg: EncoderConfig, outdir: str):
    cfg.keep_frac = 1.0 if cfg.keep_frac is None else cfg.keep_frac
    cfg.keep_best = False if cfg.keep_best is None else cfg.keep_best
    cfg.bc = True if cfg.bc is None else cfg.bc
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    train_rng = np.random.default_rng(cfg.seed + 3_000_003)
    logf = open(os.path.join(outdir, "fallback.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    env = LiftEnv(seed=cfg.seed, cube_offset=tuple(cfg.cube_offset), enable_render=True)
    encoder = ImageEncoder(enc_cfg)
    calibrate_encoder(env, encoder, np.random.default_rng(cfg.seed + 2_000_003), cfg.rot_mode,
                      enc_cfg.n_calib, cfg.calib_restart)
    encoder.save_stats(os.path.join(outdir, "encoder_stats.npz"))
    env.reset()
    log(f"PICK task (lift_env): cube {env.cube_start.round(3)} yaw {np.degrees(env.cube_yaw0):.0f} deg | "
        f"s_0 {env.eef_pos().round(3)} | actions: closed-loop 1 cm / 0.1 rad [{cfg.rot_mode}] + gripper | "
        f"fallback rotation: {'none (orientation held at s_1, gripper open)' if cfg.explore_rot == 'none' else cfg.explore_rot} | "
        f"encoder calibrated on {enc_cfg.n_calib} states")
    log(f"config: {json.dumps(asdict(cfg))}")

    def new_T_exp():
        return ImageCostTransformer(cfg, encoder.n_views, encoder.emb_dim, action_dim=7, action_scale=1.0,
                                    token_fusion=cfg.token_fusion).to(cfg.device)

    tau1 = make_tau1(env, encoder, cfg, log)
    D: List[LiftTraj] = [tau1]
    env.reset()
    s0_p, s0_R = env.eef_pose()
    bccfg = BCLiftConfig(n_aug=cfg.bc_n_aug, epochs=cfg.bc_epochs, lookahead=cfg.bc_lookahead, seed=cfg.seed)
    bc_cache: List[dict] = []
    multi = cfg.n_iters > 1
    it_rows, summary = [], None

    for it in range(1, cfg.n_iters + 1):
        t_it = time.time()
        it_dir = os.path.join(outdir, f"iter{it:02d}") if multi else outdir
        os.makedirs(it_dir, exist_ok=True)
        D_iter = list(D)
        D_P = np.concatenate([np.asarray(tr.eefs) for tr in D_iter])
        D_R = np.concatenate([np.asarray(tr.rots) for tr in D_iter])
        D_owner = np.concatenate([np.full(len(tr.eefs), j) for j, tr in enumerate(D_iter)])
        if multi:
            log(f"\n{'=' * 24} iteration {it}/{cfg.n_iters} | |D| = {len(D)} {'=' * 24}")

        # ---- T_D: DNN ensemble on all of D ----
        models = train_dnn_ensemble(
            np.concatenate([np.stack([t[:-7] for t in tr.tokens]) for tr in D]),
            np.concatenate([np.stack([t[-7:] for t in tr.tokens]) for tr in D]),
            np.concatenate([tr.labels for tr in D]), cfg.n_dnn, cfg.dnn_hidden, cfg.dnn_epochs,
            cfg.dnn_lr, cfg.seed, 1.0, None)
        dnn = DNNEnsemble(models, 1.0, cfg.n_probe_actions, np.random.default_rng(cfg.seed + 8_000_003 + 1000 * (it - 1)),
                          action_sampler=lambda r: sample_explore_action(r, cfg))
        qu_D = [[dnn.state_u(e) for e in [t[:-7] for t in tr.tokens] + [np.asarray(tr.end_emb).reshape(-1)]]
                for tr in D]
        mu_states = np.array([u for qu in qu_D for _, u in qu])
        mu_tol = cfg.mu_tol if cfg.mu_tol is not None else float(mu_states.mean())
        log(f"T_D = {cfg.n_dnn} DNNs on {len(mu_states)} states of D | stop when U(path end) < {mu_tol:.4f} "
            f"(mean U over D; min {mu_states.min():.4f} max {mu_states.max():.4f})")

        # ---- BC on all of D (corrective data cached per trajectory) ----
        policy = None
        if cfg.bc:
            torch_state = torch.get_rng_state()
            for j, tr in enumerate(D[len(bc_cache):], start=len(bc_cache)):
                bc_cache.append(build_bc_lift_dataset(env, encoder, [tr], replace(bccfg, seed=cfg.seed + 1000 * j), log))
            policy = BCLiftPolicy(train_bc_lift(merge_bc_lift_datasets(bc_cache), bccfg, log))
            torch.set_rng_state(torch_state)

        # ---- T_explore: cold start ----
        torch.manual_seed(cfg.seed + 1 + 1000 * (it - 1))
        T_exp = new_T_exp()

        # ---- s_1 ----
        s1 = None
        for attempt in range(1, S1_ATTEMPTS + 1):
            try:
                s1 = sample_s1(env, cfg, rng, D_P, D_R)
                break
            except RuntimeError:
                log(f"s_1 sampling attempt {attempt}/{S1_ATTEMPTS} failed")
        if s1 is None:
            log(f"no valid s_1 -- skipping iteration {it}")
            it_rows.append(dict(iteration=it, s1=None, s1_side=None, s1_dist_tau1_cm=None, s1_dist_D_cm=None,
                                stop="no_s1", rounds=0, path_end_dist_D_cm=None, path_end_nearest_traj=None,
                                bc=None, bc_steps=None, added=False, D_size=len(D), sec=round(time.time() - t_it, 1),
                                **_lm_cols(None)))
            continue
        s1_state, s1_p, s1_R, s1_d = s1
        emb1 = encoder.embed(env)
        log(f"s_1 = {s1_p.round(3)} ({100 * np.linalg.norm(s1_p - s0_p):.1f} cm from s_0, pose distance to D "
            f"{100 * s1_d:.1f} cm, position distance {100 * np.linalg.norm(D_P - s1_p, axis=1).min():.1f} cm)")
        R_hold = s1_R if cfg.explore_rot == "none" else None

        # ---- fallback rounds (same logic as the reaching loop) ----
        start_state, start_emb, start_p, start_R = s1_state, emb1, s1_p, s1_R
        start_hist, start_hist_qu = [], []
        start_u = None
        chain_tokens, chain_eefs, chain_rots, chain_states = [], [s1_p], [s1_R], [s1_state]
        path_state, path_emb = s1_state, emb1
        explore_buffer, rounds_trajs, s_stars, rows, state_scores, round_moved = [], [], [], [], [], []
        stop_reason = "max_rounds"
        def pos_D(p, Rm):                    # position distance to D (m) and the angle (deg) at that state
            j = int(np.linalg.norm(D_P - np.asarray(p), axis=1).argmin())
            return float(np.linalg.norm(D_P[j] - p)), float(np.degrees(rot_angle(Rm, D_R[j])))
        log(f"\n{'rnd':>3} {'steps':>5} {'stuck':>5} {'rej':>5} {'blocked':>7} {'U(end)':>8} {'d(end,D)':>8} "
            f"{'dmin,D':>7} {'pos(end)':>8} {'ang(end)':>8} {'pmin,D':>7} {'sec':>6}" + (f" {'move':>5}" if cfg.keep_best else ""))
        for rnd in range(1, cfg.max_rounds + 1):
            t_r = time.time()
            if cfg.keep_best or start_u is None:
                env.set_state(start_state); _, start_u = dnn.state_u(start_emb)
            keep_n = cfg.max_len - cfg.m
            hist = start_hist[-keep_n:] if keep_n > 0 else []
            hist_qu = start_hist_qu[-keep_n:] if keep_n > 0 else []
            trajs = [collect_lift_trajectory(T_exp, dnn, env, encoder, cfg, rng, start_state, start_emb, hist, hist_qu,
                                             R_hold=R_hold)
                     for _ in range(cfg.k)]
            explore_buffer.extend(trajs)
            loss_e = train_q(T_exp, explore_sequences(explore_buffer, cfg), cfg.explore_epochs, cfg, train_rng)
            S = []
            for t in trajs:
                if t.n_new == 0:
                    continue
                _, mu_e_all = mc_all(T_exp, t.tokens, cfg)
                for j in range(1, t.n_new + 1):
                    p_ = t.hist_len + j - 1
                    S.append((t, j, t.tok_qu[p_][1], float(mu_e_all[p_])))
            if not S:
                stop_reason = "stuck"
                log(f"{rnd:>3}  every trajectory was stuck at its first step -- stopping")
                break
            S_sorted = sorted(S, key=lambda c: -c[3])[:max(1, int(np.ceil(cfg.keep_frac * len(S))))]
            best, j_star, u_star, _ = min(S_sorted, key=lambda c: c[2])
            moved = (not cfg.keep_best) or u_star < start_u
            dist_D = lambda p, Rm: float(dist_to_poses(p, Rm, D_P, D_R).min())
            if moved:
                cut = best.hist_len + j_star
                chain_tokens.extend(best.tokens[best.hist_len:cut])
                chain_eefs.extend(best.eefs[1:j_star + 1]); chain_rots.extend(best.rots[1:j_star + 1])
                chain_states.extend(best.qs[1:j_star + 1])
                start_state, start_emb = best.qs[j_star], best.embs[j_star]
                start_p, start_R = best.eefs[j_star], best.rots[j_star]
                start_hist, start_hist_qu, start_u = best.tokens[:cut], best.tok_qu[:cut], u_star
                path_state, path_emb = start_state, start_emb
            round_moved.append(bool(moved)); s_stars.append(np.asarray(start_p)); rounds_trajs.append(trajs)
            state_scores.append([dict(traj=next(i for i, tt in enumerate(trajs) if tt is c[0]), step=c[1],
                                      q_td=float("nan"), mu_td=c[2], q_exp=float("nan"), mu_exp=c[3],
                                      in_S_prime=True, dist_tau1=dist_D(c[0].eefs[c[1]], c[0].rots[c[1]]),
                                      dist_pos=pos_D(c[0].eefs[c[1]], c[0].rots[c[1]])[0])
                                 for c in S])
            new_d = [dist_D(p, Rm) for t in trajs for p, Rm in zip(t.eefs[1:], t.rots[1:])]
            new_p = [pos_D(p, Rm)[0] for t in trajs for p, Rm in zip(t.eefs[1:], t.rots[1:])]
            sp_pos, sp_ang = pos_D(start_p, start_R)
            row = dict(round=rnd, steps=int(sum(t.n_new for t in trajs)), stuck=int(sum(t.stuck for t in trajs)),
                       rejects=int(sum(t.n_reject for t in trajs)), blocked=int(sum(t.n_sample_fail for t in trajs)),
                       loss_exp=loss_e, moved=bool(moved), mu_td_sstar=float(start_u),
                       sstar=np.asarray(start_p).round(4).tolist(), sstar_dist_tau1=dist_D(start_p, start_R),
                       min_dist_tau1=min(new_d) if new_d else float("nan"), sstar_pos_dist=sp_pos,
                       sstar_angle_deg=sp_ang, min_pos_dist=min(new_p) if new_p else float("nan"),
                       sec=time.time() - t_r)
            rows.append(row)
            log(f"{rnd:>3} {row['steps']:>5} {row['stuck']:>5} {row['rejects']:>5} {row['blocked']:>7} "
                f"{start_u:>8.4f} {100 * row['sstar_dist_tau1']:>7.1f}c {100 * row['min_dist_tau1']:>6.1f}c "
                f"{100 * sp_pos:>7.1f}c {sp_ang:>7.1f}d {100 * row['min_pos_dist']:>6.1f}c "
                f"{row['sec']:>6.1f}" + (f" {'yes' if moved else 'stay':>5}" if cfg.keep_best else ""))
            if start_u < mu_tol:
                stop_reason = "mu_td_certain"
                log(f"T_D certain at the path end: U = {start_u:.4f} < {mu_tol:.4f} -> BC takes over")
                break

        # ---- how well U tracked distance to D over every scored state (Spearman) ----
        def _spear(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if len(a) < 3:
                return float("nan")
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])
        _sc = [c for rs in state_scores for c in rs]
        u_spear = dict(pose=_spear([c["mu_td"] for c in _sc], [c["dist_tau1"] for c in _sc]),
                       position=_spear([c["mu_td"] for c in _sc], [c["dist_pos"] for c in _sc]), n=len(_sc))
        if _sc:
            log(f"U vs distance to D over {len(_sc)} visited states (Spearman): pose {u_spear['pose']:+.2f}, "
                f"position {u_spear['position']:+.2f}  (+1 = higher U farther from D)")

        # ---- BC hand-off ----
        bc_info, bc_r, tau_new = None, None, None
        if cfg.bc and rows:
            max_steps = int(cfg.bc_max_steps_mult * len(tau1.tokens))
            r = bc_r = bc_lift_rollout(policy, env, encoder, path_state, -1.0, (D_P, D_R), bccfg, max_steps,
                                       rng=np.random.default_rng(cfg.seed + 12_000_003 + it), start_emb=path_emb)
            bc_info = dict(success=bool(r["success"]), end=r["end"], steps=len(r["norm_actions"]),
                           start_dist=float(r["dist"][0]), max_dev=r["max_dev"], mean_dev=r["mean_dev"],
                           cube_lift_cm=100 * float(env.cube_pos()[2] - env.cube_start[2]),
                           eefs=np.asarray(r["eefs"]).tolist(), dist=r["dist"].tolist(),
                           grip_cmd=[float(g) for g in r["grip_cmd"][1:]], block_grip=r["block_grip"],
                           block_reasons=list(r["block_reasons"]))
            bc_info["start_pos_dist"] = 100 * pos_D(np.asarray(r["eefs"][0]), np.asarray(r["rots"][0]))[0]
            log(f"\nBC takes over ({stop_reason}; path end {r['dist'][0]:.1f} cm from D, position "
                f"{bc_info['start_pos_dist']:.1f} cm): {r['end']} after "
                f"{len(r['norm_actions'])} actions | cube lifted {bc_info['cube_lift_cm']:.1f} cm")
            log("BC gripper commands (taken actions): "
                + ("".join("C" if g > 0 else "o" for g in bc_info["grip_cmd"]) or "-") + "  (o open, C close)")
            if r["end"] == "blocked":
                log(f"blocked action: gripper {'close' if r['block_grip'] > 0 else 'open'} | every try failed: "
                    + ", ".join(r["block_reasons"]))
            if r["success"]:
                toks = list(chain_tokens) + list(r["tokens"])
                n = len(toks)
                bc_ph = bc_action_phases(r["eefs"], [float(a[6]) for a in r["norm_actions"]], env.cube_start)
                tau_new = LiftTraj(toks, list(chain_eefs) + list(r["eefs"][1:]),
                                   list(chain_states) + list(r["states"][1:]), r["embs"][-1], reached=True,
                                   labels=cfg.gamma ** (n - 1 - np.arange(n, dtype=float)),
                                   rots=list(chain_rots) + list(r["rots"][1:]),
                                   grip_cmd_act=[-1.0] * len(chain_tokens) + [float(a[6]) for a in r["norm_actions"]],
                                   phases=["approach"] * len(chain_tokens) + bc_ph)
                if cfg.smooth:
                    tau_new, sm = smooth_lift_traj(env, encoder, tau_new, env.cube_start, len(chain_tokens),
                                                   SmoothConfig(gamma=cfg.gamma), log)
                    if sm.get("applied"):
                        log(f"smooth: {sm['len_before']} -> {sm['len_after']} actions | "
                            + " | ".join(f"{k}: {v}" for k, v in sm["pieces"].items())
                            + f" | hover states collapsed {sm['hover_collapsed']}, {sm['n_new_states']} new states")
                    else:
                        log(f"smooth skipped: {sm.get('reason')}")
                    n = len(tau_new.tokens)
                elif cfg.shortcut:
                    tau_new, sc = shortcut_lift_traj(env, encoder, tau_new, env.cube_start,
                                                     ShortcutConfig(gamma=cfg.gamma), log)
                    if sc.get("applied"):
                        log(f"shortcut: {sc['len_before']} -> {sc['len_after']} actions (before the final grasp "
                            f"{sc['pre_grasp_before']} -> {sc['pre_grasp_after']}, {sc['waypoints']} waypoints, "
                            f"{sc['kept_original_steps']} original steps kept; open hover at the grasp removed: "
                            f"{sc['open_hover_before']} actions; {sc['new_states_encoded']} new states, "
                            f"{sc['ik_checks']} IK checks)")
                    else:
                        log(f"shortcut skipped: {sc.get('reason')}")
                    n = len(tau_new.tokens)
                D.append(tau_new)
                n_exec = len(chain_tokens) + len(r["norm_actions"])
                log(f"tau_{len(D)}: executed fallback ({len(chain_tokens)} actions) + BC ({len(r['norm_actions'])}) = "
                    f"{n_exec} actions"
                    + (f" -> {n} actions after {'smoothing' if cfg.smooth else 'shortcutting'}" if n != n_exec else "")
                    + f", cube lifted -> added to D (|D| = {len(D)}) | BC phases (executed) "
                    f"{ {p: bc_ph.count(p) for p in ('approach', 'descend', 'grasp', 'lift')} }")
            else:
                log(f"BC did not lift the cube ({r['end']}): not added to D (still plotted)")

        # ---- outputs ----
        with open(os.path.join(it_dir, "rounds.csv"), "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summary = dict(
            stop_reason=stop_reason, rounds=len(rows), mu_tol=mu_tol, iteration=it, task="pick",
            tau1_len=len(tau1.tokens), s0=s0_p.tolist(), s1=np.asarray(s1_p).tolist(), cube=env.cube_start.tolist(),
            config=asdict(cfg), wall_sec=time.time() - t_it, tau1_eefs=np.asarray(tau1.eefs).tolist(),
            tau1_mu_td=[u for _, u in qu_D[0][1:]], s_stars=[np.asarray(s).tolist() for s in s_stars],
            round_moved=round_moved, chain_eefs=np.asarray(chain_eefs).tolist(), state_scores=state_scores,
            bc_rollout=bc_info, u_dist_spearman=u_spear, D_eefs=[np.asarray(tr.eefs).tolist() for tr in D_iter],
            tau2=None if tau_new is None else dict(eefs=np.asarray(tau_new.eefs).tolist(), join_idx=-1,
                                                   labels=tau_new.labels.tolist(),
                                                   # state of tau_2 where the fallback part ends and BC begins
                                                   # (the hand-off state is kept exactly by smoothing / shortcutting)
                                                   handoff_idx=int(np.argmin(np.linalg.norm(
                                                       np.asarray(tau_new.eefs) - np.asarray(bc_r["eefs"][0]), axis=1)))),
            exploration=[[np.asarray(t.eefs).tolist() for t in trajs] for trajs in rounds_trajs])
        with open(os.path.join(it_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        _lm = iteration_metrics(summary)
        if rows:
            write_html(it_dir, summary)
            write_html(it_dir, summary, path_only=True)
            if cfg.gif:
                try:
                    cur_states = list(chain_states) + (list(bc_r["states"][1:]) if bc_r is not None else [])
                    cur_eefs = list(chain_eefs) + (list(bc_r["eefs"][1:]) if bc_r is not None else [])
                    labels = ["fallback"] * len(chain_states) + ["BC"] * (len(cur_states) - len(chain_states))
                    write_replay_gif(env, os.path.join(it_dir, "replay.gif"), [np.asarray(tr.eefs) for tr in D_iter],
                                     cur_states, cur_eefs, n_fallback=len(chain_eefs),
                                     title=f"iter {it} | " if multi else "", set_frame=env.set_state,
                                     phase_labels=labels, log=log)
                except Exception as e:
                    log(f"replay gif failed ({type(e).__name__}: {e}) -- continuing")
        end_p = np.asarray(chain_eefs[-1])
        j_near = int(D_owner[dist_to_poses(end_p, chain_rots[-1], D_P, D_R).argmin()])
        it_rows.append(dict(iteration=it, s1=np.asarray(s1_p).round(4).tolist(), s1_side=None,
                            s1_dist_tau1_cm=round(100 * float(dist_to_poses(s1_p, s1_R, np.asarray(tau1.eefs), np.asarray(tau1.rots)).min()), 2),
                            s1_dist_D_cm=round(100 * s1_d, 2), stop=stop_reason, rounds=len(rows),
                            path_end_dist_D_cm=round(100 * float(dist_to_poses(end_p, chain_rots[-1], D_P, D_R).min()), 2) if rows else None,
                            path_end_nearest_traj=j_near + 1 if rows else None,
                            bc=None if bc_info is None else bc_info["end"], bc_steps=None if bc_info is None else bc_info["steps"],
                            added=tau_new is not None, D_size=len(D), sec=round(time.time() - t_it, 1),
                            **_lm_cols(_lm)))
        log(f"learning: lifted={_lm['lifted']} | actions fallback {_lm['fallback_actions']} + BC {_lm['bc_actions']} | "
            f"return episode {_lm['return_episode']:.3g}, BC "
            f"{'-' if _lm['return_bc'] is None else format(_lm['return_bc'], '.3g')} | fallback distance to D "
            f"{_lm['s1_dist_D_cm']:.1f} -> {_lm['end_dist_D_cm']:.1f} cm ({'closer' if _lm['fallback_closer'] else 'not closer'})")
        log(f"stop={stop_reason} rounds={len(rows)} bc={bc_info['end'] if bc_info else '-'} "
            f"new trajectory={'yes' if tau_new else 'no'} | |D|={len(D)} | {time.time() - t_it:.0f}s")

    if multi:
        write_iteration_outputs(outdir, it_rows, D, env.cube_start, cfg.seed, log)
    log(f"total {time.time() - t0:.0f}s")
    env.close(); logf.close()
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Online fallback loop for the pick-up task (v6: fixed-orientation fallback + orientation-holding tau_1).")
    d = LiftConfig()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--n-iters", type=int, default=d.n_iters)
    p.add_argument("--k", type=int, default=d.k)
    p.add_argument("--m", type=int, default=d.m)
    p.add_argument("--max-rounds", type=int, default=d.max_rounds)
    p.add_argument("--max-retries", type=int, default=d.max_retries)
    p.add_argument("--rot-mode", choices=["xyz", "yaw"], default=d.rot_mode, help="tau_1 / calibration rotation axis")
    p.add_argument("--explore-rot", choices=["none", "xyz", "yaw"], default=d.explore_rot,
                   help="fallback rotation: none = orientation held at s_1 (v5 default); xyz/yaw = v4 behaviour")
    p.add_argument("--demo-noise", type=float, default=d.demo_noise)
    p.add_argument("--demo-rot-noise", type=float, default=d.demo_rot_noise)
    p.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=d.smooth,
                   help="smooth + resample the fallback and BC parts of each successful tau_k (default on)")
    p.add_argument("--shortcut", action=argparse.BooleanOptionalAction, default=d.shortcut,
                   help="greedy-shortcut each successful tau_k before adding it to D (default on)")
    p.add_argument("--demo-hold-orient", action=argparse.BooleanOptionalAction, default=d.demo_hold_orient,
                   help="v6: tau_1 holds the start orientation until the hover point (default on)")
    p.add_argument("--s1-min-dist", type=float, default=d.s1_min_dist)
    p.add_argument("--s1-max-dist", type=float, default=d.s1_max_dist)
    p.add_argument("--cube-offset", type=float, nargs=2, default=list(d.cube_offset), metavar=("DX", "DY"))
    p.add_argument("--n-dnn", type=int, default=d.n_dnn)
    p.add_argument("--mu-tol", type=float, default=None)
    p.add_argument("--keep-best", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bc-n-aug", type=int, default=d.bc_n_aug)
    p.add_argument("--bc-epochs", type=int, default=d.bc_epochs)
    p.add_argument("--bc-max-steps-mult", type=float, default=d.bc_max_steps_mult)
    p.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default="pick_results/online")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = LiftConfig(seed=a.seed, n_iters=a.n_iters, k=a.k, m=a.m, max_rounds=a.max_rounds,
                     max_retries=a.max_retries, rot_mode=a.rot_mode, explore_rot=a.explore_rot, demo_noise=a.demo_noise,
                     demo_rot_noise=a.demo_rot_noise, demo_hold_orient=a.demo_hold_orient, shortcut=a.shortcut, smooth=a.smooth, s1_min_dist=a.s1_min_dist, s1_max_dist=a.s1_max_dist,
                     cube_offset=tuple(a.cube_offset), n_dnn=a.n_dnn, mu_tol=a.mu_tol, keep_best=a.keep_best,
                     bc=a.bc, bc_n_aug=a.bc_n_aug, bc_epochs=a.bc_epochs, bc_max_steps_mult=a.bc_max_steps_mult,
                     gif=a.gif)
    enc_cfg = EncoderConfig(n_calib=a.n_calib)
    if a.smoke:
        cfg.k, cfg.m, cfg.max_rounds, cfg.explore_epochs, cfg.mc_samples = 2, 3, 2, 10, 8
        cfg.bc_n_aug, cfg.bc_epochs, cfg.dnn_epochs = 2, 100, 100
        enc_cfg.n_calib = 20
    return cfg, enc_cfg, a.outdir


if __name__ == "__main__":
    cfg, enc_cfg, outdir = parse_args()
    main(cfg, enc_cfg, outdir)
