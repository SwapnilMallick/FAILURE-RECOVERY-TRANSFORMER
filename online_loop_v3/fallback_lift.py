"""
Online fallback loop for the PICK-UP task (v3, Stage B) -- the lift counterpart
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
    python online_loop_v3/fallback_lift.py --seed 10 --no-keep-best --outdir pick_results/online/seed10
    python online_loop_v3/fallback_lift.py --smoke --seed 10 --outdir /tmp/lift_smoke
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

from bc_lift import BCLiftConfig, BCLiftPolicy, bc_lift_rollout, build_bc_lift_dataset, merge_bc_lift_datasets, train_bc_lift
from dnn_ensemble import DNNEnsemble, train_dnn_ensemble
from fallback_explore import (
    S1_ATTEMPTS, FallbackConfig, Traj, explore_sequences, mc_all, train_q, write_iteration_outputs,
)
from image_encoder import EncoderConfig, ImageEncoder
from lift_env import LiftEnv, ROT_W, dist_to_poses, in_workspace
from online_cost_transformer import ImageCostTransformer, fuse, predict_last_mc
from plot_fallback_3d import write_html
from replay_gif import write_replay_gif
from scripted_pick import ROT_STEP, STEP, scripted_pick_cl, to_norm


@dataclass
class LiftConfig(FallbackConfig):
    rot_mode: str = "xyz"                # exploration / demo rotation axis: random 3-D | vertical
    demo_noise: float = 1.0              # tau_1 approach: translation direction noise
    demo_rot_noise: float = 1.0          # tau_1 approach: rotation axis noise
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
                         approach_noise=cfg.demo_noise, rot_noise=cfg.demo_rot_noise, rot_mode=cfg.rot_mode)
    if not r["success"]:
        raise RuntimeError("scripted pick did not lift the cube")
    embs = []
    for s in r["states"]:
        env.set_state(s)
        embs.append(encoder.embed(env).reshape(-1))
    T = len(r["norm_actions"])
    tokens = [fuse(embs[i], r["norm_actions"][i]) for i in range(T)]
    labels = cfg.gamma ** (T - 1 - np.arange(T, dtype=float))
    log(f"tau_1 (scripted pick, approach noise {cfg.demo_noise}, rotation noise {cfg.demo_rot_noise} "
        f"[{cfg.rot_mode}]): {T} actions | phases "
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


def collect_lift_trajectory(T_exp, dnn, env, encoder, cfg, rng, start_state, start_emb, hist, hist_qu):
    env.set_state(start_state, last_grip=-1.0)
    tokens = list(hist)
    tr = LiftTraj(tokens, [env.eef_pos()], [env.get_state()], start_emb, embs=[start_emb],
                  hist_len=len(hist), tok_qu=list(hist_qu), rots=[env.eef_rot()])
    emb = start_emb
    for _ in range(cfg.m):
        accepted = None
        for _retry in range(cfg.max_retries):
            a = sample_action(rng, cfg.rot_mode)
            s_b, lg = env.get_state(), env.last_grip
            out = env.act(a[:3] * STEP, a[3:6] * ROT_STEP, -1.0)
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
                          action_sampler=lambda r: sample_action(r, cfg.rot_mode))
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
                                bc=None, bc_steps=None, added=False, D_size=len(D), sec=round(time.time() - t_it, 1)))
            continue
        s1_state, s1_p, s1_R, s1_d = s1
        emb1 = encoder.embed(env)
        log(f"s_1 = {s1_p.round(3)} ({100 * np.linalg.norm(s1_p - s0_p):.1f} cm from s_0, pose distance to D "
            f"{100 * s1_d:.1f} cm)")

        # ---- fallback rounds (same logic as the reaching loop) ----
        start_state, start_emb, start_p, start_R = s1_state, emb1, s1_p, s1_R
        start_hist, start_hist_qu = [], []
        start_u = None
        chain_tokens, chain_eefs, chain_rots, chain_states = [], [s1_p], [s1_R], [s1_state]
        path_state, path_emb = s1_state, emb1
        explore_buffer, rounds_trajs, s_stars, rows, state_scores, round_moved = [], [], [], [], [], []
        stop_reason = "max_rounds"
        log(f"\n{'rnd':>3} {'steps':>5} {'stuck':>5} {'rej':>5} {'blocked':>7} {'U(end)':>8} {'d(end,D)':>8} "
            f"{'dmin,D':>7} {'sec':>6}" + (f" {'move':>5}" if cfg.keep_best else ""))
        for rnd in range(1, cfg.max_rounds + 1):
            t_r = time.time()
            if cfg.keep_best or start_u is None:
                env.set_state(start_state); _, start_u = dnn.state_u(start_emb)
            keep_n = cfg.max_len - cfg.m
            hist = start_hist[-keep_n:] if keep_n > 0 else []
            hist_qu = start_hist_qu[-keep_n:] if keep_n > 0 else []
            trajs = [collect_lift_trajectory(T_exp, dnn, env, encoder, cfg, rng, start_state, start_emb, hist, hist_qu)
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
                                      in_S_prime=True, dist_tau1=dist_D(c[0].eefs[c[1]], c[0].rots[c[1]]))
                                 for c in S])
            new_d = [dist_D(p, Rm) for t in trajs for p, Rm in zip(t.eefs[1:], t.rots[1:])]
            row = dict(round=rnd, steps=int(sum(t.n_new for t in trajs)), stuck=int(sum(t.stuck for t in trajs)),
                       rejects=int(sum(t.n_reject for t in trajs)), blocked=int(sum(t.n_sample_fail for t in trajs)),
                       loss_exp=loss_e, moved=bool(moved), mu_td_sstar=float(start_u),
                       sstar=np.asarray(start_p).round(4).tolist(), sstar_dist_tau1=dist_D(start_p, start_R),
                       min_dist_tau1=min(new_d) if new_d else float("nan"), sec=time.time() - t_r)
            rows.append(row)
            log(f"{rnd:>3} {row['steps']:>5} {row['stuck']:>5} {row['rejects']:>5} {row['blocked']:>7} "
                f"{start_u:>8.4f} {100 * row['sstar_dist_tau1']:>7.1f}c {100 * row['min_dist_tau1']:>6.1f}c "
                f"{row['sec']:>6.1f}" + (f" {'yes' if moved else 'stay':>5}" if cfg.keep_best else ""))
            if start_u < mu_tol:
                stop_reason = "mu_td_certain"
                log(f"T_D certain at the path end: U = {start_u:.4f} < {mu_tol:.4f} -> BC takes over")
                break

        # ---- BC hand-off ----
        bc_info, bc_r, tau_new = None, None, None
        if cfg.bc and rows:
            max_steps = int(cfg.bc_max_steps_mult * len(tau1.tokens))
            r = bc_r = bc_lift_rollout(policy, env, encoder, path_state, -1.0, (D_P, D_R), bccfg, max_steps,
                                       rng=np.random.default_rng(cfg.seed + 12_000_003 + it), start_emb=path_emb)
            bc_info = dict(success=bool(r["success"]), end=r["end"], steps=len(r["norm_actions"]),
                           start_dist=float(r["dist"][0]), max_dev=r["max_dev"], mean_dev=r["mean_dev"],
                           cube_lift_cm=100 * float(env.cube_pos()[2] - env.cube_start[2]),
                           eefs=np.asarray(r["eefs"]).tolist(), dist=r["dist"].tolist())
            log(f"\nBC takes over ({stop_reason}; path end {r['dist'][0]:.1f} cm from D): {r['end']} after "
                f"{len(r['norm_actions'])} actions | cube lifted {bc_info['cube_lift_cm']:.1f} cm")
            if r["success"]:
                toks = list(chain_tokens) + list(r["tokens"])
                n = len(toks)
                tau_new = LiftTraj(toks, list(chain_eefs) + list(r["eefs"][1:]),
                                   list(chain_states) + list(r["states"][1:]), r["embs"][-1], reached=True,
                                   labels=cfg.gamma ** (n - 1 - np.arange(n, dtype=float)),
                                   rots=list(chain_rots) + list(r["rots"][1:]),
                                   grip_cmd_act=[-1.0] * len(chain_tokens) + [float(a[6]) for a in r["norm_actions"]],
                                   phases=["approach"] * n)
                D.append(tau_new)
                log(f"tau_{len(D)} = fallback ({len(chain_tokens)} actions) + BC ({len(r['norm_actions'])}) = "
                    f"{n} actions, cube lifted -> added to D (|D| = {len(D)})")
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
            bc_rollout=bc_info, D_eefs=[np.asarray(tr.eefs).tolist() for tr in D_iter],
            tau2=None if tau_new is None else dict(eefs=np.asarray(tau_new.eefs).tolist(), join_idx=-1,
                                                   labels=tau_new.labels.tolist()),
            exploration=[[np.asarray(t.eefs).tolist() for t in trajs] for trajs in rounds_trajs])
        with open(os.path.join(it_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
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
                            added=tau_new is not None, D_size=len(D), sec=round(time.time() - t_it, 1)))
        log(f"stop={stop_reason} rounds={len(rows)} bc={bc_info['end'] if bc_info else '-'} "
            f"new trajectory={'yes' if tau_new else 'no'} | |D|={len(D)} | {time.time() - t_it:.0f}s")

    if multi:
        write_iteration_outputs(outdir, it_rows, D, env.cube_start, cfg.seed, log)
    log(f"total {time.time() - t0:.0f}s")
    env.close(); logf.close()
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Online fallback loop for the pick-up task (v3).")
    d = LiftConfig()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--n-iters", type=int, default=d.n_iters)
    p.add_argument("--k", type=int, default=d.k)
    p.add_argument("--m", type=int, default=d.m)
    p.add_argument("--max-rounds", type=int, default=d.max_rounds)
    p.add_argument("--max-retries", type=int, default=d.max_retries)
    p.add_argument("--rot-mode", choices=["xyz", "yaw"], default=d.rot_mode)
    p.add_argument("--demo-noise", type=float, default=d.demo_noise)
    p.add_argument("--demo-rot-noise", type=float, default=d.demo_rot_noise)
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
                     max_retries=a.max_retries, rot_mode=a.rot_mode, demo_noise=a.demo_noise,
                     demo_rot_noise=a.demo_rot_noise, s1_min_dist=a.s1_min_dist, s1_max_dist=a.s1_max_dist,
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
