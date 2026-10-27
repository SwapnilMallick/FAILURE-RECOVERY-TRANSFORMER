"""
Greedy shortcutting of a successful pick trajectory before it joins D
(online_loop_arc_shortcutting; online_loop_arc_shortcut_fallback: called with
upto = the BC hand-off index, so only the fallback part is shortcut and the BC
part -- approach, descent, any hover, grasp, lift -- stays exactly as executed).

Why: in the multi-iteration arc runs BC got worse as successful trajectories
were added -- each tau_k carried (a) the fallback's random walk, imitated as
"approach", and (b) BC's own open-gripper hover at the grasp pose (median 10,
up to 220 actions; tau_1 has 1). Shortcutting replaces everything BEFORE the
final grasp with a short, clean path between states of the same trajectory,
and keeps the grasp + lift exactly as executed.

Greedy (from state i, jump to the FARTHEST later state j the straight segment
i -> j can reach, then continue from j), over the pre-grasp states 0..k (k =
the state where the final close run starts = the pose that actually grasped):
  segment rules (checked analytically first, cheap):
    * every point inside the workspace;
    * below hover - 1 cm above the cube, the path must stay inside the
      vertical column above the grasp pose (<= col_xy sideways) -- approach
      high, descend vertically, like tau_1 (no diagonal swoop at the cube);
    * a segment that turns the gripper by more than rot_min_deg must be at
      most rot_seg_max_len long -- turning stays where the trajectory turned
      (near the hover point), so the approach region keeps one orientation;
  then physically: the arm is placed by 6-D IK at every <= 1 cm / <= 0.1 rad
  point along the segment (cube resting, gripper open); any forbidden contact,
  cube contact or failed IK rejects it. At most max_ik_checks segments are
  IK-checked per i; if none passes, the original transition i -> i+1 is kept.
The new path's intermediate states are rendered and encoded; its actions are
the 1-cm / 0.1-rad steps between consecutive states (gripper open). Segment
end points reuse the original states (exact simulator state + embedding).
Labels gamma^(T-1-t) are recomputed for the new length; phases: open within
3 cm (sideways) of the grasp pose and below hover + 1 cm = descend, else
approach; grasp / lift keep theirs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from lift_env import in_workspace, mat_to_rotvec, rot_angle, rotvec_to_mat
from scripted_pick import ROT_STEP, STEP, to_norm


@dataclass
class ShortcutConfig:
    hover: float = 0.10              # tau_1's hover height above the cube
    col_xy: float = 0.015            # column radius around the grasp pose below hover - 1 cm
    rot_min_deg: float = 5.0         # a segment turning more than this ...
    rot_seg_max_len: float = 0.03    # ... must be at most this long (m)
    max_ik_checks: int = 40          # IK-checked candidate segments per start state
    gamma: float = 0.97


def _fuse(e, a):
    return np.concatenate([np.asarray(e, np.float32).reshape(-1), np.asarray(a, np.float32)])


def _final_close_start(grip) -> int:
    k = len(grip)
    while k > 0 and grip[k - 1] > 0:
        k -= 1
    return k


def _segment_points(p0, R0, p1, R1):
    """Interpolated poses (excluding the start) every <= STEP / <= ROT_STEP."""
    rv = mat_to_rotvec(R1 @ R0.T)
    n = max(int(np.ceil(np.linalg.norm(p1 - p0) / STEP - 1e-9)),
            int(np.ceil(np.linalg.norm(rv) / ROT_STEP - 1e-9)), 1)
    return [(p0 + (p1 - p0) * m / n, rotvec_to_mat(rv * m / n) @ R0) for m in range(1, n + 1)]


def shortcut_lift_traj(env, encoder, tr, cube, cfg: ShortcutConfig, log=print,
                       upto: Optional[int] = None) -> Tuple[object, Dict]:
    """tr: LiftTraj (tokens, eefs, states, rots, grip_cmd_act, phases, end_emb).
    upto (arc_shortcut_fallback): shortcut only states 0..upto (the fallback
    part, ending at the BC hand-off state) and keep every action after it as
    executed; new states are "approach" and the low-height column rule is
    centred on the cube. upto=None: everything before the final close (arc_shortcutting).
    Returns (new LiftTraj, info). The simulator state is left modified."""
    P = [np.asarray(p, float) for p in tr.eefs]
    R = [np.asarray(r, float) for r in tr.rots]
    S = list(tr.states)
    T = len(tr.tokens)
    embs = [np.asarray(t[:-7]) for t in tr.tokens] + [np.asarray(tr.end_emb).reshape(-1)]
    k = _final_close_start(tr.grip_cmd_act) if upto is None else int(upto)
    if k <= 1:
        return tr, dict(applied=False, reason="nothing to shortcut (part has <= 1 action)")
    cube = np.asarray(cube, float)
    col_ref = P[k][:2] if upto is None else cube[:2]
    z_low = cube[2] + cfg.hover - 0.01

    def analytic_ok(i, j):
        if rot_angle(R[i], R[j]) > np.radians(cfg.rot_min_deg) and np.linalg.norm(P[j] - P[i]) > cfg.rot_seg_max_len:
            return False
        for p, _ in _segment_points(P[i], R[i], P[j], R[j]):
            if not in_workspace(p):
                return False
            if p[2] < z_low and np.linalg.norm(p[:2] - col_ref) > cfg.col_xy:
                return False
        return True

    def ik_path(i, j):
        env.set_state(S[i], last_grip=-1.0)
        out = []
        pts = _segment_points(P[i], R[i], P[j], R[j])
        for m, (p, Rm) in enumerate(pts):
            if not env.place_arm(p, Rm) or env.cube_contact():
                return None
            if m < len(pts) - 1:
                out.append((env.get_state(), *env.eef_pose()))
        return out                                       # intermediate states (end point = original S[j])

    wps, segs, n_ik = [0], [], 0
    i = 0
    while i < k:
        best, tries = None, 0
        for j in range(k, i + 1, -1):
            if not analytic_ok(i, j):
                continue
            tries += 1; n_ik += 1
            mid = ik_path(i, j)
            if mid is not None:
                best = (j, mid)
                break
            if tries >= cfg.max_ik_checks:
                break
        if best is None:
            best = (i + 1, None)                         # keep the original transition
        segs.append((i, *best)); wps.append(best[0]); i = best[0]

    # ---- rebuild the trajectory ----
    toks, eefs, rots, states, grip, phases, new_embs = [], [P[0]], [R[0]], [S[0]], [], [], [embs[0]]
    n_new_states = 0

    def phase_of(p):
        if upto is not None:
            return "approach"                            # fallback part: approach, as before
        return "descend" if (np.linalg.norm(p[:2] - P[k][:2]) < 0.03 and p[2] < cube[2] + cfg.hover + 0.01) else "approach"

    for (a, b, mid) in segs:
        if mid is None:                                  # original transition a -> a+1
            toks.append(tr.tokens[a]); grip.append(tr.grip_cmd_act[a]); phases.append(tr.phases[a])
            states.append(S[b]); eefs.append(P[b]); rots.append(R[b]); new_embs.append(embs[b])
            continue
        chain = [(st, p, Rm, None) for (st, p, Rm) in mid] + [(S[b], P[b], R[b], embs[b])]
        prev_p, prev_R, prev_e = eefs[-1], rots[-1], new_embs[-1]
        for (st, p, Rm, e) in chain:
            if e is None:
                env.set_state(st, last_grip=-1.0)
                e = encoder.embed(env).reshape(-1); n_new_states += 1
            act = to_norm(p - prev_p, mat_to_rotvec(Rm @ prev_R.T), -1.0)
            toks.append(_fuse(prev_e, act)); grip.append(-1.0); phases.append(phase_of(prev_p))
            states.append(st); eefs.append(p); rots.append(Rm); new_embs.append(e)
            prev_p, prev_R, prev_e = p, Rm, e
    # the rest exactly as executed (grasp + lift; or, with upto, the whole BC part)
    for t in range(k, T):
        toks.append(tr.tokens[t]); grip.append(tr.grip_cmd_act[t]); phases.append(tr.phases[t])
        states.append(S[t + 1]); eefs.append(P[t + 1]); rots.append(R[t + 1]); new_embs.append(embs[t + 1])
    n = len(toks)
    new = replace(tr, tokens=toks, eefs=eefs, qs=states, rots=rots, grip_cmd_act=grip, phases=phases,
                  embs=new_embs, labels=cfg.gamma ** (n - 1 - np.arange(n, dtype=float)), tok_qu=[])
    hover_before = int(sum(1 for t in range(k) if tr.grip_cmd_act[t] < 0
                           and np.linalg.norm(P[t][:2] - P[k][:2]) < 0.015 and P[t][2] < P[k][2] + 0.015))
    info = dict(applied=True, len_before=T, len_after=n, pre_grasp_before=k, pre_grasp_after=n - (T - k),
                waypoints=len(wps), kept_original_steps=sum(1 for s in segs if s[2] is None),
                new_states_encoded=n_new_states, ik_checks=n_ik, open_hover_before=hover_before)
    return new, info
