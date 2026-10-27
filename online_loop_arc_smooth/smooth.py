"""
Smoothing + arc-length resampling of a successful pick trajectory before it
joins D (online_loop_arc_smooth).

The executed trajectory is split into pieces, each handled separately:
  1. fallback        : s_1 -> BC hand-off state                       (sigma_fallback, ~2 cm)
  2a. BC approach    : hand-off -> where BC enters the column above
                       the grasp pose for good                         (sigma_bc, ~0.5 cm)
  2b. BC descent     : column entry -> grasp pose (start of the final
                       close run)                                      (sigma_bc)
  3. grasp + lift    : kept exactly as executed.
Per piece (1, 2a, 2b):
  * consecutive positions closer than dedup (3 mm) are merged -- pauses and
    BC's open hover at the grasp collapse to one point;
  * a least-squares cubic B-spline (scipy splprep, smoothing s = m * sigma^2
    for m points; end points weighted heavily) is fitted -- wiggles of the
    order of sigma disappear, the overall route stays;
  * the curve is resampled into n = ceil(length / 1 cm) equal steps (<= 1 cm);
    the piece's first / last point are the exact original states (pins:
    s_1, hand-off, column entry, grasp pose), so pieces join without gaps;
  * orientation of a new point = the original orientation at the same
    fraction of the piece's (deduplicated) path length, slerped -- the turn to
    the cube's yaw stays where BC turned; steps turning > 0.1 rad are split;
  * every new point is checked by placing the arm with 6-D IK (no forbidden /
    cube contact, inside the workspace); on failure the piece is retried with
    sigma / 2, then sigma / 4, then kept exactly as executed.
New points are rendered + encoded; actions = the steps between consecutive
points (gripper open). Phases: open within 3 cm (sideways) of the grasp pose
and below hover + 1 cm = descend, else approach (piece 1 = approach). Labels
gamma^(T-1-t) are recomputed for the new length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep

from lift_env import in_workspace, mat_to_rotvec, rot_angle, rotvec_to_mat
from scripted_pick import ROT_STEP, STEP, to_norm


@dataclass
class SmoothConfig:
    sigma_fallback: float = 0.02     # allowed smoothing deviation, fallback piece (m)
    sigma_bc: float = 0.005          # allowed smoothing deviation, BC pieces (m)
    dedup: float = 0.003             # merge consecutive positions closer than this (m)
    col_xy: float = 0.015            # column radius around the grasp pose (m)
    hover: float = 0.10              # tau_1's hover height above the cube (m)
    gamma: float = 0.97


def _fuse(e, a):
    return np.concatenate([np.asarray(e, np.float32).reshape(-1), np.asarray(a, np.float32)])


def _final_close_start(grip) -> int:
    k = len(grip)
    while k > 0 and grip[k - 1] > 0:
        k -= 1
    return k


def _dedup(idx: List[int], P) -> List[int]:
    """Indices kept after merging near-duplicate consecutive positions; the
    first and last index are always kept."""
    keep = [idx[0]]
    for i in idx[1:-1]:
        if np.linalg.norm(P[i] - P[keep[-1]]) >= _dedup.d:
            keep.append(i)
    if len(idx) > 1:
        if len(keep) > 1 and np.linalg.norm(P[idx[-1]] - P[keep[-1]]) < _dedup.d:
            keep[-1] = idx[-1]
        else:
            keep.append(idx[-1])
    return keep


def _resample(Q: np.ndarray, sigma: float) -> np.ndarray:
    """Smoothed, arc-length resampled positions (first / last = Q's ends)."""
    m = len(Q)
    L0 = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    if m < 2 or L0 < 1e-6:
        return Q[[0, -1]] if m >= 2 else Q
    if m >= 4 and sigma > 0:
        w = np.ones(m); w[0] = w[-1] = 100.0
        try:
            tck, _ = splprep(Q.T, w=w, s=m * sigma ** 2, k=3)
            dense = np.array(splev(np.linspace(0, 1, max(400, 40 * m)), tck)).T
        except Exception:                                # degenerate data: use the polyline
            return _resample(Q, 0.0)
    else:
        dense = np.vstack([Q[i] + (Q[i + 1] - Q[i]) * t for i in range(m - 1) for t in np.linspace(0, 1, 20, endpoint=False)] + [Q[-1:]])
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(cum[-1] / STEP - 1e-9)), 1)
    targets = np.linspace(0, cum[-1], n + 1)
    out = np.array([np.array([np.interp(t, cum, dense[:, d]) for d in range(3)]) for t in targets])
    out[0], out[-1] = Q[0], Q[-1]
    return out


def _orient(newP, Q, RQ):
    """Orientation of each new point: the original orientation at the same
    fraction of the path length (slerp between neighbouring original states)."""
    cq = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    cn = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(newP, axis=0), axis=1))])
    out = []
    for u in (cn / cn[-1] if cn[-1] > 0 else np.zeros(len(newP))):
        t = u * cq[-1]
        j = int(np.clip(np.searchsorted(cq, t, side="right") - 1, 0, len(Q) - 2)) if len(Q) > 1 else 0
        if len(Q) == 1:
            out.append(RQ[0]); continue
        f = 0.0 if cq[j + 1] == cq[j] else (t - cq[j]) / (cq[j + 1] - cq[j])
        out.append(rotvec_to_mat(mat_to_rotvec(RQ[j + 1] @ RQ[j].T) * f) @ RQ[j])
    return out


def _split_rot(pts):
    """Split steps turning more than ROT_STEP (positions interpolated)."""
    out = [pts[0]]
    for (p, R) in pts[1:]:
        p0, R0 = out[-1]
        rv = mat_to_rotvec(R @ R0.T)
        n = max(int(np.ceil(np.linalg.norm(rv) / ROT_STEP - 1e-9)), 1)
        for m in range(1, n + 1):
            out.append((p0 + (p - p0) * m / n, rotvec_to_mat(rv * m / n) @ R0))
    return out


def smooth_lift_traj(env, encoder, tr, cube, n_fb: int, cfg: SmoothConfig, log=print) -> Tuple[object, Dict]:
    """tr: LiftTraj = fallback part (n_fb actions) + BC rollout. Returns
    (new LiftTraj, info). The simulator state is left modified."""
    _dedup.d = cfg.dedup
    P = [np.asarray(p, float) for p in tr.eefs]
    R = [np.asarray(r, float) for r in tr.rots]
    S = list(tr.states)
    T = len(tr.tokens)
    embs = [np.asarray(t[:-7]) for t in tr.tokens] + [np.asarray(tr.end_emb).reshape(-1)]
    cube = np.asarray(cube, float)
    k = _final_close_start(tr.grip_cmd_act)
    if k <= n_fb:
        return tr, dict(applied=False, reason="BC closed at the hand-off (no open BC part)")
    # column entry: start of the final stretch of BC states within col_xy of the grasp pose
    c = k
    while c - 1 >= n_fb and np.linalg.norm(P[c - 1][:2] - P[k][:2]) < cfg.col_xy:
        c -= 1
    pieces = [("fallback", 0, n_fb, cfg.sigma_fallback), ("bc_approach", n_fb, c, cfg.sigma_bc),
              ("bc_descent", c, k, cfg.sigma_bc)]

    def phase_of(p, piece):
        if piece == "fallback":
            return "approach"
        return "descend" if (np.linalg.norm(p[:2] - P[k][:2]) < 0.03 and p[2] < cube[2] + cfg.hover + 0.01) else "approach"

    toks, eefs, rots, states, grip, phases, new_embs = [], [P[0]], [R[0]], [S[0]], [], [], [embs[0]]
    info = dict(applied=True, len_before=T, n_new_states=0, pieces={})
    for name, a, b, sigma in pieces:
        if b - a < 1:
            info["pieces"][name] = "empty"
            continue
        idx = _dedup(list(range(a, b + 1)), P)
        Q, RQ = np.array([P[i] for i in idx]), [R[i] for i in idx]
        result, used = None, None
        for sg in (sigma, sigma / 2, sigma / 4):
            newP = _resample(Q, sg)
            pts = _split_rot(list(zip(newP, _orient(newP, Q, RQ))))
            env.set_state(S[a], last_grip=-1.0)
            mids, ok = [], True
            for (p, Rm) in pts[1:-1]:
                if not in_workspace(p) or not env.place_arm(p, Rm) or env.cube_contact():
                    ok = False
                    break
                mids.append((env.get_state(), *env.eef_pose()))
            if ok:
                result, used = mids, sg
                break
        if result is None:                               # keep the piece exactly as executed
            for t in range(a, b):
                toks.append(tr.tokens[t]); grip.append(tr.grip_cmd_act[t]); phases.append(tr.phases[t])
                states.append(S[t + 1]); eefs.append(P[t + 1]); rots.append(R[t + 1]); new_embs.append(embs[t + 1])
            info["pieces"][name] = f"{b - a} actions kept as executed (smoothing failed the IK / contact checks)"
            continue
        chain = [(st, p, Rm, None) for (st, p, Rm) in result] + [(S[b], P[b], R[b], embs[b])]
        pp, pR, pe = eefs[-1], rots[-1], new_embs[-1]
        for (st, p, Rm, e) in chain:
            if e is None:
                env.set_state(st, last_grip=-1.0)
                e = encoder.embed(env).reshape(-1); info["n_new_states"] += 1
            toks.append(_fuse(pe, to_norm(p - pp, mat_to_rotvec(Rm @ pR.T), -1.0)))
            grip.append(-1.0); phases.append(phase_of(pp, name))
            states.append(st); eefs.append(p); rots.append(Rm); new_embs.append(e)
            pp, pR, pe = p, Rm, e
        L_old = float(np.linalg.norm(np.diff(np.array(P[a:b + 1]), axis=0), axis=1).sum())
        L_new = float(np.linalg.norm(np.diff(np.array([q for q in [P[a]] + [x[1] for x in chain]]), axis=0), axis=1).sum())
        info["pieces"][name] = (f"{b - a} -> {len(chain)} actions, path {100 * L_old:.1f} -> {100 * L_new:.1f} cm"
                                f" (sigma {100 * used:.2g} cm)")
    for t in range(k, T):                                # grasp + lift exactly as executed
        toks.append(tr.tokens[t]); grip.append(tr.grip_cmd_act[t]); phases.append(tr.phases[t])
        states.append(S[t + 1]); eefs.append(P[t + 1]); rots.append(R[t + 1]); new_embs.append(embs[t + 1])
    n = len(toks)
    info["len_after"] = n
    info["hover_collapsed"] = int(sum(1 for t in range(n_fb, k) if np.linalg.norm(P[t] - P[k]) < 0.003))
    new = replace(tr, tokens=toks, eefs=eefs, qs=states, rots=rots, grip_cmd_act=grip, phases=phases,
                  embs=new_embs, labels=cfg.gamma ** (n - 1 - np.arange(n, dtype=float)), tok_qu=[])
    return new, info
