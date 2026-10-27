"""
Per-iteration learning metrics of the pick online loop (arc_shortcutting).

Sparse reward, consistent with the Q-labels gamma^(T-1-t) that T_D is trained
on: reward 1 on the action that lifts the cube, 0 otherwise. So

    episode return = gamma^(L - 1) if lifted else 0,  L = fallback + BC actions (from s_1)
    BC return      = gamma^(L_BC - 1) if lifted, 0 if BC ran and failed,
                     None if BC never ran (fallback stuck in round 1)

L counts the actions actually EXECUTED (before any shortcutting of the trajectory
that joins D). Fallback effectiveness: position distance (cm) from s_1, and from
the fallback's path end (= BC's start), to the nearest state of D at that
iteration; the fallback "worked" if it ended at least `closer_cm` closer.
A fallback stuck in round 1 never moved (delta 0, did not work).

Everything is computed from the iteration's summary.json, so it also applies
to runs made before this module existed.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

CLOSER_CM = 1.0


def iteration_metrics(summary: Dict, gamma: Optional[float] = None, closer_cm: float = CLOSER_CM) -> Dict:
    g = gamma if gamma is not None else float((summary.get("config") or {}).get("gamma", 0.97))
    D = np.concatenate([np.asarray(t, float) for t in summary["D_eefs"]])
    chain = np.asarray(summary.get("chain_eefs") or [summary["s1"]], float)
    b = summary.get("bc_rollout") or None
    fb_actions = max(len(chain) - 1, 0)
    bc_ran = b is not None
    bc_actions = int(b["steps"]) if bc_ran else 0
    lifted = bool(bc_ran and b.get("end") == "reached")
    total = fb_actions + bc_actions
    s1_d = 100 * float(np.linalg.norm(D - np.asarray(summary["s1"], float), axis=1).min())
    end_d = 100 * float(np.linalg.norm(D - chain[-1], axis=1).min())
    return dict(
        iteration=int(summary.get("iteration", 0)), D_before=len(summary["D_eefs"]), stop=summary.get("stop_reason"),
        rounds=int(summary.get("rounds", 0)), bc_ran=bc_ran, bc_end=(b or {}).get("end", "-"), lifted=lifted,
        fallback_actions=fb_actions, bc_actions=bc_actions, total_actions=total,
        return_episode=(g ** (total - 1) if lifted else 0.0),
        return_bc=(g ** (bc_actions - 1) if lifted else (0.0 if bc_ran else None)),
        s1_dist_D_cm=round(s1_d, 2), end_dist_D_cm=round(end_d, 2), fallback_delta_cm=round(end_d - s1_d, 2),
        fallback_closer=bool(end_d < s1_d - closer_cm))
