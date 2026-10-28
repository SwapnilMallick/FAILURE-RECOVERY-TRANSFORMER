"""
Stage A check for the pick-up task: does the scripted pick (scripted_pick.py)
succeed on the physics Lift environment (lift_env.py) across seeds?

Per seed: build LiftEnv (cube position + yaw drawn per seed, +10 cm cube
offset as in v2/v3), run the scripted pick, and record success, the step it
was reached, phase lengths, cube lift, final gripper opening (~4.2-4.4 cm =
holding the cube across its faces) and whether any forbidden contact
(robot <-> table / robot <-> robot) occurred. A replay GIF (front / side /
isometric) of the saved simulator states shows the cube actually moving.

--approach-noise > 0 makes the approach phase varied, as the reaching
setup's varied demo did (grasp and lift stay clean).

Outputs in --outdir: pick.log, pick.csv, seed<N>_pick.gif, seed<N>_tau1.npz.

Usage (from the project root):
    python online_loop_v3/stageA_pick_eval.py
    python online_loop_v3/stageA_pick_eval.py --approach-noise 1.0 --outdir pick_results/stageA_varied
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from lift_env import LiftEnv
from replay_gif import write_replay_gif
from lift_env import rot_angle
from scripted_pick import scripted_pick, scripted_pick_cl


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "pick.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"scripted pick, approach noise {a.approach_noise}, cube offset {a.cube_offset}, seeds {a.seeds}")
    rows = []
    for seed in a.seeds:
        t0 = time.time()
        env = LiftEnv(seed=seed, cube_offset=tuple(a.cube_offset), enable_render=a.gif)
        if a.closed_loop:
            r = scripted_pick_cl(env, rng=np.random.default_rng(seed + 6_000_003),
                                 approach_noise=a.approach_noise, rot_noise=a.rot_noise,
                                 rot_mode=a.rot_mode)
            r["actions"] = r["norm_actions"]
        else:
            r = scripted_pick(env, rng=np.random.default_rng(seed + 6_000_003),
                              approach_noise=a.approach_noise)
        forbidden = 0
        for s in r["states"]:                                  # any table / self contact?
            env.set_state(s)
            forbidden += env.forbidden_contact()
        ph = {p: int((r["phase"] == p).sum()) for p in ("approach", "descend", "grasp", "lift")}
        row = dict(seed=seed, success=r["success"], success_step=r["success_at"],
                   steps=len(r["actions"]), **{f"{p}_steps": n for p, n in ph.items()},
                   cube_lift_cm=round(100 * r["cube_lift"], 2),
                   gripper_final_cm=round(100 * r["gripper"][-1], 2),
                   cube_yaw_deg=round(float(np.degrees(r["cube_yaw"])), 1),
                   start_to_cube_cm=round(100 * float(np.linalg.norm(r["cube"][0] - r["eef"][0])), 1),
                   forbidden_contact_states=forbidden)
        if a.closed_loop:                                     # orientation variety + execution cost
            R_last = r["rot"][-1]
            row.update(max_tilt_from_final_deg=round(float(max(np.degrees(rot_angle(Rk, R_last))
                                                                 for Rk in r["rot"])), 1),
                       substeps_total=int(r["substeps"].sum()),
                       substeps_per_action=round(float(r["substeps"].mean()), 1))
        np.savez(os.path.join(a.outdir, f"seed{seed}_tau1.npz"), **{k: r[k] for k in
                 ("actions", "states", "eef", "cube", "gripper", "phase")})
        if a.gif:
            write_replay_gif(env, os.path.join(a.outdir, f"seed{seed}_pick.gif"), [],
                             list(r["states"]), r["eef"], n_fallback=len(r["eef"]),
                             title=f"seed {seed} | ", set_frame=env.set_state,
                             phase_labels=list(r["phase"]) + ["done"], log=log)
        env.close()
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        log(f"seed {seed:>3}: {'SUCCESS' if r['success'] else 'FAILED '} at {'action' if a.closed_loop else 'step'} "
            f"{r['success_at']} of {row['steps']} | phases {ph} | cube lifted {row['cube_lift_cm']} cm | gripper "
            f"{row['gripper_final_cm']} cm | cube yaw {row['cube_yaw_deg']} deg | start->cube "
            f"{row['start_to_cube_cm']} cm | forbidden-contact states {forbidden} | {row['sec']}s"
            + (f" | max orientation deviation {row['max_tilt_from_final_deg']} deg | "
               f"{row['substeps_per_action']} substeps/action" if a.closed_loop else ""))

    with open(os.path.join(a.outdir, "pick.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["success"]]
    log(f"\nsuccess {len(ok)}/{len(rows)} | steps median {np.median([r['steps'] for r in rows]):.0f} "
        f"({min(r['steps'] for r in rows)}-{max(r['steps'] for r in rows)}) | cube lift median "
        f"{np.median([r['cube_lift_cm'] for r in rows]):.1f} cm | runs with forbidden contact "
        f"{sum(r['forbidden_contact_states'] > 0 for r in rows)}")
    log(f"outputs -> {a.outdir}/ (pick.log, pick.csv, seed<N>_pick.gif, seed<N>_tau1.npz)")
    logf.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage A: scripted pick-and-lift check across seeds.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--approach-noise", type=float, default=0.0)
    p.add_argument("--closed-loop", action="store_true",
                   help="Stage B: closed-loop actions (1 cm / 0.1 rad, env.act) instead of raw OSC steps")
    p.add_argument("--rot-noise", type=float, default=0.0,
                   help="closed loop: random rotation-axis noise during the approach (varied tau_1)")
    p.add_argument("--rot-mode", choices=["xyz", "yaw"], default="xyz")
    p.add_argument("--cube-offset", type=float, nargs=2, default=[0.10, 0.0], metavar=("DX", "DY"))
    p.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--outdir", default="pick_results/stageA")
    main(p.parse_args())
