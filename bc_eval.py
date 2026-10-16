"""
Stand-alone evaluation of the BC policy (bc_policy.py), before it is hooked
into the fallback loop.

Rebuilds exactly what a DNN fallback run with the same seed uses -- cube,
image calibration, varied tau_1 with fixed 1 cm steps (fallback_explore.py
defaults) -- trains the BC policy on tau_1 plus corrective examples, then
rolls it out from three kinds of start:
  (a) s_0: does it reproduce tau_1 and reach the cube?
  (b) perturbed starts: --dists cm off tau_1 (default 0.5, 1, 2, 3), at
      random points along the first 70% of tau_1, --n-starts per distance;
      the arm is placed by IK seeded from the nearest tau_1 configuration
  (c) fallback path ends: the last state of every run for this seed in
      --runs-dir (default dnn_fallback_results_10_ensemble), placed the same
      way (only joint-free EEF positions are saved); skipped if that run's
      tau_1 differs from this one
A rollout succeeds when the gripper gets within the task's 3 cm goal radius.

Outputs in --outdir (default bc_results/seed<seed>): bc.log, bc_eval.csv
(one row per rollout), bc_rollouts.json (paths), bc_3d.html (interactive),
bc_model.pt, bc_dataset.npz.

Usage:
    python bc_eval.py --seed 10
    python bc_eval.py --seed 10 --n-aug 40 --lookahead 3 --dists 0.5,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from bc_policy import BCConfig, BCPolicy, bc_rollout, build_bc_dataset, nearest_idx, train_bc
from fallback_explore import FallbackConfig, make_varied_demo
from image_encoder import EncoderConfig, ImageEncoder
from plot_fallback_3d import EPS_DIST, PLOTLY_JS, _sphere, fallback_path
from robosuite_env import WorldConfig
from task import RobosuiteTask, TaskConfig


def build_scene(seed: int, n_calib: int, log):
    """Cube, calibrated encoder and tau_1 exactly as fallback_explore.main
    builds them with its DNN defaults (varied demo, fixed 1 cm steps)."""
    cfg = FallbackConfig(seed=seed, fixed_step=True)
    torch.manual_seed(seed)
    task = RobosuiteTask(TaskConfig(world=WorldConfig(step_size=cfg.action_step)),
                         seed=seed, enable_render=True)
    encoder = ImageEncoder(EncoderConfig(n_calib=n_calib))
    task.world.cfg.step_size = WorldConfig().step_size
    try:
        encoder.calibrate(task, np.random.default_rng(seed + 2_000_003))
    finally:
        task.world.cfg.step_size = cfg.action_step
    task.world.cfg.fixed_step = True
    tau1 = make_varied_demo(cfg.demo_noise, seed, log)(task, encoder, cfg)
    return cfg, task, encoder, tau1


def place(task, tau1, p):
    """Arm to EEF position p by IK seeded from the nearest tau_1
    configuration; returns (q, eef) or None if unreachable / colliding."""
    ik, world = task.ik, task.world
    T = np.asarray(tau1.eefs)
    q, ok = ik.solve_ik(p, q_init=tau1.qs[nearest_idx(p, T)])
    if not ok:
        return None
    ik.set_q(q)
    if world._forbidden_contact():
        return None
    return q, ik.eef_pos()


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "bc.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    fcfg, task, encoder, tau1 = build_scene(a.seed, a.n_calib, log)
    T = np.asarray(tau1.eefs)
    max_steps = int(a.max_steps_mult * len(tau1.tokens))
    log(f"seed {a.seed} | tau_1: {len(tau1.tokens)} steps | cube {task.cube_pos.round(3)} | "
        f"rollout step limit {max_steps}")

    bcfg = BCConfig(n_aug=a.n_aug, lookahead=a.lookahead, epochs=a.epochs,
                    aug_max=a.aug_max / 100.0, seed=a.seed)
    data = build_bc_dataset(task, encoder, tau1, bcfg, log)
    np.savez(os.path.join(a.outdir, "bc_dataset.npz"), **data)
    model = train_bc(data, bcfg, log)
    torch.save(model.state_dict(), os.path.join(a.outdir, "bc_model.pt"))
    policy = BCPolicy(model, fcfg.action_step)
    rng = np.random.default_rng(a.seed + 13_000_003)
    roll_rng = np.random.default_rng(a.seed + 12_000_003)

    starts = [("a: s_0", 0.0, tau1.qs[0], None)]           # (label, start dist cm, q, note)
    dists = [float(x) for x in a.dists.split(",") if x]
    for dcm in dists:                                       # (b) perturbed starts
        got = 0
        for _ in range(200 * a.n_starts):
            if got == a.n_starts:
                break
            i = int(rng.integers(0, max(1, int(0.7 * len(T)))))
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            pl = place(task, tau1, T[i] + v * dcm / 100.0)
            if pl is None:
                continue
            q, pe = pl
            actual = 100 * np.linalg.norm(T - pe, axis=1).min()
            if abs(actual - dcm) > 0.25 * dcm + 0.1:        # nearest tau_1 state is closer
                continue
            starts.append((f"b: {dcm:g} cm off", actual, q, f"near step {nearest_idx(pe, T)}"))
            got += 1
    for f in sorted(glob.glob(os.path.join(a.runs_dir, f"seed{a.seed}_*", "summary.json"))):
        d = json.load(open(f))                              # (c) fallback path ends
        if not np.allclose(np.asarray(d["tau1_eefs"]), T, atol=1e-6):
            log(f"skip {f}: its tau_1 differs from this one")
            continue
        path = fallback_path(d)
        if path is None or len(path) < 2:
            continue
        pl = place(task, tau1, path[-1])
        if pl is None:
            log(f"skip {f}: path end not reachable by IK from a tau_1 configuration")
            continue
        q, pe = pl
        starts.append((f"c: fallback end ({os.path.basename(os.path.dirname(f))})",
                       100 * np.linalg.norm(T - pe, axis=1).min(), q, None))

    results = []
    log(f"\n{'start':<44} {'start d':>7} | {'result':>9} {'steps':>5} {'max dev':>7} "
        f"{'mean dev':>8} {'reached step':>12}")
    for label, d0, q, note in starts:
        r = bc_rollout(policy, task, encoder, q, T, bcfg, max_steps, rng=roll_rng)
        r.update(label=label, start_dist=d0, note=note)
        results.append(r)
        log(f"{label:<44} {d0:6.1f}  | {r['end']:>9} {len(r['actions']):>5} {r['max_dev']:6.1f}  "
            f"{r['mean_dev']:7.1f}  {int(r['progress'][-1]):>6}/{len(T) - 1}")

    log("\nsummary by start type:")
    groups = {}
    for r in results:
        groups.setdefault(r["label"].split(" (")[0] if r["label"].startswith("c") else r["label"], []).append(r)
    for g, rs in groups.items():
        log(f"  {g:<22} success {sum(x['success'] for x in rs)}/{len(rs)} | max deviation "
            f"median {np.median([x['max_dev'] for x in rs]):.1f} cm | mean deviation median "
            f"{np.median([x['mean_dev'] for x in rs]):.1f} cm")

    with open(os.path.join(a.outdir, "bc_eval.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "start_dist_cm", "result", "steps", "max_dev_cm", "mean_dev_cm",
                    "end_dist_to_cube_cm", "end_tau1_step", "note"])
        for r in results:
            w.writerow([r["label"], round(r["start_dist"], 2), r["end"], len(r["actions"]),
                        round(r["max_dev"], 2), round(r["mean_dev"], 2),
                        round(100 * task.dist_to_cube(r["eefs"][-1]), 2),
                        int(r["progress"][-1]), r["note"] or ""])
    with open(os.path.join(a.outdir, "bc_rollouts.json"), "w") as fh:
        json.dump(dict(seed=a.seed, tau1_eefs=T.tolist(), cube=task.cube_pos.tolist(),
                       rollouts=[dict(label=r["label"], start_dist=r["start_dist"], end=r["end"],
                                      eefs=r["eefs"].tolist(), dist=r["dist"].tolist())
                                 for r in results]), fh)
    write_html(a.outdir, a.seed, T, task.cube_pos, results)
    log(f"\noutputs -> {a.outdir}/ (bc.log, bc_eval.csv, bc_rollouts.json, bc_3d.html, "
        f"bc_model.pt, bc_dataset.npz) | {time.time() - t0:.0f}s")
    task.close()
    logf.close()


def write_html(outdir, seed, T, cube, results):
    colors = {"a": "#1a9850", "b": "#2c7bb6", "c": "#d61fd6"}
    traces = [dict(type="scatter3d", mode="lines+markers", name="τ₁", x=T[:, 0].tolist(),
                   y=T[:, 1].tolist(), z=T[:, 2].tolist(), text=[f"τ₁ step {i}" for i in range(len(T))],
                   hovertemplate="%{text}<extra></extra>", line=dict(color="black", width=7),
                   marker=dict(color="black", size=3))]
    for k, r in enumerate(results):
        E = r["eefs"]
        col = colors[r["label"][0]]
        traces.append(dict(
            type="scatter3d", mode="lines+markers",
            name=f"{r['label']} → {r['end']}", legendgroup=r["label"][0],
            x=E[:, 0].tolist(), y=E[:, 1].tolist(), z=E[:, 2].tolist(),
            text=[f"{r['label']}<br>step {i} · {dd:.2f} cm from τ₁" for i, dd in enumerate(r["dist"])],
            hovertemplate="%{text}<extra></extra>",
            line=dict(color=col, width=4, dash="solid" if r["success"] else "dash"),
            marker=dict(color=col, size=2)))
        traces.append(dict(type="scatter3d", mode="markers", showlegend=False, x=[E[0, 0]],
                           y=[E[0, 1]], z=[E[0, 2]], hoverinfo="skip",
                           marker=dict(color=col, size=5, symbol="circle",
                                       line=dict(color="black", width=1))))
    traces.append(dict(type="scatter3d", mode="markers", name="cube", x=[float(cube[0])],
                       y=[float(cube[1])], z=[float(cube[2])], hoverinfo="skip",
                       marker=dict(color="red", size=9, symbol="square")))
    traces.append(_sphere(np.asarray(cube), EPS_DIST))
    n_ok = sum(r["success"] for r in results)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BC policy rollouts</title>
<script src="{PLOTLY_JS}"></script>
<style>
 html, body {{ margin: 0; height: 100%; background: #fafafa; color: #222;
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
 header {{ padding: 10px 16px 4px; }} h1 {{ font-size: 17px; margin: 0 0 2px; }}
 .sub {{ font-size: 13px; color: #555; }}
 #plot {{ height: calc(100% - 60px); min-height: 420px; }}
</style></head><body>
<header><h1>BC policy rollouts along τ₁ (seed {seed})</h1>
<div class="sub">{n_ok}/{len(results)} reached the cube · green = from s₀, blue = perturbed starts,
 magenta = fallback path ends · solid = reached, dashed = failed · hover for distance to τ₁</div></header>
<div id="plot"></div>
<script>
Plotly.newPlot("plot", {json.dumps(traces)}, {{
  margin: {{l: 0, r: 0, t: 0, b: 0}}, paper_bgcolor: "#fafafa",
  legend: {{x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)", font: {{size: 11}}}},
  scene: {{aspectmode: "data", xaxis: {{title: "x (m)"}}, yaxis: {{title: "y (m)"}},
           zaxis: {{title: "z (m)"}}, camera: {{eye: {{x: 1.4, y: 1.4, z: 0.9}}}}}}
}}, {{responsive: true, displaylogo: false}});
</script></body></html>
"""
    with open(os.path.join(outdir, "bc_3d.html"), "w") as fh:
        fh.write(page)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stand-alone BC policy evaluation along tau_1.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--n-aug", type=int, default=20, help="corrective examples per tau_1 state")
    p.add_argument("--aug-max", type=float, default=2.0, help="max corrective offset, cm")
    p.add_argument("--lookahead", type=int, default=2, help="label = direction to the tau_1 "
                   "state this many steps ahead of the nearest one")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--dists", default="0.5,1,2,3", help="perturbed-start distances, cm")
    p.add_argument("--n-starts", type=int, default=3, help="perturbed starts per distance")
    p.add_argument("--max-steps-mult", type=float, default=2.0,
                   help="rollout step limit = this x tau_1's length")
    p.add_argument("--runs-dir", default="dnn_fallback_results_10_ensemble")
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    args.outdir = args.outdir or f"bc_results/seed{args.seed}"
    main(args)
