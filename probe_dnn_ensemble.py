"""
Probe, DNN version: does a small ensemble of plain feed-forward Q-networks
become more uncertain with distance from tau_1?

Same scene, tau_1 and band sampling as probe_td_ensemble.py (same seed and
--demo -> same cube, image calibration and demo), but the models are not
transformers:
  * Model. n_models (default 2) fully connected networks Q(s, a): input = the
    current state's standardized image embedding (all views flattened) plus
    the 3-D action (scaled by 1/action_step, as T_D does), 2 hidden layers,
    scalar Q output. No history, no dropout. Members differ only in their
    random initial weights.
  * Training. tau_1's (state, action) pairs with T_D's labels
    gamma^(T-1-t); squared error, Adam, full batch, --epochs.
  * Uncertainty. U(s, a) = max_i Q^i(s, a) - min_i Q^i(s, a) over the models
    (|Q^1 - Q^2| for two). The task text sums max - min over the 3 action
    components, which presumes models that OUTPUT actions; these models
    output a single Q-value, so the same max - min spread is taken over Q.
  * As in probe_td_ensemble.py: 1 cm distance bands around the tau_1 path,
    one point per equal-length stretch of tau_1 in every band, IK from the
    nearest tau_1 configuration, collision / workspace checks, and --action
    random (exploration-style) or tau1 (nearest tau_1 step's action).

Outputs in --outdir: dnn.csv, dnn.log, dnn_3d.html, dnn.png.

Usage:
    python probe_dnn_ensemble.py --seed 7
    python probe_dnn_ensemble.py --seed 7 --demo varied --action random
    python probe_dnn_ensemble.py --seed 7 --n-models 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from dnn_ensemble import train_dnn_ensemble
from fallback_explore import FallbackConfig, collect_scripted_trajectory, make_varied_demo
from image_encoder import EncoderConfig, ImageEncoder
from plot_fallback_3d import MU_SCALE, PLOTLY_JS
from probe_td_bands import polyline_dist
from probe_td_ensemble import ACTION_TEXT, point_at_arclength
from robosuite_env import WorldConfig
from task import RobosuiteTask, TaskConfig


def build_scene(cfg: FallbackConfig, enc_cfg: EncoderConfig, demo_fn):
    """Scene, calibrated encoder and tau_1, built with the same RNG streams as
    fallback_explore.main (so the same seed gives the same cube, image
    standardization and straight demo; the varied demo has its own stream)."""
    torch.manual_seed(cfg.seed)
    task = RobosuiteTask(TaskConfig(world=WorldConfig(step_size=cfg.action_step)),
                         seed=cfg.seed, enable_render=True)
    encoder = ImageEncoder(enc_cfg)
    task.world.cfg.step_size = WorldConfig().step_size
    try:
        encoder.calibrate(task, np.random.default_rng(cfg.seed + 2_000_003))
    finally:
        task.world.cfg.step_size = cfg.action_step
    tau1 = (demo_fn or collect_scripted_trajectory)(task, encoder, cfg)
    return task, encoder, tau1


def main(a):
    cfg = FallbackConfig(seed=a.seed)
    enc_cfg = EncoderConfig(n_calib=a.n_calib)
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "dnn.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    demo_fn = make_varied_demo(a.demo_noise, cfg.seed, log) if a.demo == "varied" else None
    task, encoder, tau1 = build_scene(cfg, enc_cfg, demo_fn)
    ik, world = task.ik, task.world
    T = np.asarray(tau1.eefs)
    acts = np.asarray([tok[-3:] for tok in tau1.tokens])
    log(f"tau_1 ({a.demo} demo): {len(tau1.tokens)} steps")

    # ---- train the DNN ensemble on tau_1's (state, action) -> Q pairs ----
    S = np.stack([tok[:-3] for tok in tau1.tokens])
    models = train_dnn_ensemble(S, acts, tau1.labels, a.n_models, a.hidden, a.epochs, a.lr,
                                cfg.seed, cfg.action_step, log)

    @torch.no_grad()
    def q_all(emb, act):
        s = torch.tensor(emb.reshape(1, -1), dtype=torch.float32)
        av = torch.tensor(np.asarray(act).reshape(1, 3), dtype=torch.float32)
        return np.array([float(m(s, av)) for m in models])

    fit = np.array([q_all(S[t], acts[t]) for t in range(len(acts))])
    log(f"on tau_1's own (state, action) pairs: U mean {np.ptp(fit, axis=1).mean():.4f} | "
        f"max |Q - label| {np.abs(fit.mean(1) - tau1.labels).max():.4f}")

    rng = np.random.default_rng(cfg.seed + 4_000_003)
    act_rng = np.random.default_rng(cfg.seed + 5_000_003)

    def probe(emb, near):
        if a.action == "tau1":
            v = acts[min(near, len(acts) - 1)]
        else:
            v = act_rng.normal(size=3)
            v *= cfg.action_step * act_rng.random() ** (1 / 3) / np.linalg.norm(v)
        qs = q_all(emb, v)
        return dict(ax=v[0], ay=v[1], az=v[2], q_mean=float(qs.mean()),
                    u=float(qs.max() - qs.min()))

    rows = []
    for i in range(len(tau1.tokens)):                             # reference: tau_1 states
        ik.set_q(tau1.qs[i])
        rows.append(dict(band="tau_1", x=T[i, 0], y=T[i, 1], z=T[i, 2], dist_cm=0.0,
                         tau1_idx=i, **probe(encoder.embed(ik), i)))

    L = float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum())
    n = a.n_per_band
    bands = [(b, b + 1) for b in range(a.bands)]
    for lo, hi in bands:
        for k in range(n):                                        # one point per stretch
            for _ in range(a.max_tries):
                c = point_at_arclength(T, rng.uniform(k * L / n, (k + 1) * L / n))
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                p = c + v * rng.uniform(lo, hi) / 100.0
                d, near = polyline_dist(p[None], T)
                if not (lo / 100 <= d[0] < hi / 100) or not world._in_bounds(p) or task.reached(p):
                    continue
                q_ik, ok = ik.solve_ik(p, q_init=tau1.qs[int(near[0])])
                if not ok:
                    continue
                ik.set_q(q_ik)
                if world._forbidden_contact():
                    continue
                pe = ik.eef_pos()
                d, near = polyline_dist(pe[None], T)
                if not (lo / 100 <= d[0] < hi / 100):
                    continue
                rows.append(dict(band=f"{lo}-{hi} cm", x=pe[0], y=pe[1], z=pe[2],
                                 dist_cm=100 * float(d[0]), tau1_idx=int(near[0]),
                                 **probe(encoder.embed(ik), int(near[0]))))
                break
            else:
                log(f"band {lo}-{hi} cm, stretch {k}: no valid point in {a.max_tries} tries")

    with open(os.path.join(a.outdir, "dnn.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- stats ----
    def rank(x):
        return np.argsort(np.argsort(x))

    def spear(x, y):
        return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])

    names = ["tau_1"] + [f"{lo}-{hi} cm" for lo, hi in bands]
    log(f"\nno history, {ACTION_TEXT[a.action]}; U = max - min of Q over {a.n_models} DNNs")
    log(f"{'band':>8} {'n':>3} {'step idx':>8} | {'U mean':>8} {'median':>8} {'std':>8}")
    for name in names:
        g = [r for r in rows if r["band"] == name]
        if not g:
            continue
        u = np.array([r["u"] for r in g])
        log(f"{name:>8} {len(g):>3} {np.mean([r['tau1_idx'] for r in g]):>8.1f} | "
            f"{u.mean():>8.4f} {np.median(u):>8.4f} {u.std():>8.4f}")
    br = [r for r in rows if r["band"] != "tau_1"]
    rho = spear([r["dist_cm"] for r in br], [r["u"] for r in br])
    log(f"\nSpearman with distance (cm): {rho:+.2f} | with position on tau_1: "
        f"{spear([r['tau1_idx'] for r in br], [r['u'] for r in br]):+.2f}")
    # the DNNs fit tau_1's own (state, action) pairs almost exactly, so U on the
    # tau_1 row can be ~0 and a ratio against it is meaningless; compare the
    # outermost band with the innermost one (both off tau_1) instead
    inner = [r["u"] for r in rows if r["band"] == names[1]]
    outer = [r["u"] for r in rows if r["band"] == names[-1]]
    log(f"median U, outermost band / innermost band ({names[1]}): "
        f"{np.median(outer) / np.median(inner):.2f}x")
    log("expected if U grows with distance from tau_1: clearly positive distance "
        "correlation, ratio well above 1")

    write_html(a, rows, T, task.cube_pos, rho)
    write_png(a, rows, T, task.cube_pos)
    log(f"\noutputs -> {a.outdir}/ (dnn.log, dnn.csv, dnn_3d.html, dnn.png) | "
        f"{time.time() - t0:.0f}s")
    task.close()
    logf.close()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _range(rows):
    return (float(v) for v in np.percentile([r["u"] for r in rows], [2, 98]))


def write_html(a, rows, T, cube, rho):
    names = ["tau_1"] + sorted({r["band"] for r in rows} - {"tau_1"})
    cmin, cmax = _range(rows)
    bar = dict(title=dict(text=f"U = max − min Q<br>({a.n_models} DNNs)<br>blue = certain",
                          side="right"), len=0.6, thickness=14)
    traces = [dict(type="scatter3d", mode="lines", name="τ₁ path", x=T[:, 0].tolist(),
                   y=T[:, 1].tolist(), z=T[:, 2].tolist(), hoverinfo="skip",
                   line=dict(color="black", width=5))]
    for k, name in enumerate(names):
        g = [r for r in rows if r["band"] == name]
        is_t1 = name == "tau_1"
        traces.append(dict(
            type="scatter3d", mode="markers",
            name="τ₁ states (reference)" if is_t1 else f"band {name}",
            x=[r["x"] for r in g], y=[r["y"] for r in g], z=[r["z"] for r in g],
            text=[(f"τ₁ step {r['tau1_idx']}" if is_t1 else
                   f"band {name} · {r['dist_cm']:.2f} cm from τ₁ (near step {r['tau1_idx']})")
                  + f"<br>U {r['u']:.4f} · Q (mean) {r['q_mean']:.3f}"
                    f"<br>action ({100 * r['ax']:.2f}, {100 * r['ay']:.2f}, "
                    f"{100 * r['az']:.2f}) cm" for r in g],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=7 if is_t1 else 5, symbol="square" if is_t1 else "circle",
                        color=[r["u"] for r in g], colorscale=MU_SCALE, cmin=cmin, cmax=cmax,
                        showscale=k == 0, colorbar=bar,
                        line=dict(color="black", width=1 if is_t1 else 0))))
    traces.append(dict(type="scatter3d", mode="markers", name="cube", x=[cube[0]], y=[cube[1]],
                       z=[cube[2]], hoverinfo="skip",
                       marker=dict(color="red", size=9, symbol="square")))
    tbl = "".join(
        f"<tr><td>{nm}</td><td>{len(g)}</td><td>{np.median([r['u'] for r in g]):.4f}</td></tr>"
        for nm in names for g in [[r for r in rows if r["band"] == nm]] if g)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DNN ensemble probe</title>
<script src="{PLOTLY_JS}"></script>
<style>
 html, body {{ margin: 0; height: 100%; background: #fafafa; color: #222;
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
 header {{ padding: 10px 16px 4px; }} h1 {{ font-size: 17px; margin: 0 0 2px; }}
 .sub {{ font-size: 13px; color: #555; }}
 .wrap {{ display: flex; height: calc(100% - 64px); }}
 #plot {{ flex: 1; min-height: 420px; }}
 table {{ border-collapse: collapse; font-size: 12px; margin: 8px 16px; align-self: flex-start; }}
 td, th {{ padding: 3px 8px; border-bottom: 1px solid #ddd; text-align: right; }}
 td:first-child, th:first-child {{ text-align: left; }}
 @media (max-width: 800px) {{ .wrap {{ flex-direction: column; height: auto; }}
   #plot {{ height: 70vh; }} }}
</style></head><body>
<header><h1>DNN ensemble ({a.n_models} Q-networks): uncertainty around τ₁ (seed {a.seed},
 {a.demo} demo, no history)</h1>
<div class="sub">Spearman(distance, U) = {rho:+.2f} · {ACTION_TEXT[a.action]} ·
 click legend entries to show/hide bands · hover for details</div></header>
<div class="wrap"><div id="plot"></div>
<table><tr><th>band</th><th>n</th><th>median U</th></tr>{tbl}</table></div>
<script>
Plotly.newPlot("plot", {json.dumps(traces)}, {{
  margin: {{l: 0, r: 0, t: 0, b: 0}}, paper_bgcolor: "#fafafa",
  legend: {{x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)"}},
  scene: {{aspectmode: "data", xaxis: {{title: "x (m)"}}, yaxis: {{title: "y (m)"}},
           zaxis: {{title: "z (m)"}}, camera: {{eye: {{x: 1.4, y: 1.4, z: 0.9}}}}}}
}}, {{responsive: true, displaylogo: false}});
</script></body></html>
"""
    with open(os.path.join(a.outdir, "dnn_3d.html"), "w") as f:
        f.write(page)


def write_png(a, rows, T, cube):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = LinearSegmentedColormap.from_list("u", [c for _, c in MU_SCALE])
    cmin, cmax = _range(rows)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2], "k-", lw=2, label=r"$\tau_1$")
    t1 = [r for r in rows if r["band"] == "tau_1"]
    bp = [r for r in rows if r["band"] != "tau_1"]
    sc = ax.scatter([r["x"] for r in bp], [r["y"] for r in bp], [r["z"] for r in bp],
                    c=[r["u"] for r in bp], cmap=cmap, vmin=cmin, vmax=cmax, s=18,
                    label="band points")
    ax.scatter([r["x"] for r in t1], [r["y"] for r in t1], [r["z"] for r in t1],
               c=[r["u"] for r in t1], cmap=cmap, vmin=cmin, vmax=cmax, s=45,
               marker="s", edgecolors="k", label=r"$\tau_1$ states")
    ax.scatter(*cube, color="red", s=80, label="cube")
    fig.colorbar(sc, ax=ax, shrink=0.6,
                 label=f"U = max − min Q over {a.n_models} DNNs (blue = certain)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(rf"DNN-ensemble uncertainty around $\tau_1$ ({a.demo} demo)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "dnn.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DNN-ensemble Q uncertainty probe around tau_1.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-models", type=int, default=2, help="DNNs in the ensemble (>= 2)")
    p.add_argument("--hidden", type=int, default=256, help="units per hidden layer")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--demo", choices=["straight", "varied"], default="straight")
    p.add_argument("--demo-noise", type=float, default=1.0)
    p.add_argument("--action", choices=["random", "tau1"], default="random")
    p.add_argument("--bands", type=int, default=5)
    p.add_argument("--n-per-band", type=int, default=12)
    p.add_argument("--max-tries", type=int, default=400)
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    if args.n_models < 2:
        p.error("--n-models must be at least 2")
    tag = "" if args.n_models == 2 else f"_n{args.n_models}"
    tag += "" if args.epochs == 400 else f"_ep{args.epochs}"
    tag += "" if args.action == "random" else "_tau1act"
    if args.demo == "varied":
        tag += "_varied" + ("" if args.demo_noise == 1.0 else f"{args.demo_noise:g}")
    args.outdir = args.outdir or f"dnn_results/seed{args.seed}{tag}"
    main(args)
