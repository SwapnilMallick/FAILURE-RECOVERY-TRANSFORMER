"""
Probe: does T_D's uncertainty grow with distance from tau_1?

Rebuilds exactly what fallback_explore.py does before exploring (same seed ->
same cube, encoder calibration, tau_1 and T_D weights), then samples arm
states in 1 cm distance bands around tau_1 and asks T_D for its MC-dropout
uncertainty mu_TD at each one, with NO history (a single-token sequence).

  * Distance to tau_1 = distance from the EEF to the tau_1 polyline (its
    segments, not only its states), so bands are tubes around the path.
  * A band point is drawn by picking a uniform point along tau_1, a uniform
    random direction and a radius in the band, then keeping it only if its
    true polyline distance lands in the band, it is inside the workspace, IK
    reaches it (started from the nearest tau_1 configuration, so the arm
    posture stays tau_1-like) and the arm is collision-free there.
  * T_D scores (state, action) tokens, so a history-free query still needs an
    action. Each point gets ONE random action drawn like exploration's
    (uniform direction, length uniform in the action_step ball), i.e. the
    kind of query T_D answers during a fallback run. The action is saved in
    probe.csv (ax, ay, az).
  * Each estimate uses --mc-samples-probe dropout passes (default 200, vs 20
    in the fallback run) so MC noise does not hide a trend.
  * Reference: tau_1's own states probed the same way (history-free).

Outputs in --outdir: probe.csv (one row per point), probe.log (per-band
stats + Spearman correlation), probe_3d.html (interactive, coloured by mu_TD)
and probe.png (static 3D view).

Usage:
    python probe_td_bands.py --seed 7
    python probe_td_bands.py --seed 42 --n-per-band 15 --bands 5
    python probe_td_bands.py --seed 7 --td-epochs 2000   # -> probe_results/seed7_ep2000
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch

from fallback_explore import (
    FallbackConfig, collect_scripted_trajectory, predict_last_mc, td_sequences, train_q,
)
from image_encoder import EncoderConfig, ImageEncoder
from online_cost_transformer import ImageCostTransformer, fuse
from plot_fallback_3d import MU_SCALE, PLOTLY_JS
from robosuite_env import WorldConfig
from task import RobosuiteTask, TaskConfig


def polyline_dist(P: np.ndarray, T: np.ndarray):
    """Distance from each point in P (N,3) to the polyline T (M,3), and the
    index of the nearest tau_1 STATE (for the IK seed / action)."""
    A, B = T[:-1], T[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=-1).min(1)
    near = np.linalg.norm(P[:, None, :] - T[None], axis=-1).argmin(1)
    return d, near


def sample_point_on_path(T: np.ndarray, rng) -> np.ndarray:
    seg = np.linalg.norm(np.diff(T, axis=0), axis=1)
    s = rng.uniform(0, seg.sum())
    i = min(int(np.searchsorted(np.cumsum(seg), s)), len(seg) - 1)
    f = (s - (np.cumsum(seg)[i] - seg[i])) / max(seg[i], 1e-12)
    return T[i] + f * (T[i + 1] - T[i])


def build_td(cfg: FallbackConfig, enc_cfg: EncoderConfig, log, demo_fn=None):
    """Same construction order and RNG streams as fallback_explore.main, so
    T_D here is identical to the T_D of a fallback run with the same seed.
    demo_fn(task, encoder, cfg) -> Traj replaces the straight scripted demo
    (then T_D is trained on that demo instead and no longer matches a run's)."""
    torch.manual_seed(cfg.seed)
    calib_rng = np.random.default_rng(cfg.seed + 2_000_003)
    train_rng = np.random.default_rng(cfg.seed + 3_000_003)
    task = RobosuiteTask(TaskConfig(world=WorldConfig(step_size=cfg.action_step)),
                         seed=cfg.seed, enable_render=True)
    encoder = ImageEncoder(enc_cfg)
    task.world.cfg.step_size = WorldConfig().step_size
    try:
        encoder.calibrate(task, calib_rng)
    finally:
        task.world.cfg.step_size = cfg.action_step

    def new_model():
        return ImageCostTransformer(
            cfg, encoder.n_views, encoder.emb_dim, action_dim=3,
            action_scale=1.0 / task.world.cfg.step_size,
            token_fusion=cfg.token_fusion).to(cfg.device)

    torch.manual_seed(cfg.seed)
    T_D = new_model()
    torch.manual_seed(cfg.seed + 1)
    new_model()                       # T_explore in the real run; built only to keep RNG in step
    tau1 = (demo_fn or collect_scripted_trajectory)(task, encoder, cfg)
    loss = train_q(T_D, td_sequences(tau1, cfg), cfg.td_epochs, cfg, train_rng)
    mu_hist = [predict_last_mc(T_D, tau1.tokens[:t + 1], cfg)[1] for t in range(len(tau1.tokens))]
    log(f"T_D trained on tau_1 ({len(tau1.tokens)} steps, final loss {loss:.2e}); "
        f"mu_TD on tau_1 WITH history (as in the fallback run, {cfg.mc_samples} passes): "
        f"min {min(mu_hist):.4f} mean {np.mean(mu_hist):.4f} max {max(mu_hist):.4f}")
    return task, encoder, T_D, tau1


def main(a):
    cfg = FallbackConfig(seed=a.seed, td_epochs=a.td_epochs)
    enc_cfg = EncoderConfig(n_calib=a.n_calib)
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "probe.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    task, encoder, T_D, tau1 = build_td(cfg, enc_cfg, log)
    ik, world = task.ik, task.world
    pcfg = copy.copy(cfg)
    pcfg.mc_samples = a.mc_samples_probe
    T = np.asarray(tau1.eefs)
    act_rng = np.random.default_rng(cfg.seed + 5_000_003)
    rng = np.random.default_rng(cfg.seed + 4_000_003)

    def probe(emb):
        """History-free (Q_TD, mu_TD, action) for one random exploration-style
        action (same distribution as world.sample_free's actions)."""
        v = act_rng.normal(size=3)
        v *= cfg.action_step * act_rng.random() ** (1 / 3) / np.linalg.norm(v)
        q, mu = predict_last_mc(T_D, [fuse(emb.reshape(-1), v)], pcfg)
        return q, mu, v

    rows = []
    # reference: tau_1's own states, history-free, same probe
    for i in range(len(tau1.tokens)):
        ik.set_q(tau1.qs[i])
        q, mu, v = probe(encoder.embed(ik))
        rows.append(dict(band="tau_1", x=T[i, 0], y=T[i, 1], z=T[i, 2], dist_cm=0.0,
                         tau1_idx=i, ax=v[0], ay=v[1], az=v[2], q_td=q, mu_td=mu))

    bands = [(b, b + 1) for b in range(a.bands)]
    for lo, hi in bands:
        got, tries = 0, 0
        while got < a.n_per_band:
            tries += 1
            if tries > 400 * a.n_per_band:
                log(f"band {lo}-{hi} cm: only {got} valid points after {tries} tries")
                break
            c = sample_point_on_path(T, rng)
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = c + v * rng.uniform(lo, hi) / 100.0
            d, near = polyline_dist(p[None], T)
            d, near = float(d[0]), int(near[0])
            if not (lo / 100 <= d < hi / 100) or not world._in_bounds(p) or task.reached(p):
                continue
            q_ik, ok = ik.solve_ik(p, q_init=tau1.qs[near])
            if not ok:
                continue
            ik.set_q(q_ik)
            if world._forbidden_contact():
                continue
            pe = ik.eef_pos()
            d, near = polyline_dist(pe[None], T)
            d, near = float(d[0]), int(near[0])
            if not (lo / 100 <= d < hi / 100):
                continue
            q, mu, v = probe(encoder.embed(ik))
            rows.append(dict(band=f"{lo}-{hi} cm", x=pe[0], y=pe[1], z=pe[2],
                             dist_cm=100 * d, tau1_idx=near, ax=v[0], ay=v[1], az=v[2],
                             q_td=q, mu_td=mu))
            got += 1

    with open(os.path.join(a.outdir, "probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- stats ----
    def rank(x):
        return np.argsort(np.argsort(x))

    def spear(a_, b_):
        return float(np.corrcoef(rank(np.asarray(a_)), rank(np.asarray(b_)))[0, 1])

    log(f"\nhistory-free mu_TD, one random action per point (<= {cfg.action_step} m), "
        f"{a.mc_samples_probe} MC passes per estimate")
    log(f"{'band':>8} {'n':>3} {'dist cm':>9} | {'mu mean':>9} {'median':>9} {'std':>9} | "
        f"{'step idx':>8}")
    for name in ["tau_1"] + [f"{lo}-{hi} cm" for lo, hi in bands]:
        g = [r for r in rows if r["band"] == name]
        if not g:
            continue
        m = np.array([r["mu_td"] for r in g])
        dd = np.array([r["dist_cm"] for r in g])
        ix = np.array([r["tau1_idx"] for r in g])
        log(f"{name:>8} {len(g):>3} {dd.min():>4.1f}-{dd.max():<4.1f} | {m.mean():>9.4f} "
            f"{np.median(m):>9.4f} {m.std():>9.4f} | {ix.mean():>8.1f}")
    band_rows = [r for r in rows if r["band"] != "tau_1"]
    rho = spear([r["dist_cm"] for r in band_rows], [r["mu_td"] for r in band_rows])
    log(f"\nSpearman(distance, mu_TD) over the {len(band_rows)} band points: {rho:+.2f}")
    col = lambda k: [r[k] for r in band_rows]
    log(f"Spearman(step idx, mu_TD) {spear(col('tau1_idx'), col('mu_td')):+.2f} "
        f"(step idx = nearest tau_1 step, 0 = s_0 end; column = band mean)")
    log("expected if mu_TD tracks closeness to tau_1: clearly positive, band means rising")

    write_html(a.outdir, rows, T, task.cube_pos, cfg.seed, rho)
    write_png(a.outdir, rows, T, task.cube_pos)
    log(f"\noutputs -> {a.outdir}/ (probe.log, probe.csv, probe_3d.html, probe.png) | "
        f"{time.time() - t0:.0f}s")
    task.close()
    logf.close()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _cscale(rows):
    m = [r["mu_td"] for r in rows]
    return (float(v) for v in np.percentile(m, [2, 98]))


def write_html(outdir, rows, T, cube, seed, rho):
    cmin, cmax = _cscale(rows)
    bar = dict(title=dict(text="μ_TD<br>(blue = certain)", side="right"),
               len=0.6, thickness=14)
    traces = [dict(type="scatter3d", mode="lines", name="τ₁ path", x=T[:, 0].tolist(),
                   y=T[:, 1].tolist(), z=T[:, 2].tolist(), hoverinfo="skip",
                   line=dict(color="black", width=5))]
    names = ["tau_1"] + sorted({r["band"] for r in rows} - {"tau_1"})
    for k, name in enumerate(names):
        g = [r for r in rows if r["band"] == name]
        is_t1 = name == "tau_1"
        traces.append(dict(
            type="scatter3d", mode="markers",
            name="τ₁ states (reference)" if is_t1 else f"band {name}",
            x=[r["x"] for r in g], y=[r["y"] for r in g], z=[r["z"] for r in g],
            text=[(f"τ₁ step {r['tau1_idx']}" if is_t1 else
                   f"band {r['band']} · {r['dist_cm']:.2f} cm from τ₁ (near step "
                   f"{r['tau1_idx']})") +
                  f"<br>μ_TD {r['mu_td']:.4f} (certainty {-r['mu_td']:.4f})"
                  f"<br>Q_TD {r['q_td']:.3f} · action ({100 * r['ax']:.2f}, "
                  f"{100 * r['ay']:.2f}, {100 * r['az']:.2f}) cm"
                  for r in g],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=7 if is_t1 else 5, symbol="square" if is_t1 else "circle",
                        color=[r["mu_td"] for r in g], colorscale=MU_SCALE, cmin=cmin,
                        cmax=cmax, showscale=k == 0, colorbar=bar,
                        line=dict(color="black", width=1 if is_t1 else 0))))
    traces.append(dict(type="scatter3d", mode="markers", name="cube", x=[cube[0]], y=[cube[1]],
                       z=[cube[2]], hoverinfo="skip",
                       marker=dict(color="red", size=9, symbol="square")))
    # band summary table
    tbl = "".join(
        f"<tr><td>{n}</td><td>{len(g)}</td><td>{np.mean([r['mu_td'] for r in g]):.4f}</td>"
        f"<td>{np.median([r['mu_td'] for r in g]):.4f}</td></tr>"
        for n in names for g in [[r for r in rows if r["band"] == n]])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T_D distance probe</title>
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
<header><h1>Does T_D's uncertainty grow with distance from τ₁? (seed {seed}, no history)</h1>
<div class="sub">Spearman(distance, μ_TD) = {rho:+.2f} · colour = μ_TD (one random action per point) ·
 click legend entries to show/hide bands · hover for details</div></header>
<div class="wrap"><div id="plot"></div>
<table><tr><th>band</th><th>n</th><th>mean μ_TD</th><th>median μ_TD</th></tr>
{tbl}</table></div>
<script>
Plotly.newPlot("plot", {json.dumps(traces)}, {{
  margin: {{l: 0, r: 0, t: 0, b: 0}}, paper_bgcolor: "#fafafa",
  legend: {{x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)"}},
  scene: {{aspectmode: "data", xaxis: {{title: "x (m)"}}, yaxis: {{title: "y (m)"}},
           zaxis: {{title: "z (m)"}}, camera: {{eye: {{x: 1.4, y: 1.4, z: 0.9}}}}}}
}}, {{responsive: true, displaylogo: false}});
</script></body></html>
"""
    with open(os.path.join(outdir, "probe_3d.html"), "w") as f:
        f.write(page)


def write_png(outdir, rows, T, cube):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = LinearSegmentedColormap.from_list("mu", [c for _, c in MU_SCALE])
    cmin, cmax = _cscale(rows)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2], "k-", lw=2, label=r"$\tau_1$")
    t1 = [r for r in rows if r["band"] == "tau_1"]
    bp = [r for r in rows if r["band"] != "tau_1"]
    sc = ax.scatter([r["x"] for r in bp], [r["y"] for r in bp], [r["z"] for r in bp],
                    c=[r["mu_td"] for r in bp], cmap=cmap, vmin=cmin, vmax=cmax, s=18,
                    label="band points")
    ax.scatter([r["x"] for r in t1], [r["y"] for r in t1], [r["z"] for r in t1],
               c=[r["mu_td"] for r in t1], cmap=cmap, vmin=cmin, vmax=cmax, s=45,
               marker="s", edgecolors="k", label=r"$\tau_1$ states")
    ax.scatter(*cube, color="red", s=80, label="cube")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=r"$\mu_{T_D}$ (no history; blue = certain)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(r"T_D uncertainty around $\tau_1$ (0–5 cm bands)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "probe.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Probe T_D's uncertainty in distance bands around tau_1.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bands", type=int, default=5, help="number of 1 cm bands (0-1, 1-2, ...)")
    p.add_argument("--n-per-band", type=int, default=12)
    p.add_argument("--td-epochs", type=int, default=FallbackConfig().td_epochs,
                   help="T_D training epochs (default matches fallback_explore.py; "
                        "change it and T_D no longer matches a default fallback run)")
    p.add_argument("--mc-samples-probe", type=int, default=200)
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    suffix = "" if args.td_epochs == FallbackConfig().td_epochs else f"_ep{args.td_epochs}"
    args.outdir = args.outdir or f"probe_results/seed{args.seed}{suffix}"
    main(args)
