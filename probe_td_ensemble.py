"""
Probe, ensemble version: does an ENSEMBLE of T_D models become more uncertain
with distance from tau_1, where MC dropout on a single T_D did not?

Same setup as probe_td_bands.py (same seed -> same cube, encoder, tau_1), but:
  * Ensemble. n_members T_D models, all trained on tau_1 exactly as in
    fallback_explore.py (same data, epochs, optimiser). Member 0 IS the
    fallback run's T_D (rebuilt with the same RNG streams); members 1.. differ
    only in their random initial weights (and dropout noise while training).
    mu_ens = std of the members' Q predictions with dropout OFF; Q_ens = mean.
  * Comparison. mu_mc = member 0's MC-dropout std (--mc-samples-probe passes)
    at the same point with the same action -- the uncertainty the fallback
    method uses today.
  * Balanced bands. probe_td_bands.py drew band points uniformly along tau_1,
    so with ~12 points a band could land mostly near the s_0 end (high mu) or
    the cube end (low mu) and its mean said more about WHERE along tau_1 it
    was than how FAR from it. Here tau_1 is cut into n_per_band equal-length
    stretches and every band takes one point per stretch.
  * Action (--action). "random" (default): one random exploration-style
    action per point (uniform direction, length uniform in the action_step
    ball) -- what T_D is asked during exploration. "tau1": the action of the
    nearest tau_1 step -- the only action T_D was trained on -- so the image
    is the only thing that differs between points.
  * Demo (--demo). "straight" (default): fallback_explore.py's scripted demo,
    1 cm straight at the cube every step, so every tau_1 action is the same
    vector. "varied": every step aims at the cube plus Gaussian direction
    noise (--demo-noise, relative to a unit vector) and its length is drawn
    like an exploration action (uniform in the action_step ball), so T_D is
    trained on many different actions; the demo still has to reach the cube
    within script_max_steps (retried with fresh noise otherwise). Bands are
    then measured around this wiggly path. With "varied", member 0 is no
    longer the T_D of a fallback run.
  * As before: no history (single token),
    IK from the nearest tau_1 configuration, collision/workspace checks.

Outputs in --outdir: ensemble.csv, ensemble.log, ensemble_3d.html
(interactive; colour toggle ensemble / MC dropout), ensemble.png.

Usage:
    python probe_td_ensemble.py --seed 7                   # random actions
    python probe_td_ensemble.py --seed 7 --action tau1     # tau_1's action
    python probe_td_ensemble.py --seed 7 --demo varied     # tau_1 with varied actions
    python probe_td_ensemble.py --seed 42 --n-members 10 --td-epochs 400
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

from fallback_explore import FallbackConfig, Traj, predict_last_mc, td_sequences, train_q
from image_encoder import EncoderConfig
from online_cost_transformer import ImageCostTransformer, causal_mask, fuse
from plot_fallback_3d import MU_SCALE, PLOTLY_JS
from probe_td_bands import build_td, polyline_dist


ACTION_TEXT = {"random": "one random action per point",
               "tau1": "τ₁'s own action (nearest step)"}


def make_varied_demo(noise: float, seed: int, log, tries_per_step: int = 30,
                     max_demos: int = 20):
    """Returns demo_fn(task, encoder, cfg) -> Traj: a scripted reach whose
    actions vary step to step (see module docstring)."""
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
                    a_ = dirv * cfg.action_step * rng.random() ** (1 / 3)
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


def point_at_arclength(T: np.ndarray, s: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(T, axis=0), axis=1)
    cum = np.cumsum(seg)
    i = min(int(np.searchsorted(cum, s)), len(seg) - 1)
    f = (s - (cum[i] - seg[i])) / max(seg[i], 1e-12)
    return T[i] + np.clip(f, 0.0, 1.0) * (T[i + 1] - T[i])


@torch.no_grad()
def predict_members(members, tokens, cfg) -> np.ndarray:
    """Deterministic (dropout OFF) last-token Q of every member."""
    x = torch.tensor(np.array(tokens[-cfg.max_len:]), dtype=torch.float32,
                     device=cfg.device).unsqueeze(0)
    out = []
    for mdl in members:
        mdl.eval()
        out.append(float(mdl(x, causal_mask(x.shape[1], cfg.device), None)[0, -1]))
    return np.array(out)


def main(a):
    cfg = FallbackConfig(seed=a.seed, td_epochs=a.td_epochs)
    enc_cfg = EncoderConfig(n_calib=a.n_calib)
    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, "ensemble.log"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    demo_fn = make_varied_demo(a.demo_noise, cfg.seed, log) if a.demo == "varied" else None
    task, encoder, T_D, tau1 = build_td(cfg, enc_cfg, log, demo_fn)   # member 0
    ik, world = task.ik, task.world

    # ---- members 1..n-1: same data / epochs, different initial weights ----
    seqs = td_sequences(tau1, cfg)
    members = [T_D]
    for i in range(1, a.n_members):
        torch.manual_seed(cfg.seed + 1000 * i)
        m = ImageCostTransformer(cfg, encoder.n_views, encoder.emb_dim, action_dim=3,
                                 action_scale=1.0 / task.world.cfg.step_size,
                                 token_fusion=cfg.token_fusion).to(cfg.device)
        loss = train_q(m, seqs, cfg.td_epochs, cfg, np.random.default_rng(cfg.seed + 7_000_003 + i))
        members.append(m)
        log(f"member {i} trained (final loss {loss:.2e})")

    pcfg = copy.copy(cfg)
    pcfg.mc_samples = a.mc_samples_probe
    T = np.asarray(tau1.eefs)
    rng = np.random.default_rng(cfg.seed + 4_000_003)
    act_rng = np.random.default_rng(cfg.seed + 5_000_003)
    acts = np.asarray([tok[-3:] for tok in tau1.tokens])          # tau_1's actions

    # sanity: on tau_1 WITH history (what training saw), members should agree
    agree = [predict_members(members, tau1.tokens[:t + 1], cfg).std(ddof=1)
             for t in range(len(tau1.tokens))]
    log(f"ensemble of {a.n_members}: mu_ens on tau_1 WITH history (training inputs): "
        f"mean {np.mean(agree):.4f} max {np.max(agree):.4f}")

    def probe(emb, near):
        if a.action == "tau1":
            v = acts[min(near, len(acts) - 1)]
        else:
            v = act_rng.normal(size=3)
            v *= cfg.action_step * act_rng.random() ** (1 / 3) / np.linalg.norm(v)
        tok = [fuse(emb.reshape(-1), v)]
        qs = predict_members(members, tok, cfg)
        _, mu_mc = predict_last_mc(T_D, tok, pcfg)
        return dict(ax=v[0], ay=v[1], az=v[2], q_ens=float(qs.mean()),
                    mu_ens=float(qs.std(ddof=1)), mu_mc=mu_mc)

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

    with open(os.path.join(a.outdir, "ensemble.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- stats ----
    def rank(x):
        return np.argsort(np.argsort(x))

    def spear(x, y):
        return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])

    names = ["tau_1"] + [f"{lo}-{hi} cm" for lo, hi in bands]
    log(f"\nhistory-free, {ACTION_TEXT[a.action]}; mu_ens = std of {a.n_members} members' Q "
        f"(dropout off); mu_mc = member 0 MC dropout ({a.mc_samples_probe} passes)")
    log(f"{'band':>8} {'n':>3} {'step idx':>8} | {'mu_ens mean':>11} {'median':>8} | "
        f"{'mu_mc mean':>10} {'median':>8}")
    for name in names:
        g = [r for r in rows if r["band"] == name]
        if not g:
            continue
        col = lambda k: np.array([r[k] for r in g])
        log(f"{name:>8} {len(g):>3} {col('tau1_idx').mean():>8.1f} | {col('mu_ens').mean():>11.4f} "
            f"{np.median(col('mu_ens')):>8.4f} | {col('mu_mc').mean():>10.4f} "
            f"{np.median(col('mu_mc')):>8.4f}")
    br = [r for r in rows if r["band"] != "tau_1"]
    c = lambda k: [r[k] for r in br]
    rho_ens, rho_mc = spear(c("dist_cm"), c("mu_ens")), spear(c("dist_cm"), c("mu_mc"))
    log(f"\nSpearman with distance (cm):     ensemble {rho_ens:+.2f} | MC dropout {rho_mc:+.2f}")
    log(f"Spearman with position on tau_1: ensemble {spear(c('tau1_idx'), c('mu_ens')):+.2f} | "
        f"MC dropout {spear(c('tau1_idx'), c('mu_mc')):+.2f}")
    t1 = [r for r in rows if r["band"] == "tau_1"]
    outer = [r for r in rows if r["band"] == names[-1]]
    for k, lab in [("mu_ens", "ensemble"), ("mu_mc", "MC dropout")]:
        ratio = np.median([r[k] for r in outer]) / np.median([r[k] for r in t1])
        log(f"median {lab} uncertainty, outermost band / tau_1 states: {ratio:.2f}x")
    log("expected if uncertainty tracks closeness to tau_1: clearly positive distance "
        "correlation, ratio well above 1")

    write_html(a.outdir, rows, T, task.cube_pos, cfg.seed, a.n_members, rho_ens, rho_mc,
               a.action)
    write_png(a.outdir, rows, T, task.cube_pos, a.n_members)
    log(f"\noutputs -> {a.outdir}/ (ensemble.log, ensemble.csv, ensemble_3d.html, ensemble.png)"
        f" | {time.time() - t0:.0f}s")
    task.close()
    logf.close()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _range(rows, k):
    return (float(v) for v in np.percentile([r[k] for r in rows], [2, 98]))


def write_html(outdir, rows, T, cube, seed, n_members, rho_ens, rho_mc, action):
    names = ["tau_1"] + sorted({r["band"] for r in rows} - {"tau_1"})
    traces = [dict(type="scatter3d", mode="lines", name="τ₁ path", x=T[:, 0].tolist(),
                   y=T[:, 1].tolist(), z=T[:, 2].tolist(), hoverinfo="skip",
                   line=dict(color="black", width=5))]
    groups = {"mu_ens": [], "mu_mc": []}
    for key, label in [("mu_ens", f"ensemble of {n_members}"), ("mu_mc", "MC dropout")]:
        cmin, cmax = _range(rows, key)
        bar = dict(title=dict(text=f"{'μ_ens' if key == 'mu_ens' else 'μ_mc'}<br>({label})"
                              "<br>blue = certain", side="right"), len=0.6, thickness=14)
        for k, name in enumerate(names):
            g = [r for r in rows if r["band"] == name]
            is_t1 = name == "tau_1"
            groups[key].append(len(traces))
            traces.append(dict(
                type="scatter3d", mode="markers", visible=key == "mu_ens",
                name="τ₁ states (reference)" if is_t1 else f"band {name}",
                x=[r["x"] for r in g], y=[r["y"] for r in g], z=[r["z"] for r in g],
                text=[(f"τ₁ step {r['tau1_idx']}" if is_t1 else
                       f"band {name} · {r['dist_cm']:.2f} cm from τ₁ (near step {r['tau1_idx']})")
                      + f"<br>μ_ens {r['mu_ens']:.4f} · μ_mc {r['mu_mc']:.4f}"
                        f"<br>Q (ensemble mean) {r['q_ens']:.3f}"
                        f"<br>action ({100 * r['ax']:.2f}, "
                        f"{100 * r['ay']:.2f}, {100 * r['az']:.2f}) cm" for r in g],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=7 if is_t1 else 5, symbol="square" if is_t1 else "circle",
                            color=[r[key] for r in g], colorscale=MU_SCALE, cmin=cmin,
                            cmax=cmax, showscale=k == 0, colorbar=bar,
                            line=dict(color="black", width=1 if is_t1 else 0))))
    traces.append(dict(type="scatter3d", mode="markers", name="cube", x=[cube[0]], y=[cube[1]],
                       z=[cube[2]], hoverinfo="skip",
                       marker=dict(color="red", size=9, symbol="square")))

    def med(g, k):
        return np.median([r[k] for r in g])

    tbl = "".join(
        f"<tr><td>{n}</td><td>{len(g)}</td><td>{med(g, 'mu_ens'):.4f}</td>"
        f"<td>{med(g, 'mu_mc'):.4f}</td></tr>"
        for n in names for g in [[r for r in rows if r["band"] == n]] if g)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T_D ensemble probe</title>
<script src="{PLOTLY_JS}"></script>
<style>
 html, body {{ margin: 0; height: 100%; background: #fafafa; color: #222;
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
 header {{ padding: 10px 16px 4px; }} h1 {{ font-size: 17px; margin: 0 0 2px; }}
 .sub {{ font-size: 13px; color: #555; }}
 .bar {{ padding: 4px 16px; font-size: 13px; color: #555; display: flex; gap: 6px;
         align-items: center; flex-wrap: wrap; }}
 button {{ font: inherit; padding: 4px 10px; border: 1px solid #bbb; border-radius: 5px;
           background: white; cursor: pointer; }}
 button:hover {{ background: #eee; }}
 .wrap {{ display: flex; height: calc(100% - 96px); }}
 #plot {{ flex: 1; min-height: 420px; }}
 table {{ border-collapse: collapse; font-size: 12px; margin: 8px 16px; align-self: flex-start; }}
 td, th {{ padding: 3px 8px; border-bottom: 1px solid #ddd; text-align: right; }}
 td:first-child, th:first-child {{ text-align: left; }}
 @media (max-width: 800px) {{ .wrap {{ flex-direction: column; height: auto; }}
   #plot {{ height: 70vh; }} }}
</style></head><body>
<header><h1>Ensemble vs MC dropout: uncertainty around τ₁ (seed {seed}, no history)</h1>
<div class="sub">Spearman(distance, uncertainty): ensemble {rho_ens:+.2f} · MC dropout
 {rho_mc:+.2f} · {ACTION_TEXT[action]} · hover for details</div></header>
<div class="bar">Colour by:
 <button id="ens">Ensemble μ_ens ({n_members} T_D models)</button>
 <button id="mc">MC dropout μ_mc (single T_D)</button></div>
<div class="wrap"><div id="plot"></div>
<table><tr><th>band</th><th>n</th><th>median μ_ens</th><th>median μ_mc</th></tr>
{tbl}</table></div>
<script>
const groups = {json.dumps(groups)};
Plotly.newPlot("plot", {json.dumps(traces)}, {{
  margin: {{l: 0, r: 0, t: 0, b: 0}}, paper_bgcolor: "#fafafa",
  legend: {{x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)"}},
  scene: {{aspectmode: "data", xaxis: {{title: "x (m)"}}, yaxis: {{title: "y (m)"}},
           zaxis: {{title: "z (m)"}}, camera: {{eye: {{x: 1.4, y: 1.4, z: 0.9}}}}}}
}}, {{responsive: true, displaylogo: false}});
const show = k => {{
  Plotly.restyle("plot", {{visible: true}}, groups[k]);
  Plotly.restyle("plot", {{visible: false}}, groups[k === "mu_ens" ? "mu_mc" : "mu_ens"]);
}};
document.getElementById("ens").onclick = () => show("mu_ens");
document.getElementById("mc").onclick = () => show("mu_mc");
</script></body></html>
"""
    with open(os.path.join(outdir, "ensemble_3d.html"), "w") as f:
        f.write(page)


def write_png(outdir, rows, T, cube, n_members):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = LinearSegmentedColormap.from_list("mu", [c for _, c in MU_SCALE])
    cmin, cmax = _range(rows, "mu_ens")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2], "k-", lw=2, label=r"$\tau_1$")
    t1 = [r for r in rows if r["band"] == "tau_1"]
    bp = [r for r in rows if r["band"] != "tau_1"]
    sc = ax.scatter([r["x"] for r in bp], [r["y"] for r in bp], [r["z"] for r in bp],
                    c=[r["mu_ens"] for r in bp], cmap=cmap, vmin=cmin, vmax=cmax, s=18,
                    label="band points")
    ax.scatter([r["x"] for r in t1], [r["y"] for r in t1], [r["z"] for r in t1],
               c=[r["mu_ens"] for r in t1], cmap=cmap, vmin=cmin, vmax=cmax, s=45,
               marker="s", edgecolors="k", label=r"$\tau_1$ states")
    ax.scatter(*cube, color="red", s=80, label="cube")
    fig.colorbar(sc, ax=ax, shrink=0.6,
                 label=rf"$\mu_{{ens}}$ (std of {n_members} T_D models; blue = certain)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(r"Ensemble uncertainty around $\tau_1$ (0–5 cm bands)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ensemble.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    d = FallbackConfig()
    p = argparse.ArgumentParser(description="Ensemble-of-T_D uncertainty probe around tau_1.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-members", type=int, default=5, help="T_D models in the ensemble (>= 2)")
    p.add_argument("--demo", choices=["straight", "varied"], default="straight",
                   help="tau_1: straight scripted reach (all actions equal) or varied actions")
    p.add_argument("--demo-noise", type=float, default=1.0,
                   help="--demo varied: std of the direction noise added to the unit "
                        "cube direction (larger = more varied, longer demo)")
    p.add_argument("--action", choices=["random", "tau1"], default="random",
                   help="random: exploration-style action per point; tau1: nearest tau_1 step's action")
    p.add_argument("--td-epochs", type=int, default=d.td_epochs)
    p.add_argument("--bands", type=int, default=5, help="number of 1 cm bands (0-1, 1-2, ...)")
    p.add_argument("--n-per-band", type=int, default=12,
                   help="points per band = equal-length stretches of tau_1")
    p.add_argument("--mc-samples-probe", type=int, default=200)
    p.add_argument("--max-tries", type=int, default=400, help="sampling attempts per point")
    p.add_argument("--n-calib", type=int, default=200)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    if args.n_members < 2:
        p.error("--n-members must be at least 2")
    tag = "" if args.td_epochs == d.td_epochs else f"_ep{args.td_epochs}"
    tag += "" if args.n_members == 5 else f"_n{args.n_members}"
    tag += "" if args.action == "random" else "_tau1act"
    if args.demo == "varied":
        tag += "_varied" + ("" if args.demo_noise == 1.0 else f"{args.demo_noise:g}")
    args.outdir = args.outdir or f"ensemble_results/seed{args.seed}{tag}"
    main(args)
