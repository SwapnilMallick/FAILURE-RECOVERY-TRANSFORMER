"""
Interactive 3D view of a fallback_explore.py run, as a standalone HTML page.

Reads <run_dir>/summary.json and writes <run_dir>/fallback_3d.html: tau_1,
s_0 / s_1, every round's exploration paths, the s* of each round, tau_2 (if
built), the cube with its eps_dist goal sphere, and the table top. Drag to
rotate, scroll to zoom, right-drag (or the toolbar) to pan, click legend
entries to hide/show, hover a point for its round / trajectory / step and
distance to tau_1, and -- for runs that saved them -- T_D's uncertainty mu_TD
(certainty = -mu_TD), mu_explore and S' membership. Buttons switch between
preset camera views and between colouring by round and colouring every
state by mu_TD.

Plotly.js is loaded from its CDN, so opening the page needs an internet
connection; nothing beyond numpy is needed on the Python side.

fallback_explore.main() calls write_html() at the end of every run; for
older results folders run it directly:
    python plot_fallback_3d.py fallback_results_2/seed7 [more run dirs ...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
EPS_DIST = 0.03          # task.TaskConfig.eps_dist (goal radius around the cube)
TABLE_Z = 0.80           # robosuite Lift table top (table_offset z)


def _dist_to(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(P[:, None, :] - ref[None, :, :], axis=-1).min(axis=1)


def _viridis(i: int, n: int) -> str:
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    t = 0.0 if n <= 1 else i / (n - 1)
    x = t * (len(stops) - 1)
    k = min(int(x), len(stops) - 2)
    f = x - k
    c = [round(a + (b - a) * f) for a, b in zip(stops[k], stops[k + 1])]
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _line(P, name, color, width, hover, mode="lines+markers", size=3, group=None,
          showlegend=True, dash=None):
    P = np.asarray(P, float)
    tr = dict(type="scatter3d", mode=mode, name=name, x=P[:, 0].tolist(),
              y=P[:, 1].tolist(), z=P[:, 2].tolist(), text=hover,
              hovertemplate="%{text}<extra></extra>", showlegend=showlegend,
              line=dict(color=color, width=width), marker=dict(color=color, size=size))
    if dash:
        tr["line"]["dash"] = dash
    if group:
        tr["legendgroup"] = group
    return tr


def _point(p, name, color, symbol, size, hover):
    p = np.asarray(p, float)
    return dict(type="scatter3d", mode="markers", name=name, x=[p[0]], y=[p[1]], z=[p[2]],
                text=[hover], hovertemplate="%{text}<extra></extra>",
                marker=dict(color=color, size=size, symbol=symbol,
                            line=dict(color="black", width=1)))


def _sphere(center, r, n=18):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
    x = center[0] + r * np.cos(u) * np.sin(v)
    y = center[1] + r * np.sin(u) * np.sin(v)
    z = center[2] + r * np.cos(v)
    return dict(type="surface", x=x.tolist(), y=y.tolist(), z=z.tolist(), opacity=0.18,
                showscale=False, colorscale=[[0, "red"], [1, "red"]], name="goal (eps_dist)",
                showlegend=True, hoverinfo="skip")


PATH_COLOR = "#d61fd6"    # fallback path / tau_2: magenta, unlike every other trace


def fallback_path(d: dict):
    """(N, 3) EEF positions of s_1 -> s*_1 -> ... -> last s*. Read from
    summary['chain_eefs'] (newer runs) or tau_2, else rebuilt from the saved
    exploration: each round's s* is a state of one of that round's
    trajectories, and the path follows that trajectory from the round's start
    up to s*."""
    if d.get("chain_eefs"):
        return np.asarray(d["chain_eefs"], float)
    if d.get("tau2"):
        return np.asarray(d["tau2"]["eefs"], float)
    rounds, s_stars = d.get("exploration", []), d.get("s_stars", [])
    if not s_stars:
        return None
    path = [np.asarray(d["s1"], float)]
    for trajs, s in zip(rounds, s_stars):
        s = np.asarray(s, float)
        hit = next(((P, k) for P in (np.asarray(t, float) for t in trajs if len(t) > 1)
                    for k in range(1, len(P)) if np.allclose(P[k], s, atol=1e-9)), None)
        if hit is None:
            return None                       # cannot rebuild reliably
        P, k = hit
        path.extend(P[1:k + 1])
    return np.asarray(path)


MU_SCALE = [[0.0, "#2c7bb6"], [0.5, "#ffffbf"], [1.0, "#d7191c"]]   # low mu (certain) = blue


def _score_text(sc) -> str:
    if sc is None:
        return ""
    return (f"<br>μ_TD {sc['mu_td']:.4f} (certainty {-sc['mu_td']:.4f})"
            f"<br>μ_explore {sc['mu_exp']:.4f} · {'in' if sc['in_S_prime'] else 'not in'} S′")


def build_figure(d: dict):
    """Returns (traces, round_idx, mu_idx): indices of the traces shown in
    'colour by round' mode and in 'colour by mu_TD' mode (the rest are shown
    in both)."""
    T = np.asarray(d["tau1_eefs"], float)
    cube = np.asarray(d["cube"], float)
    traces, round_idx, mu_idx = [], [], []
    # per-round {(traj, step): scores}; step j >= 1 is the j-th new state
    scores = [{(sc["traj"], sc["step"]): sc for sc in rnd} for rnd in d.get("state_scores", [])]
    # tau_1 state j >= 1 is scored at token j - 1 (the step that led into it)
    tau1_mu = d.get("tau1_mu_td")
    mu_tol = d.get("mu_tol")

    # exploration, one legend entry per round (trajectories joined by None gaps)
    rounds = d.get("exploration", [])
    s_stars = [np.asarray(s, float) for s in d.get("s_stars", [])]
    for r, trajs in enumerate(rounds):
        xs, ys, zs, hov = [], [], [], []
        for j, tr in enumerate(trajs):
            P = np.asarray(tr, float)
            if len(P) < 2:
                continue
            dd = _dist_to(P, T)
            for t, (p, di) in enumerate(zip(P, dd)):
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                sc = scores[r].get((j, t)) if r < len(scores) else None
                hov.append(f"round {r + 1} · traj {j} · step {t}<br>"
                           f"{100 * di:.1f} cm from τ₁" + _score_text(sc))
            xs.append(None); ys.append(None); zs.append(None); hov.append("")
        col = _viridis(r, len(rounds))
        round_idx.append(len(traces))
        traces.append(dict(type="scatter3d", mode="lines+markers", name=f"round {r + 1}",
                           x=xs, y=ys, z=zs, text=hov, hovertemplate="%{text}<extra></extra>",
                           legendgroup=f"r{r}", connectgaps=False, opacity=0.75,
                           line=dict(color=col, width=2), marker=dict(color=col, size=2)))

    # tau_1
    dT = [f"τ₁ step {i}" + (f"<br>μ_TD {tau1_mu[i - 1]:.4f} (certainty {-tau1_mu[i - 1]:.4f})"
                             if tau1_mu and i >= 1 else "") for i in range(len(T))]
    traces.append(_line(T, "τ₁ (demo)", "black", 7, dT, size=4))

    # ---- 'colour by mu_TD' mode ----
    pts, mus, syms, hovs = [], [], [], []
    for r, trajs in enumerate(rounds):
        for j, tr in enumerate(trajs):
            P = np.asarray(tr, float)
            for t in range(1, len(P)):
                sc = scores[r].get((j, t)) if r < len(scores) else None
                if sc is None:
                    continue
                pts.append(P[t]); mus.append(sc["mu_td"])
                syms.append("circle" if sc["in_S_prime"] else "circle-open")
                hovs.append(f"round {r + 1} · traj {j} · step {t}<br>"
                            f"{100 * sc['dist_tau1']:.1f} cm from τ₁" + _score_text(sc))
    all_mu = mus + (list(tau1_mu) if tau1_mu else [])
    if all_mu:
        cmin, cmax = (float(v) for v in np.percentile(all_mu, [2, 98]))
        bar = dict(title=dict(text="μ_TD<br>(blue = certain)" +
                              (f"<br>stop &lt; {mu_tol:.4f}" if mu_tol else ""), side="right"),
                   len=0.6, thickness=14, x=1.0)
        if pts:
            # grey paths underneath, so the coloured states keep their context
            xs, ys, zs = [], [], []
            for trajs in rounds:
                for tr in trajs:
                    for p in tr:
                        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                    xs.append(None); ys.append(None); zs.append(None)
            mu_idx.append(len(traces))
            traces.append(dict(type="scatter3d", mode="lines", name="exploration paths",
                               x=xs, y=ys, z=zs, hoverinfo="skip", visible=False,
                               line=dict(color="rgba(120,120,120,0.45)", width=1.5)))
            P = np.asarray(pts)
            mu_idx.append(len(traces))
            traces.append(dict(type="scatter3d", mode="markers",
                               name="explored states: μ_TD (filled = in S′)",
                               x=P[:, 0].tolist(), y=P[:, 1].tolist(), z=P[:, 2].tolist(),
                               text=hovs, hovertemplate="%{text}<extra></extra>", visible=False,
                               marker=dict(size=3.5, color=mus, symbol=syms, colorscale=MU_SCALE,
                                           cmin=cmin, cmax=cmax, colorbar=bar,
                                           showscale=True)))
        if tau1_mu:
            mu_idx.append(len(traces))
            traces.append(dict(type="scatter3d", mode="markers", name="τ₁ states: μ_TD",
                               x=T[1:, 0].tolist(), y=T[1:, 1].tolist(), z=T[1:, 2].tolist(),
                               text=[f"τ₁ step {i}<br>μ_TD {m:.4f} (certainty {-m:.4f})"
                                     for i, m in enumerate(tau1_mu, start=1)],
                               hovertemplate="%{text}<extra></extra>", visible=False,
                               marker=dict(size=6, symbol="square", color=list(tau1_mu),
                                           colorscale=MU_SCALE, cmin=cmin, cmax=cmax,
                                           showscale=not pts, colorbar=bar,
                                           line=dict(color="black", width=1))))

    # fallback path s_1 -> s*_1 -> ... -> last s* (= tau_2 once it is built),
    # drawn whether or not the run finished
    tau2 = d.get("tau2")
    chain = fallback_path(d)
    if chain is not None and len(chain) > 1:
        dc = _dist_to(chain, T)
        name = (f"fallback path = τ₂ ({len(chain) - 1} steps)" if tau2 else
                f"fallback path ({len(chain) - 1} steps, unfinished: no τ₂)")
        hov = [f"fallback path step {i}<br>{100 * di:.1f} cm from τ₁" for i, di in enumerate(dc)]
        traces.append(_line(chain, name, PATH_COLOR, 9, hov, size=4))
        j = tau2.get("join_idx", -1) if tau2 else -1
        if 0 <= j < len(T):
            traces.append(_line(np.stack([chain[-1], T[j]]), f"τ₂ end → τ₁ state {j}",
                                PATH_COLOR, 3, ["τ₂ end", f"τ₁ state {j}"], mode="lines",
                                dash="dash"))

    # s* per round
    for r, s in enumerate(s_stars):
        di = _dist_to(s[None], T)[0]
        sc = None
        if r < len(scores) and r < len(rounds):
            for (j, t), v in scores[r].items():
                if np.allclose(rounds[r][j][t], s, atol=1e-9):
                    sc = v
                    break
        tr = _point(s, f"s* round {r + 1}", "orange", "diamond", 6,
                    f"s* round {r + 1}<br>{100 * di:.1f} cm from τ₁" + _score_text(sc))
        tr["legendgroup"] = "sstar"
        tr["showlegend"] = r == 0
        if r == 0:
            tr["name"] = "s* (per round)"
        traces.append(tr)

    # landmarks
    traces.append(_point(d["s0"], "s₀ (τ₁ start)", "black", "square", 7, "s₀"))
    s1 = np.asarray(d["s1"], float)
    traces.append(_point(s1, "s₁ (exploration start)", "royalblue", "circle", 7,
                         f"s₁<br>{100 * _dist_to(s1[None], T)[0]:.1f} cm from τ₁"))
    traces.append(_point(cube, "cube", "red", "square", 9, "cube centre"))
    traces.append(_sphere(cube, EPS_DIST))

    # table top under everything that was plotted
    allp = [T, cube[None], s1[None]] + [np.asarray(t, float) for tr in rounds for t in tr if t]
    A = np.concatenate(allp)
    pad = 0.05
    x0, x1 = A[:, 0].min() - pad, A[:, 0].max() + pad
    y0, y1 = A[:, 1].min() - pad, A[:, 1].max() + pad
    traces.append(dict(type="surface", x=[[x0, x1], [x0, x1]], y=[[y0, y0], [y1, y1]],
                       z=[[TABLE_Z, TABLE_Z], [TABLE_Z, TABLE_Z]], opacity=0.25,
                       showscale=False, colorscale=[[0, "#8b6b4a"], [1, "#8b6b4a"]],
                       name="table top", showlegend=True, hoverinfo="skip"))
    return traces, round_idx, mu_idx


def _box(P: np.ndarray, pad: float):
    """Axis ranges around P plus a true-proportion aspect ratio for them."""
    lo, hi = P.min(axis=0) - pad, P.max(axis=0) + pad
    ext = hi - lo
    ratio = ext / ext.max()
    return dict(range=[[float(a), float(b)] for a, b in zip(lo, hi)],
                ratio=[float(r) for r in ratio])


def write_html(run_dir: str, summary: dict = None) -> str:
    if summary is None:
        with open(os.path.join(run_dir, "summary.json")) as f:
            summary = json.load(f)
    traces, round_idx, mu_idx = build_figure(summary)
    # close-up box: exploration, s_1, s* and tau_2 (tau_1 / table / cube are
    # still drawn, just clipped to this box)
    pts = [np.asarray(summary["s1"], float)[None]]
    pts += [np.asarray(t, float) for trajs in summary.get("exploration", [])
            for t in trajs if len(t)]
    if summary.get("tau2"):
        pts.append(np.asarray(summary["tau2"]["eefs"], float))
    # ...plus the stretch of tau_1 nearest to them, so the fall-back target stays in view
    E = np.concatenate(pts)
    T = np.asarray(summary["tau1_eefs"], float)
    nearest = np.linalg.norm(E[:, None, :] - T[None, :, :], axis=-1).argmin(axis=1)
    close = _box(np.concatenate([E, T[nearest.min():nearest.max() + 1]]), pad=0.02)
    c = summary.get("config", {})
    stats = (f"seed {c.get('seed', '?')} · k={c.get('k', '?')}, m={c.get('m', '?')} · "
             f"{summary.get('rounds', '?')} round(s) · stop: {summary.get('stop_reason', '?')} · "
             f"τ₂: {'yes' if summary.get('tau2') else 'no'}")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fallback 3D view</title>
<script src="{PLOTLY_JS}"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: #fafafa; color: #222;
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  header {{ padding: 10px 16px 4px; }}
  h1 {{ font-size: 17px; margin: 0 0 2px; }}
  .sub {{ font-size: 13px; color: #555; }}
  .views {{ padding: 6px 16px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
            font-size: 13px; color: #555; }}
  button {{ font: inherit; padding: 4px 10px; border: 1px solid #bbb; border-radius: 5px;
            background: white; cursor: pointer; }}
  button:hover {{ background: #eee; }}
  #plot {{ height: calc(100% - 92px); min-height: 420px; }}
</style>
</head>
<body>
<header>
  <h1>Fallback to τ₁: exploration in 3D (EEF positions)</h1>
  <div class="sub">{stats} · drag = rotate, scroll = zoom, right-drag = pan,
    click legend = show/hide, hover = details</div>
</header>
<div class="views">View:
  <button data-eye="1.4,1.4,0.9">3D</button>
  <button data-eye="0,0,2.4" data-up="1,0,0">Top</button>
  <button data-eye="2.4,0,0.1">Front (+x)</button>
  <button data-eye="0,-2.4,0.1">Side (−y)</button>
  <button id="reset">Reset</button>
  <span style="margin-left:10px">Zoom:</span>
  <button id="close">Exploration close-up</button>
  <button id="full">Full scene</button>
  <span class="colour" style="margin-left:10px">Colour:</span>
  <button class="colour" id="by-round">By round</button>
  <button class="colour" id="by-mu">By T_D uncertainty (μ_TD)</button>
</div>
<div id="plot"></div>
<script>
const traces = {json.dumps(traces)};
const layout = {{
  margin: {{l: 0, r: 0, t: 0, b: 0}},
  paper_bgcolor: "#fafafa",
  legend: {{x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.8)", font: {{size: 12}}}},
  scene: {{
    aspectmode: "data",
    xaxis: {{title: "x (m)"}}, yaxis: {{title: "y (m)"}}, zaxis: {{title: "z (m)"}},
    camera: {{eye: {{x: 1.4, y: 1.4, z: 0.9}}, up: {{x: 0, y: 0, z: 1}}}},
  }},
}};
Plotly.newPlot("plot", traces, layout, {{responsive: true, displaylogo: false}});
const vec = s => {{ const [x, y, z] = s.split(",").map(Number); return {{x, y, z}}; }};
document.querySelectorAll("button[data-eye]").forEach(b => b.onclick = () =>
  Plotly.relayout("plot", {{"scene.camera": {{eye: vec(b.dataset.eye),
    up: vec(b.dataset.up || "0,0,1"), center: {{x: 0, y: 0, z: 0}}}}}}));
document.getElementById("reset").onclick = () =>
  Plotly.relayout("plot", {{"scene.camera": layout.scene.camera}});
const close = {json.dumps(close)};
document.getElementById("close").onclick = () => Plotly.relayout("plot", {{
  "scene.aspectmode": "manual",
  "scene.aspectratio": {{x: close.ratio[0], y: close.ratio[1], z: close.ratio[2]}},
  "scene.xaxis.range": close.range[0], "scene.yaxis.range": close.range[1],
  "scene.zaxis.range": close.range[2]}});
const roundIdx = {json.dumps(round_idx)}, muIdx = {json.dumps(mu_idx)};
if (!muIdx.length) document.querySelectorAll(".colour").forEach(e => e.style.display = "none");
const colourMode = mu => {{
  Plotly.restyle("plot", {{visible: !mu}}, roundIdx);
  Plotly.restyle("plot", {{visible: mu}}, muIdx);
}};
document.getElementById("by-round").onclick = () => colourMode(false);
document.getElementById("by-mu").onclick = () => colourMode(true);
document.getElementById("full").onclick = () => Plotly.relayout("plot", {{
  "scene.aspectmode": "data", "scene.xaxis.autorange": true,
  "scene.yaxis.autorange": true, "scene.zaxis.autorange": true}});
</script>
</body>
</html>
"""
    path = os.path.join(run_dir, "fallback_3d.html")
    with open(path, "w") as f:
        f.write(page)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python plot_fallback_3d.py <run_dir> [<run_dir> ...]")
    for run_dir in sys.argv[1:]:
        print("wrote", write_html(run_dir))
