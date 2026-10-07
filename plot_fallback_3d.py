"""
Interactive 3D view of a fallback_explore.py run, as a standalone HTML page.

Reads <run_dir>/summary.json and writes <run_dir>/fallback_3d.html: tau_1,
s_0 / s_1, every round's exploration paths, the s* of each round, tau_2 (if
built), the cube with its eps_dist goal sphere, and the table top. Drag to
rotate, scroll to zoom, right-drag (or the toolbar) to pan, click legend
entries to hide/show, hover a point for its round / trajectory / step and
distance to tau_1. Buttons switch between preset camera views.

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


def build_figure(d: dict):
    T = np.asarray(d["tau1_eefs"], float)
    cube = np.asarray(d["cube"], float)
    traces = []

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
                hov.append(f"round {r + 1} · traj {j} · step {t}<br>"
                           f"{100 * di:.1f} cm from τ₁")
            xs.append(None); ys.append(None); zs.append(None); hov.append("")
        col = _viridis(r, len(rounds))
        traces.append(dict(type="scatter3d", mode="lines+markers", name=f"round {r + 1}",
                           x=xs, y=ys, z=zs, text=hov, hovertemplate="%{text}<extra></extra>",
                           legendgroup=f"r{r}", connectgaps=False, opacity=0.75,
                           line=dict(color=col, width=2), marker=dict(color=col, size=2)))

    # tau_1
    dT = [f"τ₁ step {i}" for i in range(len(T))]
    traces.append(_line(T, "τ₁ (demo)", "black", 7, dT, size=4))

    # tau_2
    tau2 = d.get("tau2")
    if tau2:
        P2 = np.asarray(tau2["eefs"], float)
        d2 = _dist_to(P2, T)
        hov = [f"τ₂ step {i}<br>{100 * di:.1f} cm from τ₁" for i, di in enumerate(d2)]
        traces.append(_line(P2, f"τ₂ ({len(P2) - 1} steps)", "crimson", 8, hov, size=4))
        j = tau2.get("join_idx", -1)
        if 0 <= j < len(T):
            traces.append(_line(np.stack([P2[-1], T[j]]), f"τ₂ end → τ₁ state {j}",
                                "crimson", 3, ["τ₂ end", f"τ₁ state {j}"], mode="lines",
                                dash="dash"))

    # s* per round
    for r, s in enumerate(s_stars):
        di = _dist_to(s[None], T)[0]
        tr = _point(s, f"s* round {r + 1}", "orange", "diamond", 6,
                    f"s* round {r + 1}<br>{100 * di:.1f} cm from τ₁")
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
    return traces


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
    traces = build_figure(summary)
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
