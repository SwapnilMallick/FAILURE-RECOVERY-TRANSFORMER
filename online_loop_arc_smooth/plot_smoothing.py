"""
Executed vs smoothed trajectory of one successful iteration (arc_smooth).

Reads an iteration's summary.json: chain_eefs (executed fallback path),
bc_rollout.eefs (executed BC rollout) and tau2.eefs (the smoothed trajectory
that joined D); tau1_eefs for reference. Three orthographic views (top x-y,
front x-z, side y-z), positions in cm relative to the cube.

    python online_loop_arc_smooth/plot_smoothing.py pick_results/online/arc_smooth_local/seed3
    (an iteration folder, or a run folder with a single iteration)
"""
import glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FALLBACK, BC, SMOOTH = "#eb6834", "#2a78d6", "#1baf7a"            # categorical slots 2, 1, 3
INK, INK2, MUTED, GRID = "#1f1f1e", "#55544f", "#8a8983", "#e4e3df"


def main(path):
    f = os.path.join(path, "summary.json")
    if not os.path.exists(f):
        cands = sorted(glob.glob(os.path.join(path, "iter*", "summary.json")))
        f = next((c for c in cands if json.load(open(c)).get("tau2")), cands[0] if cands else None)
    j = json.load(open(f))
    if not j.get("tau2"):
        sys.exit(f"{f}: no trajectory was added in this iteration (BC did not lift) -- nothing to compare")
    c = np.array(j["cube"])
    rel = lambda a: 100 * (np.asarray(a, float) - c)
    fb, bc, sm, t1 = rel(j["chain_eefs"]), rel(j["bc_rollout"]["eefs"]), rel(j["tau2"]["eefs"]), rel(j["tau1_eefs"])
    pathlen = lambda a: float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
    views = [(0, 1, "Top view (x-y)", "x (cm)", "y (cm)"), (0, 2, "Front view (x-z)", "x (cm)", "z above cube (cm)"),
             (1, 2, "Side view (y-z)", "y (cm)", "z above cube (cm)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (a, b, title, xl, yl) in zip(axes, views):
        ax.plot(t1[:, a], t1[:, b], color=MUTED, linewidth=1.2, linestyle="--", label=f"tau_1 ({len(t1) - 1} actions)")
        ax.plot(fb[:, a], fb[:, b], color=FALLBACK, linewidth=1.3, alpha=0.9, label=f"executed fallback ({len(fb) - 1} actions, {pathlen(fb):.0f} cm)")
        ax.plot(bc[:, a], bc[:, b], color=BC, linewidth=1.3, alpha=0.9, label=f"executed BC ({len(bc) - 1} actions, {pathlen(bc):.0f} cm)")
        ax.plot(sm[:, a], sm[:, b], color=SMOOTH, linewidth=2.4, marker="o", markersize=2.8,
                label=f"smoothed, joins D ({len(sm) - 1} actions, {pathlen(sm):.0f} cm)")
        ax.scatter([fb[0, a]], [fb[0, b]], s=70, color=INK, marker="^", zorder=5, label="s_1")
        ax.scatter([bc[0, a]], [bc[0, b]], s=60, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=5, label="hand-off to BC")
        ax.scatter([0], [0], s=140, color=INK, marker="s", zorder=5, label="cube")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel(xl, fontsize=9, color=INK2); ax.set_ylabel(yl, fontsize=9, color=INK2)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    it = j.get("iteration")
    fig.suptitle(f"Seed {j['config']['seed']}, iteration {it}: executed vs smoothed trajectory "
                 f"(positions relative to the cube)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    out = os.path.join(os.path.dirname(f), "smoothing.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
