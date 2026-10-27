"""
Learning curves of multi-iteration pick runs (arc_shortcutting), from the
per-iteration summary.json files -- works for any run of arc / arc_shortcutting
(and older ones), no extra logging needed. Metrics: learning_metrics.py.

Outputs (in --out):
  per_iteration.csv            one row per (run, seed, iteration)
  per_seed.csv                 one row per (run, seed): task success rate (lifts /
                               iterations), BC success rate (lifts / BC runs),
                               fallback-closer fraction, final |D|, mean returns
  <run>_returns_per_seed.png   one panel per seed: BC and episode return per iteration
  overview.png                 all runs overlaid: task success rate per iteration,
                               mean BC return per iteration, BC success by |D|,
                               fallback-closer fraction per seed

    python online_loop_arc_shortcutting/learning_curves.py \\
        --run no_shortcut=pick_results/online/arc_band4_6_20iters \\
        --run shortcut=pick_results/online/arc_shortcut_band4_6_20iters \\
        --out pick_results/online/learning_curves
"""
import argparse, csv, glob, json, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learning_metrics import CLOSER_CM, iteration_metrics            # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import matplotlib.ticker                                             # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                 # categorical slots 1-4, fixed order
INK, INK2, MUTED, GRID = "#1f1f1e", "#55544f", "#8a8983", "#e4e3df"
FLOOR = 1e-5                                                         # log-axis floor where "no lift" is drawn
D_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 9, "6-9"), (10, 10 ** 6, "10+")]


def load_run(path):
    rows = []
    for f in glob.glob(os.path.join(path, "seed*", "iter*", "summary.json")):
        s = int(re.search(r"seed(\d+)", f).group(1))
        m = iteration_metrics(json.load(open(f)))
        m["seed"] = s
        rows.append(m)
    return sorted(rows, key=lambda r: (r["seed"], r["iteration"]))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2); ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def per_seed_table(name, rows):
    out = []
    for s in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["seed"] == s]
        bc = [r for r in rr if r["bc_ran"]]
        moved = [r for r in rr if r["rounds"] > 0]
        out.append(dict(run=name, seed=s, iterations=len(rr), lifts=sum(r["lifted"] for r in rr),
                        task_success_rate=round(sum(r["lifted"] for r in rr) / len(rr), 3),
                        bc_runs=len(bc), bc_success_rate=round(sum(r["lifted"] for r in bc) / len(bc), 3) if bc else None,
                        fallback_closer_frac=round(sum(r["fallback_closer"] for r in rr) / len(rr), 3),
                        fallback_closer_frac_when_moved=round(sum(r["fallback_closer"] for r in moved) / len(moved), 3) if moved else None,
                        stuck_round1=sum(r["rounds"] == 0 for r in rr),
                        final_D=max(r["D_before"] for r in rr) + int(rr[-1]["lifted"]),
                        mean_return_bc=round(float(np.mean([r["return_bc"] for r in bc])), 4) if bc else None,
                        mean_return_episode=round(float(np.mean([r["return_episode"] for r in rr])), 5)))
    return out


def plot_returns_per_seed(name, rows, path):
    seeds = sorted({r["seed"] for r in rows})
    nc = 5; nr = int(np.ceil(len(seeds) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.3 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, seeds):
        rr = [r for r in rows if r["seed"] == s]
        it = np.array([r["iteration"] for r in rr])
        for key, col, lab in (("return_bc", SERIES[0], "BC return"), ("return_episode", SERIES[1], "episode return")):
            y = np.array([np.nan if r[key] is None else max(r[key], FLOOR) for r in rr], float)
            ax.plot(it, y, color=col, linewidth=1.6, marker="o", markersize=4, label=lab)
        for r in rr:
            if r["rounds"] == 0:
                ax.axvspan(r["iteration"] - 0.5, r["iteration"] + 0.5, color=GRID, alpha=0.7, linewidth=0)
        style(ax, f"seed {s}: {sum(r['lifted'] for r in rr)}/{len(rr)} lifted", "", "")
        ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.5, 1.5)
        ax.set_yticks([FLOOR, 1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["no lift", "1e-4", "1e-3", "0.01", "0.1", "1"])
        ax.minorticks_off(); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in axes.flat[len(seeds):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("return (log)", fontsize=9, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel("iteration", fontsize=9, color=INK2)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=9, frameon=False, ncol=2)
    fig.suptitle(f"{name}: sparse return per iteration (reward 1 on the lifting action, gamma^(L-1)); "
                 f"'no lift' row = BC ran but did not lift; grey band = fallback stuck in round 1 (BC never ran)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_overview(runs, seed_tables, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (a1, a2), (a3, a4) = axes
    names = list(runs)
    for k, name in enumerate(names):
        rows, col = runs[name], SERIES[k % len(SERIES)]
        its = sorted({r["iteration"] for r in rows})
        sr = [np.mean([r["lifted"] for r in rows if r["iteration"] == i]) for i in its]
        a1.plot(its, sr, color=col, linewidth=2, marker="o", markersize=4, label=name)
        mb = [np.mean([r["return_bc"] for r in rows if r["iteration"] == i and r["bc_ran"]] or [np.nan]) for i in its]
        a2.plot(its, mb, color=col, linewidth=2, marker="o", markersize=4, label=name)
        w = 0.8 / len(names)
        rates, ns = [], []
        for lo, hi, _ in D_BINS:
            rr = [r for r in rows if r["bc_ran"] and lo <= r["D_before"] <= hi]
            rates.append(np.mean([r["lifted"] for r in rr]) if rr else np.nan); ns.append(len(rr))
        x = np.arange(len(D_BINS)) + (k - (len(names) - 1) / 2) * w
        a3.bar(x, rates, width=w * 0.92, color=col, label=name)
        for xi, rt, n in zip(x, rates, ns):
            if n:
                a3.annotate(f"{n}", (xi, rt), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        st = seed_tables[name]
        xs = np.array([t["seed"] for t in st]) + (k - (len(names) - 1) / 2) * w
        a4.bar(xs, [t["fallback_closer_frac"] for t in st], width=w * 0.92, color=col, label=name)
    style(a1, "Task success rate per iteration (fraction of seeds that lifted)", "iteration", "success rate")
    a1.set_ylim(0, 1.05)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style(a2, "Mean BC return per iteration (iterations where BC ran)", "iteration", "BC return  gamma^(L_BC-1)")
    a2.set_ylim(bottom=0)
    style(a3, "BC success by |D| at the iteration (numbers = BC runs)", "|D| (trajectories in D before the iteration)", "BC success rate")
    a3.set_xticks(range(len(D_BINS))); a3.set_xticklabels([b[2] for b in D_BINS]); a3.set_ylim(0, 1.1)
    style(a4, f"Fallback ended >= {CLOSER_CM:g} cm closer to D (fraction of a seed's iterations)", "seed", "fraction")
    a4.set_xticks(range(1, 21)); a4.set_ylim(0, 1.05)
    for ax in axes.flat:
        if len(names) > 1:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        runs[name] = load_run(path)
        if not runs[name]:
            sys.exit(f"no iter*/summary.json under {path}")
    with open(os.path.join(a.out, "per_iteration.csv"), "w", newline="") as f:
        cols = ["run", "seed"] + list(next(iter(runs.values()))[0].keys() - {"seed"})
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for name, rows in runs.items():
            for r in rows:
                w.writerow(dict(run=name, **r))
    seed_tables = {name: per_seed_table(name, rows) for name, rows in runs.items()}
    with open(os.path.join(a.out, "per_seed.csv"), "w", newline="") as f:
        allr = [t for ts in seed_tables.values() for t in ts]
        w = csv.DictWriter(f, fieldnames=list(allr[0].keys())); w.writeheader(); w.writerows(allr)
    for name, rows in runs.items():
        plot_returns_per_seed(name, rows, os.path.join(a.out, f"{name}_returns_per_seed.png"))
    plot_overview(runs, seed_tables, os.path.join(a.out, "overview.png"))

    for name, rows in runs.items():
        bc = [r for r in rows if r["bc_ran"]]
        print(f"\n== {name}: {len(rows)} iterations, {len({r['seed'] for r in rows})} seeds | task success "
              f"{sum(r['lifted'] for r in rows)}/{len(rows)} ({100 * np.mean([r['lifted'] for r in rows]):.0f}%) | "
              f"BC success {sum(r['lifted'] for r in bc)}/{len(bc)} ({100 * np.mean([r['lifted'] for r in bc]):.0f}%) | "
              f"fallback closer {100 * np.mean([r['fallback_closer'] for r in rows]):.0f}% of iterations "
              f"({100 * np.mean([r['fallback_closer'] for r in rows if r['rounds'] > 0]):.0f}% when it moved)")
        print(f"   {'seed':>4} {'iters':>5} {'lifts':>5} {'task SR':>7} {'BC SR':>6} {'closer':>6} {'closer|moved':>12} "
              f"{'stuck r1':>8} {'final|D|':>8} {'BC ret':>7}")
        for t in seed_tables[name]:
            print(f"   {t['seed']:>4} {t['iterations']:>5} {t['lifts']:>5} {t['task_success_rate']:>7.2f} "
                  f"{(t['bc_success_rate'] if t['bc_success_rate'] is not None else float('nan')):>6.2f} "
                  f"{t['fallback_closer_frac']:>6.2f} {(t['fallback_closer_frac_when_moved'] if t['fallback_closer_frac_when_moved'] is not None else float('nan')):>12.2f} "
                  f"{t['stuck_round1']:>8} {t['final_D']:>8} "
                  f"{(t['mean_return_bc'] if t['mean_return_bc'] is not None else float('nan')):>7.3f}")
    print(f"\nwritten: {a.out}/ per_iteration.csv, per_seed.csv, overview.png, <run>_returns_per_seed.png")


if __name__ == "__main__":
    main()
