"""Per-seed reward vs iteration for an online-loop run.

Reward of an iteration = 1 if the policy lifted the cube (tau_k added to D),
else 0. One subplot per seed.

Usage:
  python plot_rewards.py RUN_DIR [RUN_DIR ...]
Writes RUN_DIR/reward_plots/rewards_per_seed.png
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(run_dir):
    """seed -> (rows, running). iterations.csv is only written when a seed
    finishes; for a seed still running, the per-iteration 'stop=... new
    trajectory=yes|no' lines of its fallback.log are used instead."""
    seeds = {}
    for d in glob.glob(os.path.join(run_dir, "seed*")):
        sd = int(re.search(r"seed(\d+)$", d).group(1))
        f = os.path.join(d, "iterations.csv")
        if os.path.exists(f):
            rows = list(csv.DictReader(open(f)))
            seeds[sd] = ([(int(r["iteration"]), int(r["added"] == "True")) for r in rows], False)
        elif os.path.exists(os.path.join(d, "fallback.log")):
            ends = [m.group(1) for m in (re.match(r"stop=.* new trajectory=(yes|no)", l)
                                         for l in open(os.path.join(d, "fallback.log"))) if m]
            seeds[sd] = ([(i + 1, int(e == "yes")) for i, e in enumerate(ends)], True)
    return dict(sorted(seeds.items()))


def plot(run_dir):
    seeds = load(run_dir)
    n_it = max(len(v[0]) for v in seeds.values())
    fig, axes = plt.subplots(5, 4, figsize=(18, 16), sharex=True, sharey=True)
    for ax, (sd, (rows, running)) in zip(axes.flat, seeds.items()):
        it = [r[0] for r in rows]
        rew = [r[1] for r in rows]
        ax.plot(it, rew, color="tab:blue", lw=0.8, marker="o", ms=3)
        ax.set_title(f"seed {sd}: {sum(rew)}/{len(rew)} lifted" + (" (still running)" if running else ""), fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlim(0, n_it + 1)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("reward")
    fig.suptitle(f"Reward per iteration — {os.path.basename(os.path.normpath(run_dir))}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = os.path.join(run_dir, "reward_plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "rewards_per_seed.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        plot(d)
