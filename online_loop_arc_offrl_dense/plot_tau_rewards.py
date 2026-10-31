"""Per-transition IQL rewards along every trajectory in a seed's D.

Recomputes robosuite's shaping terms on the stored simulator states
(D.npz tau<k>_qs, LiftEnv.shaping_terms), then the rewards with
iql_policy.traj_rewards. The dashed line marks the hand-off from the
fallback to the policy (from the 'tau_k = fallback (N actions)' log lines).

Usage:
  python plot_tau_rewards.py RUN_DIR SEED [--reward pbrs|absorbing|sparse]
Writes RUN_DIR/reward_plots/seed<SEED>_tau_rewards_<reward>.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iql_policy import IQLConfig, traj_rewards
from lift_env import LiftEnv


class _Traj:
    def __init__(self, tokens, shaping):
        self.tokens, self.shaping = tokens, shaping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("seed", type=int)
    p.add_argument("--reward", default="pbrs", choices=["sparse", "pbrs", "absorbing"])
    a = p.parse_args()
    sd = os.path.join(a.run_dir, f"seed{a.seed}")
    d = np.load(os.path.join(sd, "D.npz"))
    n = int(d["n"])
    handoff = {int(m.group(1)): int(m.group(2)) for m in
               (re.match(r"tau_(\d+) = fallback \((\d+) actions\)", l) for l in open(os.path.join(sd, "fallback.log"))) if m}
    env = LiftEnv(seed=a.seed, enable_render=False)
    cfg = IQLConfig(reward=a.reward)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), sharey=True, squeeze=False)
    for k in range(1, n + 1):
        S = np.array([env.shaping_terms(s) for s in d[f"tau{k}_qs"]], float)
        rw = traj_rewards(_Traj(list(d[f"tau{k}_tokens"]), S), cfg)
        ax = axes.flat[k - 1]
        ax.axhline(0, color="0.6", lw=0.6)
        ax.plot(np.arange(len(rw)), rw, color="tab:blue", lw=0.9)
        if k in handoff:
            ax.axvline(handoff[k] - 0.5, color="tab:red", ls="--", lw=0.8)
        ax.set_title(f"tau_{k}: {len(rw)} actions" + (f" (fallback {handoff[k]})" if k in handoff else " (demo)"), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("transition")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{a.reward} reward")
    env.close()
    fig.suptitle(f"{os.path.basename(os.path.normpath(a.run_dir))}, seed {a.seed}: per-transition {a.reward} reward along each "
                 f"trajectory in D (red dashed = hand-off to the policy)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir = os.path.join(a.run_dir, "reward_plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"seed{a.seed}_tau_rewards_{a.reward}.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
