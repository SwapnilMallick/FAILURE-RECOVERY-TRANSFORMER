"""
BC-only test (lift_mcnn): MLP-BC and/or MCNN-BC trained on the same tau_1
dataset (on-trajectory + approach/descend corrective examples), rolled out
from perturbed starts: a tau_1 state of the chosen phase pushed
start-min..start-max sideways/up and rotated <= 11 deg (cube resting, gripper
open). No fallback, no ensemble. Prints each rollout's end, length, gripper
commands, block reasons and (MCNN) memory weights, and the lift count per model.
The default starts (descend, 1-2 cm) are the same as the v4 test's.

    python online_loop_lift_mcnn/bc_descend_test.py --seed 7                      # MCNN and MLP
    python online_loop_lift_mcnn/bc_descend_test.py --seed 7 --models mcnn
    python online_loop_lift_mcnn/bc_descend_test.py --seed 7 --phase approach --start-min 0.05 --start-max 0.10
"""
import argparse, time
import numpy as np, torch
from lift_env import LiftEnv, rotvec_to_mat
from image_encoder import ImageEncoder, EncoderConfig
from fallback_lift import LiftConfig, calibrate_encoder, make_tau1
from bc_lift import BCLiftConfig, BCLiftPolicy, build_bc_lift_dataset, train_bc_lift, bc_lift_rollout
from mcnn import MCNNConfig, train_bc_lift_mcnn

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=8, help="number of perturbed starts")
ap.add_argument("--models", default="mcnn,mlp", help="comma list of mcnn / mlp")
ap.add_argument("--phase", choices=["descend", "approach"], default="descend", help="tau_1 phase the starts come from")
ap.add_argument("--start-min", type=float, default=0.01, help="start offset from the tau_1 state, m")
ap.add_argument("--start-max", type=float, default=0.02)
ap.add_argument("--mcnn-mem-frac", type=float, default=0.5)
ap.add_argument("--mcnn-lambda", type=float, default=None)
args = ap.parse_args()
seed, n = args.seed, args.n
cfg = LiftConfig(seed=seed); enc_cfg = EncoderConfig()
torch.manual_seed(seed)
env = LiftEnv(seed=seed, cube_offset=tuple(cfg.cube_offset), enable_render=True)
enc = ImageEncoder(enc_cfg)
calibrate_encoder(env, enc, np.random.default_rng(seed + 2_000_003), cfg.rot_mode, enc_cfg.n_calib, cfg.calib_restart)
tau1 = make_tau1(env, enc, cfg, lambda *a: None)
E, Rs = np.asarray(tau1.eefs), np.asarray(tau1.rots)
P_D, R_D = E, Rs
src = [i for i, p in enumerate(tau1.phases) if p == args.phase]
rng = np.random.default_rng(seed + 77)
starts = []
while len(starts) < n:
    i = int(rng.choice(src)); env.set_state(tau1.states[i])
    d = rng.normal(size=3); d[2] = abs(d[2]); d /= np.linalg.norm(d)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    if env.place_arm(E[i] + d * rng.uniform(args.start_min, args.start_max), rotvec_to_mat(ax * rng.uniform(0, 0.2)) @ Rs[i]) \
            and not env.cube_contact() and not env.forbidden_contact():
        starts.append((i, env.get_state()))
bccfg = BCLiftConfig(seed=seed)
data = build_bc_lift_dataset(env, enc, [tau1], bccfg, print)
for name in [m.strip() for m in args.models.split(",") if m.strip()]:
    t0 = time.time()
    if name == "mcnn":
        pol = train_bc_lift_mcnn(data, bccfg, MCNNConfig(mem_frac=args.mcnn_mem_frac, lam=args.mcnn_lambda), print)
    else:
        pol = BCLiftPolicy(train_bc_lift(data, bccfg, print))
    res = []
    for k, (i, st) in enumerate(starts):
        r = bc_lift_rollout(pol, env, enc, st, -1.0, (P_D, R_D), bccfg, 5 * len(tau1.tokens),
                            rng=np.random.default_rng(seed + k))
        g = "".join("C" if x > 0 else "o" for x in r["grip_cmd"][1:])
        res.append(r["end"])
        mw = np.array([m[0] for m in r["mem_info"]])
        print(f"  start {k} ({args.phase} state {i}): {r['end']:9s} {len(r['norm_actions']):3d} actions | grip {g[-25:]}"
              + (f" | memory weight median {np.median(mw):.2f} (min {mw.min():.2f})" if len(mw) else "")
              + (f" | blocked ({'close' if r['block_grip'] > 0 else 'open'}): {sorted(set(r['block_reasons']))}"
                 if r["end"] == "blocked" else ""))
    print(f"== seed {seed} {name.upper()} ({args.phase} starts {100 * args.start_min:.0f}-{100 * args.start_max:.0f} cm): "
          f"{res.count('reached')}/{n} lifted, {res.count('blocked')} blocked, {res.count('max_steps')} max_steps "
          f"({time.time() - t0:.0f}s)\n", flush=True)
