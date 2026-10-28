"""
BC-only test (v4): BC trained on tau_1 with / without the descend-phase
corrective examples, rolled out from perturbed descend-phase starts (a descend
state of tau_1 pushed 1-2 cm sideways/up and rotated <= 11 deg, cube resting,
gripper open). No fallback, no ensemble. Prints each rollout's end, length,
gripper commands and block reasons, and the lift count per variant.

    python online_loop_v4/bc_descend_test.py --seed 7
    python online_loop_v4/bc_descend_test.py --seed 7 --n 8 --both   # also the approach-only (v3-style) BC
"""
import argparse, sys, time
from dataclasses import replace
import numpy as np, torch
from lift_env import LiftEnv, rotvec_to_mat
from image_encoder import ImageEncoder, EncoderConfig
from fallback_lift import LiftConfig, calibrate_encoder, make_tau1
from bc_lift import BCLiftConfig, BCLiftPolicy, build_bc_lift_dataset, train_bc_lift, bc_lift_rollout

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=8, help="number of perturbed descend starts")
ap.add_argument("--both", action="store_true", help="also run BC without descend corrective examples")
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
desc = [i for i, p in enumerate(tau1.phases) if p == "descend"]
# starts: descend states pushed 1-2 cm sideways/up + <= 11 deg, cube resting, gripper open
rng = np.random.default_rng(seed + 77)
starts = []
while len(starts) < n:
    i = int(rng.choice(desc)); env.set_state(tau1.states[i])
    d = rng.normal(size=3); d[2] = abs(d[2]); d /= np.linalg.norm(d)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    if env.place_arm(E[i] + d * rng.uniform(0.01, 0.02), rotvec_to_mat(ax * rng.uniform(0, 0.2)) @ Rs[i]) \
            and not env.cube_contact():
        starts.append((i, env.get_state()))
arms = (("approach-only (v3)", False), ("approach+descend (v4)", True))
if not args.both:
    arms = arms[1:]
for name, aug_desc in arms:
    t0 = time.time()
    bccfg = BCLiftConfig(seed=seed, aug_descend=aug_desc)
    data = build_bc_lift_dataset(env, enc, [tau1], bccfg, print)
    pol = BCLiftPolicy(train_bc_lift(data, bccfg, print))
    res = []
    for k, (i, st) in enumerate(starts):
        r = bc_lift_rollout(pol, env, enc, st, -1.0, (P_D, R_D), bccfg, 5 * len(tau1.tokens),
                            rng=np.random.default_rng(seed + k))
        g = "".join("C" if x > 0 else "o" for x in r["grip_cmd"][1:])
        res.append(r["end"])
        print(f"  start {k} (descend state {i}): {r['end']:9s} {len(r['norm_actions']):3d} actions | grip {g[-25:]}"
              + (f" | blocked ({'close' if r['block_grip'] > 0 else 'open'}): {sorted(set(r['block_reasons']))}"
                 if r["end"] == "blocked" else ""))
    print(f"== seed {seed} {name}: {res.count('reached')}/{n} lifted, {res.count('blocked')} blocked, "
          f"{res.count('max_steps')} max_steps ({time.time() - t0:.0f}s)\n", flush=True)
