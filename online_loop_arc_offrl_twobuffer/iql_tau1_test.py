"""
Policy-only test (arc_offrl): IQL trained on tau_1 alone vs BC with / without
corrective examples, all rolled out (bc_lift.bc_lift_rollout) from the same
perturbed starts: a tau_1 state of the chosen phase pushed start-min..start-max
(random 3-D direction, upward for descend) and rotated <= 11 deg, cube resting,
gripper open. No fallback, no ensemble.

    python online_loop_arc_offrl/iql_tau1_test.py --seed 10
    python online_loop_arc_offrl/iql_tau1_test.py --seed 10 --phase approach --start-min 0.03 --start-max 0.06
"""
import argparse, time
import numpy as np, torch
from lift_env import LiftEnv, rotvec_to_mat
from image_encoder import ImageEncoder, EncoderConfig
from fallback_lift import LiftConfig, calibrate_encoder, make_tau1
from bc_lift import BCLiftConfig, BCLiftPolicy, build_bc_lift_dataset, train_bc_lift, bc_lift_rollout
from iql_policy import IQLConfig, train_iql

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--phase", choices=["descend", "approach"], default="descend")
ap.add_argument("--start-min", type=float, default=0.01)
ap.add_argument("--start-max", type=float, default=0.02)
ap.add_argument("--models", default="iql,bc_corrective,bc_plain")
ap.add_argument("--iql-steps", type=int, default=5000)
a = ap.parse_args()
seed = a.seed
cfg = LiftConfig(seed=seed); enc_cfg = EncoderConfig()
torch.manual_seed(seed)
env = LiftEnv(seed=seed, cube_offset=tuple(cfg.cube_offset), enable_render=True)
enc = ImageEncoder(enc_cfg)
calibrate_encoder(env, enc, np.random.default_rng(seed + 2_000_003), cfg.rot_mode, enc_cfg.n_calib, cfg.calib_restart)
env.reset()
tau1 = make_tau1(env, enc, cfg, print)
E, Rs = np.asarray(tau1.eefs), np.asarray(tau1.rots)
src = [i for i, p in enumerate(tau1.phases) if p == a.phase]
rng = np.random.default_rng(seed + 77)
starts = []
while len(starts) < a.n:
    i = int(rng.choice(src)); env.set_state(tau1.states[i])
    d = rng.normal(size=3)
    if a.phase == "descend":
        d[2] = abs(d[2])
    d /= np.linalg.norm(d)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    if env.place_arm(E[i] + d * rng.uniform(a.start_min, a.start_max), rotvec_to_mat(ax * rng.uniform(0, 0.2)) @ Rs[i]) \
            and not env.cube_contact():
        starts.append((i, env.get_state()))
print(f"{a.n} starts: {a.phase} states of tau_1 pushed {100 * a.start_min:.0f}-{100 * a.start_max:.0f} cm, <= 11 deg")
bccfg = BCLiftConfig(seed=seed)
for name in [m.strip() for m in a.models.split(",") if m.strip()]:
    t0 = time.time()
    if name == "iql":
        pol = train_iql([tau1], IQLConfig(seed=seed, steps=a.iql_steps), print)
    else:
        c = BCLiftConfig(seed=seed, n_aug=20 if name == "bc_corrective" else 0)
        pol = BCLiftPolicy(train_bc_lift(build_bc_lift_dataset(env, enc, [tau1], c, print), c, print))
    res = []
    for k, (i, st) in enumerate(starts):
        r = bc_lift_rollout(pol, env, enc, st, -1.0, (E, Rs), bccfg, 5 * len(tau1.tokens), rng=np.random.default_rng(seed + k))
        g = "".join("C" if x > 0 else "o" for x in r["grip_cmd"][1:])
        res.append(r["end"])
        print(f"  start {k} ({a.phase} state {i}): {r['end']:9s} {len(r['norm_actions']):3d} actions | grip {g[-20:]}"
              + (f" | blocked: {sorted(set(r['block_reasons']))}" if r["end"] == "blocked" else ""))
    print(f"== seed {seed} {name.upper()}: {res.count('reached')}/{a.n} lifted, {res.count('blocked')} blocked, "
          f"{res.count('max_steps')} step limit ({time.time() - t0:.0f}s)\n", flush=True)
