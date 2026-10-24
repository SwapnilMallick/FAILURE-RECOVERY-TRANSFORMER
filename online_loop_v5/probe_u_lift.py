"""
Probe (v5): does the ensemble's U track distance to tau_1 in the pick task,
and do the probe actions matter? No fallback, no T_explore, no BC.

Setup = fallback_lift.py's iteration 1: same seed -> same cube, tau_1 (varied
pick demo) and encoder calibration; T_D = the same 10-DNN ensemble on tau_1.
Test states (cube resting on the table, gripper open), placed by 6-D IK
around tau_1's APPROACH states (the region the fallback explores):
  pos      : position offsets 0..10 cm (random 3-D direction), orientation =
             that tau_1 state's
  pos_s1   : the same offsets, orientation = the reset pose's (what the v5
             fallback holds: s_1 is placed with s_0's orientation)
  rot      : orientation offsets 0..30 deg (random axis) at that tau_1 state's
             position
Distance = the actual position distance (cm) to the nearest tau_1 state (and
the angle there) after placement. U is computed with 3 probe actions of each
kind, on the same states:
  rot3d    : 1 cm random direction + 0.1 rad random-axis rotation, gripper open (v4 exploration)
  trans    : 1 cm random direction, zero rotation, gripper open (v5 exploration)
  tau1     : 3 actions drawn from tau_1's own (normalized) actions
Prints mean U per distance bin and Spearman(U, distance) per set and probe kind.

    python online_loop_v5/probe_u_lift.py --seed 15
"""
import argparse, json, os
import numpy as np, torch
from lift_env import LiftEnv, rot_angle, rotvec_to_mat
from image_encoder import ImageEncoder, EncoderConfig
from fallback_lift import LiftConfig, calibrate_encoder, make_tau1, sample_action
from dnn_ensemble import DNNEnsemble, train_dnn_ensemble

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=20, help="placements per offset level")
ap.add_argument("--out", default=None, help="optional JSON with every probed state")
args = ap.parse_args()
seed = args.seed
cfg = LiftConfig(seed=seed); enc_cfg = EncoderConfig()
torch.manual_seed(seed)
env = LiftEnv(seed=seed, cube_offset=tuple(cfg.cube_offset), enable_render=True)
enc = ImageEncoder(enc_cfg)
calibrate_encoder(env, enc, np.random.default_rng(seed + 2_000_003), cfg.rot_mode, enc_cfg.n_calib, cfg.calib_restart)
env.reset()
tau1 = make_tau1(env, enc, cfg, lambda *a: None)
env.reset(); _, R_reset = env.eef_pose()
T_E, T_R = np.asarray(tau1.eefs), np.asarray(tau1.rots)
models = train_dnn_ensemble(np.stack([t[:-7] for t in tau1.tokens]), np.stack([t[-7:] for t in tau1.tokens]),
                            tau1.labels, cfg.n_dnn, cfg.dnn_hidden, cfg.dnn_epochs, cfg.dnn_lr, cfg.seed, 1.0, None)
dnn = DNNEnsemble(models, 1.0, cfg.n_probe_actions, np.random.default_rng(seed + 5))
tau_acts = np.stack([t[-7:] for t in tau1.tokens])
appr = [i for i, p in enumerate(tau1.phases) if p == "approach"]
print(f"seed {seed}: tau_1 {len(tau1.tokens)} actions ({len(appr)} approach) | probing around approach states")

rng = np.random.default_rng(seed + 99)


def probe_actions(kind, r):
    if kind == "rot3d":
        return [sample_action(r, "xyz") for _ in range(3)]
    if kind == "trans":
        out = []
        for _ in range(3):
            d = r.normal(size=3); d /= np.linalg.norm(d)
            out.append(np.concatenate([d, np.zeros(3), [-1.0]]))
        return out
    return list(tau_acts[r.choice(len(tau_acts), size=3, replace=False)])


KINDS = ("rot3d", "trans", "tau1")
sets = {"pos": [0, 1, 2, 4, 6, 8, 10], "pos_s1": [0, 1, 2, 4, 6, 8, 10], "rot": [0, 5, 10, 20, 30]}
records = []
for set_name, levels in sets.items():
    for lev in levels:
        got, tries = 0, 0
        while got < args.n and tries < 50 * args.n:
            tries += 1
            i = int(rng.choice(appr)); env.set_state(tau1.states[i])
            if set_name == "rot":
                ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                p, Rm = T_E[i], rotvec_to_mat(ax * np.radians(lev)) @ T_R[i]
            else:
                d = rng.normal(size=3); d /= np.linalg.norm(d)
                p = T_E[i] + d * lev / 100
                Rm = T_R[i] if set_name == "pos" else R_reset
            if not env.place_arm(p, Rm) or env.cube_contact():
                continue
            pe, Re = env.eef_pose()
            j = int(np.linalg.norm(T_E - pe, axis=1).argmin())
            emb = enc.embed(env).reshape(-1)
            rec = dict(set=set_name, level=lev, pos_cm=100 * float(np.linalg.norm(T_E[j] - pe)),
                       ang_deg=float(np.degrees(rot_angle(Re, T_R[j]))))
            r_pr = np.random.default_rng(seed * 1000 + len(records))
            for kind in KINDS:
                rec[kind] = dnn.state_u(emb, actions=probe_actions(kind, r_pr))[1]
            records.append(rec); got += 1


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


for set_name, levels in sets.items():
    R = [r for r in records if r["set"] == set_name]
    unit = "deg" if set_name == "rot" else "cm"
    print(f"\n== {set_name}: mean U per offset level ({unit}); actual distance to tau_1 shown as position cm / angle deg")
    print(f"{'level':>6} {'n':>3} {'pos':>6} {'ang':>6} " + " ".join(f"{k:>8}" for k in KINDS))
    for lev in levels:
        rr = [r for r in R if r["level"] == lev]
        if not rr:
            continue
        print(f"{lev:>6} {len(rr):>3} {np.mean([r['pos_cm'] for r in rr]):6.1f} {np.mean([r['ang_deg'] for r in rr]):6.1f} "
              + " ".join(f"{np.mean([r[k] for r in rr]):8.4f}" for k in KINDS))
    x = [r["ang_deg"] if set_name == "rot" else r["pos_cm"] for r in R]
    print(f"  Spearman(U, {'angle' if set_name == 'rot' else 'position distance'}): "
          + " | ".join(f"{k} {spear(np.array([r[k] for r in R]), np.array(x)):+.2f}" for k in KINDS))
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=1)
