"""
Scripted pick-and-lift policy for the reference trajectory tau_1 (v3, Stage A).

robosuite ships no scripted Lift policy, so this is a standard waypoint
policy on top of the Panda's OSC_POSE controller (lift_env.LiftEnv):

  1. approach : gripper open, move to 10 cm above the cube while turning the
                gripper about the vertical axis to line its fingers up with
                the cube's faces (the cube is 90-degree symmetric, so the
                closest aligned yaw is chosen);
  2. descend  : straight down to grasp height (grip site at the cube centre);
  3. grasp    : close the gripper and hold position for `grasp_steps` steps;
  4. lift     : raise the cube until robosuite reports success (cube centre
                > 4 cm above the table top, i.e. ~1 cm of lift), then
                `extra_lift_steps` more so the lift is clearly visible.

Grasp alignment: yaw_offset = 0 lines the fingers up with the cube's faces
(checked: the fingers close to ~4.2-4.4 cm, the cube's width; 45 degrees
grabs the corners, closing at ~6 cm).

Every step is a 7-D action [dpos(3), drot(3), gripper] in [-1, 1] (pos 1 = 5
cm, rot 1 = 0.5 rad, gripper -1 open / +1 closed). Position commands are
proportional to the distance to the current waypoint and capped at
`max_step` metres per step (default 1 cm, matching the reaching setup);
rotation commands are proportional to the yaw error, capped at `max_rot`.

`approach_noise` > 0 adds random direction noise to the approach phase only
(the varied-demo idea of the reaching setup); grasping and lifting stay
clean.

Returns a dict: actions (T, 7), states (T+1, flattened sim states), eef
(T+1, 3), cube (T+1, 3), gripper (T+1,), phase (T,) labels, success, and the
step at which success was first reached.
"""

from __future__ import annotations

import numpy as np

from lift_env import LiftEnv, mat_to_rotvec, rotvec_to_mat

POS_SCALE, ROT_SCALE = 0.05, 0.5          # OSC_POSE: action 1 = 5 cm / 0.5 rad


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def target_yaw(env: LiftEnv, yaw_offset: float) -> float:
    """Gripper yaw that lines the fingers up with the cube's faces: the cube
    yaw (+ offset) modulo 90 degrees, closest to the current gripper yaw."""
    base = env.cube_yaw() + yaw_offset
    cur = env.eef_yaw()
    k = np.round((cur - base) / (np.pi / 2))
    return float(base + k * np.pi / 2)


def scripted_pick(env: LiftEnv, rng=None, approach_noise: float = 0.0, max_step: float = 0.01,
                  max_rot: float = 0.1, yaw_offset: float = 0.0, hover: float = 0.10,
                  lift_height: float = 0.15, grasp_steps: int = 15, extra_lift_steps: int = 20,
                  max_steps: int = 400, kp: float = 1.0, kr: float = 1.0) -> dict:
    rng = rng or np.random.default_rng(0)
    env.reset()
    cube0 = env.cube_pos()
    yaw_t = target_yaw(env, yaw_offset)
    acts, phases = [], []
    states, eef, cube, grip = [env.get_state()], [env.eef_pos()], [env.cube_pos()], [env.gripper_opening()]
    phase, hold, success_at, extra = "approach", 0, None, 0

    for t in range(max_steps):
        p = env.eef_pos()
        if phase == "approach":
            goal, g = cube0 + np.array([0, 0, hover]), -1.0
        elif phase == "descend":
            goal, g = np.array([cube0[0], cube0[1], cube0[2]]), -1.0
        elif phase == "grasp":
            goal, g = p, 1.0
        else:                                                  # lift
            goal, g = np.array([cube0[0], cube0[1], cube0[2] + lift_height]), 1.0

        err = goal - p
        dpos = kp * err
        n = np.linalg.norm(dpos)
        if n > max_step:
            dpos *= max_step / n
        if phase == "approach" and approach_noise > 0 and n > 0.02:
            v = dpos / max(np.linalg.norm(dpos), 1e-9) + approach_noise * rng.normal(size=3)
            dpos = v / np.linalg.norm(v) * min(max_step, np.linalg.norm(dpos))
        yerr = _wrap(yaw_t - env.eef_yaw()) if phase in ("approach", "descend") else 0.0
        drot = np.array([0.0, 0.0, np.clip(kr * yerr, -max_rot, max_rot)])
        a = np.concatenate([dpos / POS_SCALE, drot / ROT_SCALE, [g]])
        ok = env.step(a)
        acts.append(a); phases.append(phase)
        states.append(env.get_state()); eef.append(env.eef_pos()); cube.append(env.cube_pos())
        grip.append(env.gripper_opening())

        # phase transitions
        if phase == "approach" and np.linalg.norm(err) < 0.01 and abs(yerr) < np.radians(3):
            phase = "descend"
        elif phase == "descend" and np.linalg.norm(err) < 0.005:
            phase = "grasp"
        elif phase == "grasp":
            hold += 1
            if hold >= grasp_steps:
                phase = "lift"
        if ok and success_at is None:
            success_at = t + 1
        if success_at is not None:
            extra += 1
            if extra > extra_lift_steps:
                break

    return dict(actions=np.asarray(acts), states=np.asarray(states), eef=np.asarray(eef),
                cube=np.asarray(cube), gripper=np.asarray(grip), phase=np.asarray(phases),
                success=success_at is not None, success_at=success_at,
                cube_lift=float(cube[-1][2] - cube0[2]), yaw_target=yaw_t, cube_yaw=env.cube_yaw0)


# =========================================================================== #
# Stage B: the same pick, expressed in CLOSED-LOOP actions (lift_env.act)
# =========================================================================== #
STEP, ROT_STEP = 0.01, 0.1              # one action: 1 cm translation, 0.1 rad rotation


def to_norm(dpos, drot, grip) -> np.ndarray:
    """Normalized 7-D action used as model input: [dpos / STEP, drot / ROT_STEP,
    grip], each roughly in [-1, 1]."""
    return np.concatenate([np.asarray(dpos, float) / STEP, np.asarray(drot, float) / ROT_STEP, [grip]])


def from_norm(a):
    a = np.asarray(a, float)
    return a[:3] * STEP, a[3:6] * ROT_STEP, float(a[6])


def _cap(v, m):
    n = np.linalg.norm(v)
    return v if n <= m else v * (m / n)


def _unit(rng):
    u = rng.normal(size=3)
    return u / np.linalg.norm(u)


def scripted_pick_cl(env, rng=None, approach_noise: float = 0.0, rot_noise: float = 0.0,
                     rot_mode: str = "xyz", yaw_offset: float = 0.0, hover: float = 0.10,
                     grasp_actions: int = 2, extra_lift_actions: int = 5,
                     max_actions: int = 200) -> dict:
    """Scripted pick in closed-loop actions. Every action is <= 1 cm and
    <= 0.1 rad, executed with env.act until reached.

    Varied approach (only while > 3 cm from the hover point): approach_noise
    adds direction noise to the 1 cm translation; rot_noise turns each
    rotation into a fixed 0.1 rad step whose axis is the correction direction
    plus rot_noise x a random unit vector (rot_mode "xyz": any 3-D axis,
    "yaw": vertical axis only). Within 3 cm the approach corrects position and
    orientation cleanly, so the descent, grasp and lift are always
    top-down and face-aligned.

    Returns dict: norm_actions (T, 7), states (T+1), eef (T+1, 3), rot (T+1, 3, 3),
    cube (T+1, 3), gripper (T+1,), phase (T,), grip_cmd (T+1,) (command in
    force at each state), substeps, success, success_at."""
    rng = rng or np.random.default_rng(0)
    env.reset()
    cube0 = env.cube_pos()
    R0 = env.eef_rot()
    yaw_t = target_yaw(env, yaw_offset)
    R_goal = rotvec_to_mat([0.0, 0.0, _wrap(yaw_t - env.eef_yaw())]) @ R0      # top-down, aligned
    acts, phases, subs = [], [], []
    states, eef, rots, cube = [env.get_state()], [env.eef_pos()], [env.eef_rot()], [env.cube_pos()]
    grip, grip_cmd = [env.gripper_opening()], [None]
    phase, n_grasp, success_at, extra = "approach", 0, None, 0

    for t in range(max_actions):
        p, Rm = env.eef_pose()
        e_r = mat_to_rotvec(R_goal @ Rm.T)
        g = -1.0
        if phase == "approach":
            goal = cube0 + np.array([0.0, 0.0, hover])
            to_goal = goal - p
            dist = np.linalg.norm(to_goal)
            dpos = _cap(to_goal, STEP)
            drot = _cap(e_r, ROT_STEP)
            if dist > 0.03:
                if approach_noise > 0:
                    v = to_goal / dist + approach_noise * _unit(rng)
                    dpos = v / np.linalg.norm(v) * min(STEP, dist)
                if rot_noise > 0:
                    u = _unit(rng) if rot_mode == "xyz" else np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
                    w = _cap(e_r / ROT_STEP, 1.0) + rot_noise * u
                    drot = w / np.linalg.norm(w) * ROT_STEP
        elif phase == "descend":
            dpos, drot = _cap(cube0 - p, STEP), _cap(e_r, ROT_STEP)
        elif phase == "grasp":
            dpos, drot, g = np.zeros(3), np.zeros(3), 1.0
        else:                                                   # lift
            dpos, drot, g = np.array([0.0, 0.0, STEP]), np.zeros(3), 1.0
        out = env.act(dpos, drot, g)
        acts.append(to_norm(dpos, drot, g)); phases.append(phase); subs.append(out["substeps"])
        states.append(env.get_state()); eef.append(env.eef_pos()); rots.append(env.eef_rot())
        cube.append(env.cube_pos()); grip.append(env.gripper_opening()); grip_cmd.append(g)

        p, Rm = env.eef_pose()
        ang = np.linalg.norm(mat_to_rotvec(R_goal @ Rm.T))
        if phase == "approach" and np.linalg.norm(cube0 + [0, 0, hover] - p) < 0.01 and ang < np.radians(2):
            phase = "descend"
        elif phase == "descend" and np.linalg.norm(cube0 - p) < 0.003:
            phase = "grasp"
        elif phase == "grasp":
            n_grasp += 1
            if n_grasp >= grasp_actions:
                phase = "lift"
        if out["success"] and success_at is None:
            success_at = t + 1
        if success_at is not None:
            extra += 1
            if extra > extra_lift_actions:
                break

    return dict(norm_actions=np.asarray(acts), states=np.asarray(states), eef=np.asarray(eef),
                rot=np.asarray(rots), cube=np.asarray(cube), gripper=np.asarray(grip),
                phase=np.asarray(phases), grip_cmd=grip_cmd, substeps=np.asarray(subs),
                success=success_at is not None, success_at=success_at,
                cube_lift=float(cube[-1][2] - cube0[2]), yaw_target=yaw_t, cube_yaw=env.cube_yaw0)
