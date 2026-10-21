"""
Physics-based robosuite Lift environment for the pick-up task (v3, Stage A).

Unlike task.RobosuiteTask (reaching), the arm is NOT teleported: every action
goes through robosuite's own controller and physics, env.step(action), so the
gripper can close, friction can hold the cube and the cube moves with the hand.

  * Action (7-D, each in [-1, 1]): Panda OSC_POSE delta -- position (x, y, z;
    1 = 5 cm), rotation (axis-angle x, y, z; 1 = 0.5 rad) -- and the gripper
    (-1 = open, +1 = closed).
  * Full simulator state (arm, gripper, cube: positions + velocities) is saved
    / restored with get_state() / set_state(); this replaces the reaching
    code's joint-angle teleports for undoing actions and replaying.
  * Cube: position AND yaw are drawn once per seed from robosuite's own
    placement sampler (yaw uniform in [0, 2 pi)) and then pinned, so the same
    seed always gives the same cube; cube_offset (dx, dy) is added to the
    position as in v2 (default +10 cm away from the robot).
  * Success: robosuite's Lift check -- cube centre more than 4 cm above the
    table top.
  * forbidden_contact(): robot <-> table or robot <-> robot contact (gripper
    <-> cube is allowed), same classification as robosuite_env.py.

Stage B additions
  * act(dpos, drot, grip): a CLOSED-LOOP action. The target gripper pose is
    the current one moved by dpos (m, world frame) and rotated by drot (rad,
    world-frame axis-angle); the OSC controller is stepped until the pose is
    reached (~1 mm, ~1 deg) and the fingers have stopped, capped at max_sub
    low-level steps. A single 20 Hz OSC step from rest only moves ~1 mm for a
    "1 cm" command, which is why actions are executed this way. A forbidden
    contact or leaving the workspace aborts the action (the caller restores
    the saved state).
  * place_arm(pos, R): 6-D damped-least-squares IK on the arm joints, then
    zero velocities -- a static pose with the cube untouched (used for s_1
    and BC's approach-phase corrective examples).
  * pose_dist(): gripper position difference + orientation difference, with
    ROT_W metres per radian (10 deg ~ 1 cm).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import mujoco
import numpy as np
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

logging.getLogger("robosuite_logs").setLevel(logging.WARNING)

POS_SCALE, ROT_SCALE = 0.05, 0.5            # OSC_POSE: action 1 = 5 cm / 0.5 rad
ROT_W = 0.01 / np.radians(10.0)             # pose distance: 10 deg counts as 1 cm
WORKSPACE = ((-0.25, 0.25), (-0.25, 0.25), (0.80, 1.10))   # same box as the reaching setup


def rotvec_to_mat(v) -> np.ndarray:
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(Rm) -> np.ndarray:
    Rm = np.asarray(Rm, float)
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:                       # ~180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((Rm + np.eye(3)) / 2)
        return V[:, np.argmax(w)] * th
    return th / (2 * np.sin(th)) * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0],
                                             Rm[1, 0] - Rm[0, 1]])


def rot_angle(R1, R2) -> float:
    """Angle (rad) of the rotation taking R1 to R2."""
    return float(np.linalg.norm(mat_to_rotvec(np.asarray(R2) @ np.asarray(R1).T)))


def pose_dist(p1, R1, p2, R2) -> float:
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)) + ROT_W * rot_angle(R1, R2))


def dist_to_poses(p, R, P, Rs) -> np.ndarray:
    """Pose distance from (p, R) to every (P[i], Rs[i]): position difference +
    ROT_W x rotation angle (vectorized)."""
    P, Rs = np.asarray(P, float), np.asarray(Rs, float)
    c = (np.einsum("nij,ij->n", Rs, np.asarray(R, float)) - 1.0) / 2.0     # trace(Rs_i^T R)
    return np.linalg.norm(P - np.asarray(p, float), axis=1) + ROT_W * np.arccos(np.clip(c, -1.0, 1.0))


def in_workspace(p) -> bool:
    return all(lo <= x <= hi for x, (lo, hi) in zip(p, WORKSPACE))


def _quat_wxyz_to_yaw(q) -> float:
    """Yaw (rotation about world z) of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _pinned_sampler(x, y, yaw):
    return UniformRandomSampler(
        name="ObjectSampler", x_range=[x, x], y_range=[y, y], rotation=float(yaw),
        rotation_axis="z", ensure_object_boundary_in_range=False,
        ensure_valid_placement=False, reference_pos=(0.0, 0.0, 0.8), z_offset=0.01)


class LiftEnv:
    def __init__(self, seed: int, cube_offset: Tuple[float, float] = (0.10, 0.0),
                 enable_render: bool = True, control_freq: int = 20):
        # phase 1: the seeded default sampler gives this seed's cube xy + yaw
        probe = suite.make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
                           use_camera_obs=False, ignore_done=True, seed=seed)
        probe.reset()
        cid = probe.sim.model.body_name2id("cube_main")
        xy = np.array(probe.sim.data.body_xpos[cid][:2]) + np.asarray(cube_offset, float)
        yaw = _quat_wxyz_to_yaw(probe.sim.data.body_xquat[cid])
        probe.close()
        # phase 2: the real env, cube pinned there
        self.env = suite.make("Lift", robots="Panda", has_renderer=False,
                              has_offscreen_renderer=enable_render, use_camera_obs=False,
                              ignore_done=True, control_freq=control_freq, hard_reset=False,
                              placement_initializer=_pinned_sampler(xy[0], xy[1], yaw), seed=seed)
        self.sim = self.env.sim
        m = self.sim.model
        self.cube_body = m.body_name2id("cube_main")
        grip = [n for n in m.site_names if n.endswith("grip_site")]
        self.grip_site = m.site_name2id(grip[0])
        self.table_z = float(self.env.model.mujoco_arena.table_offset[2])
        self._geom_class = [self._classify(i) for i in range(m.ngeom)]
        self._renderer, self._renderer_size = None, None
        robot = self.env.robots[0]
        self.arm_qpos = np.array(robot._ref_joint_pos_indexes)
        self.arm_dof = np.array(robot._ref_joint_vel_indexes)
        jids = [m.joint_name2id(n) for n in robot.robot_model.joints]
        self.jnt_low, self.jnt_high = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        self.finger_dof = np.array(robot._ref_gripper_joint_vel_indexes["right"])
        self.reset()
        self.cube_start = self.cube_pos()
        self.cube_yaw0 = self.cube_yaw()
        self.state0 = self.get_state()

    # ------------------------------------------------------------------ core
    def reset(self):
        self.last_grip = None
        return self.env.reset()

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.env.step(a)
        return self.success()

    def get_state(self) -> np.ndarray:
        return np.array(self.sim.get_state().flatten())

    def set_state(self, state, last_grip=None) -> None:
        """Restore a full simulator state. last_grip: the gripper command in
        force in that state (so act() knows whether the next command changes
        it); None = unknown, which makes the next act() wait for the fingers."""
        self.sim.set_state_from_flattened(np.asarray(state, float))
        self.sim.forward()
        self.last_grip = last_grip

    def close(self):
        self.env.close()

    # -------------------------------------------------------------- queries
    def eef_pos(self) -> np.ndarray:
        return self.sim.data.site_xpos[self.grip_site].copy()

    def eef_rot(self) -> np.ndarray:
        return self.sim.data.site_xmat[self.grip_site].reshape(3, 3).copy()

    def eef_yaw(self) -> float:
        """Yaw of the gripper's closing axis (its site x-axis) in the table plane."""
        R = self.eef_rot()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def gripper_opening(self) -> float:
        q = self.sim.data.qpos[self.env.robots[0]._ref_gripper_joint_pos_indexes["right"]]
        return float(abs(q[0] - q[1]))

    def cube_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.cube_body])

    def cube_yaw(self) -> float:
        return _quat_wxyz_to_yaw(self.sim.data.body_xquat[self.cube_body])

    def success(self) -> bool:
        return bool(self.env._check_success())

    # -------------------------------------------------- closed-loop actions
    def act(self, dpos, drot, grip: float, tol_pos: float = 0.0015, tol_rot: float = np.radians(1.5),
            settle_pos: float = 0.0003, settle_rot: float = np.radians(0.3), max_sub: int = 40,
            min_grip_sub: int = 12, kp: float = 1.0, kr: float = 1.0) -> dict:
        """Move the gripper by dpos (m) and rotate it by drot (rad, world-frame
        axis-angle) with the gripper command `grip` (-1 open, +1 closed),
        stepping OSC until the target pose is within tol and the arm has
        stopped moving (< settle per low-level step); when the gripper command
        changes, at least min_grip_sub steps are taken so the fingers finish
        (they need ~10-20). Gain 1 was best: higher gains oscillate (measured:
        35/40 random 1 cm + 0.1 rad actions reached, 1.02 +- 0.07 cm moved,
        1.2 mm end error, 0.5 mm drift afterwards, ~26 steps / ~130 ms).
        An action that hits max_sub still ends within a few mm and is kept.
        Returns dict(reached, collided, out_of_bounds, substeps, success)."""
        p_t = self.eef_pos() + np.asarray(dpos, float)
        R_t = rotvec_to_mat(drot) @ self.eef_rot()
        out = dict(reached=False, collided=False, out_of_bounds=not in_workspace(p_t),
                   substeps=0, success=False)
        if out["out_of_bounds"]:
            return out
        min_sub = min_grip_sub if (self.last_grip is None or np.sign(grip) != np.sign(self.last_grip)) else 1
        prev_p, prev_R = self.eef_pose()
        for k in range(max_sub):
            e_p = p_t - self.eef_pos()
            e_r = mat_to_rotvec(R_t @ self.eef_rot().T)
            a = np.concatenate([np.clip(kp * e_p / POS_SCALE, -1, 1), np.clip(kr * e_r / ROT_SCALE, -1, 1),
                                [grip]])
            self.env.step(a)
            out["substeps"] = k + 1
            if self.forbidden_contact():
                out["collided"] = True
                return out
            p, R = self.eef_pose()
            moving = np.linalg.norm(p - prev_p) > settle_pos or rot_angle(prev_R, R) > settle_rot
            prev_p, prev_R = p, R
            if (k + 1 >= min_sub and not moving and np.linalg.norm(p_t - p) < tol_pos
                    and rot_angle(R_t, R) < tol_rot):
                out["reached"] = True
                break
        self.last_grip = float(grip)
        out["success"] = self.success()
        return out

    def eef_pose(self):
        return self.eef_pos(), self.eef_rot()

    # ------------------------------------------------------------ placement
    def place_arm(self, pos, R=None, q_init=None, iters: int = 200, tol: float = 1e-4,
                  damping: float = 1e-2, max_step: float = 0.1) -> bool:
        """6-D DLS IK on the 7 arm joints to put the grip site at pos with
        orientation R (default: keep the current one); velocities are zeroed
        afterwards. Returns whether it converged (and the pose is
        collision-free)."""
        m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
        d = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        R = self.eef_rot() if R is None else np.asarray(R, float)
        pos = np.asarray(pos, float)
        q = (self.sim.data.qpos[self.arm_qpos].copy() if q_init is None
             else np.asarray(q_init, float).copy())
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        ok = False
        for _ in range(iters):
            self.sim.data.qpos[self.arm_qpos] = q
            self.sim.forward()
            e = np.concatenate([pos - self.eef_pos(), mat_to_rotvec(R @ self.eef_rot().T)])
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
                ok = True
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.grip_site)
            J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.jnt_low, self.jnt_high)
        self.sim.data.qpos[self.arm_qpos] = q
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        return ok and not self.forbidden_contact()

    def camera_names(self):
        m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
        return [m.camera(i).name for i in range(m.ncam)]

    # ------------------------------------------------------------ contacts
    def _classify(self, gid) -> str:
        m = self.sim.model
        body = m.body_id2name(m.geom_bodyid[gid]) or ""
        if body == "table":
            return "table"
        if body == "cube_main":
            return "cube"
        if body.startswith(("robot0_", "gripper0_", "fixed_mount0_")):
            return "robot"
        return "other"

    def forbidden_contact(self) -> bool:
        d = self.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = self._geom_class[c.geom1], self._geom_class[c.geom2]
            if {g1, g2} == {"table", "robot"} or (g1 == "robot" and g2 == "robot"):
                return True
        return False

    # -------------------------------------------------------------- render
    def render(self, camera, height: int = 256, width: int = 256) -> np.ndarray:
        m = self.sim.model._model if hasattr(self.sim.model, "_model") else self.sim.model
        d = self.sim.data._data if hasattr(self.sim.data, "_data") else self.sim.data
        if self._renderer is None or self._renderer_size != (height, width):
            self._renderer = mujoco.Renderer(m, height=height, width=width)
            self._renderer_size = (height, width)
            self._opt = mujoco.MjvOption()
            self._opt.geomgroup[0] = 0
        self._renderer.update_scene(d, camera=camera, scene_option=self._opt)
        return self._renderer.render()
