"""
Per-iteration replay GIF for the online loop (fallback_explore.py --gif).

Three free-camera views side by side -- front, side, isometric -- of the
robot replaying the CURRENT iteration's trajectory from its start s_1:
the fallback path (s_1 -> path end) followed by the BC rollout (path end ->
cube), whether or not it was added to D. Drawn in the scene as 3-D tubes:
  * every trajectory already in D: tau_1 black, the others in distinct,
    semi-transparent colours (static);
  * the current trajectory as a growing trail: magenta while on the
    fallback path, green on the BC rollout;
  * a blue sphere at s_1.
The robot is rendered semi-transparent so the paths behind it stay visible.
That is done by temporarily lowering the robot geoms' alpha in the shared
MuJoCo model -- the same model the image encoder renders -- so the original
colours are always restored before returning.

Framing: the cameras look at the centre of everything drawn and back off in
proportion to its extent, so the paths fill the frame (the upper arm may be
cut off). Long trajectories are subsampled to at most `max_frames` frames;
the last frame is held.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import mujoco
import numpy as np
from PIL import Image, ImageDraw

D_COLOURS = [(0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
             (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.09, 0.75, 0.81), (0.74, 0.74, 0.13),
             (0.50, 0.50, 0.50)]
FALLBACK_RGBA = (0.84, 0.12, 0.84, 1.0)         # matches plot_fallback_3d.PATH_COLOR
BC_RGBA = (0.10, 0.60, 0.31, 1.0)               # matches plot_fallback_3d.BC_COLOR
S1_RGBA = (0.25, 0.41, 0.88, 1.0)
VIEWS = (("front", 180.0, -12.0, 1.0), ("side", 90.0, -12.0, 1.0), ("isometric", 135.0, -32.0, 1.1))


def _add_tube(scene, P, rgba, width):
    for a, b in zip(P[:-1], P[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float),
                             np.asarray(b, float))
        scene.ngeom += 1


def _add_sphere(scene, p, rgba, r):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([r, r, r]), np.asarray(p, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def write_replay_gif(ik, out_path: str, D_eefs: Sequence[np.ndarray], cur_qs: Sequence[np.ndarray],
                     cur_eefs: Sequence[np.ndarray], n_fallback: int, title: str = "",
                     size: int = 256, max_frames: int = 120, frame_ms: int = 80,
                     hold_ms: int = 1500, robot_alpha: float = 0.35, log=None,
                     set_frame=None, phase_labels: Optional[Sequence[str]] = None) -> Optional[str]:
    """ik: the run's RobosuiteIK (its sim is used and left at the last frame's
    configuration). D_eefs: EEF paths of the trajectories in D (tau_1 first).
    cur_qs / cur_eefs: the current trajectory's joint configs / EEF positions
    from s_1; the first n_fallback points are the fallback path (the rest is
    the BC rollout)."""
    cur_qs = [np.asarray(q, float) for q in cur_qs]
    cur_eefs = np.asarray(cur_eefs, float)
    if len(cur_qs) < 1:
        return None
    m = ik.sim.model._model if hasattr(ik.sim.model, "_model") else ik.sim.model
    dat = ik.sim.data._data if hasattr(ik.sim.data, "_data") else ik.sim.data

    # semi-transparent robot (restored in `finally`)
    robot = [g for g in range(m.ngeom)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
             .startswith(("robot0_", "gripper0_"))]
    saved_alpha = m.geom_rgba[robot, 3].copy()
    renderer = None
    try:
        m.geom_rgba[robot, 3] = np.minimum(saved_alpha, robot_alpha)   # never reveal hidden geoms
        renderer = mujoco.Renderer(m, height=size, width=size, max_geom=20000)
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0                                   # hide collision meshes

        pts = np.concatenate([np.asarray(E, float) for E in D_eefs] + [cur_eefs])
        lo, hi = pts.min(0), pts.max(0)
        center, extent = (lo + hi) / 2, float(np.max(hi - lo))
        cams = []
        for name, az, el, scale in VIEWS:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = center
            c.azimuth, c.elevation = az, el
            c.distance = float(np.clip(2.4 * extent, 0.8, 1.3) * scale)   # >= 0.8: the gripper
                                                                         # must not fill the view
            cams.append((name, c))

        idx = np.unique(np.linspace(0, len(cur_qs) - 1, min(max_frames, len(cur_qs))).round().astype(int))
        frames = []
        for k in idx:
            # set_frame(item) restores a full simulator state (physics task, so the
            # cube moves too); default = arm joint configuration (reaching task)
            (set_frame or ik.set_q)(cur_qs[k])
            if phase_labels is not None:
                phase = phase_labels[min(k, len(phase_labels) - 1)]
            else:
                phase = "fallback" if k < n_fallback else "BC"
            panels = []
            for name, c in cams:
                renderer.update_scene(dat, camera=c, scene_option=opt)
                sc = renderer.scene
                for j, E in enumerate(D_eefs):
                    rgba = (0, 0, 0, 1) if j == 0 else (*D_COLOURS[(j - 1) % len(D_COLOURS)], 0.6)
                    _add_tube(sc, np.asarray(E, float), rgba, 0.0035 if j == 0 else 0.003)
                _add_tube(sc, cur_eefs[:min(k, n_fallback - 1) + 1], FALLBACK_RGBA, 0.0055)
                if k >= n_fallback:
                    _add_tube(sc, cur_eefs[n_fallback - 1:k + 1], BC_RGBA, 0.0055)
                _add_sphere(sc, cur_eefs[0], S1_RGBA, 0.009)
                im = Image.fromarray(renderer.render())
                dr = ImageDraw.Draw(im)
                dr.text((6, 5), name, fill=(0, 0, 0))
                dr.text((6, size - 15), f"{title}step {k}/{len(cur_qs) - 1} ({phase})", fill=(0, 0, 0))
                panels.append(np.asarray(im))
            frames.append(np.concatenate(panels, axis=1))
    finally:
        m.geom_rgba[robot, 3] = saved_alpha
        if renderer is not None:
            renderer.close()

    imgs = [Image.fromarray(f) for f in frames]
    pal = imgs[-1].quantize(colors=256, method=Image.MEDIANCUT)      # last frame has every colour
    q = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
    q[0].save(out_path, save_all=True, append_images=q[1:],
              duration=[frame_ms] * (len(q) - 1) + [hold_ms], loop=0)
    if log:
        log(f"replay gif: {len(q)} frames x 3 views ({size} px) -> {out_path} "
            f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path
