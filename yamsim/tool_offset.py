"""Tool-frame shifts of commanded arm targets (sim runner; written so the executor can reuse it).

Motivation (2026-09-19): in sim the policy stops ~1.5-2 cm short along the gripper's x axis (across the pad
width, +x = the side away from the wrist camera) at grasp onset, for the absolute and the relative checkpoints.

Two executor-side interventions, both a rigid translation of the gripper in its OWN frame (orientation kept):
  constant offset   : every commanded arm target is shifted by d_tool. Meaningful for ABSOLUTE-action policies.
                      With relative actions the shift becomes part of the observed state and the next chunk is
                      anchored to it, so a constant shift would be re-applied every replan (it compounds):
                      refused for relative checkpoints unless forced.
  pre-close advance : when the policy starts to close a gripper, hold the gripper open for N ticks and advance the
                      tool by d over those ticks, then let the close proceed. For RELATIVE policies the advance is
                      absorbed into the state; only targets from chunks OBSERVED before/during the advance still
                      need the part of the advance they have not seen: shift = A(now) - A(t_obs). For ABSOLUTE
                      policies the advance is held while the gripper command stays closed and ramps back afterwards.

`ToolShifter.shift(side, q6, d_tool)` = a few damped-least-squares steps from q6 (numeric Jacobian on the URDF
FK, the same arm model as arm_arbiter / the renderer); falls back to q6 if the residual stays above 1 mm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .kinematics import GripperCalibration, YamArm, rotation_error

URDF = str(Path(__file__).resolve().parent.parent / "assets/yam/yam.urdf")
OFF = {"left": 0, "right": 7}


def base_tf(x: float, height: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Same arm placement as mjworld.build_world_mjcf: R = Rx(pitch) @ Rz(90 deg), position (x, 0, height)."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("x", np.radians(pitch_deg)).as_matrix() @ Rot.from_euler("z", np.pi / 2).as_matrix()
    T[:3, 3] = (x, 0.0, height)
    return T


class ToolShifter:
    def __init__(self, arm_spacing: float = 0.61, arm_height: float = 0.0, base_pitch_deg: float = 0.0, link: str = "gripper"):
        self.link = link
        self.arms = {s: YamArm(URDF, GripperCalibration(6.351, 1.45), base_tf=base_tf(x, arm_height, base_pitch_deg))
                     for s, x in (("left", -arm_spacing / 2), ("right", arm_spacing / 2))}
        self.last_err_mm = {"left": 0.0, "right": 0.0}

    def pose(self, side: str, q6: np.ndarray) -> np.ndarray:
        return self.arms[side].link_poses(np.r_[np.asarray(q6, float), 1.45])[self.link]

    def shift(self, side: str, q6: np.ndarray, d_tool, iters: int = 4, damping: float = 1e-3, max_step: float = 0.08) -> np.ndarray:
        """Joint target whose gripper pose = pose(q6) translated by d_tool (m, gripper frame), same orientation."""
        d_tool = np.asarray(d_tool, float)
        q0 = np.asarray(q6, float)
        if not np.any(np.abs(d_tool) > 1e-6):
            return q0.copy()
        T0 = self.pose(side, q0)
        p_t, R_t = T0[:3, 3] + T0[:3, :3] @ d_tool, T0[:3, :3]
        lim = self.arms[side].joint_limits
        q = q0.copy()
        for _ in range(iters):
            T = self.pose(side, q)
            e = np.r_[p_t - T[:3, 3], 0.3 * rotation_error(T[:3, :3], R_t)]
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            J = np.zeros((6, 6)); h = 1e-4
            for j in range(6):
                qp = q.copy(); qp[j] += h; Tp = self.pose(side, qp)
                J[:, j] = np.r_[Tp[:3, 3] - T[:3, 3], 0.3 * rotation_error(T[:3, :3], Tp[:3, :3])] / h
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -max_step, max_step), lim[:, 0], lim[:, 1])
        err = float(np.linalg.norm(p_t - self.pose(side, q)[:3, 3])) * 1000
        self.last_err_mm[side] = err
        return q if err < 1.0 else q0.copy()


PAD_CENTRE = np.array([0.0, 0.0, -0.1335, 1.0])   # pad-patch centre in the gripper frame


def floor_limited(shifter: "ToolShifter", side: str, q6: np.ndarray, d_tool, pad_floor_m: float) -> np.ndarray:
    """Remove the part of a tool-frame shift that would LOWER the pad centre below `pad_floor_m` above the arm base
    plane (= the work surface on the rig). If the pads are already below the floor the shift may not lower them at
    all. The horizontal part of the shift is kept. Returns the adjusted shift in the tool frame."""
    d_tool = np.asarray(d_tool, float)
    T = shifter.pose(side, q6); R = T[:3, :3]
    z_pad = float((T @ PAD_CENTRE)[2]); dw = R @ d_tool
    lowest = min(z_pad, pad_floor_m)          # never below the floor; if already below it, never lower than now
    dw[2] = max(dw[2], lowest - z_pad)
    return R.T @ dw


TIP_END = np.array([0.0, 0.0, -0.147, 1.0])      # fingertip end in the gripper frame


class DownSpeedCap:
    """Final-stage limiter on what is SENT to an arm: the fingertips may not be commanded downward faster than
    v(z) = v_hi above z_hi, tapering linearly to v_lo at/below z_lo (z = fingertip height above the arm base plane = the work
    surface). When a step would descend faster, the whole joint step is scaled (the path direction is kept). The step is
    also bounded by `max_step` rad per joint, because the sent stream can lag the policy-frame command while it is capped.
    Demonstrations (6 episodes): downward fingertip speed max 35 cm/s, p99.9 22-31 cm/s below 10 cm; the 2026-09-19 crash
    plunged at ~55 cm/s from 21 cm."""

    def __init__(self, shifter: "ToolShifter", v_hi_cm_s: float = 35.0, v_lo_cm_s: float = 25.0, z_hi_cm: float = 15.0, z_lo_cm: float = 5.0,
                 max_step: float = 0.10, hz: float = 30.0):
        self.sh, self.v_hi, self.v_lo, self.z_hi, self.z_lo, self.max_step, self.dt = shifter, v_hi_cm_s / 100, v_lo_cm_s / 100, z_hi_cm / 100, z_lo_cm / 100, max_step, 1.0 / hz
        self.last = {"left": None, "right": None}; self.z = {"left": 0.0, "right": 0.0}; self.n_capped = {"left": 0, "right": 0}

    def tip_z(self, side: str, q6: np.ndarray) -> float:
        return float((self.sh.pose(side, q6) @ TIP_END)[2])

    def vmax(self, z: float) -> float:
        return float(np.interp(z, [self.z_lo, self.z_hi], [self.v_lo, self.v_hi]))

    def apply(self, side: str, q6: np.ndarray) -> np.ndarray:
        q = np.asarray(q6, float)
        if self.last[side] is None:
            self.last[side] = q.copy(); self.z[side] = self.tip_z(side, q); return q
        q = self.last[side] + np.clip(q - self.last[side], -self.max_step, self.max_step)
        z_new = self.tip_z(side, q); drop = self.z[side] - z_new; allowed = self.vmax(self.z[side]) * self.dt
        if drop > allowed:
            for _ in range(3):   # joint step -> height is only locally linear: a few scalings converge
                q = self.last[side] + (q - self.last[side]) * (allowed / max(drop, 1e-9))
                z_new = self.tip_z(side, q); drop = self.z[side] - z_new
                if drop <= allowed * 1.02:
                    break
            self.n_capped[side] += 1
        self.last[side] = q.copy(); self.z[side] = z_new
        return q


class PreCloseAdvance:
    """Bookkeeping of the one-shot advance at close onset, per arm. Call `step` once per control tick with the
    policy-frame command; it returns the gripper commands to send and the advance (m, along +x of the tool) to add to
    a target that comes from a chunk observed at tick `t_obs[side]`."""

    def __init__(self, advance_m: float, ticks: int = 6, relative: bool = True, close_thr: float = 0.20, rearm_ticks: int = 15):
        self.d, self.n, self.rel, self.thr, self.rearm = float(advance_m), max(1, int(ticks)), bool(relative), close_thr, rearm_ticks
        self.A = {s: [0.0] for s in OFF}          # cumulative advance per tick (index = tick + 1; A[0] = before tick 0)
        self.phase = {s: 0 for s in OFF}          # ticks left in the running advance
        self.open_ticks = {s: rearm_ticks for s in OFF}
        self.hold_grip = {s: 0.0 for s in OFF}
        self.n_advances = {s: 0 for s in OFF}

    def _A_at(self, side: str, tick: int) -> float:
        a = self.A[side]
        return a[int(np.clip(tick + 1, 0, len(a) - 1))]

    def step(self, tick: int, cmd14: np.ndarray, t_obs: dict) -> tuple[dict, dict]:
        grips, shifts = {}, {}
        for side, o in OFF.items():
            g = float(cmd14[o + 6]); a_prev = self.A[side][-1]; a = a_prev
            if self.phase[side] == 0 and g > self.thr and self.open_ticks[side] >= self.rearm:
                self.phase[side] = self.n; self.n_advances[side] += 1           # close onset: start an advance
            if self.phase[side] > 0:
                a = a_prev + self.d / self.n; self.phase[side] -= 1
                grips[side] = min(g, self.hold_grip[side])                       # keep the gripper where it was
            else:
                grips[side] = g
                if g <= self.thr:
                    self.hold_grip[side] = g
                if not self.rel and g < 0.10 and a_prev > 0:                     # absolute policy: give the advance back after the release
                    a = max(0.0, a_prev - self.d / self.n)
            self.open_ticks[side] = self.open_ticks[side] + 1 if g <= self.thr else 0
            self.A[side].append(a)
            # relative policy: a chunk observed at t_obs already contains A(t_obs); absolute policy: it contains none of it
            shifts[side] = a - (self._A_at(side, t_obs[side]) if self.rel else 0.0)
        return grips, shifts
