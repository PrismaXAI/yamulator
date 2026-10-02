"""Arm arbitration for the bimanual YAM policy executor / simulator: at most ONE arm in the shared work zone.

Why: with few fruits both arms reach for the same one; the waiting arm hovers over the plate, blocks the other wrist
camera and risks gripper/camera collisions. The policy itself takes turns eventually; this makes the turn-taking explicit.

How (no perception needed, only forward kinematics of both arms):
  * zone      = gripper pad within `zone_w` metres (lateral) of the rig centre line (rest pads sit at +-0.30 m).
  * intent    = the policy's TARGET joints for that arm put its pad inside the zone.
  * owner     = decided from the policy's own PLAN when a lookahead (the rest of the current chunk) is given: among
                the arms whose plan enters the zone, the one that plans to close its gripper, else the one planning
                the lower / deeper descent. (First-to-cross assignment gave the zone to the wrong arm: sim, 25 of 28
                first grasps by the left arm and +4 s to first grasp.) Without a lookahead: first to enter.
  * hold_mode = "freeze": the other arm is stopped at the zone boundary like a virtual wall — it keeps its
                last target that was outside the zone (the raised waiting posture the policy itself produces) and
                follows the policy again whenever its target is outside. "rest": it is sent back to the rest pose
                (sim 2026-09-18: a partner going home made the working arm hesitate, first grasp 6 s -> 10-13 s).
  * other arm = HELD: its arm-joint target is replaced as above and its gripper kept open. It really
                sits at rest, so the state and images the policy sees stay truthful (one arm waiting at rest is common
                in the training data). With relative-action checkpoints a released arm resumes from its true state.
  * release   = owner out of the zone (state AND intent) for `release_ticks`; e.g. while it carries fruit to its own
                container, so the other arm can already start on the next fruit.
  * timeout   = owner inside the zone for `timeout_s` without commanding a close -> ownership is dropped and that arm
                is blocked for `cooldown_s`, so a dithering arm cannot starve the other one.
Pure numpy + the sim's URDF FK; imports nothing from the robot SDK.
"""
import numpy as np

from yamulator import paths
from yamulator.sim.kinematics import GripperCalibration, YamArm

URDF = str(paths.URDF)
REST = np.array([0.0, 0.0, 0.0, -0.08, 0.0, 0.0])
SIDES = ("left", "right")
OFF = {"left": 0, "right": 7}


def _base_tf(x):
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; T[0, 3] = x
    return T


class ZoneArbiter:
    def __init__(self, zone_w: float = 0.20, release_ticks: int = 15, timeout_s: float = 8.0, cooldown_s: float = 6.0,
                 arm_spacing: float = 0.61, hz: float = 30.0, hold_mode: str = "rest"):
        self.zone_w, self.release_ticks, self.hz, self.hold_mode = zone_w, release_ticks, hz, hold_mode
        self.last_ok = {"left": None, "right": None}   # last arm-joint target whose pad was outside the zone
        self.timeout_ticks, self.cooldown_ticks = int(timeout_s * hz), int(cooldown_s * hz)
        self.arms = {s: YamArm(URDF, GripperCalibration(6.351, 1.45), base_tf=_base_tf(x))
                     for s, x in (("left", -arm_spacing / 2), ("right", arm_spacing / 2))}
        self.owner = None
        self.out_count = 0          # ticks the owner has been fully out of the zone
        self.own_ticks = 0          # ticks the owner has been in the zone without closing
        self.blocked = {"left": 0, "right": 0}
        self.held_ticks = {"left": 0, "right": 0}

    def pad(self, side, q6):
        P = self.arms[side].link_poses(np.r_[q6, 1.45])["gripper"]
        return (P @ np.array([0, 0, -0.134, 1.0]))[:3]

    def pad_x(self, side, q6):
        return float(self.pad(side, q6)[0])

    def plan_intent(self, side, lookahead):
        """(enters zone?, max planned close command, lowest planned pad height, deepest planned |x|) over the plan."""
        o = OFF[side]; rows = lookahead[::4]
        pads = np.array([self.pad(side, r[o:o + 6]) for r in rows])
        return bool((np.abs(pads[:, 0]) < self.zone_w).any()), float(lookahead[:, o + 6].max()), float(pads[:, 2].min()), float(np.abs(pads[:, 0]).min())

    def step(self, state14, target14, lookahead=None, prefer=None):
        """state14: measured [L q1..q6, grip | R ...] (arm joints rad); target14: policy target in action space
        (gripper 0=open..1=closed). Returns (target14 possibly modified, info dict)."""
        state14, target14 = np.asarray(state14, float), np.asarray(target14, float).copy()
        inz, want, depth = {}, {}, {}
        for s in SIDES:
            o = OFF[s]
            xs, xt = self.pad_x(s, state14[o:o + 6]), self.pad_x(s, target14[o:o + 6])
            inz[s], want[s], depth[s] = abs(xs) < self.zone_w, abs(xt) < self.zone_w, abs(xt)
            if not want[s]:
                self.last_ok[s] = target14[o:o + 6].copy()
            if self.blocked[s] > 0:
                self.blocked[s] -= 1
        if self.owner is None:
            cand = [s for s in SIDES if (inz[s] or want[s]) and self.blocked[s] == 0]
            if cand and prefer in SIDES and self.blocked[prefer] == 0:
                # perception hint (which side of the scene the fruit is on): that arm gets the zone as soon as anyone wants it
                self.owner = prefer
            elif cand and lookahead is not None and len(lookahead) >= 4:
                # someone is about to enter: ask the PLAN which arm means to work, instead of who crosses first
                intent = {s: self.plan_intent(s, np.asarray(lookahead, float)) for s in SIDES if self.blocked[s] == 0}
                enter = [s for s in intent if intent[s][0] or inz[s]]
                if len(enter) == 1:
                    self.owner = enter[0]
                elif len(enter) == 2:   # plans to close wins; else the lower planned descent; else the deeper one
                    self.owner = max(enter, key=lambda s: (round(intent[s][1], 1), -round(intent[s][2], 2), -intent[s][3]))
            elif len(cand) == 1:
                self.owner = cand[0]
            elif len(cand) == 2:
                self.owner = min(cand, key=lambda s: depth[s])
            self.out_count = self.own_ticks = 0
        else:
            s = self.owner; o = OFF[s]
            if not inz[s] and not want[s]:
                self.out_count += 1
            else:
                self.out_count = 0
            closing = target14[o + 6] > 0.3
            self.own_ticks = 0 if (closing or not inz[s]) else self.own_ticks + 1
            if self.out_count >= self.release_ticks:
                self.owner = None
            elif self.own_ticks >= self.timeout_ticks:
                self.blocked[s] = self.cooldown_ticks; self.owner = None
        held = None
        if self.owner is not None:
            held = "right" if self.owner == "left" else "left"
            o = OFF[held]
            if self.hold_mode == "rest":
                target14[o:o + 6] = REST      # retreat to / wait at the standby pose (rate limiting is the caller's)
                target14[o + 6] = 0.0; self.held_ticks[held] += 1
            elif want[held] or inz[held]:     # virtual wall: stay at (or go back to) the last pose outside the zone
                target14[o:o + 6] = self.last_ok[held] if self.last_ok[held] is not None else REST
                target14[o + 6] = 0.0; self.held_ticks[held] += 1
            # else: its target is outside the zone anyway -> follow the policy
        return target14, {"owner": self.owner, "held": held, "in_zone": dict(inz), "want": dict(want)}
