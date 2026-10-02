"""ContactGraspSim: drop-in replacement for physics.AppleGraspSim backed by the MuJoCo contact world.

Same interface used by run_policy_sim (reset / step / events / success / max_lift / placed_in / plate_xy) but the
apple, the pinch and the release are physical. Arm dynamics are inside the world (SDK motor law), so SimArms
should not step its own arm models when using this class: `step(left7, right7)` takes the commanded targets.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .mjworld import YamContactWorld, PAD_Z
from .physics import GraspEvent
from .scene import YamStation, tf


class ContactGraspSim:
    def __init__(self, station: YamStation, lift_threshold: float = 0.05, contact=None, plate: bool = True, plate_raise: float = 0.0):
        self.st = station
        c = station.cfg
        self.n_apples_max = int(getattr(contact, "n_apples", 1)) if contact is not None else 1
        conts = tuple((-c.container_offset_x if n.endswith("left") else c.container_offset_x, 0.0) for n in station.container_names) or ()
        self.world = YamContactWorld(arm_spacing=c.arm_spacing, plate_xy=tuple(c.plate_xy), containers=conts, container_size=c.container_size, arm_base_height=c.arm_base_height,
                                     contact=contact, plate=plate, arm_base_pitch_deg=c.arm_base_pitch_deg, plate_height=0.0075 + plate_raise, plate_radius=c.plate_diameter / 2)
        self.has_plate = plate
        self.r = c.apple_diameter / 2
        self.lift_threshold = lift_threshold
        self.plate_xy = np.array(c.plate_xy)
        self.plate_R = c.plate_diameter / 2
        self.stroke = self.world.stroke
        self.cal = c.gripper_cal
        self.rest = np.array([0.0, 0.0, 0.0, -0.08, 0.0, 0.0])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, apple_xy=None, apple_yaw: float = 0.7, extra_xy=(), extra_yaw=()):
        """apple_xy: apple 0 (plate centre if None); extra_xy: xy of apples 1..n-1 (world n_apples is fixed at
        construction; apples without a position are parked off the table)."""
        xy0 = self.plate_xy if apple_xy is None else np.asarray(apple_xy, float)
        xys = [xy0] + [np.asarray(e, float) for e in extra_xy][: self.world.n_apples - 1]
        yaws = [apple_yaw] + list(extra_yaw) + [0.7] * len(xys)
        # the world's plate is at its own plate_xy; our per-episode plate jitter moves the world plate too
        self.world.move_plate(self.plate_xy)
        P, Q = [], []
        for xy, yaw in zip(xys, yaws):
            on_plate = self.has_plate and np.linalg.norm(xy - self.plate_xy) < self.plate_R * 0.95
            P.append([xy[0], xy[1], (self.world.plate_height if on_plate else 0.0) + self.r + 0.001])
            q = Rot.from_euler("z", yaw).as_quat(); Q.append([q[3], q[0], q[1], q[2]])
        self.world.reset({"left": np.concatenate([self.rest, [1.0]]), "right": np.concatenate([self.rest, [1.0]])}, apple_pos=P, apple_quat=Q)
        for _ in range(10):
            self.world.step()   # settle
        self.n = len(xys)                                   # active apples this episode
        self.apples = np.array([self.world.apple_pose(k)[0] for k in range(self.n)])
        self.apple = self.apples[0]
        self.start_xy = self.apples[:, :2].copy()
        self.rest_zs = self.apples[:, 2].copy(); self.rest_z = float(self.rest_zs[0])
        self.apple_yaw = apple_yaw
        self.holder = [None] * self.n                       # which gripper pinches apple k
        self.grasped_by = None                              # legacy: a side that holds any apple
        self.step_i = -1
        self.events: list[GraspEvent] = []
        self.was_between = {"left": False, "right": False}
        self._released_at = [-999] * self.n; self._released_by = [None] * self.n
        self.lifted_each = [False] * self.n; self.max_lift_each = [0.0] * self.n; self.placed_each = [None] * self.n
        self.lifted = False; self.max_lift = 0.0; self.placed_in = None
        self._unpinched_ticks = [0] * self.n
        self._sync_render()

    def _sync_render(self):
        for k in range(self.world.n_apples):
            name = "apple" if k == 0 else f"apple{k}"
            if name not in self.st.instances:
                continue
            p, q = self.world.apple_pose(k)
            T = np.eye(4); T[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            # the render apple mesh has its origin at the bottom of the apple; rotate about its centre
            T[:3, 3] = p - T[:3, :3] @ np.array([0, 0, self.r])
            self.st.set_object_pose(name, T)
        for side in ("left", "right"):
            s7 = self.world.arm_state(side)
            self.st.arm_state[side] = np.concatenate([s7[:6], [self.cal[side].to_rad(s7[6])]])
        self.st._pose_arms()

    def arm_state7(self, side: str) -> np.ndarray:
        """[q1..q6, gripper open state 0..1] as measured in the world."""
        return self.world.arm_state(side)

    # ------------------------------------------------------------------ step
    def step(self, left_target7: np.ndarray, right_target7: np.ndarray) -> dict:
        """Targets: [q1..q6 rad, gripper open state 0..1] per arm (the executor's command). Advances 1/30 s.
        With several apples every apple has its own grasp / lift / release / place state; the per-side info keys
        (pad force, pinched, between pads, apple in the gripper frame) refer to that side's FOCUS apple = the one it
        pinches hardest, else the one nearest its pads ('<side>_apple' gives its index)."""
        self.step_i += 1
        self.world.set_target("left", left_target7); self.world.set_target("right", right_target7)
        self.world.step(1.0 / 30.0)
        self.apples = np.array([self.world.apple_pose(k)[0] for k in range(self.n)]); self.apple = self.apples[0]
        supports = [self.world.apple_support(k) for k in range(self.n)]
        info = {"apple": self.apple.copy(), "apples": self.apples.copy(), "supports": list(supports)}
        pinched_by = [None] * self.n
        for side in ("left", "right"):
            s7 = self.world.arm_state(side); d = s7[6] * self.stroke
            best = None
            for k in range(self.n):
                p = self.world.apple_in_gripper(side, k); con = self.world.apple_contacts(side, k)
                force = float(min(con.get("left_pad", 0), con.get("right_pad", 0)))
                in_footprint = (abs(p[0]) < 0.014 + 0.5 * self.r) and (PAD_Z[0] - 0.5 * self.r < p[2] < PAD_Z[1] + 0.5 * self.r)
                between = in_footprint and abs(p[1]) + self.r < d + 0.02 + 0.5 * self.r
                pinched = con.get("left_pad", 0) > 1.0 and con.get("right_pad", 0) > 1.0
                if pinched and pinched_by[k] is None:
                    pinched_by[k] = side
                key = (force, -float(np.linalg.norm(p - np.array([0.0, 0.0, 0.5 * (PAD_Z[0] + PAD_Z[1])]))))
                if best is None or key > best[0]:
                    best = (key, k, p, force, between, pinched)
            _, kf, p, force, between, pinched = best
            info[f"{side}_gap"] = 2 * d; info[f"{side}_gap_effective"] = 2 * d
            info[f"{side}_between_pads"] = bool(between); info[f"{side}_pinched"] = bool(pinched)
            info[f"{side}_pad_force"] = force; info[f"{side}_apple"] = int(kf)
            info[f"{side}_apple_in_gripper_frame"] = p.copy()
            if between and not self.was_between[side]:
                self.events.append(GraspEvent(self.step_i, side, "between_pads", {"gap": 2 * d, "apple_index": kf}))
            self.was_between[side] = bool(between)
        for k in range(self.n):
            support = supports[k]; holder = pinched_by[k]
            if holder is not None and self.holder[k] is None:
                self.holder[k] = holder; self._unpinched_ticks[k] = 0
                self.events.append(GraspEvent(self.step_i, holder, "grasp", {"apple": self.apples[k].copy(), "apple_index": k}))
            elif holder is None and self.holder[k] is not None and (self._unpinched_ticks[k] < 4 and support == "air"):
                self._unpinched_ticks[k] += 1      # hysteresis: a momentary force dropout while carried is not a release
            elif holder is None and self.holder[k] is not None:
                side = self.holder[k]; self.holder[k] = None; self._unpinched_ticks[k] = 0
                self.events.append(GraspEvent(self.step_i, side, "release", {"apple": self.apples[k].copy(), "support": support, "apple_index": k}))
                self._released_at[k] = self.step_i; self._released_by[k] = side
            if self.holder[k] is None and support.startswith("container") and self.placed_each[k] is None and self.step_i > 30:
                self.placed_each[k] = "container_left" if support == "container0" else "container_right"
                # placed by the gripper if released within the last second, otherwise knocked/rolled in
                by = self._released_by[k] if self.step_i - self._released_at[k] <= 30 else "none"
                self.events.append(GraspEvent(self.step_i, by or "none", "place", {"container": self.placed_each[k], "apple_index": k}))
            lift = float(self.apples[k][2] - self.rest_zs[k])
            if self.holder[k] is not None:
                self.max_lift_each[k] = max(self.max_lift_each[k], lift)
                if lift > self.lift_threshold and not self.lifted_each[k]:
                    self.lifted_each[k] = True
                    self.events.append(GraspEvent(self.step_i, self.holder[k], "lift", {"height": lift, "apple_index": k}))
        holders = [h for h in self.holder if h is not None]
        self.grasped_by = holders[0] if holders else None
        self.lifted = any(self.lifted_each); self.max_lift = max(self.max_lift_each)
        self.placed_in = next((c for c in self.placed_each if c is not None), None)
        self._sync_render()
        info.update({"grasped_by": self.grasped_by, "lifted": self.lifted, "max_lift": self.max_lift, "support": supports[0],
                     "on_plate": supports[0] == "plate", "placed_in": self.placed_in, "holders": list(self.holder),
                     "lifted_each": list(self.lifted_each), "placed_each": list(self.placed_each), "all_lifted": all(self.lifted_each)})
        return info

    def success(self) -> bool:
        return self.lifted
