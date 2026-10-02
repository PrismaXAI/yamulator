"""Minimal contact model for the apple-on-plate grasp task.

Deliberately simple: the apple is a rigid sphere that either rests on its support surface (plate well
or table), is pushed sideways by a finger that intersects it, or is rigidly attached to a gripper once
both finger pads close onto it. Enough to answer "did the policy ever try to close the gripper around
the apple, and did it lift it?" -- not a dynamics simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import FINGER_JOINTS
from .scene import YamStation, tf


@dataclass
class FingerPads:
    """Inner contact faces of the two fingers, in the gripper link frame (fingers extend along -z,
    open along +/-y; finger travel d puts the pad faces at y = +d (tip_left) and y = -d (tip_right)).
    Extents measured from the tip STL: pad/tip region z in [-0.150, -0.105], half width 14 mm in x."""
    x_half: float = 0.014
    z_range: tuple = (-0.150, -0.105)
    thickness: float = 0.020    # finger body outward of the pad face
    grasp_margin: float = 0.002  # commanded gap may be this much larger than the apple and still hold
    release_margin: float = 0.008


@dataclass
class GraspEvent:
    step: int
    side: str
    kind: str   # "between_pads", "grasp", "release", "lift", "push"
    detail: dict = field(default_factory=dict)


class AppleGraspSim:
    def __init__(self, station: YamStation, pads: FingerPads | None = None, lift_threshold: float = 0.05):
        self.st = station
        self.pads = pads or FingerPads()
        self.r = station.cfg.apple_diameter / 2
        self.lift_threshold = lift_threshold
        self.plate_xy = np.array(station.cfg.plate_xy)
        self.plate_R = station.cfg.plate_diameter / 2
        self.plate_top = station.plate_top_z
        self.containers = {n: station.object_poses[n][:3, 3].copy() for n in getattr(station, "container_names", [])}
        self.container_half = np.array(station.cfg.container_size[:2]) / 2
        self.container_depth = station.cfg.container_size[2]
        self.stroke = station.arms["left"].finger_stroke
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, apple_xy: np.ndarray | None = None, apple_yaw: float = 0.7):
        xy = self.plate_xy if apple_xy is None else np.asarray(apple_xy, float)
        self.apple = np.array([xy[0], xy[1], self.support_z(xy) + self.r])
        self.apple_yaw = apple_yaw
        self.grasped_by: str | None = None
        self.grasp_offset = None
        self.rest_z = self.apple[2]
        self.step_i = -1
        self.was_between = {"left": False, "right": False}
        self.events: list[GraspEvent] = []
        self.prev_gap = {"left": None, "right": None}
        self.lifted = False
        self.max_lift = 0.0
        self.placed_in = None
        self._sync_render()

    def in_container(self, xy, margin: float = 0.0) -> str | None:
        for n, c in self.containers.items():
            if np.all(np.abs(np.asarray(xy) - c[:2]) < self.container_half + margin):
                return n
        return None

    def support_z(self, xy) -> float:
        if np.linalg.norm(np.asarray(xy) - self.plate_xy) < self.plate_R * 0.9:
            return self.plate_top
        # inside a container: on its floor; on the edge: perched on the rim
        if self.in_container(xy, margin=-self.r * 0.6) is not None:
            return 0.0005 + 0.0003
        if self.in_container(xy, margin=self.r * 0.6) is not None:
            return self.container_depth
        return 0.0

    def _sync_render(self):
        self.st.set_object_pose("apple", tf(xyz=(self.apple[0], self.apple[1], self.apple[2] - self.r), rpy=(0, 0, self.apple_yaw)))

    # ------------------------------------------------------------------ geometry helpers
    def _sphere_box(self, p, lo, hi):
        """Closest point on an axis-aligned box to p and the penetration depth (>0 if intersecting)."""
        q = np.clip(p, lo, hi)
        d = p - q
        dist = np.linalg.norm(d)
        return q, self.r - dist, (d / dist if dist > 1e-9 else np.array([0.0, 1.0, 0.0]))

    def finger_boxes(self, d: float):
        P = self.pads
        left = (np.array([-P.x_half, d, P.z_range[0]]), np.array([P.x_half, d + P.thickness, P.z_range[1]]))
        right = (np.array([-P.x_half, -d - P.thickness, P.z_range[0]]), np.array([P.x_half, -d, P.z_range[1]]))
        return left, right

    # ------------------------------------------------------------------ step
    def step(self, left_action: np.ndarray, right_action: np.ndarray) -> dict:
        """Apply one commanded joint state (actuator radians, 7 per arm) and update the apple."""
        self.step_i += 1
        self.st.set_arm_state(left_action, right_action)
        info = {"apple": self.apple.copy(), "grasped_by": self.grasped_by}
        sides = {"left": np.asarray(left_action, float), "right": np.asarray(right_action, float)}
        for side, act in sides.items():
            g = self.st.cfg.gripper_cal[side].to_open_state(act[6])
            d = g * self.stroke                 # commanded finger travel (half gap)
            G = self.st.link_pose(side, "gripper")
            p = (np.linalg.inv(G) @ np.append(self.apple, 1.0))[:3]   # apple centre in gripper frame
            P = self.pads
            in_footprint = (abs(p[0]) < P.x_half + 0.5 * self.r) and (P.z_range[0] - 0.5 * self.r < p[2] < P.z_range[1] + 0.5 * self.r)
            between = in_footprint and abs(p[1]) + self.r < d + P.thickness + 0.5 * self.r
            closing = self.prev_gap[side] is not None and d < self.prev_gap[side] - 1e-4
            info[f"{side}_gap"] = 2 * d                                   # commanded
            info[f"{side}_gap_effective"] = min(2 * d, 2 * self.r) if self.grasped_by == side else 2 * d
            info[f"{side}_between_pads"] = bool(between)
            info[f"{side}_closing"] = bool(closing)
            info[f"{side}_apple_in_gripper_frame"] = p.copy()
            if between and not self.was_between[side]:
                self.events.append(GraspEvent(self.step_i, side, "between_pads", {"gap": 2 * d}))
            self.was_between[side] = bool(between)

            if self.grasped_by == side:
                if 2 * d > 2 * self.r + P.release_margin:   # opened: release and drop
                    self.grasped_by = None; self.grasp_offset = None
                    xy = self.apple[:2].copy()
                    self.apple = np.array([xy[0], xy[1], self.support_z(xy) + self.r])
                    self.events.append(GraspEvent(self.step_i, side, "release", {"gap": 2 * d, "apple": self.apple.copy()}))
                    tgt = self.in_container(self.apple[:2], margin=0.03)
                    if tgt is not None:
                        self.placed_in = tgt
                        self.events.append(GraspEvent(self.step_i, side, "place", {"container": tgt}))
                else:                                          # carried; fingers stop at the apple surface
                    self.apple = (G @ np.append(self.grasp_offset, 1.0))[:3]
                    self.st.set_gripper_open_state(side, min(g, self.r / self.stroke))
                    lift = self.apple[2] - self.rest_z
                    self.max_lift = max(self.max_lift, lift)
                    if lift > self.lift_threshold and not self.lifted:
                        self.lifted = True
                        self.events.append(GraspEvent(self.step_i, side, "lift", {"height": lift}))
            elif self.grasped_by is None:
                if between and 2 * d <= 2 * self.r + P.grasp_margin and abs(p[1]) < d + 0.5 * self.r:
                    # pads have closed onto the apple: snap it between the pads and attach
                    p_c = np.array([p[0], 0.0, p[2]])
                    self.apple = (G @ np.append(p_c, 1.0))[:3]
                    self.grasped_by = side; self.grasp_offset = p_c
                    self.st.set_gripper_open_state(side, min(g, self.r / self.stroke))
                    self.events.append(GraspEvent(self.step_i, side, "grasp", {"gap": 2 * d, "apple": self.apple.copy()}))
                else:
                    # resolve finger/apple interpenetration by sliding the apple on its support
                    for lo, hi in self.finger_boxes(d):
                        q, pen, n = self._sphere_box(p, lo, hi)
                        if pen > 0:
                            n_w = G[:3, :3] @ n; n_w[2] = 0.0
                            if np.linalg.norm(n_w) > 1e-6:
                                n_w /= np.linalg.norm(n_w)
                                self.apple[:2] += n_w[:2] * pen
                                self.apple[2] = self.support_z(self.apple[:2]) + self.r
                                p = (np.linalg.inv(G) @ np.append(self.apple, 1.0))[:3]
                                self.events.append(GraspEvent(self.step_i, side, "push", {"pen": pen}))
            self.prev_gap[side] = d
        if self.grasped_by is None:
            self.apple[2] = self.support_z(self.apple[:2]) + self.r
        self._sync_render()
        info.update({"apple": self.apple.copy(), "grasped_by": self.grasped_by, "lifted": self.lifted, "max_lift": self.max_lift,
                     "on_plate": self.grasped_by is None and np.linalg.norm(self.apple[:2] - self.plate_xy) < self.plate_R * 0.9,
                     "placed_in": self.placed_in})
        return info

    def success(self) -> bool:
        return self.lifted
