"""MuJoCo world with both YAM arms, the apple, plate, containers and table: physical gripper-apple contact.

Same world frame as the renderer (z up, origin between the arm flanges on the work surface, arms yawed to face
+y). Arms are the i2rt MJCF (arm + linear_4310 gripper, via dynamics.build_combined_mjcf) instantiated twice with
name prefixes. Link meshes do not collide; collision geometry is primitives measured from the meshes:
  - finger pads (per finger): box 22 x 4 x 27 mm on the tip face, z in [-147, -120] mm below the gripper origin
  - finger bodies: box above the pad (z -120..-80 mm), the carriages/palm: one box around the gripper body
  - apple: sphere r = 37.5 mm, 150 g, friction 0.6; plate: static cylinder; containers: static boxes (3 mm walls
    for collision, the visual walls are 0.5 mm); table: plane.
Arm joints: SDK motor law (see dynamics.py). Fingers: position actuators with a force cap (the real gripper
runs a force limiter), so a pinch applies a bounded squeeze rather than a kinematic clamp.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .dynamics import YamMotorConfig, build_combined_mjcf, YamArmDynamics

PAD_Z = (-0.147, -0.120)


def _mjq_to_R(q):  # MuJoCo wxyz
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _R_to_mjq(R):
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


@dataclass
class ContactConfig:
    apple_radius: float = 0.0375
    apple_mass: float = 0.15
    friction: float = 0.6            # pad/apple and apple/plate sliding friction
    rolling: float = 0.002           # rolling friction (m): an apple rolling at 0.3 m/s stops in ~10 cm on a plate
    plate_rim_height: float = 0.020  # the plate's rim above the well (collision ring), keeps a nudged apple on the plate
    grip_force_max: float = 35.0     # N per finger, force limiter of the real gripper (assumed)
    grip_kp: float = 3000.0          # N/m position servo on the finger travel (saturates at grip_force_max)
    grip_kv: float = 60.0
    solref: tuple = (0.002, 1.0)     # 'rigid' parts: 2x timestep (MuJoCo's stiffness scales with the contacting mass: ~1 mm at 35 N for the apple)
    pad_half: tuple = (0.011, 0.002, 0.0135)
    # foam pad: ~2 mm compliant layer on each tip face, modelled as a grid of soft contact bumps so a pinch forms a
    # patch (several contact points -> moment resistance) with a tuned compliance (~1.5 mm compression at 30 N)
    foam: bool = True
    foam_thickness: float = 0.002
    foam_grid: tuple = (3, 4)        # bumps across x and z
    foam_bump_r: float = 0.0025
    foam_solref: tuple = (0.006, 1.0)  # timeconst, dampratio: ~1.5 mm foam compression at 30 N (tuned)
    foam_friction: float = 0.8       # foam on apple skin
    n_apples: int = 1                # free-body apples (apple-apple and gripper-apple contact; arm LINK meshes stay visual only)


def _prefix_tree(root: ET.Element, prefix: str):
    """Prefix all names/references in an arm MJCF (bodies, joints, geoms, sites, actuators, equality, contact)."""
    for el in root.iter():
        for attr in ("name", "joint", "joint1", "joint2", "body1", "body2", "site", "body"):
            if el.tag in ("mesh",) and attr == "name":
                continue
            v = el.get(attr)
            if v is not None and not v.startswith(prefix):
                el.set(attr, prefix + v)


def _add_gripper_collision(gripper_body: ET.Element, prefix: str, cfg: ContactConfig):
    """Collision boxes measured in the gripper frame at the closed position, placed on the right bodies."""
    tips = {b.get("name"): b for b in gripper_body.findall("body")}
    tl, tr = tips[prefix + "tip_left"], tips[prefix + "tip_right"]
    # boxes in the gripper frame at closed (joint7 = joint8 = 0): (center, half-size) -- left finger on -y side
    left_boxes = [("pad", np.array([0.0, -0.002, -0.1335]), np.array(cfg.pad_half)),
                  ("finger", np.array([0.0, -0.0128, -0.100]), np.array([0.019, 0.0103, 0.020]))]
    for body, sign in ((tl, 1.0), (tr, -1.0)):
        p_b = np.array([float(x) for x in body.get("pos").split()]); R_b = _mjq_to_R([float(x) for x in body.get("quat").split()])
        for name, c, h in left_boxes:
            c = c * np.array([1.0, sign, 1.0])
            c_local = R_b.T @ (c - p_b)
            q_local = _R_to_mjq(R_b.T)   # box axes = gripper axes
            ET.SubElement(body, "geom", {"name": f"{body.get('name')}_{name}", "type": "box", "pos": _fmt(c_local), "quat": _fmt(q_local),
                                         "size": _fmt(h), "contype": "1", "conaffinity": "1", "friction": f"{cfg.friction} 0.005 0.0001",
                                         "solref": _fmt(cfg.solref), "rgba": "0.2 0.8 0.2 0.4", "group": "3"})
        if cfg.foam:
            # bumps on the pad face (face at y = 0 in the closed gripper frame, pad material on the -sign side);
            # bump centres sit foam_thickness - bump_r behind the face plane so the bump surface is the foam surface
            c_pad, h_pad = left_boxes[0][1] * np.array([1.0, sign, 1.0]), left_boxes[0][2]
            face_y = c_pad[1] - sign * (-h_pad[1])   # y of the contact face (toward the centre): c_pad.y + sign*h.y
            face_y = c_pad[1] + sign * h_pad[1] * -1 if sign < 0 else c_pad[1] + h_pad[1]
            face_y = 0.0 + (0.0)   # at closed both faces are at y = 0 by construction
            nx, nz = cfg.foam_grid
            for i, bx in enumerate(np.linspace(-h_pad[0] + cfg.foam_bump_r, h_pad[0] - cfg.foam_bump_r, nx)):
                for j, bz in enumerate(np.linspace(c_pad[2] - h_pad[2] + cfg.foam_bump_r, c_pad[2] + h_pad[2] - cfg.foam_bump_r, nz)):
                    by = -sign * (cfg.foam_thickness - cfg.foam_bump_r) * -1   # toward the centre (+y for the left pad which is on -y)
                    by = (cfg.foam_thickness - cfg.foam_bump_r) * (1.0 if sign > 0 else -1.0)
                    # left pad (sign=+1 in this builder) has material on -y, so its foam protrudes toward +y
                    c_b = np.array([bx, by, bz])
                    ET.SubElement(body, "geom", {"name": f"{body.get('name')}_foam{i}{j}", "type": "sphere", "pos": _fmt(R_b.T @ (c_b - p_b)),
                                                 "size": str(cfg.foam_bump_r), "contype": "1", "conaffinity": "1", "friction": f"{cfg.foam_friction} 0.01 0.005",
                                                 "solref": _fmt(cfg.foam_solref), "solimp": "0.9 0.95 0.001 0.5 2", "rgba": "0.9 0.9 0.2 0.6", "group": "3", "mass": "0"})
    # palm / carriages: one box around the gripper body (z -65..+10 mm)
    ET.SubElement(gripper_body, "geom", {"name": prefix + "palm", "type": "box", "pos": "0 0 -0.0275", "size": "0.0325 0.051 0.0375",
                                         "contype": "1", "conaffinity": "1", "friction": f"{cfg.friction} 0.005 0.0001", "rgba": "0.2 0.2 0.8 0.3", "group": "3"})


def build_world_mjcf(arm_spacing: float = 0.61, plate_xy=(0.0, 0.38), plate_radius=0.10, plate_height=0.0075, arm_base_height: float = 0.0, plate: bool = True, arm_base_pitch_deg: float = 0.0,
                     containers=((-0.29, 0.0), (0.29, 0.0)), container_size=(0.15, 0.20, 0.03), motor: YamMotorConfig | None = None,
                     contact: ContactConfig | None = None) -> str:
    motor = motor or YamMotorConfig(); contact = contact or ContactConfig()
    world = ET.Element("mujoco", {"model": "yam_station"})
    ET.SubElement(world, "compiler", {"angle": "radian"})
    opt = ET.SubElement(world, "option", {"timestep": str(motor.motor_dt), "integrator": "implicitfast", "gravity": "0 0 -9.81", "cone": "elliptic", "impratio": "5"})
    asset = ET.SubElement(world, "asset")
    wb = ET.SubElement(world, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2", "dir": "0 0 -1"})
    fr = f"{contact.friction} 0.005 {contact.rolling}"
    # finite work surface (matches the render: 1.2 x 1.3 m slab), floor 0.7 m below, work-cell walls
    ET.SubElement(wb, "geom", {"name": "table", "type": "box", "size": "0.6 0.65 0.015", "pos": "0 0.22 -0.015", "contype": "1", "conaffinity": "1", "friction": fr, "rgba": "0.9 0.9 0.9 1"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.01", "pos": "0 0 -0.7", "contype": "1", "conaffinity": "1", "friction": fr, "rgba": "0.3 0.3 0.3 1"})
    ET.SubElement(wb, "geom", {"name": "wall_rear", "type": "box", "size": "0.5 0.005 0.25", "pos": "0 0.805 0.25", "contype": "1", "conaffinity": "1", "rgba": "1 1 1 0.3"})
    ET.SubElement(wb, "geom", {"name": "wall_left", "type": "box", "size": "0.005 0.5 0.25", "pos": "-0.505 0.3 0.25", "contype": "1", "conaffinity": "1", "rgba": "1 1 1 0.3"})
    ET.SubElement(wb, "geom", {"name": "wall_right", "type": "box", "size": "0.005 0.5 0.25", "pos": "0.505 0.3 0.25", "contype": "1", "conaffinity": "1", "rgba": "1 1 1 0.3"})
    if not plate:
        plate_height = 0.0
    if plate:
        ET.SubElement(wb, "geom", {"name": "plate", "type": "cylinder", "size": f"{plate_radius} {plate_height / 2}", "pos": f"{plate_xy[0]} {plate_xy[1]} {plate_height / 2}",
                                   "contype": "1", "conaffinity": "1", "friction": fr, "rgba": "1 1 1 1"})
    # plate rim: ring of box segments just inside the outer radius (the real rim flares up ~2 cm)
    n_seg = 24; r_rim = plate_radius - 0.006
    for k in (range(n_seg) if (plate and contact.plate_rim_height > 0) else ()):
        a = 2 * np.pi * k / n_seg
        ET.SubElement(wb, "geom", {"name": f"plate_rim{k}", "type": "box", "size": f"{np.pi * r_rim / n_seg * 1.05} 0.004 {contact.plate_rim_height / 2}",
                                   "pos": f"{plate_xy[0] + r_rim * np.cos(a)} {plate_xy[1] + r_rim * np.sin(a)} {plate_height + contact.plate_rim_height / 2}",
                                   "quat": _fmt(_R_to_mjq(Rot.from_euler("z", a + np.pi / 2).as_matrix())), "contype": "1", "conaffinity": "1", "friction": fr, "rgba": "1 1 1 0.5"})
    sx, sy, sz = container_size; wall = 0.003
    for i, (dx, dy) in enumerate(containers):
        cx, cy = plate_xy[0] + dx, plate_xy[1] + dy
        ET.SubElement(wb, "geom", {"name": f"container{i}_floor", "type": "box", "size": f"{sx/2} {sy/2} 0.0015", "pos": f"{cx} {cy} 0.0015", "contype": "1", "conaffinity": "1", "rgba": "0.8 0.8 1 0.3"})
        for j, (ox, oy, hx, hy) in enumerate(((-(sx - wall) / 2, 0, wall / 2, sy / 2), ((sx - wall) / 2, 0, wall / 2, sy / 2), (0, -(sy - wall) / 2, sx / 2, wall / 2), (0, (sy - wall) / 2, sx / 2, wall / 2))):
            ET.SubElement(wb, "geom", {"name": f"container{i}_wall{j}", "type": "box", "size": f"{hx} {hy} {sz/2}", "pos": f"{cx + ox} {cy + oy} {sz/2}", "contype": "1", "conaffinity": "1", "rgba": "0.8 0.8 1 0.3"})
    for k in range(max(1, int(contact.n_apples))):   # apple 0 keeps the historical names; extra apples are apple1, apple2, ...
        nm = "apple" if k == 0 else f"apple{k}"
        apple = ET.SubElement(wb, "body", {"name": nm, "pos": f"{plate_xy[0] + 0.09 * k} {plate_xy[1]} {plate_height + contact.apple_radius}"})
        ET.SubElement(apple, "freejoint", {"name": f"{nm}_free"})
        ET.SubElement(apple, "geom", {"name": f"{nm}_geom", "type": "sphere", "size": str(contact.apple_radius), "mass": str(contact.apple_mass), "contype": "1", "conaffinity": "1",
                                      "friction": fr, "solref": _fmt(contact.solref), "condim": "6", "rgba": "0.8 0.1 0.1 1"})
    act = ET.SubElement(world, "actuator"); eq = ET.SubElement(world, "equality"); con = ET.SubElement(world, "contact")
    mesh_names = set()
    for side, x in (("left", -arm_spacing / 2), ("right", arm_spacing / 2)):
        prefix = side + "_"
        arm = ET.fromstring(build_combined_mjcf(cfg=motor))
        for m in arm.find("asset"):
            if m.get("name") not in mesh_names:
                asset.append(m); mesh_names.add(m.get("name"))
        # link meshes: visual only
        for g in arm.iter("geom"):
            if g.get("type") == "mesh":
                g.set("contype", "0"); g.set("conaffinity", "0"); g.set("group", "1")
        for sec in ("actuator", "equality", "contact"):
            s = arm.find(sec)
            if s is not None:
                arm.remove(s)
        _prefix_tree(arm, prefix)
        base = arm.find("worldbody/body")
        base.set("pos", f"{x} 0 {arm_base_height}")
        base.set("quat", _fmt(_R_to_mjq(Rot.from_euler("x", np.radians(arm_base_pitch_deg)).as_matrix() @ Rot.from_euler("z", np.pi / 2).as_matrix())))
        gb = None
        for b in base.iter("body"):
            if b.get("name") == prefix + "gripper":
                gb = b
        _add_gripper_collision(gb, prefix, contact)
        wb.append(base)
        for i in range(6):
            ET.SubElement(act, "motor", {"name": f"{prefix}m{i+1}", "joint": f"{prefix}joint{i+1}", "gear": "1"})
        for j in ("joint7", "joint8"):
            ET.SubElement(act, "position", {"name": f"{prefix}g_{j}", "joint": prefix + j, "kp": str(contact.grip_kp), "kv": str(contact.grip_kv),
                                            "forcerange": f"-{contact.grip_force_max} {contact.grip_force_max}", "ctrlrange": "0 0.0475"})
        ET.SubElement(eq, "joint", {"joint1": prefix + "joint7", "joint2": prefix + "joint8", "polycoef": "0 1 0 0 0"})
        for b1, b2 in (("gripper", "tip_left"), ("gripper", "tip_right")):
            ET.SubElement(con, "exclude", {"body1": prefix + b1, "body2": prefix + b2})
    return ET.tostring(world, encoding="unicode")


class YamContactWorld:
    """Both arms + apple in one MuJoCo model, driven with the SDK motor law; physical gripper-apple contact."""

    def __init__(self, arm_spacing=0.61, plate_xy=(0.0, 0.38), containers=((-0.29, 0.0), (0.29, 0.0)), container_size=(0.15, 0.20, 0.03),
                 motor: YamMotorConfig | None = None, contact: ContactConfig | None = None, arm_base_height: float = 0.0, plate: bool = True, arm_base_pitch_deg: float = 0.0, plate_height: float = 0.0075, plate_radius: float = 0.10):
        self.motor = motor or YamMotorConfig(); self.contact = contact or ContactConfig(); self.has_plate = plate; self.plate_radius = plate_radius
        self.model = mujoco.MjModel.from_xml_string(build_world_mjcf(arm_spacing, plate_xy, plate_radius=plate_radius, containers=containers, container_size=container_size, motor=self.motor, contact=self.contact, arm_base_height=arm_base_height, plate=plate, arm_base_pitch_deg=arm_base_pitch_deg, plate_height=plate_height))
        self.data = mujoco.MjData(self.model)
        self.plate_xy = np.array(plate_xy); self.plate_height = plate_height if plate else 0.0
        self.containers = [(np.array([plate_xy[0] + dx, plate_xy[1] + dy]), np.array(container_size)) for dx, dy in containers]
        self.sides = ("left", "right")
        self.q_ids = {s: [self.model.joint(f"{s}_joint{i+1}").qposadr[0] for i in range(6)] for s in self.sides}
        self.v_ids = {s: [self.model.joint(f"{s}_joint{i+1}").dofadr[0] for i in range(6)] for s in self.sides}
        self.g_ids = {s: (self.model.joint(f"{s}_joint7").qposadr[0], self.model.joint(f"{s}_joint8").qposadr[0]) for s in self.sides}
        self.act_ids = {s: [self.model.actuator(f"{s}_m{i+1}").id for i in range(6)] for s in self.sides}
        self.gact_ids = {s: [self.model.actuator(f"{s}_g_joint7").id, self.model.actuator(f"{s}_g_joint8").id] for s in self.sides}
        self.stroke = float(self.model.joint("left_joint7").range[1])
        self.n_apples = max(1, int(self.contact.n_apples))
        names = ["apple" if k == 0 else f"apple{k}" for k in range(self.n_apples)]
        self.apple_qadrs = [self.model.joint(f"{n}_free").qposadr[0] for n in names]
        self.apple_geoms = [self.model.geom(f"{n}_geom").id for n in names]
        self.apple_qadr, self.apple_geom = self.apple_qadrs[0], self.apple_geoms[0]   # apple 0 (single-apple API)
        self.pad_geoms = {s: {self.model.geom(f"{s}_tip_left_pad").id: "left_pad", self.model.geom(f"{s}_tip_right_pad").id: "right_pad",
                              self.model.geom(f"{s}_tip_left_finger").id: "left_finger", self.model.geom(f"{s}_tip_right_finger").id: "right_finger",
                              self.model.geom(f"{s}_palm").id: "palm"} for s in self.sides}
        for s in self.sides:   # foam bumps count toward their pad
            for i in range(self.model.ngeom):
                nm = self.model.geom(i).name
                if nm.startswith(f"{s}_tip_left_foam"):
                    self.pad_geoms[s][i] = "left_pad"
                elif nm.startswith(f"{s}_tip_right_foam"):
                    self.pad_geoms[s][i] = "right_pad"
        self.gripper_body = {s: self.model.body(f"{s}_gripper").id for s in self.sides}
        # arm-only helper models for gravity compensation (the SDK computes it from the arm MJCF, no objects)
        self.comp = {s: YamArmDynamics(self.motor) for s in self.sides}
        self.target = {s: np.array([0, 0, 0, 0, 0, 0, 1.0]) for s in self.sides}
        self.t = 0.0

    # ---------------------------------------------------------------- state
    def reset(self, q7: dict, apple_pos, apple_quat=(1, 0, 0, 0)):
        mujoco.mj_resetData(self.model, self.data)
        for s in self.sides:
            q = np.asarray(q7[s], float)
            self.data.qpos[self.q_ids[s]] = q[:6]
            g = float(np.clip(q[6], 0, 1)) * self.stroke
            self.data.qpos[list(self.g_ids[s])] = g
            self.target[s] = q.copy()
            self.data.ctrl[self.gact_ids[s]] = g
        P = np.atleast_2d(np.asarray(apple_pos, float)); Q = np.atleast_2d(np.asarray(apple_quat, float))
        for k, a in enumerate(self.apple_qadrs):   # apples without a given pose are parked under the table edge, out of everyone's way
            self.data.qpos[a:a + 3] = P[k] if k < len(P) else (0.9 + 0.1 * k, -0.6, -0.66)
            self.data.qpos[a + 3:a + 7] = Q[k] if k < len(Q) else (1, 0, 0, 0)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0

    def move_plate(self, plate_xy):
        """Move the plate (and its rim ring) for per-episode plate jitter."""
        d = np.asarray(plate_xy, float) - self.plate_xy
        for name in ["plate"] + [f"plate_rim{k}" for k in range(24)]:
            try:
                gid = self.model.geom(name).id
            except KeyError:
                continue   # no plate / no rim in this world
            self.model.geom_pos[gid][:2] += d
        self.plate_xy = np.asarray(plate_xy, float).copy()

    def set_target(self, side: str, q7):
        self.target[side] = np.asarray(q7, float).copy()

    def step(self, dt: float = 1.0 / 30.0):
        n = max(1, int(round(dt / self.motor.motor_dt))); c = self.motor
        for _ in range(n):
            for s in self.sides:
                q = self.data.qpos[self.q_ids[s]]; v = self.data.qvel[self.v_ids[s]]
                comp = self.comp[s]; comp.data.qpos[comp.q_ids] = q
                g = comp.gravity_torques() * c.gravity_comp_factor
                tau = np.clip(c.kp * (self.target[s][:6] - q) - c.kd * v + g, -c.tau_max, c.tau_max)
                self.data.ctrl[self.act_ids[s]] = tau
                self.data.ctrl[self.gact_ids[s]] = float(np.clip(self.target[s][6], 0, 1)) * self.stroke
            mujoco.mj_step(self.model, self.data)
            self.t += c.motor_dt

    def arm_state(self, side: str) -> np.ndarray:
        g = float(np.mean(self.data.qpos[list(self.g_ids[side])]))
        return np.concatenate([self.data.qpos[self.q_ids[side]], [float(np.clip(g / self.stroke, 0, 1))]])

    def apple_pose(self, k: int = 0):
        a = self.apple_qadrs[k]
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def apple_in_gripper(self, side: str, k: int = 0) -> np.ndarray:
        b = self.gripper_body[side]
        R = self.data.xmat[b].reshape(3, 3); p = self.data.xpos[b]
        return R.T @ (self.apple_pose(k)[0] - p)

    def apple_contacts(self, side: str, k: int = 0) -> dict:
        """Which gripper parts touch apple k, with total normal force per part."""
        out = {}; ag = self.apple_geoms[k]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            other = g2 if g1 == ag else (g1 if g2 == ag else None)
            if other is None or other not in self.pad_geoms[side]:
                continue
            f = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, f)
            name = self.pad_geoms[side][other]
            out[name] = out.get(name, 0.0) + float(abs(f[0]))
        return out

    def apple_support(self, k: int = 0) -> str:
        """'plate' / 'container0' / 'container1' / 'table' / 'apple' / 'air': what apple k rests on. A container counts
        only when the apple centre is inside its footprint and it rests on the container floor/walls from inside or on
        another apple that does (not touching a container wall from outside). 'apple' = resting only on another apple."""
        names = []; ag = self.apple_geoms[k]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if ag in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == ag else con.geom1
                names.append("apple" if other in self.apple_geoms else self.model.geom(other).name)
        p = self.apple_pose(k)[0]
        for ci, (c, size) in enumerate(self.containers):
            inside = np.all(np.abs(p[:2] - c) < size[:2] / 2) and p[2] < size[2] + 3 * self.contact.apple_radius + 0.005
            if inside and any(n.startswith(f"container{ci}") or n == "apple" for n in names):
                return f"container{ci}"
        if any(n.startswith("plate") for n in names):
            return "plate"
        if any(n == "table" for n in names):
            return "table"
        if any(n == "apple" for n in names):
            return "apple"
        return "air"
