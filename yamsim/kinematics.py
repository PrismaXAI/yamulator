"""Minimal URDF loader + forward kinematics for the I2RT YAM arm.

Joint/actuator conventions follow the i2rt SDK: 6 revolute joints in radians (joint1..joint6)
plus a gripper motor angle. The gripper motor angle is mapped linearly onto a normalized
open state g in [0, 1] via (closed_rad, open_rad) -> (0, 1), and g drives the two prismatic
finger joints (joint7/joint8) which each travel 0 .. FINGER_STROKE (from the URDF).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_tf(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)  # list of (mesh_path, origin 4x4)


class URDF:
    def __init__(self, path: str):
        self.path = path
        root = ET.parse(path).getroot()
        base = os.path.dirname(path)
        self.links: dict[str, Link] = {}
        self.joints: dict[str, Joint] = {}
        for l in root.findall("link"):
            link = Link(l.get("name"))
            for v in l.findall("visual"):
                o = v.find("origin")
                xyz = [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
                rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
                m = v.find("geometry/mesh")
                if m is not None:
                    link.visuals.append((os.path.join(base, m.get("filename")), make_tf(xyz, rpy)))
            self.links[link.name] = link
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            lim = j.find("limit")
            lower = float(lim.get("lower", 0)) if lim is not None else 0.0
            upper = float(lim.get("upper", 0)) if lim is not None else 0.0
            self.joints[j.get("name")] = Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                                               j.find("child").get("link"), make_tf(xyz, rpy), axis, lower, upper)
        self.child_joints: dict[str, list[Joint]] = {}
        for jt in self.joints.values():
            self.child_joints.setdefault(jt.parent, []).append(jt)
        children = {jt.child for jt in self.joints.values()}
        self.root_link = next(n for n in self.links if n not in children)

    def link_transforms(self, q: dict[str, float], base_tf: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """World transform of every link frame given joint values (missing joints = 0)."""
        tfs = {self.root_link: np.eye(4) if base_tf is None else base_tf.copy()}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for jt in self.child_joints.get(parent, []):
                T = tfs[parent] @ jt.origin
                val = float(q.get(jt.name, 0.0))
                if jt.jtype == "revolute" or jt.jtype == "continuous":
                    T = T @ axis_angle(jt.axis, val)
                elif jt.jtype == "prismatic":
                    P = np.eye(4)
                    P[:3, 3] = jt.axis / np.linalg.norm(jt.axis) * val
                    T = T @ P
                tfs[jt.child] = T
                stack.append(jt.child)
        return tfs


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ("joint7", "joint8")


@dataclass
class GripperCalibration:
    """Linear map from gripper motor radians to normalized open state: closed_rad -> 0, open_rad -> 1."""
    closed_rad: float
    open_rad: float

    def to_open_state(self, rad: float) -> float:
        g = (rad - self.closed_rad) / (self.open_rad - self.closed_rad)
        return float(np.clip(g, 0.0, 1.0))

    def to_rad(self, g: float) -> float:
        return self.closed_rad + float(np.clip(g, 0, 1)) * (self.open_rad - self.closed_rad)


# Gripper motor conventions. The LeRobot training data stores the gripper in the leader/teleop
# radian convention (closed ~6.35, open ~1.45); the physical followers on our station were
# calibrated (GRIP_CAL in prism-pi's dryrun_inference.py) to closed ~0.01 / open ~-5.09 rad.
# i2rt's linear_4310.yml: motor_stroke 6.57 rad <-> gripper_stroke 0.096 m (linear).
TRAIN_GRIPPER_CAL = {"left": GripperCalibration(6.351, 1.45), "right": GripperCalibration(6.392, 1.34)}
STATION_GRIPPER_CAL = {"left": GripperCalibration(0.011, -5.094), "right": GripperCalibration(0.060, -5.041)}


class YamArm:
    """One YAM arm: FK from a 7-vector [q1..q6 (rad), gripper (rad)] to link poses."""

    def __init__(self, urdf_path: str, gripper_cal: GripperCalibration, base_tf: np.ndarray | None = None):
        self.urdf = URDF(urdf_path)
        self.gripper_cal = gripper_cal
        self.base_tf = np.eye(4) if base_tf is None else base_tf
        j7 = self.urdf.joints["joint7"]
        self.finger_stroke = abs(j7.upper - j7.lower)  # 0.04695 m per finger in the yam v1 URDF
        self.joint_limits = np.array([[self.urdf.joints[j].lower, self.urdf.joints[j].upper] for j in ARM_JOINTS])

    def joint_dict(self, action7: np.ndarray) -> dict[str, float]:
        action7 = np.asarray(action7, dtype=float)
        q = {n: float(v) for n, v in zip(ARM_JOINTS, action7[:6])}
        g = self.gripper_cal.to_open_state(action7[6]) if len(action7) > 6 else 1.0
        # Finger joints: 0 = closed (pad faces touching at y = 0), positive travel opens. In yam.urdf joint7's
        # axis is +y with limits [-stroke, 0] and joint8's is -y: the left finger (whose pad sits on the -y side
        # of its face) opens toward -y, so both joints take -d. (The SDK's MJCF uses the same convention with a
        # rotated tip frame and range [0, stroke].) Gap between the pad faces = 2 d.
        d = g * self.finger_stroke
        q["joint7"] = -d
        q["joint8"] = -d
        return q

    def link_poses(self, action7: np.ndarray) -> dict[str, np.ndarray]:
        return self.urdf.link_transforms(self.joint_dict(action7), self.base_tf)

    def gripper_open_state(self, action7: np.ndarray) -> float:
        return self.gripper_cal.to_open_state(float(action7[6]))


# ---------------------------------------------------------------------------- inverse kinematics
def rotation_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector taking R_cur to R_tgt (world frame)."""
    R = R_tgt @ R_cur.T
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax * ang


def solve_ik(arm: "YamArm", target_T: np.ndarray, q0: np.ndarray, link: str = "gripper", pos_weight: float = 1.0,
             rot_weight: float = 0.3, iters: int = 150, damping: float = 0.05, tol: float = 1e-4, mode: str = "full",
             restarts: int = 8, seed: int = 0) -> tuple[np.ndarray, float]:
    """Damped least-squares IK on the 6 arm joints (finite-difference Jacobian).

    mode="full": match position + full orientation of target_T.
    mode="axis": match position and align the link's -z axis (finger direction) with target_T's -z; the
                 rotation about that axis is left free (well suited to top-down grasps).
    Runs from q0 plus `restarts` random initial configurations and returns the best (q6, position error m).
    """
    lim = arm.joint_limits
    grip = float(q0[6]) if len(q0) > 6 else arm.gripper_cal.open_rad
    rng = np.random.default_rng(seed)

    def fk(qq):
        return arm.link_poses(np.concatenate([qq, [grip]]))[link]

    def residual(T):
        e_pos = target_T[:3, 3] - T[:3, 3]
        if mode == "full":
            e_rot = rotation_error(T[:3, :3], target_T[:3, :3])
        else:
            a, b = -T[:3, 2], -target_T[:3, 2]
            e_rot = np.cross(a, b)  # small-angle rotation vector taking a toward b
            if np.dot(a, b) < 0:   # anti-parallel: push consistently
                e_rot = e_rot + 1e-3 if np.linalg.norm(e_rot) < 1e-6 else e_rot * (np.pi / max(np.linalg.norm(e_rot), 1e-9))
        return np.concatenate([pos_weight * e_pos, rot_weight * e_rot]), np.linalg.norm(e_pos), np.linalg.norm(e_rot)

    def run(q):
        for _ in range(iters):
            T = fk(q)
            err, ep, er = residual(T)
            if ep < tol and er < 10 * tol:
                break
            J = np.zeros((6, 6)); h = 1e-5
            for j in range(6):
                dq = q.copy(); dq[j] += h
                J[:, j] = (residual(fk(dq))[0] - err) / -h  # d(residual)/dq = -(r(q+h)-r(q))/h since residual = target - f(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.25, 0.25), lim[:, 0] + 1e-3, lim[:, 1] - 1e-3)
        err, ep, er = residual(fk(q))
        return q, ep, er

    best = None
    starts = [np.array(q0[:6], dtype=float)]
    for _ in range(restarts):
        starts.append(lim[:, 0] + rng.random(6) * (lim[:, 1] - lim[:, 0]))
    for qs in starts:
        q, ep, er = run(qs)
        score = ep + 0.05 * er
        if best is None or score < best[0]:
            best = (score, q, ep)
        if ep < tol and er < 10 * tol:
            break
    return best[1], float(best[2])


def top_down_grasp_frame(point_world: np.ndarray, yaw: float = 0.0, standoff: float = 0.0) -> np.ndarray:
    """Gripper-link target pose for a top-down grasp at `point_world`.

    In the yam.urdf gripper link frame the fingers extend along -z (tips ~0.147 m below the link origin)
    and open along +/-y. The returned frame has link +z = world +z (fingers pointing down) and, at yaw=0,
    the finger-opening axis along world x. `standoff` lifts the frame along world +z.
    """
    T = np.eye(4)
    Rz = rpy_to_matrix(0, 0, yaw)
    R = Rz @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)  # columns: link x, y, z in world
    T[:3, :3] = R
    T[:3, 3] = point_world + np.array([0, 0, standoff])
    return T


FINGER_TIP_OFFSET = 0.147  # m, from the gripper link origin down (-z) to the finger-tip end (from the tip meshes)
