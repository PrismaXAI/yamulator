"""MuJoCo dynamics of one YAM arm driven exactly like the i2rt SDK drives the real motors.

Real rig (i2rt MotorChainRobot + DaMiao motors in MIT mode): every motor runs an onboard loop
    tau = kp * (p_des - q) + kd * (0 - qdot) + tau_ff
with tau_ff = gravity compensation (MuJoCo inverse dynamics x per-joint factor) [+ Coulomb feed-forward],
clipped to the motor's torque range (DM4340: 28 Nm on joints 1-3, DM4310: 10 Nm on joints 4-6 and the
gripper). p_des is whatever the executor last sent (30 Hz, zero-order hold). Gains from
i2rt/robots/config/yam_v1.yml: kp 80/80/80/10/10/10, kd 5/5/5/1.5/1.5/1.5, gravity_comp_factor
1.0/1.1/1.1/1.2/1.0/1.0, coulomb 0.3/0.3/0.3/0.06/0.06/0.06 Nm; gripper (linear_4310.yml) kp 20, kd 0.5,
motor_stroke 6.57 rad <-> gripper_stroke 0.096 m.

The model is the SDK's own MJCF (yam.xml + linear_4310.xml merged the way i2rt.robots.utils does it),
with joint armature/frictionloss added for the geared actuators (parameters, not measured). Contacts are
disabled: the apple interaction is handled by the kinematic contact model in physics.py.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass, field

import mujoco
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_XML = os.path.join(_ROOT, "assets", "yam", "yam.xml")
GRIPPER_XML = os.path.join(_ROOT, "assets", "yam", "gripper", "linear_4310.xml")


@dataclass
class YamMotorConfig:
    kp: np.ndarray = field(default_factory=lambda: np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0]))
    kd: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5]))
    tau_max: np.ndarray = field(default_factory=lambda: np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0]))
    gravity_comp_factor: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0]))
    coulomb: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06]))
    use_coulomb_ff: bool = False          # SDK default: use_coulomb_friction=False
    armature: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.005, 0.005, 0.005]))  # reflected rotor inertia, kg m^2 (assumed)
    damping: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05]))     # viscous, Nm s/rad (assumed)
    # gripper: the DM4310 + leadscrew stage is fast and stiff relative to the arm; modelled kinematically
    # as a rate-limited first-order lag on the finger travel (the fingers are driven, not simulated).
    gripper_tau_s: float = 0.05           # first-order time constant
    gripper_max_speed: float = 0.20       # m/s of finger travel (full 94 mm stroke in ~0.5 s)
    motor_dt: float = 0.001               # onboard MIT loop / sim timestep
    command_latency_s: float = 0.0        # extra CAN/processing delay on p_des (0 = none)


def build_combined_mjcf(arm_xml: str = ARM_XML, gripper_xml: str = GRIPPER_XML, cfg: YamMotorConfig | None = None) -> str:
    """Merge arm + gripper like i2rt.robots.utils (gripper children go into the arm's mount body)."""
    cfg = cfg or YamMotorConfig()
    arm = ET.parse(arm_xml).getroot()
    grip = ET.parse(gripper_xml).getroot()
    # absolute mesh paths
    for root, path in ((arm, arm_xml), (grip, gripper_xml)):
        comp = root.find("compiler"); meshdir = comp.get("meshdir", "") if comp is not None else ""
        for m in root.findall("asset/mesh"):
            m.set("file", os.path.abspath(os.path.join(os.path.dirname(path), meshdir, m.get("file"))))
        if comp is not None and "meshdir" in comp.attrib:
            del comp.attrib["meshdir"]
    arm_asset = arm.find("asset")
    have = {(c.tag, c.get("name")) for c in arm_asset}
    for c in grip.find("asset"):
        if (c.tag, c.get("name")) not in have:
            arm_asset.append(deepcopy(c))
    # deepest body of the arm = mount ("gripper")
    body = arm.find("worldbody/body")
    while body.find("body") is not None:
        body = body.find("body")
    assert body.get("name") == "gripper"
    gbody = grip.find(".//body[@name='gripper']")
    placeholder = body.find("inertial")
    if placeholder is not None:
        body.remove(placeholder)
    body.insert(0, deepcopy(gbody.find("inertial")))
    for child in gbody:
        if child.tag != "inertial":
            body.append(deepcopy(child))
    for sec in ("equality", "contact"):
        gs = grip.find(sec)
        if gs is not None:
            a = arm.find(sec)
            if a is None:
                a = ET.SubElement(arm, sec)
            for c in gs:
                a.append(deepcopy(c))
    # joint armature / damping / frictionloss for the arm joints; motor actuators
    joints = {j.get("name"): j for j in arm.iter("joint")}
    for i in range(6):
        j = joints[f"joint{i+1}"]
        j.set("armature", str(cfg.armature[i])); j.set("damping", str(cfg.damping[i])); j.set("frictionloss", str(cfg.coulomb[i]))
        # i2rt's MJCF carries a placeholder actuatorfrcrange of +-10 Nm on every joint; use the motor limits instead
        j.set("actuatorfrcrange", f"-{cfg.tau_max[i]} {cfg.tau_max[i]}")
    joints["joint7"].set("damping", "5.0"); joints["joint7"].set("frictionloss", "2.0")   # linear stage friction (N)
    joints["joint8"].set("damping", "5.0"); joints["joint8"].set("frictionloss", "2.0")
    act = ET.SubElement(arm, "actuator")
    for i in range(6):
        ET.SubElement(act, "motor", {"name": f"m{i+1}", "joint": f"joint{i+1}", "gear": "1", "ctrllimited": "false"})
    ET.SubElement(act, "motor", {"name": "m_grip", "joint": "joint7", "gear": "1", "ctrllimited": "false"})
    opt = arm.find("option")
    if opt is None:
        opt = ET.SubElement(arm, "option")
    opt.set("timestep", str(cfg.motor_dt)); opt.set("integrator", "implicitfast"); opt.set("gravity", "0 0 -9.81")
    ET.SubElement(opt, "flag", {"contact": "disable"})
    return ET.tostring(arm, encoding="unicode")


class YamArmDynamics:
    """One arm. Command interface mirrors the executor: target 7-vector [q1..q6 rad, gripper open 0..1]."""

    def __init__(self, cfg: YamMotorConfig | None = None):
        self.cfg = cfg or YamMotorConfig()
        self.model = mujoco.MjModel.from_xml_string(build_combined_mjcf(cfg=self.cfg))
        self.data = mujoco.MjData(self.model)
        self.q_ids = [self.model.joint(f"joint{i+1}").qposadr[0] for i in range(6)]
        self.v_ids = [self.model.joint(f"joint{i+1}").dofadr[0] for i in range(6)]
        self.g_qid = self.model.joint("joint7").qposadr[0]; self.g_vid = self.model.joint("joint7").dofadr[0]
        self.stroke = float(self.model.joint("joint7").range[1])
        self._gdata = mujoco.MjData(self.model)   # scratch for gravity comp
        self.target = np.zeros(7); self.target[6] = 1.0
        self._delay_buf: list[tuple[float, np.ndarray]] = []
        self.t = 0.0
        self.last_tau = np.zeros(7)
        self.reset(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    def reset(self, q7: np.ndarray):
        q7 = np.asarray(q7, float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q_ids] = q7[:6]
        g = float(np.clip(q7[6], 0, 1)) * self.stroke
        self.data.qpos[self.g_qid] = g; self.data.qpos[self.model.joint("joint8").qposadr[0]] = g
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.target = q7.copy(); self._delay_buf = []; self.t = 0.0

    def gravity_torques(self) -> np.ndarray:
        """SDK gravity comp: inverse dynamics of the current pose at zero velocity/acceleration."""
        d = self._gdata
        d.qpos[:] = self.data.qpos; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse[self.v_ids].copy()

    def set_target(self, q7: np.ndarray):
        q7 = np.asarray(q7, float).copy()
        if self.cfg.command_latency_s > 0:
            self._delay_buf.append((self.t + self.cfg.command_latency_s, q7))
        else:
            self.target = q7

    def step(self, dt: float = 1.0 / 30.0):
        n = max(1, int(round(dt / self.cfg.motor_dt)))
        c = self.cfg
        for _ in range(n):
            while self._delay_buf and self._delay_buf[0][0] <= self.t:
                self.target = self._delay_buf.pop(0)[1]
            q = self.data.qpos[self.q_ids]; v = self.data.qvel[self.v_ids]
            g = self.gravity_torques() * c.gravity_comp_factor
            tau = c.kp * (self.target[:6] - q) - c.kd * v + g
            if c.use_coulomb_ff:
                tau += c.coulomb * np.sign(v)
            tau = np.clip(tau, -c.tau_max, c.tau_max)
            # gripper: kinematic rate-limited lag on the finger travel
            gq = self.data.qpos[self.g_qid]
            dg = (self.target[6] * self.stroke - gq) * (c.motor_dt / (c.motor_dt + c.gripper_tau_s))
            dg = float(np.clip(dg, -c.gripper_max_speed * c.motor_dt, c.gripper_max_speed * c.motor_dt))
            gq = float(np.clip(gq + dg, 0.0, self.stroke))
            self.data.qpos[self.g_qid] = gq; self.data.qpos[self.model.joint("joint8").qposadr[0]] = gq
            self.data.qvel[self.g_vid] = 0.0; self.data.qvel[self.model.joint("joint8").dofadr[0]] = 0.0
            self.data.ctrl[:6] = tau; self.data.ctrl[6] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.t += c.motor_dt
        self.last_tau = np.concatenate([tau, [0.0]])

    def state(self) -> np.ndarray:
        """[q1..q6, gripper open state 0..1] as measured."""
        return np.concatenate([self.data.qpos[self.q_ids], [float(np.clip(self.data.qpos[self.g_qid] / self.stroke, 0, 1))]])

    def velocity(self) -> np.ndarray:
        return self.data.qvel[self.v_ids].copy()
