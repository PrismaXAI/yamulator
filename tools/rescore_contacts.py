"""Re-score saved episodes with a precise finger-pad contact test, by deterministic replay of the logged
commands through the same arm dynamics + contact model (no policy, no rendering).

Pad geometry measured from tip_left.stl / tip_right.stl in the gripper frame (fingers along -z):
  contact faces at y = +-(d - 4 mm)   (the tip faces sit 4 mm inside the finger frames: fingers touch at d = 4 mm)
  contact patch: z in [-147, -120] mm; half-width 7.7 mm at the tip end, widening to 13.7 mm at z = -127 mm.
A sphere pinched between two such pads is held securely when its centre projects inside the patch (contact
normals pass through the centre). If the projection falls outside by e, the pads touch the sphere on the
patch edge and the contact normal tilts by phi = asin(e / r): the pinch holds only while phi is inside the
friction cone, tan(phi) <= mu. Classification per grasp: secure (e = 0), marginal (phi <= 20 deg, holds for
mu >= 0.36), fail (phi > 20 deg: the apple squirts out / is pushed down by the pads).
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yamulator.sim.scene import YamStation, StationConfig, tf
from yamulator.sim.physics import AppleGraspSim

PAD_Z = (-0.147, -0.120)
PAD_FACE_OFFSET = 0.004
PHI_MARGINAL_DEG = 20.0


def pad_half_width(z):
    """Half-width of the tip contact patch at height z (gripper frame), from the mesh."""
    return float(np.interp(z, [-0.147, -0.140, -0.127, -0.120], [0.0077, 0.0077, 0.0137, 0.0137]))


def contact_geometry(p, r):
    """p = apple centre in the gripper frame. Returns (e, ex, ez, phi_deg): distance from the centre's
    projection to the pad patch (0 if inside), its x and z components, and the contact-normal tilt."""
    px, py, pz = p
    z_lo, z_hi = PAD_Z
    dz = 0.0 if z_lo <= pz <= z_hi else (z_lo - pz if pz < z_lo else pz - z_hi)
    w = pad_half_width(np.clip(pz, z_lo, z_hi))
    dx = max(0.0, abs(px) - w)
    e = float(np.hypot(dx, dz))
    phi = float(np.degrees(np.arcsin(min(1.0, e / r))))
    return e, dx, dz, phi


def split(a14):
    return (np.concatenate([a14[0:6], [np.clip(1.0 - a14[6], 0, 1)]]),
            np.concatenate([a14[7:13], [np.clip(1.0 - a14[13], 0, 1)]]))


class Replayer:
    def __init__(self, dynamics="mujoco", plate_xy=(0.0, 0.38)):
        from yamulator.run_policy_sim import SimArms, REST_POSE
        self.REST = REST_POSE
        self.st = YamStation(StationConfig(plate_xy=tuple(plate_xy)), log_level=0)
        self.sim = AppleGraspSim(self.st)
        self.arms = SimArms(self.st, self.sim, dynamics=dynamics)
        self.plate0 = np.array(plate_xy)
        self.r = self.sim.r

    def replay(self, ep_dir):
        s = json.load(open(os.path.join(ep_dir, "summary.json")))
        z = np.load(os.path.join(ep_dir, "run.npz"))
        var = s["variation"]
        plate_xy = self.plate0 + np.array(var["plate_jitter"])
        self.st.cfg.plate_xy = tuple(plate_xy); self.sim.plate_xy = plate_xy
        self.st.set_object_pose("plate", tf(xyz=(plate_xy[0], plate_xy[1], 0.0)))
        self.sim.reset(apple_xy=plate_xy + np.array(var["apple_offset"]), apple_yaw=var["apple_yaw"])
        self.arms.reset()
        last = np.concatenate([self.REST, [0.0], self.REST, [0.0]])
        info = self.arms.apply(*split(last), instant=True)
        recs = []
        for t, cmd in enumerate(z["cmds"]):
            info = self.arms.apply(*split(cmd))
            side = info["grasped_by"]
            rec = {"tick": t, "grasped_by": side}
            for sd in ("left", "right"):
                p = info[f"{sd}_apple_in_gripper_frame"]
                rec[sd] = (p, info[f"{sd}_gap"])
            recs.append(rec)
        ev = [(e.step, e.side, e.kind) for e in self.sim.events if e.kind != "push"]
        logged = [tuple(e) for e in s["events"]]
        return s, recs, ev, logged, z


def score_episode(rep, ep_dir):
    s, recs, ev, logged, z = rep.replay(ep_dir)
    fidelity = ev[: len(logged)] == logged[: len(ev)] and len(ev) == len(logged)
    # grasp events: contact geometry at the grasp tick (the apple is rigid to the gripper afterwards)
    grasps = []
    for step, side, kind in ev:
        if kind != "grasp":
            continue
        p, gap = recs[step][side]
        e, dx, dz, phi = contact_geometry(p, rep.r)
        cls = "secure" if e == 0 else ("marginal" if phi <= PHI_MARGINAL_DEG else "fail")
        # did this grasp lead to the logged lift?
        rel = [st for st, sd, k in ev if k == "release" and st > step]
        lift = [st for st, sd, k in ev if k == "lift" and st > step and (not rel or st < rel[0])]
        grasps.append({"tick": step, "side": side, "p_mm": (p * 1000).round(1).tolist(), "gap_mm": round(gap * 1000, 1), "e_mm": round(e * 1000, 1),
                       "dx_mm": round(dx * 1000, 1), "dz_mm": round(dz * 1000, 1), "phi_deg": round(phi, 1), "class": cls, "lifted": bool(lift)})
    strict_success = any(g["lifted"] and g["class"] != "fail" for g in grasps)
    secure_success = any(g["lifted"] and g["class"] == "secure" for g in grasps)
    return {"ep": os.path.basename(ep_dir), "fidelity": fidelity, "loose_success": s["success"], "strict_success": strict_success,
            "secure_success": secure_success, "grasps": grasps, "n_events_replay": len(ev), "n_events_logged": len(logged)}


def main():
    root = sys.argv[1]
    dynamics = "lag" if "lag" in root else "mujoco"
    rep = Replayer(dynamics=dynamics)
    out = {}
    for run in sorted(glob.glob(os.path.join(root, "*", "aggregate.json"))):
        name = os.path.basename(os.path.dirname(run))
        eps = sorted(glob.glob(os.path.join(root, name, "ep*")))
        res = [score_episode(rep, e) for e in eps]
        out[name] = res
        n = len(res); fid = sum(r["fidelity"] for r in res)
        loose = sum(r["loose_success"] for r in res); strict = sum(r["strict_success"] for r in res); secure = sum(r["secure_success"] for r in res)
        lifting = [g for r in res for g in r["grasps"] if g["lifted"]]
        cls = {c: sum(g["class"] == c for g in lifting) for c in ("secure", "marginal", "fail")}
        ex = np.array([g["dx_mm"] for g in lifting]); ez = np.array([g["dz_mm"] for g in lifting]); pz = np.array([g["p_mm"][2] for g in lifting])
        print(f"{name:20s} n={n} replay-fidelity {fid}/{n} | loose lift {loose}/{n} -> strict {strict}/{n}, secure-only {secure}/{n} | "
              f"lifting grasps: {cls} | offset beyond pad: x median {np.median(ex) if len(ex) else 0:.0f} mm, z median {np.median(ez) if len(ez) else 0:.0f} mm; "
              f"apple centre z vs pad [-147,-120]: median {np.median(pz) if len(pz) else 0:.0f} mm")
    json.dump(out, open(os.path.join(root, "rescore.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
