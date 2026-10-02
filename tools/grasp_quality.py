"""Grasp-QUALITY scoring for saved sim episodes (finer-grained than lift / no lift).

Motivation (real rig, 2026-09-18): many failures were not missed grasps but POOR ones - the pads touched the
apple off-centre, so it slipped or was squeezed out during the carry. A confident grasp has its two contact
patches roughly diametrically opposite on the apple (the grip axis passes through the apple centre).

Method: no policy, no rendering. Every tick in which the apple is pinched (logged pad force > 1 N on both
pads) is re-posed in the MuJoCo contact world from the logged joint angles + apple pose; mj_forward gives the
actual pad/apple contact points. Per tick:
  antipodal error  = 180 deg - angle between (centre -> left-pad contact) and (centre -> right-pad contact);
                     0 = diametrically opposite. For parallel pads this is 2 x the contact-normal tilt, so the
                     pinch leaves the friction cone at about 2*atan(mu) (mu 0.8 foam -> ~77 deg).
  axis miss        = distance from the apple centre to the line through the two contact centroids (mm).
  p (gripper frame)= apple centre: x across the pad width, y along the closing axis, z along the fingers
                     (pad patch spans z -147..-120 mm; z < -147 = pinched beyond the finger tips).
Per hold (a run of pinched ticks, gaps <= 4 ticks merged): onset / median / worst antipodal error, in-hand slip
(motion of the apple in the gripper frame), pad force, lift, and how it ended: 'released' (the policy opened the
gripper) or 'lost' (the apple left the pads while the gripper command was still closed). Grades:
  confident: onset error <= 15 deg, worst <= 30 deg, slip <= 8 mm, not lost
  poor     : onset error > 40 deg, or lost, or slip > 20 mm
  marginal : everything else
Score per hold (0-100) = 100 * max(0, 1 - onset_err/60) * max(0, 1 - slip/25) * (0.5 if lost).

usage: python grasp_quality.py RUN_DIR [RUN_DIR ...] [--per-grasp] [--merge-shards] [--json OUT.json] [--min-hold 0.3]
       RUN_DIR contains ep*/run.npz + summary.json (run_policy_sim.py --dynamics mujoco_contact).
"""
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yamulator.sim.mjworld import YamContactWorld, PAD_Z  # noqa: E402

HZ = 30.0
PINCH_N = 1.0          # same threshold as ContactGraspSim ("pinched")
CONF = dict(onset=15.0, worst=30.0, slip=8.0)
POOR = dict(onset=40.0, slip=20.0)
OFF = {"left": 0, "right": 7}
PAD_HALF_W = 0.011     # pad box half width (x), mjworld ContactConfig.pad_half


class Reposer:
    """Static re-posing of the contact world: logged state in, pad/apple contact geometry out."""

    def __init__(self, arm_spacing=0.61, arm_height=0.0, base_pitch_deg=0.0, n_apples=1, apple_diameter=0.075):
        from yamulator.sim.mjworld import ContactConfig
        self.w = YamContactWorld(arm_spacing=arm_spacing, arm_base_height=arm_height, arm_base_pitch_deg=base_pitch_deg, contact=ContactConfig(n_apples=n_apples, apple_radius=apple_diameter / 2))
        self.r = float(self.w.contact.apple_radius)

    def contacts(self, q7, apple_pos, apple_quat, side, k=0):
        """apple_pos/quat: (3,)/(4,) or (N,3)/(N,4); k = the pinched apple.
        Returns (p_gripper_frame, {pad: contact centroid in the WORLD frame}, touching_other_parts)."""
        w = self.w
        w.reset(q7, apple_pos, apple_quat)
        acc, other_parts = {}, set(); ag = w.apple_geoms[k]
        for i in range(w.data.ncon):
            con = w.data.contact[i]
            g = con.geom2 if con.geom1 == ag else (con.geom1 if con.geom2 == ag else None)
            if g is None or g not in w.pad_geoms[side]:
                continue
            name = w.pad_geoms[side][g]
            if name not in ("left_pad", "right_pad"):
                other_parts.add(name); continue
            wt = max(-float(con.dist), 1e-5)   # deeper penetration = more of the load
            s = acc.setdefault(name, [np.zeros(3), 0.0]); s[0] += wt * np.asarray(con.pos); s[1] += wt
        cents = {k: v[0] / v[1] for k, v in acc.items()}
        return w.apple_in_gripper(side, k), cents, other_parts

    def geometric(self, p):
        """Fallback when the re-posed world shows no contact on a pad (finger joints are logged as their mean):
        nearest point of each pad patch to the apple centre, in the gripper frame -> same error definition."""
        cx = float(np.clip(p[0], -PAD_HALF_W, PAD_HALF_W)); cz = float(np.clip(p[2], PAD_Z[0], PAD_Z[1]))
        e = float(np.hypot(p[0] - cx, p[2] - cz))
        return 2.0 * float(np.degrees(np.arcsin(min(1.0, e / self.r)))), e


def antipodal(center, cl, cr):
    ul, ur = cl - center, cr - center
    ul, ur = ul / (np.linalg.norm(ul) + 1e-12), ur / (np.linalg.norm(ur) + 1e-12)
    err = 180.0 - float(np.degrees(np.arccos(np.clip(ul @ ur, -1, 1))))
    axis = cr - cl; axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = center - cl
    miss = float(np.linalg.norm(v - (v @ axis) * axis))
    return err, miss


def hold_segments(pinched, max_gap=4):
    idx = np.flatnonzero(pinched)
    if len(idx) == 0:
        return []
    segs, a, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            segs.append((a, prev)); a = i
        prev = i
    segs.append((a, prev))
    return segs


def score_episode(rp, ep_dir, min_hold_s=0.3):
    s = json.load(open(os.path.join(ep_dir, "summary.json")))
    z = np.load(os.path.join(ep_dir, "run.npz"), allow_pickle=True)
    if "pad_force" not in z.files or "arm_q" not in z.files or not np.any(z["pad_force"]):
        pass  # kinematic-apple runs have no pad forces -> no holds; handled by the empty result below
    q, apple, quat, cmds, pf, support = z["arm_q"], z["apple"], z["apple_quat"], z["cmds"], z["pad_force"], z["support"]
    T = min(len(q), len(apple), len(cmds), len(pf))
    multi = "apples" in z.files and z["apples"].ndim == 3 and len(z["apples"]) >= T     # --n-apples logs: every apple's pose + each side's pinched apple
    if multi:
        AP, AQ, PK, SUP = z["apples"], z["apples_quat"], z["pinch_apple"], z["supports"]
    gp = s.get("gap_probe", {})
    offs = gp.get("offsets_deg") or {"left": [0] * 6, "right": [0] * 6}
    gscale = float(gp.get("max_open_gap", 0.094)) / 0.094      # arm_q gripper = true open state * 0.094 / max_open_gap
    rest_z = float(apple[0, 2])

    def q7(t):
        return {sd: np.concatenate([q[t, OFF[sd]:OFF[sd] + 6] - np.radians(offs[sd]), [np.clip(q[t, OFF[sd] + 6] * gscale, 0, 1)]]) for sd in OFF}

    holds = []
    for k, side in enumerate(("left", "right")):
        for a, b in hold_segments(pf[:T, k] > PINCH_N):
            ticks = [t for t in range(a, b + 1) if pf[t, k] > PINCH_N]
            errs, misses, ps, n_geo, parts = [], [], [], 0, set()
            kk = int(np.bincount(PK[ticks, k].astype(int)).argmax()) if multi else 0      # the apple this hold is about
            for t in ticks:
                p, cents, other = rp.contacts(q7(t), AP[t], AQ[t], side, kk) if multi else rp.contacts(q7(t), apple[t], quat[t], side)
                parts |= other; ps.append(p)
                if "left_pad" in cents and "right_pad" in cents:
                    e, m = antipodal(AP[t, kk] if multi else apple[t], cents["left_pad"], cents["right_pad"])
                else:
                    e, m = rp.geometric(p); n_geo += 1
                errs.append(e); misses.append(m)
            errs, misses, ps = np.array(errs), np.array(misses), np.array(ps)
            n0 = min(5, len(errs))
            dur = (b - a + 1) / HZ
            # how it ended: did the policy open the gripper, or did the apple leave closed pads?
            gcmd = cmds[:T, OFF[side] + 6]
            w0, w1 = max(0, b - 3), min(T, b + 6)
            released = bool(np.min(gcmd[w0:w1]) < 0.25)
            after = str(SUP[min(T - 1, b + 30)][kk]) if multi else str(support[min(T - 1, b + 30)])
            ref = ps[min(n0, len(ps) - 1)]
            slip = float(np.linalg.norm(ps[-1] - ref)) * 1000 if len(ps) > n0 else 0.0
            lift = float(np.max(AP[a:b + 1, kk, 2]) - AP[0, kk, 2]) if multi else float(np.max(apple[a:b + 1, 2]) - rest_z)
            onset, med, worst = float(np.median(errs[:n0])), float(np.median(errs)), float(np.percentile(errs, 90))
            lost = not released
            if dur < min_hold_s:
                grade = "brief"
            elif onset > POOR["onset"] or lost or slip > POOR["slip"]:
                grade = "poor"
            elif onset <= CONF["onset"] and worst <= CONF["worst"] and slip <= CONF["slip"]:
                grade = "confident"
            else:
                grade = "marginal"
            score = 100.0 * max(0.0, 1 - onset / 60.0) * max(0.0, 1 - slip / 25.0) * (0.5 if lost else 1.0)
            holds.append({
                "side": side, "apple": kk, "t0": round(a / HZ, 2), "dur_s": round(dur, 2), "grade": grade, "score": round(score, 1),
                "antipodal_err_deg": {"onset": round(onset, 1), "median": round(med, 1), "p90": round(worst, 1)},
                "axis_miss_mm": round(float(np.median(misses[:n0])) * 1000, 1),
                "apple_in_gripper_mm": (np.median(ps[:n0], axis=0) * 1000).round(1).tolist(),
                "beyond_tips_mm": round(max(0.0, (PAD_Z[0] - float(np.median(ps[:n0, 2])))) * 1000, 1),
                "slip_mm": round(slip, 1), "pad_force_N": round(float(np.median(pf[ticks, k])), 1),
                "max_lift_cm": round(lift * 100, 1), "lifted": bool(lift > 0.05), "end": "released" if released else "lost",
                "support_after": after, "touching": sorted(parts), "geometric_ticks": n_geo, "ticks": len(ticks)})
    holds.sort(key=lambda h: h["t0"])
    real = [h for h in holds if h["grade"] != "brief"]
    return {"ep": os.path.basename(ep_dir), "lifted": bool(s.get("success")), "placed": s.get("placed_in") is not None,
            "n_apples": int(s.get("n_apples", 1)), "n_lifted": int(s.get("n_lifted", bool(s.get("success")))),
            "n_final_in": int(s.get("n_in_container_final", s.get("placed_in") is not None)),
            "holds": holds, "first_score": real[0]["score"] if real else 0.0,
            "best_score": max((h["score"] for h in real), default=0.0),
            "lift_grade": next((h["grade"] for h in real if h["lifted"]), None)}


def summarize(name, eps):
    n = len(eps)
    real = [h for e in eps for h in e["holds"] if h["grade"] != "brief"]
    brief = sum(h["grade"] == "brief" for e in eps for h in e["holds"])
    g = {k: sum(h["grade"] == k for h in real) for k in ("confident", "marginal", "poor")}
    on = np.array([h["antipodal_err_deg"]["onset"] for h in real]) if real else np.zeros(1)
    sl = np.array([h["slip_mm"] for h in real]) if real else np.zeros(1)
    lost = sum(h["end"] == "lost" for h in real)
    lost_air = sum(h["end"] == "lost" and h["lifted"] for h in real)
    conf_lift = sum(e["lift_grade"] == "confident" for e in eps)
    row = {"apples": sum(e.get("n_apples", 1) for e in eps), "apples_lifted": sum(e.get("n_lifted", 0) for e in eps), "apples_final_in": sum(e.get("n_final_in", 0) for e in eps),
           "run": name, "episodes": n, "lifts": sum(e["lifted"] for e in eps), "placed": sum(e["placed"] for e in eps),
           "confident_lifts": conf_lift, "holds": len(real), "brief_touches": brief, **g,
           "antipodal_onset_med": round(float(np.median(on)), 1), "antipodal_onset_p90": round(float(np.percentile(on, 90)), 1),
           "slip_med_mm": round(float(np.median(sl)), 1), "lost": lost, "dropped_after_lift": lost_air,
           "first_score_mean": round(float(np.mean([e["first_score"] for e in eps])), 1),
           "best_score_mean": round(float(np.mean([e["best_score"] for e in eps])), 1)}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+"); ap.add_argument("--per-grasp", action="store_true")
    ap.add_argument("--merge-shards", action="store_true", help="pool run dirs that differ only by a _s<N> suffix")
    ap.add_argument("--json", default=None); ap.add_argument("--min-hold", type=float, default=0.3)
    args = ap.parse_args()
    rows, out, reposers = [], {}, {}
    for run in args.runs:
        if not os.path.isdir(run):   # globs like runs/batch/CONF_* also match the .log files
            continue
        eps_dirs = sorted(glob.glob(os.path.join(run, "ep[0-9]*")))
        eps_dirs = [e for e in eps_dirs if os.path.exists(os.path.join(e, "run.npz")) and os.path.exists(os.path.join(e, "summary.json"))]
        if not eps_dirs:
            print(f"{run}: no episodes", file=sys.stderr); continue
        appr = json.load(open(os.path.join(eps_dirs[0], "summary.json"))).get("appearance", {})
        s0 = json.load(open(os.path.join(eps_dirs[0], "summary.json")))
        key = (float(appr.get("arm_spacing", 0.61)), float(appr.get("arm_height", 0.0)), float(appr.get("base_pitch_deg", 0.0)), int(s0.get("n_apples", 1)), float(s0.get("gap_probe", {}).get("apple_diameter", 0.075)))
        eps = []
        for e in eps_dirs:   # the number of apples can differ per episode (a crowded plate may not fit the last one): size the world per episode
            se = json.load(open(os.path.join(e, "summary.json")))
            ke = key[:3] + (int(se.get("n_apples", 1)), key[4])
            if ke not in reposers:
                reposers[ke] = Reposer(*ke)
            eps.append(score_episode(reposers[ke], e, args.min_hold))
        name = os.path.basename(os.path.normpath(run))
        if args.merge_shards:   # CONF_R4K_off0_s0..s4 -> one row "CONF_R4K_off0"
            name = re.sub(r"_s\d+$", "", name)
        out.setdefault(name, []).extend([dict(e, ep=f"{os.path.basename(os.path.normpath(run))}/{e['ep']}") for e in eps] if args.merge_shards else eps)
        if args.per_grasp:
            for e in eps:
                for h in e["holds"]:
                    a = h["antipodal_err_deg"]
                    print(f"  {name} {e['ep']} {h['side'][0].upper()} t={h['t0']:5.1f}s {h['dur_s']:4.1f}s {h['grade']:9s} score {h['score']:5.1f} | antipodal err onset {a['onset']:5.1f} med {a['median']:5.1f} p90 {a['p90']:5.1f} deg, "
                          f"axis miss {h['axis_miss_mm']:4.1f} mm, beyond tips {h['beyond_tips_mm']:4.1f} mm, slip {h['slip_mm']:5.1f} mm, {h['pad_force_N']:4.1f} N, lift {h['max_lift_cm']:4.1f} cm, {h['end']} -> {h['support_after']}"
                          + (f" [touching {','.join(h['touching'])}]" if h["touching"] else ""))
    rows = [summarize(name, eps) for name, eps in out.items()]
    hdr = f"{'run':34s} {'eps':>3s} {'lift':>4s} {'plcd':>4s} {'CONF-lift':>9s} | {'holds':>5s} {'conf':>4s} {'marg':>4s} {'poor':>4s} {'brief':>5s} | {'antipodal err onset med/p90':>27s} | {'slip med':>8s} | {'lost':>4s} {'drop':>4s} | {'first':>5s} {'best':>5s}"
    print(hdr)
    for r in rows:
        if r["apples"] > r["episodes"]:
            print(f"{'  (multi-apple) ' + r['run'][:18]:34s} apples {r['apples']}, lifted {r['apples_lifted']}, in a container at the end {r['apples_final_in']}")
        print(f"{r['run'][:34]:34s} {r['episodes']:3d} {r['lifts']:4d} {r['placed']:4d} {r['confident_lifts']:9d} | {r['holds']:5d} {r['confident']:4d} {r['marginal']:4d} {r['poor']:4d} {r['brief_touches']:5d} | "
              f"{r['antipodal_onset_med']:12.1f} / {r['antipodal_onset_p90']:5.1f} deg    | {r['slip_med_mm']:5.1f} mm | {r['lost']:4d} {r['dropped_after_lift']:4d} | {r['first_score_mean']:5.1f} {r['best_score_mean']:5.1f}")
    if args.json:
        json.dump({"summary": rows, "episodes": out}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
