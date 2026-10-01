"""Per-episode two-arm interaction stats from run.npz (arm_q per tick, apple per tick, cmds).
Usage: python dualarm_stats.py <run_dir_with_ep*/ or single ep dir> ..."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yamsim.kinematics import YamArm, GripperCalibration
from yamsim.scene import tf
URDF = str(Path(__file__).resolve().parent.parent / "assets/yam/yam.urdf")
arms = {s: YamArm(URDF, GripperCalibration(6.351, 1.45), base_tf=tf(xyz=(x, 0, 0), rpy=(0, 0, np.pi/2))) for s, x in (("left", -0.305), ("right", 0.305))}
def pad(side, q6):
    P = arms[side].link_poses(np.r_[q6, 1.45])["gripper"]; return (P @ np.array([0, 0, -0.134, 1.0]))[:3]
def episode(d: Path):
    r = np.load(d / "run.npz", allow_pickle=True); q = r["arm_q"]; ap = r["apple"]; cm = r["cmds"]
    s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
    T = len(q); step = 3
    L = np.array([pad("left", q[t, :6]) for t in range(0, T, step)]); R = np.array([pad("right", q[t, 7:13]) for t in range(0, T, step)])
    A = ap[::step]
    dL = np.linalg.norm(L - A, axis=1); dR = np.linalg.norm(R - A, axis=1); dLR = np.linalg.norm(L - R, axis=1)
    both_near = (dL < 0.15) & (dR < 0.15)
    tg = s.get("t_first_grasp"); side = None
    for e in s.get("events", []):
        if e[2] == "grasp": side = e[1]; break
    other = {"left": dR, "right": dL}.get(side)
    return dict(ep=d.name, lifted=s.get("success"), grasp_t=tg, grasp_side=side,
                both_near_frac=float(both_near.mean()), both_near_s=float(both_near.sum() * step / 30),
                min_pad_dist_cm=float(dLR.min() * 100), crowd_s=float(((dLR < 0.10)).sum() * step / 30),
                other_arm_min_cm_before_grasp=(float(other[: int(tg * 30 / step)].min() * 100) if (tg and other is not None and int(tg*30/step) > 0) else None),
                first_near_L=float(np.argmax(dL < 0.15) * step / 30) if (dL < 0.15).any() else None,
                first_near_R=float(np.argmax(dR < 0.15) * step / 30) if (dR < 0.15).any() else None)
rows = []
for root in map(Path, sys.argv[1:]):
    eps = sorted(root.glob("ep*")) or [root]
    for d in eps:
        if (d / "run.npz").exists(): rows.append(episode(d))
print(f"{'ep':6s} lift grasp@  side   both<15cm(s) minLR(cm) crowd<10cm(s) otherArm@grasp(cm) firstNear L/R")
for r in rows:
    print(f"{r['ep']:6s} {str(r['lifted']):5s} {str(r['grasp_t']):6s} {str(r['grasp_side']):6s} {r['both_near_s']:6.1f}       {r['min_pad_dist_cm']:6.1f}    {r['crowd_s']:6.1f}        "
          f"{'-' if r['other_arm_min_cm_before_grasp'] is None else f"{r['other_arm_min_cm_before_grasp']:.0f}":>8s}   {r['first_near_L']} / {r['first_near_R']}")
if rows:
    print(f"\nN={len(rows)}: both arms within 15 cm of the apple at the same time: mean {np.mean([r['both_near_s'] for r in rows]):.1f} s/episode, "
          f"episodes with any such time {sum(r['both_near_s']>0 for r in rows)}; pads within 10 cm of each other: mean {np.mean([r['crowd_s'] for r in rows]):.1f} s; "
          f"min pad-pad distance median {np.median([r['min_pad_dist_cm'] for r in rows]):.0f} cm; grasp side counts {dict(zip(*np.unique([str(r['grasp_side']) for r in rows], return_counts=True)))}")
