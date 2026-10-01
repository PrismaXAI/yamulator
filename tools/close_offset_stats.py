"""Where do the gripper closes land relative to the apple? For every close onset (close cmd crosses 0.5) report the
pad-centre offset from the apple centre at that tick: forward (+ = beyond the apple, along the reach direction y),
lateral (x) and height (z, pad centre minus apple centre). Aggregates per run directory (ep*/run.npz + summary.json).
Usage: python close_offset_stats.py <run_dir> [<run_dir> ...]"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yamsim.kinematics import YamArm, GripperCalibration
from yamsim.scene import tf
URDF = str(Path(__file__).resolve().parent.parent / "assets/yam/yam.urdf")
arms = {s: YamArm(URDF, GripperCalibration(6.351, 1.45), base_tf=tf(xyz=(x, 0, 0), rpy=(0, 0, np.pi / 2))) for s, x in (("left", -0.305), ("right", 0.305))}
def pad(side, q6):
    P = arms[side].link_poses(np.r_[q6, 1.45])["gripper"]; return (P @ np.array([0, 0, -0.134, 1.0]))[:3]
def episode(d):
    r = np.load(d / "run.npz", allow_pickle=True); q, ap, cm = r["arm_q"], r["apple"], r["cmds"]
    s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
    offs = []
    for side, o in (("left", 0), ("right", 7)):
        c = cm[:, o + 6] > 0.5
        onsets = np.where(c[1:] & ~c[:-1])[0] + 1
        for t in onsets:
            p = pad(side, q[t, o:o + 6]); a = ap[t]
            offs.append((p[0] - a[0], p[1] - a[1], p[2] - a[2]))
    return offs, bool(s.get("success")), s.get("close_cmd_ticks_on_air", 0)
for root in map(Path, sys.argv[1:]):
    eps = sorted(root.glob("ep*")) or [root]
    allo, lifts, n, air = [], 0, 0, 0
    for d in eps:
        if not (d / "run.npz").exists(): continue
        o, ok, a = episode(d); allo += o; lifts += ok; n += 1; air += a
    if not allo:
        print(f"{root.name:28s} n={n} lifts={lifts} no closes"); continue
    O = np.array(allo) * 100
    ahead = (O[:, 1] > 3).mean(); behind = (O[:, 1] < -3).mean(); high = (O[:, 2] > 3).mean()
    print(f"{root.name:28s} n={n:2d} lifts={lifts:2d} closes={len(O):3d} air/ep={air/n:5.0f} | forward offset median {np.median(O[:,1]):+5.1f} cm (IQR {np.percentile(O[:,1],25):+5.1f}..{np.percentile(O[:,1],75):+5.1f}) | "
          f"lateral median {np.median(O[:,0]):+5.1f} | height median {np.median(O[:,2]):+5.1f} | closes >3 cm AHEAD {ahead*100:3.0f}%  BEHIND {behind*100:3.0f}%  HIGH {high*100:3.0f}%")
