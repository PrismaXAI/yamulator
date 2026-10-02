"""Two-arm crowding metrics per run directory from run.npz (arm_q per tick): how close the grippers get and how long
both occupy the shared zone over the plate. Usage: python crowding_stats.py <run_dir> ..."""
import sys, json, glob, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yamulator.control.arm_arbiter import ZoneArbiter
arb = ZoneArbiter()
def pad(side, q6):
    P = arb.arms[side].link_poses(np.r_[q6, 1.45])["gripper"]; return (P @ np.array([0, 0, -0.134, 1.0]))[:3]
for root in sys.argv[1:]:
    mins, near, both, lifts, placed, held, n = [], [], [], 0, 0, [], 0
    for d in sorted(glob.glob(f"{root}/ep*")):
        r = np.load(f"{d}/run.npz", allow_pickle=True); s = json.load(open(f"{d}/summary.json")); q = r["arm_q"]; T = len(q)
        L = np.array([pad("left", q[t, :6]) for t in range(0, T, 2)]); R = np.array([pad("right", q[t, 7:13]) for t in range(0, T, 2)])
        dLR = np.linalg.norm((L - R)[:, :2], axis=1)
        mins.append(dLR.min()); near.append((dLR < 0.25).sum() * 2 / 30); both.append(((np.abs(L[:, 0]) < 0.20) & (np.abs(R[:, 0]) < 0.20)).sum() * 2 / 30)
        lifts += bool(s["success"]); placed += s["placed_in"] is not None; n += 1
        h = (s.get("gap_probe") or {}).get("arb_held_s");  held.append(sum(h.values()) if h else 0.0)
    print(f"{Path(root).name:24s} n={n:2d} lifts {lifts:2d} placed {placed:2d} | min gripper distance: median {np.median(mins)*100:4.1f} cm, worst {np.min(mins)*100:4.1f} cm, episodes <15 cm: {sum(m<0.15 for m in mins):2d} | "
          f"time within 25 cm: mean {np.mean(near):4.1f} s | both arms in the zone: mean {np.mean(both):4.1f} s, episodes with any: {sum(b>0 for b in both):2d} | held-arm time mean {np.mean(held):4.1f} s")
