"""Compare checkpoints over the seeded-variation evaluation in runs/eval/<name>/aggregate.json."""
import glob, json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs/eval"
aggs = {}
for f in sorted(glob.glob(os.path.join(root, "*", "aggregate.json"))):
    aggs[os.path.basename(os.path.dirname(f))] = json.load(open(f))
if not aggs:
    sys.exit("no aggregates yet")
print(f"{'checkpoint':20s} {'n':>3s} {'lift>5cm':>9s} {'placed':>7s} {'grasp':>6s} {'t_grasp med':>12s} {'hold med':>9s} {'air-close':>10s} {'R-arm close':>12s} {'ep with air-close':>18s}")
for name, a in aggs.items():
    eps = a["per_episode"]; n = len(eps)
    tg = [e["t_first_grasp"] for e in eps if e["t_first_grasp"] is not None]
    hold = [e["first_hold_seconds"] for e in eps if e["t_first_grasp"] is not None]
    print(f"{name:20s} {n:3d} {100*np.mean([e['success'] for e in eps]):8.0f}% {100*np.mean([e.get('placed_in') is not None for e in eps]):6.0f}% {100*np.mean([e['t_first_grasp'] is not None for e in eps]):5.0f}% "
          f"{np.median(tg) if tg else float('nan'):11.1f}s {np.median(hold) if hold else float('nan'):8.1f}s {np.mean([e['close_cmd_ticks_on_air'] for e in eps])/30:9.1f}s "
          f"{100*np.mean([e['right_arm_closes'] > 0 for e in eps]):11.0f}% {100*np.mean([e['close_cmd_ticks_on_air'] > 0 for e in eps]):17.0f}%")
# paired per-episode outcome grid: S = lifted, g = grasped but not lifted, . = never grasped
print("\nper-episode outcome (same scene variation per column):")
names = list(aggs)
n = max(len(a["per_episode"]) for a in aggs.values())
print(f"{'':20s} " + "".join(f"{i:2d}" for i in range(n)))
for name in names:
    row = ""
    for e in aggs[name]["per_episode"]:
        row += " S" if e["success"] else (" g" if e["t_first_grasp"] is not None else " .")
    print(f"{name:20s} {row}")
# which variation factors hurt: correlate success with apple offset radius
print("\nsuccess vs apple offset from plate centre (all checkpoints pooled):")
r_ok, r_bad = [], []
for a in aggs.values():
    for e in a["per_episode"]:
        r = np.hypot(*e["variation"]["apple_offset"])
        (r_ok if e["success"] else r_bad).append(r)
print(f"  lifted: n={len(r_ok)} mean offset {np.mean(r_ok)*100 if r_ok else float('nan'):.1f} cm | not lifted: n={len(r_bad)} mean offset {np.mean(r_bad)*100 if r_bad else float('nan'):.1f} cm")
