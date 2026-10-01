"""Camera sensitivity sweep table: each condition vs baseline (paired scenes)."""
import glob, json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs/camsweep"
prefix = sys.argv[2] if len(sys.argv) > 2 else ""      # e.g. "CAM_" for batch runs named runs/batch/CAM_<condition>
order = ["baseline", "tilt_m6", "tilt_m3", "tilt_p3", "tilt_p6", "height_m5cm", "height_p5cm", "fwd_m5cm", "fwd_p5cm", "lat_m5cm", "lat_p5cm",
         "yaw_m5", "yaw_p5", "roll_m3", "roll_p3", "fov_95", "fov_115", "wrist_fov_95", "wrist_fov_115", "wrist_pitch_m5", "wrist_pitch_p5", "wrist_yaw_p5", "wrist_fwd_1cm", "wrist_lat_1cm"]
# any further conditions under the prefix (e.g. INTR_*) are appended in name order
extra = sorted(os.path.basename(d)[len(prefix):] for d in glob.glob(os.path.join(root, prefix + "*")) if os.path.isdir(d))
order += [n for n in extra if n not in order]
aggs = {}
for name in order:
    f = os.path.join(root, prefix + name, "aggregate.json")
    if os.path.exists(f):
        aggs[name] = json.load(open(f))["per_episode"]
base = aggs.get("baseline")
print(f"{'condition':16s} {'n':>3s} {'lift':>6s} {'placed':>7s} {'grasp':>6s} {'t_grasp':>8s} {'air-close':>10s}  {'vs baseline (paired scenes: better/same/worse)':s}")
for name, eps in aggs.items():
    n = len(eps); lift = sum(e["success"] for e in eps); placed = sum(e.get("placed_in") is not None for e in eps)
    grasp = sum(e["t_first_grasp"] is not None for e in eps)
    tg = [e["t_first_grasp"] for e in eps if e["t_first_grasp"] is not None]
    air = np.mean([e["close_cmd_ticks_on_air"] for e in eps]) / 30
    cmp = ""
    if base is not None and name != "baseline":
        m = min(n, len(base)); b = [int(base[i]["success"]) for i in range(m)]; c = [int(eps[i]["success"]) for i in range(m)]
        cmp = f"+{sum(ci > bi for ci, bi in zip(c, b))} ={sum(ci == bi for ci, bi in zip(c, b))} -{sum(ci < bi for ci, bi in zip(c, b))}"
    print(f"{name:16s} {n:3d} {lift:3d}/{n:<2d} {placed:4d}/{n:<2d} {grasp:3d}/{n:<2d} {np.median(tg) if tg else float('nan'):7.1f}s {air:9.1f}s  {cmp}")
