"""Merge sharded batch runs (<root>/<prefix>_*_s*/aggregate.json, root default runs/batch = launch_jobs.sh output) and print
rates with 95% Wilson intervals. Usage: python tools/compare_sharded.py <prefix> [root]"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
groups = defaultdict(list)
root = sys.argv[2] if len(sys.argv) > 2 else "runs/batch"
for f in sorted(glob.glob(f"{root}/{prefix}_*_s*/aggregate.json")):
    name = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))
    groups[name] += json.load(open(f))["per_episode"]
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
print(f"{'condition':28s} {'n':>4s} {'lift':>6s} {'95% CI':>14s} {'1st-appr':>9s} {'t_grasp':>8s} {'placed':>7s} {'air-close':>10s}")
for name, eps in sorted(groups.items()):
    n = len(eps); k = sum(e["success"] for e in eps); lo, hi = wilson(k, n); g = [e for e in eps if e["t_first_grasp"] is not None]
    print(f"{name:28s} {n:4d} {100*k/n:5.0f}% {100*lo:6.0f}-{100*hi:3.0f}% {100*np.mean([(e['t_first_grasp'] is not None and e['t_first_grasp']<12) for e in eps]):8.0f}% {np.median([e['t_first_grasp'] for e in g]) if g else 0:7.1f}s {100*np.mean([e.get('placed_by_release', False) for e in eps]):6.0f}% {np.median([e['close_cmd_ticks_on_air'] for e in eps])/30:9.1f}s")
