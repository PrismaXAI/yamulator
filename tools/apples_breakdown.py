"""Per-apple outcome breakdown of --n-apples batches: python tools/apples_breakdown.py runs/batch/<name> ... (prefix; shards _s* pooled)"""
import glob, json, sys, numpy as np
for pfx in sys.argv[1:]:
    S = [json.load(open(f)) for f in sorted(glob.glob(pfx + "_s*/ep*/summary.json"))]
    if not S: continue
    n_ap = np.array([s.get("n_apples", 1) for s in S])
    print(f"== {pfx.split('/')[-1]}: {len(S)} episodes, apples per episode {dict(zip(*np.unique(n_ap, return_counts=True)))}")
    for n in sorted(set(n_ap)):
        E = [s for s in S if s.get("n_apples", 1) == n]
        fin = [x for s in E for x in s["final_support"]]; tot = len(fin)
        cnt = {k: sum(f.startswith(k) for f in fin) for k in ("container", "plate", "table", "air", "apple")}
        allin = np.mean([all(f.startswith("container") for f in s["final_support"]) for s in E])
        lifted = np.mean([x for s in E for x in s["lifted_each"]])
        disp = np.array([d for s in E if s.get("apple_displacement_before_first_grasp_m") for d in s["apple_displacement_before_first_grasp_m"]]) * 100
        dur = np.median([s["sim_seconds"] for s in E]); timeout = np.mean([s["sim_seconds"] >= 59.9 for s in E])
        print(f"   {n} apple(s), {len(E)} eps: per-apple lifted {100*lifted:.0f}% | final: container {100*cnt['container']/tot:.0f}%, plate {100*cnt['plate']/tot:.0f}%, table {100*cnt['table']/tot:.0f}%, on another apple {100*cnt['apple']/tot:.0f}%, in hand/air {100*cnt['air']/tot:.0f}% | ALL apples in containers {100*allin:.0f}% | displaced >3 cm before the first grasp {100*np.mean(disp>3):.0f}% of apples (p90 {np.percentile(disp,90):.1f} cm) | median episode {dur:.0f} s, ran to the 60 s limit {100*timeout:.0f}%")
        if n == 2:
            # geometry of the pair: lateral spread vs in-line, and outcome
            rows = []
            for s in E:
                v = s["variation"]; a = np.array(v["apple_offset"]); b = np.array(v["extra_offsets"][0]); d = b - a
                rows.append((abs(d[0]), abs(d[1]), all(f.startswith("container") for f in s["final_support"]), sum(f.startswith("container") for f in s["final_support"]), max(s.get("apple_displacement_before_first_grasp_m") or [0]) * 100))
            R = np.array(rows, float); side = R[:, 0] >= R[:, 1]
            for lab, m in (("side by side (|dx| >= |dy|)", side), ("in line front/back (|dy| > |dx|)", ~side)):
                print(f"      pair {lab}: {int(m.sum())} eps, both placed {100*R[m,2].mean():.0f}%, apples placed {100*R[m,3].mean()/2:.0f}%, a neighbour knocked >3 cm before the first grasp {100*np.mean(R[m,4]>3):.0f}%")
            # same container?
            same = [s["placed_each"][0] == s["placed_each"][1] for s in E if all(p is not None for p in s["placed_each"])]
            print(f"      when both were placed at some point: same container {100*np.mean(same):.0f}% of {len(same)}")
