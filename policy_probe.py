"""Single-step policy probe: same observation (3 saved frames + rest state), N seeds -> chunk statistics.
Run on two machines and compare. Usage: python policy_probe.py <ckpt> <frames_dir/prefix> [dtype] [n]"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_sim import Pi05Runner, assert_no_hardware_modules
from PIL import Image
ckpt, prefix = sys.argv[1], sys.argv[2]; dtype = sys.argv[3] if len(sys.argv) > 3 else "bf16"; n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
frames = {k: np.asarray(Image.open(f"{prefix}_{k}.jpg").convert("RGB")) for k in ("high", "left_wrist", "right_wrist")}
state = np.array([0, 0, 0, -0.08, 0, 0, 1.45, 0, 0, 0, -0.07, 0, 0, 1.34], np.float32)
import os
if os.environ.get('SDPA_MATH'):
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False)
r = Pi05Runner(ckpt, dtype)
task = "Pick and place various fruits into containers"
chunks = []
for s in range(n):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    chunks.append(r.infer(frames, state, task))
c = np.array(chunks)  # (n, 50, 14)
m = c.mean(0)
out = {"device": torch.cuda.get_device_name(), "dtype": dtype, "sdpa_math": bool(os.environ.get("SDPA_MATH")), "n": n,
       "mean_chunk_step0": m[0].round(4).tolist(), "mean_chunk_step25": m[25].round(4).tolist(), "mean_chunk_step49": m[49].round(4).tolist(),
       "seed_spread_rad": float(c[:, :, [0,1,2,3,4,5,7,8,9,10,11,12]].std(0).mean()),
       "close_max_L": c[:, :, 6].max(1).round(3).tolist(), "close_max_R": c[:, :, 13].max(1).round(3).tolist(),
       "lift_L_max_per_seed": c[:, :, 1].max(1).round(3).tolist()}
print(json.dumps(out))
