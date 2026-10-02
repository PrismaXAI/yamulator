"""Scripted pick of the apple with the left arm through the contact model, rendering the overhead
camera and both wrist cameras along the way (denoised, low spp)."""
import os, sys, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from yamulator.paths import RENDERS
from yamulator.sim.scene import YamStation, OverheadCamera, WristCamera
from yamulator.sim.kinematics import solve_ik, top_down_grasp_frame
from yamulator.sim.physics import AppleGraspSim

out = str(RENDERS / "grasp_demo")
os.makedirs(out, exist_ok=True)
spp = int(sys.argv[1]) if len(sys.argv) > 1 else 32

st = YamStation()
sim = AppleGraspSim(st)
cal = st.cfg.gripper_cal
right_ready = np.array([0.15, 1.15, 1.9, -0.8, 0.0, 0.0, cal["right"].open_rad])
seed = np.array([0.0, 1.5, 1.5, -1.5, 0.0, 0.0, cal["left"].open_rad])
GRASP_Z = 0.135

def arm_at(point, standoff, seed_q):
    q, err = solve_ik(st.arms["left"], top_down_grasp_frame(point, standoff=GRASP_Z + standoff), seed_q, mode="full")
    assert err < 2e-3, err
    return q

# trajectory: hover -> descend -> close -> lift -> open (drop)
apple0 = sim.apple.copy()
keyframes = []
for s in np.linspace(0.12, 0.0, 8):            # descend
    keyframes.append((arm_at(apple0, s, seed), 1.0))
for g in np.linspace(1.0, 0.05, 6):            # close
    keyframes.append((keyframes[-1][0], g))
for s in np.linspace(0.0, 0.15, 8):            # lift
    keyframes.append((arm_at(apple0, s, keyframes[-1][0]), 0.05))
for g in np.linspace(0.05, 1.0, 4):            # open -> drop
    keyframes.append((keyframes[-1][0], g))

cams = {"overhead": OverheadCamera(), "wrist_left": WristCamera("left"), "wrist_right": WristCamera("right")}
frames = []
t0 = time.time()
for i, (q, g) in enumerate(keyframes):
    left = np.concatenate([q, [cal["left"].to_rad(g)]])
    info = sim.step(left, right_ready)
    row = []
    for name, cam in cams.items():
        st.set_camera(cam)
        rgb8, lin, aux = st.render(width=640, spp=spp, seed=i, clamp=20.0, denoise=True)
        row.append(Image.fromarray(rgb8))
    frames.append((i, row, info))
    print(f"step {i:2d}: g_cmd={g:.2f} gap={info['left_gap']*1000:5.1f}mm between_pads={info['left_between_pads']!s:5} grasped={info['grasped_by']} apple z={info['apple'][2]:.3f} lift={info['max_lift']:.3f}")
print(f"{len(keyframes)} steps x 3 cameras in {time.time()-t0:.1f}s; success={sim.success()}")
print("events:", [(e.step, e.side, e.kind) for e in sim.events if e.kind != "push"][:20])

# contact sheet: every 2nd step, three cameras per row
sel = frames[::2]
W, H = 640, 400
sheet = Image.new("RGB", (3 * W, len(sel) * H), "black")
d = ImageDraw.Draw(sheet)
for r, (i, row, info) in enumerate(sel):
    for c, im in enumerate(row):
        sheet.paste(im, (c * W, r * H))
    d.text((8, r * H + 8), f"step {i}  gap {info['left_gap']*1000:.0f} mm  grasped={info['grasped_by']}  lift={info['max_lift']*100:.1f} cm", fill=(255, 255, 0))
sheet.save(f"{out}/contact_sheet.png")
for i, row, info in frames:
    for name, im in zip(cams, row):
        im.save(f"{out}/{name}_{i:02d}.png")
print("->", out)
