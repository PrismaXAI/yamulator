"""Test renders of the YAM station from the overhead camera (spec: centreline, 0.6 m up, 45 deg down,
105 deg diagonal FOV, 16:10) at rest / hover / grasp poses, plus a close-up material check view."""
import os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from yamulator.paths import RENDERS
from yamulator.sim.scene import YamStation, StationConfig, OverheadCamera, LookAtCamera
from yamulator.sim.kinematics import solve_ik, top_down_grasp_frame

out = str(RENDERS)
os.makedirs(out, exist_ok=True)
spp = int(sys.argv[1]) if len(sys.argv) > 1 else 512

t0 = time.time()
st = YamStation()
print(f"scene built in {time.time() - t0:.1f}s: {len(st.instances)} instances")
cal = st.cfg.gripper_cal
apple_center = st.object_poses["apple"][:3, 3] + [0, 0, st.cfg.apple_diameter / 2]
seed_q = np.array([0.0, 1.5, 1.5, -1.5, 0.0, 0.0, cal["left"].open_rad])
GRASP_Z = 0.135  # gripper link origin above the grasp point when the finger pads straddle it

def left_pose(standoff, g):
    q, err = solve_ik(st.arms["left"], top_down_grasp_frame(apple_center, standoff=GRASP_Z + standoff), seed_q, mode="full")
    assert err < 1e-3, err
    return np.concatenate([q, [cal["left"].to_rad(g)]])

rest = lambda side: np.array([0, 0, 0, -0.08, 0, 0, cal[side].open_rad])
right_ready = np.array([0.15, 1.15, 1.9, -0.8, 0.0, 0.0, cal["right"].open_rad])
apple_grip = st.cfg.apple_diameter / (2 * st.arms["left"].finger_stroke)  # opening == apple diameter
poses = {
    "rest": (rest("left"), rest("right")),
    "hover": (left_pose(0.10, 1.0), right_ready),
    "grasp": (left_pose(0.0, apple_grip), right_ready),
}
for name, (l, r) in poses.items():
    st.set_arm_state(l, r)
    t = time.time()
    rgb8, lin, aux = st.render(width=1280, spp=spp, seed=1, clamp=20.0)
    dt = time.time() - t
    Image.fromarray(rgb8).save(f"{out}/overhead_{name}.png")
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    print(f"{name:6s}: {dt * 1000:6.0f} ms @ {spp} spp | lum mean {lum.mean():.0f} cd/m2 p99 {np.percentile(lum, 99):.0f} | "
          f"apple px {int(st.instance_mask(aux, 'apple').sum())} plate px {int(st.instance_mask(aux, 'plate').sum())} | q_left {np.round(l, 2)}")

# close-up material check: the hovering left arm, plate and apple from the front-right
st.set_arm_state(*poses["hover"])
st.set_camera(LookAtCamera(eye=(0.25, -0.15, 0.42), target=(-0.22, 0.30, 0.12), diag_fov_deg=62.0))
rgb8, lin, aux = st.render(width=1280, spp=spp, seed=2, clamp=20.0)
Image.fromarray(rgb8).save(f"{out}/closeup_materials.png")
# tight view of the grasp
st.set_arm_state(*poses["grasp"])
st.set_camera(LookAtCamera(eye=(0.05, 0.05, 0.22), target=tuple(apple_center), diag_fov_deg=40.0))
rgb8, lin, aux = st.render(width=1280, spp=spp, seed=3, clamp=20.0)
Image.fromarray(rgb8).save(f"{out}/closeup_grasp.png")
st.set_camera(OverheadCamera())
print("done ->", out)
