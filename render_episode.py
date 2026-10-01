"""Re-render an episode's video from its state log (run.npz + summary.json), no policy or physics needed.
Usage: .venv/bin/python render_episode.py runs/<set>/<name>/epNN [--every 2] [--spp 24] [--width 640]
Reproduces the scene (variation, appearance, camera perturbation) and poses arms/apple from the per-tick log."""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from yamsim.scene import YamStation, StationConfig, OverheadCamera, WristCamera, tf
from yamsim.renderer import tonemap
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser(); ap.add_argument("ep_dir"); ap.add_argument("--every", type=int, default=2); ap.add_argument("--spp", type=int, default=24)
ap.add_argument("--width", type=int, default=640); ap.add_argument("--out", default=None)
a = ap.parse_args()
s = json.load(open(os.path.join(a.ep_dir, "summary.json"))); z = np.load(os.path.join(a.ep_dir, "run.npz"))
var, app, camp = s["variation"], s.get("appearance", {}), s.get("camera_perturbation", {})
plate_xy = z["plate_xy"] if "plate_xy" in z.files else np.array([0.0, 0.38]) + np.array(var["plate_jitter"])
st = YamStation(StationConfig(plate_xy=tuple(plate_xy), cell_wall_albedo=app.get("wall_albedo", 0.85), table_surface=app.get("table", "white"), cell_wall_h=app.get("cell_wall_h", 0.5), cell_walls=app.get("cell_walls", "all")), log_level=0)
jit = camp.get("cam_jitter", 1)
cp = camp.get("cam_principal_px", [0, 0]); wp = camp.get("wrist_principal_px", [0, 0])
wk = dict(pitch_deg=camp.get("wrist_pitch", 0), yaw_deg=camp.get("wrist_yaw", 0), offset=tuple(camp.get("wrist_offset", (0, 0, 0))), diag_fov_deg=camp.get("wrist_fov", 105.0),
          distortion=tuple(camp.get("wrist_distortion", [0])), principal_offset=(wp[0] / 320, wp[1] / 200))
cams = {"high": OverheadCamera(height=0.6 + (var["cam_height_delta"] if jit else 0) + camp.get("cam_dz", 0), tilt_deg=45 + (var["cam_tilt_delta"] if jit else 0) + camp.get("cam_tilt", 0),
                               x_offset=camp.get("cam_dx", 0), y_offset=camp.get("cam_dy", 0), yaw_deg=camp.get("cam_yaw", 0), roll_deg=camp.get("cam_roll", 0), diag_fov_deg=camp.get("cam_fov", 105.0),
                               distortion=tuple(camp.get("cam_distortion", [0])), principal_offset=(cp[0] / 320, cp[1] / 200)),
        "left_wrist": WristCamera("left", **wk), "right_wrist": WristCamera("right", **wk)}
exposure = (1 / 300) * var["exposure_scale"]; ae = app.get("auto_exposure", 0); ev = app.get("exposure_ev", 0)
cal = st.cfg.gripper_cal; r = st.cfg.apple_diameter / 2
import imageio
out = a.out or os.path.join(a.ep_dir, "rollout.mp4")
w = imageio.get_writer(out, fps=30 / a.every, codec="libx264", quality=7, macro_block_size=8)
arm_q, apple, quat, cmds = z["arm_q"], z["apple"], z["apple_quat"], z["cmds"]
for t in range(0, len(arm_q), a.every):
    q = arm_q[t]
    st.set_arm_state(np.concatenate([q[:6], [cal["left"].to_rad(q[6])]]), np.concatenate([q[7:13], [cal["right"].to_rad(q[13])]]))
    T = np.eye(4); T[:3, :3] = Rot.from_quat([quat[t][1], quat[t][2], quat[t][3], quat[t][0]]).as_matrix(); T[:3, 3] = apple[t] - T[:3, :3] @ np.array([0, 0, r])
    st.set_object_pose("apple", T)
    sheet = Image.new("RGB", (3 * a.width, int(a.width / 1.6) + 28), "black")
    for i, k in enumerate(("high", "left_wrist", "right_wrist")):
        st.set_camera(cams[k]); rgb8, lin, _ = st.render(width=a.width, spp=a.spp, seed=t, clamp=app.get("clamp", 20.0), denoise=True, exposure=exposure)
        if ae > 0:
            ex = exposure; small = lin[::6, ::6]
            for _ in range(8):
                m = tonemap(small, exposure=ex).mean()
                if abs(m - ae) < 1.0: break
                ex *= (ae / max(m, 1.0)) ** 1.2
            rgb8 = tonemap(lin, exposure=ex * 2 ** ev)
        elif ev: rgb8 = tonemap(lin, exposure=exposure * 2 ** ev)
        sheet.paste(Image.fromarray(rgb8), (i * a.width, 28))
    c = cmds[t]
    ImageDraw.Draw(sheet).text((8, 7), f"{os.path.basename(a.ep_dir)} t={t/30:5.1f}s | L close {c[6]:.2f} R close {c[13]:.2f} | apple z {apple[t][2]*100:4.1f} cm  support {z['support'][t]}  pads L/R {z['pad_force'][t][0]:.0f}/{z['pad_force'][t][1]:.0f} N", fill=(255, 255, 0))
    w.append_data(np.asarray(sheet))
w.close(); print("->", out)
