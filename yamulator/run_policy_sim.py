"""Run a pi0.5 checkpoint closed-loop in the simulator (NO hardware: this script never imports the
i2rt SDK, python-can, or the real-rig scripts; nothing here can reach the CAN buses).

Scheme = the executor's "--simple" mode: synchronous inference on a fresh observation, execute the
first `commit_ticks` of the 50-step chunk at 30 Hz with the same per-tick rate limits, repeat.
Observation = three 640x400 renders (high / left_wrist / right_wrist) + 14-dim state in the training
convention; the sim arm tracks commands with a first-order lag like the real follower (~100 ms).

Usage (policy venv = requirements-policy.txt, see AGENTS.md):
  HF_HUB_OFFLINE=1 .venv-policy/bin/python -m yamulator.run_policy_sim --ckpt <checkpoint dir> --duration 40
"""
import argparse, dataclasses, json, os, sys, time
from pathlib import Path

import numpy as np

FORBIDDEN = ("i2rt", "can", "prism_pi", "dryrun_inference", "yam_executor")  # the CAN motor path and the real-rig stack; pyserial is a lerobot import, not used


def assert_no_hardware_modules():
    bad = [m for m in sys.modules if m.split(".")[0] in FORBIDDEN]
    assert not bad, f"hardware modules imported: {bad}"


import torch  # noqa: E402
from lerobot.processor import ProcessorStep, ProcessorStepRegistry  # noqa: E402


from yamulator import paths  # noqa: E402
from yamulator.policy.relative_actions_compat import RelativeActionsProcessorStep  # noqa: E402,F401  (real step under the fork's registry name)
from yamulator.sim.scene import YamStation, StationConfig, OverheadCamera, WristCamera  # noqa: E402
from yamulator.sim.physics import AppleGraspSim  # noqa: E402
from yamulator.sim.kinematics import TRAIN_GRIPPER_CAL  # noqa: E402

CTRL_HZ = 30.0
CHUNK = 50
MAX_GRIP_STEP = 0.10
RAMP_TICKS = 15
REST_POSE = np.array([0.0, 0.0, 0.0, -0.08, 0.0, 0.0])
ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def load_train_stats(ckpt: str):
    import struct
    files = sorted(Path(ckpt).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))   # step index depends on the pipeline
    assert files, f"no policy_preprocessor_step_*_normalizer_processor.safetensors in {ckpt}"
    f = open(files[0], "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = f.tell(); out = {}
    for key, meta in hdr.items():
        if key == "__metadata__":
            continue
        s, e = meta["data_offsets"]; f.seek(base + s)
        out[key] = np.frombuffer(f.read(e - s), dtype={"F32": np.float32, "F64": np.float64}[meta["dtype"]]).copy()
    return out


class Pi05Runner:
    def __init__(self, ckpt: str, dtype: str = "bf16"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.utils import prepare_observation_for_inference
        self.policy = PI05Policy.from_pretrained(ckpt).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, ckpt)
        self._prep = prepare_observation_for_inference
        self.autocast = dtype == "bf16"

    @torch.no_grad()
    def infer(self, frames: dict, state14: np.ndarray, task: str) -> np.ndarray:
        obs = {"observation.images.high": frames["high"], "observation.images.left_wrist": frames["left_wrist"],
               "observation.images.right_wrist": frames["right_wrist"], "observation.state": state14.astype(np.float32)}
        batch = self.pre(self._prep(obs, torch.device("cuda"), task))
        from contextlib import nullcontext
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else nullcontext():
            norm_chunk = self.policy.predict_action_chunk(batch)
        return self.post(norm_chunk.float()).squeeze(0).float().cpu().numpy()  # (50, 14)


JOINT_OFFSET = {"left": np.zeros(6), "right": np.zeros(6)}   # zeroing error per arm (rad), set from --joint-offset-deg
MAX_OPEN_GAP = 0.094


class SimArms:
    """Both arms as the policy sees them. dynamics="lag": first-order lag on the commanded joints (~100 ms);
    dynamics="mujoco": the SDK controller (MIT-mode PD + gravity comp, torque limits) on the i2rt MuJoCo
    model, 1 kHz, target updated at 30 Hz. Gripper state is reported in the training convention (raw motor
    rad, closed ~6.35 / open ~1.45) from the effective finger opening (blocked by the apple when grasped)."""

    def __init__(self, st: YamStation, sim: AppleGraspSim, dynamics: str = "lag", lag_s: float = 0.1):
        self.st, self.sim, self.dynamics = st, sim, dynamics
        self.alpha = (1 / CTRL_HZ) / (1 / CTRL_HZ + lag_s)
        self.cal = st.cfg.gripper_cal
        self.q = {"left": REST_POSE.copy(), "right": REST_POSE.copy()}   # measured arm joints
        self.g = {"left": 1.0, "right": 1.0}                              # measured/commanded open state
        if dynamics == "mujoco":
            from yamulator.sim.dynamics import YamArmDynamics
            self.dyn = {"left": YamArmDynamics(), "right": YamArmDynamics()}
        self.apply(np.concatenate([REST_POSE, [1.0]]), np.concatenate([REST_POSE, [1.0]]), instant=True)

    def reset(self):
        self.q = {"left": REST_POSE.copy(), "right": REST_POSE.copy()}
        self.g = {"left": 1.0, "right": 1.0}
        if self.dynamics == "mujoco_contact":
            return   # the world (arms + apple) is reset by ContactGraspSim.reset
        if self.dynamics == "mujoco":
            for side in self.dyn:
                self.dyn[side].reset(np.concatenate([REST_POSE, [1.0]]))

    def apply(self, left7, right7, instant=False):
        """left7/right7 = [q1..q6 target rad, gripper open state 0..1]. Steps the arms and the contact model."""
        acts = {}
        if self.dynamics == "mujoco_contact":
            l7 = np.asarray(left7, float).copy(); r7 = np.asarray(right7, float).copy()
            l7[:6] -= JOINT_OFFSET["left"]; r7[:6] -= JOINT_OFFSET["right"]          # zeroing error: commands land offset
            l7[6] *= MAX_OPEN_GAP / 0.094; r7[6] *= MAX_OPEN_GAP / 0.094              # smaller physical opening if configured
            if instant:
                return self.sim.step(l7, r7)
            info = self.sim.step(l7, r7)
            for side, off in (("left", JOINT_OFFSET["left"]), ("right", JOINT_OFFSET["right"])):
                s7 = self.sim.arm_state7(side); self.q[side] = s7[:6] + off; self.g[side] = float(s7[6]) * 0.094 / MAX_OPEN_GAP   # measured = true + delta
            return info
        for side, a in (("left", left7), ("right", right7)):
            a = np.asarray(a, float).copy(); a[:6] -= JOINT_OFFSET[side]
            if self.dynamics == "mujoco" and not instant:
                d = self.dyn[side]
                d.set_target(a); d.step(1.0 / CTRL_HZ)
                s7 = d.state(); self.q[side] = s7[:6] + JOINT_OFFSET[side]; self.g[side] = float(s7[6])
            elif self.dynamics == "mujoco":
                self.dyn[side].reset(a); self.q[side] = a[:6].copy(); self.g[side] = float(np.clip(a[6], 0, 1))
            else:
                self.q[side] = a[:6].copy() if instant else self.q[side] + self.alpha * (a[:6] - self.q[side])
                self.g[side] = float(np.clip(a[6], 0, 1))
            acts[side] = np.concatenate([self.q[side], [self.cal[side].to_rad(self.g[side])]])
        return self.sim.step(acts["left"], acts["right"])

    def state14(self) -> np.ndarray:
        s = []
        for side in ("left", "right"):
            s += list(self.q[side]) + [float(self.st.arm_state[side][6])]   # effective gripper rad (clamped if grasping)
        return np.array(s, dtype=np.float32)


@dataclasses.dataclass
class Variation:
    """Per-episode scene variation (seeded, identical across checkpoints for paired comparison)."""
    apple_offset: tuple = (0.0, 0.0)   # from the plate centre, m
    apple_yaw: float = 0.7
    plate_jitter: tuple = (0.0, 0.0)   # m, added to --plate-xy
    cam_height_delta: float = 0.0      # m
    cam_tilt_delta: float = 0.0        # deg
    exposure_scale: float = 1.0
    extra_offsets: tuple = ()          # apples 1..n-1, offsets from the plate centre, m (--n-apples)
    extra_yaws: tuple = ()

    @staticmethod
    def sample(rng: np.random.Generator, strength: float = 1.0) -> "Variation":
        r = 0.06 * strength * np.sqrt(rng.random()); th = rng.random() * 2 * np.pi
        return Variation(apple_offset=(r * np.cos(th), r * np.sin(th)), apple_yaw=float(rng.random() * 2 * np.pi),
                         plate_jitter=tuple(rng.uniform(-0.03, 0.03, 2) * strength), cam_height_delta=float(rng.uniform(-0.02, 0.02) * strength),
                         cam_tilt_delta=float(rng.uniform(-2.0, 2.0) * strength), exposure_scale=float(np.exp(rng.uniform(-0.2, 0.2) * strength)))

    def with_extra_apples(self, rng: np.random.Generator, n_extra: int, r_apple: float = 0.0375, r_max: float = 0.062, gap: float = 0.006) -> "Variation":
        """Extra apples on the plate: uniform within r_max of the plate centre, at least one diameter + gap from every
        other apple (rejection sampling; drawn AFTER the historical variation so apple 0 / plate / camera stay paired
        with the single-apple batches). If apple 0 leaves no room it is pulled toward the centre first."""
        pts = [np.array(self.apple_offset, float)]
        if n_extra > 0 and np.linalg.norm(pts[0]) > 0.03:
            pts[0] = pts[0] * 0.03 / np.linalg.norm(pts[0])
        yaws = []
        for _ in range(n_extra):
            for _try in range(500):
                rr = r_max * np.sqrt(rng.random()); th = rng.random() * 2 * np.pi; c = np.array([rr * np.cos(th), rr * np.sin(th)])
                if all(np.linalg.norm(c - q) >= 2 * r_apple + gap for q in pts):
                    pts.append(c); yaws.append(float(rng.random() * 2 * np.pi)); break
        return dataclasses.replace(self, apple_offset=tuple(pts[0]), extra_offsets=tuple(tuple(map(float, q)) for q in pts[1:]), extra_yaws=tuple(yaws))


def run_episode(args, ep: int, var: Variation, runner, st, sim, arms, cams, sag, a_lo, a_hi, out: Path):
    import imageio
    from PIL import Image, ImageDraw
    from yamulator.sim.scene import tf
    out.mkdir(parents=True, exist_ok=True); (out / "frames").mkdir(exist_ok=True)
    plate_xy = np.array(args.plate_xy) + np.array(var.plate_jitter)
    st.cfg.plate_xy = tuple(plate_xy)
    sim.plate_xy = plate_xy
    st.set_object_pose("plate", tf(xyz=(plate_xy[0], plate_xy[1], -1.0 if args.no_plate else args.plate_raise)))   # no-plate: hide it under the table; raised plate: lift the render too
    if var.extra_offsets:
        sim.reset(apple_xy=plate_xy + np.array(var.apple_offset), apple_yaw=var.apple_yaw, extra_xy=[plate_xy + np.array(o) for o in var.extra_offsets], extra_yaw=list(var.extra_yaws))
    else:
        sim.reset(apple_xy=plate_xy + np.array(var.apple_offset), apple_yaw=var.apple_yaw)
    jit_h = var.cam_height_delta if args.cam_jitter else 0.0
    jit_t = var.cam_tilt_delta if args.cam_jitter else 0.0
    cams["high"] = OverheadCamera(height=0.6 + args.arm_height + jit_h + args.cam_dz, tilt_deg=45.0 + jit_t + args.cam_tilt, x_offset=args.cam_dx,
                                  y_offset=args.cam_dy, yaw_deg=args.cam_yaw, roll_deg=args.cam_roll, diag_fov_deg=args.cam_fov,
                                  distortion=tuple(args.cam_distortion), principal_offset=(args.cam_principal_px[0] / 320.0, args.cam_principal_px[1] / 200.0))
    for side in ("left", "right"):
        mir = (-1.0 if side == "left" else 1.0) * args.wrist_mirror_px   # + = image content shifted OUTWARD (left cam: left, right cam: right), as measured on the rig
        cams[f"{side}_wrist"] = WristCamera(side, pitch_deg=args.wrist_pitch, yaw_deg=args.wrist_yaw, offset=tuple(args.wrist_offset), diag_fov_deg=args.wrist_fov,
                                            distortion=tuple(args.wrist_distortion), principal_offset=((args.wrist_principal_px[0] + mir) / 320.0, args.wrist_principal_px[1] / 200.0))
    mask_box = None
    if args.cam_mask_fov > 0:   # black out everything outside the frustum of a narrower lens (probe: revealed content vs image scale)
        tx_full, ty_full = cams["high"].tan_half(); tx_m, ty_m = OverheadCamera(diag_fov_deg=args.cam_mask_fov).tan_half()
        mask_box = (int(round(320 * (1 - tx_m / tx_full))), int(round(200 * (1 - ty_m / ty_full))))
    exposure = (1.0 / 300.0) * var.exposure_scale
    arms.reset()
    last_cmd = np.concatenate([REST_POSE, [0.0], REST_POSE, [0.0]])

    from yamulator.sim.renderer import tonemap
    from scipy.spatial.transform import Rotation as _Rot
    pose_hist = []   # (arm_state {side: 7}, object_poses {apple names: 4x4}) after every sim step, for --exposure-ms
    def record_pose():
        if args.exposure_ms > 0:
            pose_hist.append(({sd: st.arm_state[sd].copy() for sd in ("left", "right")},
                              {n: st.object_poses[n].copy() for n in st.object_poses if n.startswith("apple")}))
            del pose_hist[:-8]
    def _interp(a, b, f):
        arm = {sd: (1 - f) * a[0][sd] + f * b[0][sd] for sd in a[0]}
        obj = {}
        for n in a[1]:
            Ta, Tb = a[1][n], b[1][n]; T = np.eye(4)
            ra, rb = _Rot.from_matrix(Ta[:3, :3]), _Rot.from_matrix(Tb[:3, :3])
            T[:3, :3] = (ra * (ra.inv() * rb) ** f).as_matrix(); T[:3, 3] = (1 - f) * Ta[:3, 3] + f * Tb[:3, 3]
            obj[n] = T
        return arm, obj
    def exposure_poses():
        """K poses across the exposure window ending NOW (history entries are 1 tick apart, last = now)."""
        if args.exposure_ms <= 0 or len(pose_hist) < 2:
            return None
        K = args.blur_poses; e_ticks = args.exposure_ms / 1000 * CTRL_HZ; out = []
        for i in range(K):
            back = e_ticks * (1 - (i + 0.5) / K)                   # ticks before now
            j = min(int(np.floor(back)), len(pose_hist) - 2); f = back - j
            out.append(_interp(pose_hist[-1 - j], pose_hist[-2 - j], min(f, 1.0)))
        return out
    def render_all(spp):
        fr = {}
        poses = exposure_poses()
        for name, cam in cams.items():
            st.set_camera(cam)
            if poses is not None:
                rgb8, lin, _ = st.render_blur(poses, width=640, spp=spp, seed=args.seed + ep, clamp=args.clamp, denoise=True, exposure=exposure)
            else:
                rgb8, lin, _ = st.render(width=640, spp=spp, seed=args.seed + ep, clamp=args.clamp, denoise=True, exposure=exposure)
            if args.auto_exposure > 0:
                # AE loop like a webcam: find the exposure whose mean 8-bit value hits the target (2 fixed-point iterations)
                ex = exposure; small = lin[::6, ::6]   # AE on a subsampled image (cheap), final tonemap at full res
                for _ in range(8):   # converge the mean 8-bit gray onto the target (sRGB is nonlinear)
                    m = tonemap(small, exposure=ex).mean()
                    if abs(m - args.auto_exposure) < 1.0: break
                    ex *= (args.auto_exposure / max(m, 1.0)) ** 1.2
                rgb8 = tonemap(lin, exposure=ex * (2.0 ** args.exposure_ev))
            elif args.exposure_ev != 0.0:
                rgb8 = tonemap(lin, exposure=exposure * (2.0 ** args.exposure_ev))
            if poses is not None:
                st.set_arm_state(pose_hist[-1][0]["left"], pose_hist[-1][0]["right"])
                for n_, T_ in pose_hist[-1][1].items():
                    st.set_object_pose(n_, T_)
            if name != "high" and args.wrist_virtual_pitch != 0.0:
                # software re-aim: a rotation of the camera about its optical centre is EXACTLY a homography of its image
                # (H = K R K^-1; verified against a truly pitched render, 2 grey levels). Same sign as --wrist-pitch;
                # rows that would come from outside the original frame are edge-replicated.
                import cv2 as _cv2
                _f = np.hypot(320, 200) / np.tan(np.radians(cam.diag_fov_deg) / 2)
                _K = np.array([[_f, 0, 320 + 320 * cam.principal_offset[0]], [0, _f, 200 + 200 * cam.principal_offset[1]], [0, 0, 1.0]])
                _a = -np.radians(args.wrist_virtual_pitch); _R = np.array([[1, 0, 0], [0, np.cos(_a), -np.sin(_a)], [0, np.sin(_a), np.cos(_a)]])
                rgb8 = _cv2.warpPerspective(np.ascontiguousarray(rgb8), _K @ _R @ np.linalg.inv(_K), (rgb8.shape[1], rgb8.shape[0]), flags=_cv2.INTER_LINEAR, borderMode=_cv2.BORDER_REPLICATE)
            if name == "high" and mask_box is not None:
                rgb8 = rgb8.copy(); mx, my = mask_box
                rgb8[:my], rgb8[-my:] = 0, 0; rgb8[:, :mx], rgb8[:, -mx:] = 0, 0
            fr[name] = rgb8
        return fr

    video = args.video_every > 0
    writer = imageio.get_writer(str(out / "rollout.mp4"), fps=CTRL_HZ / args.video_every, codec="libx264", quality=7, macro_block_size=8) if video else None

    def video_frame(tick, frames, info, cmd, chunk_i, n_inf):
        sheet = Image.new("RGB", (3 * 640, 400 + 28), "black")
        for i, k in enumerate(("high", "left_wrist", "right_wrist")):
            sheet.paste(Image.fromarray(frames[k]), (i * 640, 28))
        d = ImageDraw.Draw(sheet)
        st14 = arms.state14()
        txt = (f"ep{ep} t={tick/CTRL_HZ:5.1f}s inf#{n_inf} step {chunk_i:2d} | L close cmd {cmd[6]:.2f} state {st14[6]:.2f} gap {info['left_gap_effective']*1000:3.0f}mm pads={int(info['left_between_pads'])} | "
               f"R close cmd {cmd[13]:.2f} state {st14[13]:.2f} gap {info['right_gap_effective']*1000:3.0f}mm pads={int(info['right_between_pads'])} | "
               f"grasped={info['grasped_by']} lift={info['max_lift']*100:4.1f}cm {'SUCCESS' if info['lifted'] else ''}")
        d.text((8, 7), txt, fill=(255, 255, 0))
        return np.asarray(sheet)

    def split(a14):
        return (np.concatenate([a14[0:6], [np.clip(1.0 - a14[6], 0, 1)]]),
                np.concatenate([a14[7:13], [np.clip(1.0 - a14[13], 0, 1)]]))

    rate = np.array([args.max_arm_step] * 6 + [MAX_GRIP_STEP] + [args.max_arm_step] * 6 + [MAX_GRIP_STEP])
    log = {"states": [], "cmds": [], "chunks": [], "latency": [], "between": [], "grasped": [], "apple": [], "pad_force": [], "support": [], "arm_q": [], "apple_quat": [], "infer_tick": [], "apples": [], "apples_quat": [], "pinch_apple": [], "supports": []}
    tick = 0; n_inf = 0; t_success = None; prev_chunk = None
    arb = None
    if args.arbitrate:
        from yamulator.control.arm_arbiter import ZoneArbiter
        arb = ZoneArbiter(zone_w=args.arb_zone, timeout_s=args.arb_timeout, arm_spacing=args.arm_spacing, hold_mode=args.arb_hold)
    log["arb_owner"] = []
    total_ticks = int(args.duration * CTRL_HZ)
    t_wall = time.time()
    info = arms.apply(*split(last_cmd), instant=True)
    obs_prev = obs_cur = 0
    arb_rejoin = {"left": False, "right": False}
    grip_hist = [last_cmd[[6, 13]].copy()] * (args.grip_delay_ticks + 1)
    # tool-frame interventions (control/tool_offset.py): constant offset and/or a one-shot advance at close onset
    tool_const = np.array(args.tool_offset_mm, float) / 1000.0
    pre_vec = np.array(args.preclose_advance_mm, float) / 1000.0
    pre_len = float(np.linalg.norm(pre_vec)); pre_dir = pre_vec / pre_len if pre_len > 0 else np.zeros(3)
    shifter = pre = None
    if np.any(tool_const != 0) or pre_len > 0:
        from yamulator.control.tool_offset import ToolShifter, PreCloseAdvance
        shifter = ToolShifter(args.arm_spacing, args.arm_height, args.base_pitch_deg)
        pre = PreCloseAdvance(pre_len, args.preclose_ticks, relative=bool(getattr(args, "is_relative", True))) if pre_len > 0 else None
    log["tool_shift_mm"] = []; shift_fail = {"left": 0, "right": 0}

    def shaped(cmd, t_obs=0):
        """Executor-side shaping of what is SENT (the policy-frame cmd / rate limiter / logs are untouched).
        Grippers: closing is delayed by --grip-delay-ticks (opening is immediate), --grip-assist adds closure once the
        command is clearly closing, and during a --preclose-advance-mm advance the gripper is held where it was.
        Arms: --tool-offset-mm (constant) plus the part of the pre-close advance that the chunk these targets come
        from (observed at tick t_obs) has not seen yet, applied as a translation in the gripper frame."""
        nonlocal grip_hist
        grip_hist = (grip_hist + [cmd[[6, 13]].copy()])[-(args.grip_delay_ticks + 1):]
        out = cmd.copy()
        g = np.minimum(cmd[[6, 13]], grip_hist[0])
        adv = {"left": 0.0, "right": 0.0}
        if pre is not None:
            held, adv = pre.step(tick, cmd, {"left": t_obs, "right": t_obs})
            g = np.minimum(g, [held["left"], held["right"]])
        g = np.minimum(1.0, g + args.grip_assist * np.clip((g - 0.10) / 0.10, 0, 1))
        out[[6, 13]] = g
        if shifter is not None:
            for side, o in (("left", 0), ("right", 7)):
                d = tool_const + pre_dir * adv[side]
                if np.any(np.abs(d) > 1e-6):   # kinematics in the TRUE joint frame (cmd is policy frame = true + zeroing offset)
                    q_true = cmd[o:o + 6] - JOINT_OFFSET[side]
                    out[o:o + 6] = shifter.shift(side, q_true, d) + JOINT_OFFSET[side]
                    shift_fail[side] += int(shifter.last_err_mm[side] >= 1.0)
            log["tool_shift_mm"].append((1000 * adv["left"], 1000 * adv["right"]))
        return out
    try:
        early_frames = None      # --camera-latency-ticks: images rendered C ticks before the state observation
        while tick < total_ticks:
            frames = early_frames if early_frames is not None else render_all(args.spp)
            early_frames = None
            state = arms.state14()
            t0 = time.perf_counter()
            D = args.latency_ticks if prev_chunk is not None else 0      # first inference happens at rest, as on the rig
            rtc = None
            if args.rtc and prev_chunk is not None:
                rtc = {"prev_abs": prev_chunk[min(args.commit_ticks, CHUNK - 2):].copy(), "delay": D, "horizon": args.rtc_horizon}
            chunk = runner.infer(frames, state, args.task, rtc=rtc) if args.rtc else runner.infer(frames, state, args.task)
            # targets for this cycle: D ticks still from the running plan, then the new chunk entered at step D
            steps = chunk[:args.commit_ticks].copy() if D == 0 else np.concatenate([prev_chunk[args.commit_ticks:args.commit_ticks + D], chunk[D:args.commit_ticks]])
            prev_chunk = chunk
            obs_prev, obs_cur = obs_cur, tick      # observation ticks of the running plan and of the new chunk
            for _ in range(int(round(args.inference_pause_s * CTRL_HZ))):   # the real executor holds the arm while inferring
                info = arms.apply(*split(shaped(last_cmd, obs_prev))); record_pose(); tick += 1
                if video and tick % args.video_every == 0:
                    writer.append_data(video_frame(tick, render_all(max(8, args.spp // 2)), info, last_cmd, -1, n_inf))
            log["latency"].append(time.perf_counter() - t0)
            log["chunks"].append(chunk); log["states"].append(state); log["infer_tick"].append(tick)
            if n_inf % 4 == 0:
                for k, f in frames.items():
                    Image.fromarray(f).save(out / "frames" / f"{n_inf:04d}_{k}.jpg", quality=88)
            n_inf += 1
            print(f"ep{ep} inf {n_inf:3d} @ {tick/CTRL_HZ:5.1f}s: {log['latency'][-1]*1000:.0f} ms, close cmd max L {chunk[:, 6].max():.2f} R {chunk[:, 13].max():.2f} | "
                  f"apple z {info['apple'][2]:.3f} grasped={info['grasped_by']} lift={info['max_lift']*100:.1f}cm", flush=True)
            for i in range(min(args.commit_ticks, CHUNK)):
                target = np.clip(sag(steps[i], state), a_lo, a_hi)
                rate_t = rate
                if arb is not None:
                    prefer = ("left" if info["apple"][0] < 0 else "right") if args.arb_prefer_side else None
                    target, arb_info = arb.step(state, target, lookahead=np.concatenate([steps[i:], chunk[args.commit_ticks:args.commit_ticks + 15]]), prefer=prefer)
                    log["arb_owner"].append({"left": 1, "right": 2}.get(arb_info["owner"], 0))
                    if arb_info["held"] is not None:   # a held arm retreats gently (homing rate), not at the policy's rate limit
                        o = 0 if arb_info["held"] == "left" else 7
                        rate_t = rate.copy(); rate_t[o:o + 6] = np.minimum(rate_t[o:o + 6], 0.03)
                    if args.arb_soft_rejoin:   # as yam_executor.py: a released arm keeps the gentle rate until its command has caught up with the plan
                        for _side, _o in (("left", 0), ("right", 7)):
                            if arb_info["held"] == _side:
                                arb_rejoin[_side] = True
                            elif arb_rejoin[_side]:
                                if np.max(np.abs(target[_o:_o + 6] - last_cmd[_o:_o + 6])) < 0.06:
                                    arb_rejoin[_side] = False
                                else:
                                    rate_t = rate_t.copy(); rate_t[_o:_o + 6] = np.minimum(rate_t[_o:_o + 6], 0.03)
                if tick < RAMP_TICKS:
                    alpha = (tick + 1) / RAMP_TICKS
                    target = (1 - alpha) * last_cmd + alpha * target
                cmd = last_cmd + np.clip(target - last_cmd, -rate_t, rate_t)
                info = arms.apply(*split(shaped(cmd, obs_prev if i < D else obs_cur)))
                record_pose()
                last_cmd = cmd
                log["cmds"].append(cmd.copy()); log["between"].append((info["left_between_pads"], info["right_between_pads"])); log["grasped"].append(info["grasped_by"] or "")
                log["apple"].append(np.asarray(info["apple"], float).copy()); log["pad_force"].append((info.get("left_pad_force", 0.0), info.get("right_pad_force", 0.0))); log["support"].append(info.get("support", ""))
                log["arm_q"].append(np.concatenate([arms.q["left"], [arms.g["left"]], arms.q["right"], [arms.g["right"]]]))
                log["apple_quat"].append(sim.world.apple_pose()[1].copy() if hasattr(sim, "world") else np.array([1.0, 0, 0, 0]))
                if "apples" in info:   # multi-apple bookkeeping (contact sim): every apple's pose, each side's focus apple, every support
                    log["apples"].append(np.asarray(info["apples"], float).copy()); log["apples_quat"].append(np.array([sim.world.apple_pose(k)[1] for k in range(len(info["apples"]))]))
                    log["pinch_apple"].append((info.get("left_apple", 0), info.get("right_apple", 0))); log["supports"].append(list(info["supports"]))
                if video and tick % args.video_every == 0:
                    writer.append_data(video_frame(tick, render_all(max(8, args.spp // 2)), info, cmd, i, n_inf))
                tick += 1
                state = arms.state14()
                if args.camera_latency_ticks > 0 and i == min(args.commit_ticks, CHUNK) - args.camera_latency_ticks - 1:
                    early_frames = render_all(args.spp)   # the images the next observation will see: C ticks stale vs its state
                if info.get("all_lifted", info["lifted"]) and t_success is None:   # several apples: the clock starts when ALL have been lifted
                    t_success = tick
                    print(f"*** ep{ep} SUCCESS: apple lifted at t={tick/CTRL_HZ:.1f}s", flush=True)
                if t_success is not None and tick - t_success > args.stop_after_success * CTRL_HZ:
                    break
            if t_success is not None and tick - t_success > args.stop_after_success * CTRL_HZ:
                break
    finally:
        if writer is not None:
            writer.close()
    cmds = np.array(log["cmds"]); between = np.array(log["between"], dtype=bool); grasped = np.array(log["grasped"])
    closes_on_air = int(((cmds[:, 6] > 0.5) & ~between[:, 0] & (grasped != "left")).sum() + ((cmds[:, 13] > 0.5) & ~between[:, 1] & (grasped != "right")).sum())
    ev = {k: [e.step for e in sim.events if e.kind == k] for k in ("between_pads", "grasp", "lift", "release")}
    hold = 0
    if ev["grasp"]:
        g0 = ev["grasp"][0]; rel = [r for r in ev["release"] if r > g0]
        hold = ((rel[0] if rel else tick) - g0) / CTRL_HZ
    summary = {
        "episode": ep, "ckpt": args.ckpt, "variation": dataclasses.asdict(var),
        "camera_perturbation": {k: getattr(args, k) for k in ("cam_dx", "cam_dy", "cam_dz", "cam_tilt", "cam_yaw", "cam_roll", "wrist_pitch", "wrist_yaw", "wrist_offset", "cam_jitter", "cam_fov", "wrist_fov",
                                                              "cam_distortion", "wrist_distortion", "cam_principal_px", "wrist_principal_px", "cam_mask_fov", "wrist_mirror_px", "wrist_virtual_pitch")},
        "torch_seed": args.torch_seed, "appearance": {"wall_albedo": args.wall_albedo, "table": args.table, "auto_exposure": args.auto_exposure, "exposure_ev": args.exposure_ev, "light_shape": args.light_shape, "light_radius": args.light_radius, "light_xy": args.light_xy, "clamp": args.clamp, "cell_walls": args.cell_walls, "cell_wall_h": args.cell_wall_h, "rear_texture": args.rear_texture, "arm_height": args.arm_height, "arm_spacing": args.arm_spacing, "base_pitch_deg": args.base_pitch_deg},
        "gap_probe": {"joint_offset_deg": args.joint_offset_deg, "joint_offset_vec_deg": args.joint_offset_vec_deg, "offset_seed": args.offset_seed, "offsets_deg": {k: np.degrees(v).round(3).tolist() for k, v in JOINT_OFFSET.items()},
                      "tool_offset_mm": list(args.tool_offset_mm), "preclose_advance_mm": list(args.preclose_advance_mm), "preclose_ticks": args.preclose_ticks,
                      "preclose_advances": (dict(pre.n_advances) if pre is not None else None), "tool_shift_unsolved_ticks": dict(shift_fail), "plate_diameter": args.plate_diameter, "apple_diameter": args.apple_diameter, "grip_delay_ticks": args.grip_delay_ticks, "grip_assist": args.grip_assist, "inference_pause_s": args.inference_pause_s, "max_open_gap": args.max_open_gap, "plate_rim": args.plate_rim, "plate_raise": args.plate_raise, "rolling": args.rolling, "no_plate": args.no_plate, "latency_ticks": args.latency_ticks, "camera_latency_ticks": args.camera_latency_ticks, "exposure_ms": args.exposure_ms, "rtc": args.rtc, "rtc_horizon": args.rtc_horizon, "arbitrate": args.arbitrate, "arb_zone": args.arb_zone,
                      "arb_held_s": ({k: v / CTRL_HZ for k, v in arb.held_ticks.items()} if arb is not None else None)}, "ticks": tick, "sim_seconds": tick / CTRL_HZ,
        "wall_seconds": time.time() - t_wall, "inferences": n_inf, "mean_latency_ms": 1000 * float(np.mean(log["latency"])),
        "success": bool(sim.success()), "max_lift_m": float(sim.max_lift), "placed_in": sim.placed_in, "dynamics": args.dynamics,
        "placed_by_release": any(e.kind == "place" and e.side != "none" for e in sim.events),
        "n_apples": int(getattr(sim, "n", 1)), "lifted_each": [bool(x) for x in getattr(sim, "lifted_each", [sim.success()])],
        "placed_each": list(getattr(sim, "placed_each", [sim.placed_in])), "final_support": [str(x) for x in (log["supports"][-1] if log["supports"] else [])],
        "n_lifted": int(sum(getattr(sim, "lifted_each", [sim.success()]))), "n_in_container_final": int(sum(str(x).startswith("container") for x in (log["supports"][-1] if log["supports"] else []))),
        "apple_displacement_before_first_grasp_m": ([float(np.linalg.norm(np.asarray(log["apples"][min(ev["grasp"][0], len(log["apples"]) - 1) if ev["grasp"] else -1])[k, :2] - sim.start_xy[k])) for k in range(sim.n)] if log["apples"] and hasattr(sim, "start_xy") else None),
        "events_apple": [(e.step, e.side, e.kind, int(e.detail.get("apple_index", 0))) for e in sim.events if e.kind != "push"],
        "t_first_between_pads": ev["between_pads"][0] / CTRL_HZ if ev["between_pads"] else None,
        "t_first_grasp": ev["grasp"][0] / CTRL_HZ if ev["grasp"] else None,
        "t_lift": ev["lift"][0] / CTRL_HZ if ev["lift"] else None,
        "first_hold_seconds": hold, "n_grasps": len(ev["grasp"]),
        "close_cmd_ticks_on_air": closes_on_air,
        "close_cmd_max": {"left": float(cmds[:, 6].max()), "right": float(cmds[:, 13].max())},
        "right_arm_closes": int((cmds[:, 13] > 0.5).sum()),
        "events": [(e.step, e.side, e.kind) for e in sim.events if e.kind != "push"],
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    np.savez(out / "run.npz", states=np.array(log["states"]), cmds=cmds, chunks=np.array(log["chunks"]), apple=np.array(log["apple"]),
             pad_force=np.array(log["pad_force"]), support=np.array(log["support"]), arm_q=np.array(log["arm_q"]), apple_quat=np.array(log["apple_quat"]),
             apples=np.array(log["apples"]), apples_quat=np.array(log["apples_quat"]), pinch_apple=np.array(log["pinch_apple"], dtype=np.int8), supports=np.array(log["supports"]),
             tool_shift_mm=np.array(log["tool_shift_mm"]), infer_tick=np.array(log["infer_tick"]), plate_xy=plate_xy, arb_owner=np.array(log["arb_owner"], dtype=np.int8))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LeRobot pi0.5 checkpoint directory (the pretrained_model dir, or a shim made by tools/make_ckpt_shim.py)")
    ap.add_argument("--task", default="Pick and place various fruits into containers")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--episode-start", type=int, default=0, help="first episode index (scene seed = 1000 + index), for sharding large samples across GPUs")
    ap.add_argument("--vary", type=float, default=0.0, help="variation strength (0 = fixed scene; 1 = apple within 6 cm of plate centre, plate +-3 cm, cam +-2 cm/+-2 deg, exposure +-20%%)")
    ap.add_argument("--dynamics", choices=["lag", "mujoco", "mujoco_contact"], default="mujoco_contact",
                    help="lag: first-order lag + kinematic apple; mujoco: SDK-controller-on-MuJoCo + kinematic apple; mujoco_contact: same arms + physical gripper/apple contact")
    # nominal camera perturbations (sensitivity tests); --cam-jitter 0 removes the random per-episode camera jitter
    ap.add_argument("--cam-dx", type=float, default=0.0, help="overhead camera lateral offset, m")
    ap.add_argument("--cam-dy", type=float, default=0.0, help="overhead camera forward offset, m")
    ap.add_argument("--cam-dz", type=float, default=0.0, help="overhead camera height offset, m")
    ap.add_argument("--cam-tilt", type=float, default=0.0, help="overhead camera extra downward tilt, deg")
    ap.add_argument("--cam-yaw", type=float, default=0.0, help="overhead camera yaw, deg")
    ap.add_argument("--cam-roll", type=float, default=0.0, help="overhead camera roll, deg")
    ap.add_argument("--wrist-pitch", type=float, default=0.0, help="both wrist cameras: pitch mounting error, deg (+ = toward the fingers)")
    ap.add_argument("--wrist-yaw", type=float, default=0.0, help="both wrist cameras: yaw mounting error, deg")
    ap.add_argument("--wrist-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="both wrist cameras: offset in the optical frame (x right, y down, z forward), m")
    ap.add_argument("--cam-jitter", type=int, default=1, help="1 = random +-2 cm / +-2 deg overhead jitter per episode (part of --vary)")
    ap.add_argument("--cam-fov", type=float, default=105.0, help="overhead camera diagonal FOV, deg (spec 105)")
    ap.add_argument("--arm-height", type=float, default=0.0, help="arm mounting flanges above the work surface, m (arms + overhead camera rise together; table/plate stay)")
    ap.add_argument("--arm-spacing", type=float, default=0.61, help="distance between the arm bases, m (spec 0.61)")
    ap.add_argument("--base-pitch-deg", type=float, default=0.0, help="whole-arm mount tilt about the lateral axis, both arms (+ = reach direction tips up, e.g. a base plate not flat); cameras on the arms tilt with it, the overhead does not")
    ap.add_argument("--wrist-fov", type=float, default=105.0, help="wrist cameras diagonal FOV, deg (spec 105)")
    # lens intrinsics deviations (training frames were rectified: zero distortion, centred principal point)
    ap.add_argument("--cam-distortion", type=float, nargs="+", default=[0.0], help="overhead camera: uncorrected OpenCV distortion coefficients k1 [k2 p1 p2 k3] (barrel: k1 < 0); focal length stays at --cam-fov")
    ap.add_argument("--wrist-distortion", type=float, nargs="+", default=[0.0], help="wrist cameras: same, uncorrected OpenCV coefficients k1 [k2 p1 p2 k3]")
    ap.add_argument("--cam-principal-px", type=float, nargs=2, default=[0.0, 0.0], help="overhead principal point offset in pixels of the 640x400 frame (+x right, +y down)")
    ap.add_argument("--wrist-principal-px", type=float, nargs=2, default=[0.0, 0.0], help="wrist cameras principal point offset in pixels of the 640x400 frame")
    ap.add_argument("--cam-mask-fov", type=float, default=0.0, help="probe: render the overhead camera at --cam-fov but black out pixels outside the frustum of this narrower diagonal FOV")
    # sim-to-real gap probes
    ap.add_argument("--joint-offset-deg", type=float, default=0.0, help="actuator zeroing error: per-joint offsets ~U(-D, D) deg, fixed for the run; the policy sees q+delta and its commands land at cmd-delta")
    ap.add_argument("--offset-seed", type=int, default=0)
    ap.add_argument("--joint-offset-vec-deg", type=float, nargs=6, default=None, help="explicit encoder-offset hypothesis, deg per joint [pan lift elbow wr1 wr2 wr3], same for both arms: "
                    "the policy sees q+delta and its commands land at cmd-delta (= the DATA rig read true+delta). Overrides --joint-offset-deg")
    ap.add_argument("--inference-pause-s", type=float, default=0.0, help="executor-style replan pause: hold the last command for this long at every inference (real rig: 0.25-0.4 s)")
    ap.add_argument("--plate-raise", type=float, default=0.0, help="raise the plate's top surface (and the apple on it) by this many metres, e.g. 0.003 for double-sided tape under the plate; physics and render")
    ap.add_argument("--plate-rim", type=float, default=0.020, help="plate rim height, m (contact ring that keeps a nudged apple on the plate); 0 = flat plate, apple rolls off")
    ap.add_argument("--rolling", type=float, default=0.002, help="rolling friction coefficient (m) of the apple on plate/table; smaller = rolls further (laminate desk ~0.0005)")
    ap.add_argument("--no-plate", action="store_true", help="remove the plate entirely (physics + render): apple directly on the work surface, as on the rig 2026-09-17 evening")
    ap.add_argument("--max-open-gap", type=float, default=0.094, help="physical gap at gripper open state 1 (URDF: 0.094 m; use a smaller value if the real gripper opens less)")
    ap.add_argument("--commit-ticks", type=int, default=25)
    ap.add_argument("--max-arm-step", type=float, default=0.10)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--spp", type=int, default=24)
    ap.add_argument("--video-every", type=int, default=0, help="render video frames every N ticks (2 = 15 fps); 0 = no video (default; re-render later from the state log with render_episode.py)")
    ap.add_argument("--plate-xy", type=float, nargs=2, default=[0.0, 0.38], help="plate centre (data rig: centred, 0.38 m out)")
    ap.add_argument("--apple-offset", type=float, nargs=2, default=[0.0, 0.0], help="apple offset from the plate centre (episode 0 / no --vary)")
    ap.add_argument("--force-apple-offset", action="store_true", help="with --vary: keep the per-episode scene variation (plate jitter, exposure, yaw) but pin the apple at --apple-offset from the plate centre")
    ap.add_argument("--sag-comp", choices=["auto", "on", "off"], default="auto",
                    help="add the data rig's follower-sag offsets (control/sag_comp.py, needs config/sag_table.json) to leader-action checkpoints "
                         "(auto: off if the checkpoint is follower-action or no config/sag_table.json is present)")
    ap.add_argument("--stop-after-success", type=float, default=6.0, help="seconds to keep running after a lift")
    ap.add_argument("--wall-albedo", type=float, default=0.85, help="work-cell wall albedo (data rig black curtain ~0.15)")
    ap.add_argument("--table", choices=["white", "wood", "wood_matte"], default="white", help="work surface: white laminate, light varnished wood grain, or the same grain matte")
    ap.add_argument("--light-shape", choices=["rect", "disk"], default="rect", help="overhead fixtures: 0.5x1 m panels or disks of --light-radius (same lumens per fixture)")
    ap.add_argument("--cell-walls", choices=["all", "rear", "none"], default="all", help="shielding panels around the workspace")
    ap.add_argument("--cell-wall-h", type=float, default=0.5, help="shielding panel height, m (0.5 spec; taller panels keep a wide lens from seeing the room above them)")
    ap.add_argument("--rear-texture", default=None, help="image in assets/backdrops shown on the rear panel (or the room wall if --cell-walls none)")
    ap.add_argument("--light-radius", type=float, default=0.15)
    ap.add_argument("--light-xy", type=float, nargs="+", default=None, help="fixture centres x1 y1 x2 y2 ... (m, world frame, ceiling height); default = four panels at (+-0.75, +-0.75)")
    ap.add_argument("--clamp", type=float, default=500.0, help="per-sample radiance clamp after a diffuse bounce (cd/m2). 500 = near-lossless (standard since 2026-09-17); 20 = historical runs (lost ~45%% of bounce light)")
    ap.add_argument("--auto-exposure", type=float, default=0.0, help="camera-style AE: scale each frame's exposure so its mean 8-bit gray hits this target (0 = fixed exposure; training frames ~150)")
    ap.add_argument("--exposure-ev", type=float, default=0.0, help="exposure error in EV applied after AE / fixed exposure (+ = over-exposed; highlights clip like a real sensor)")
    ap.add_argument("--arbitrate", action="store_true", help="arm arbitration (control/arm_arbiter.py): at most one arm in the shared work zone; the other is held at rest")
    ap.add_argument("--arb-soft-rejoin", action="store_true", help="with --arbitrate: an arm released by the arbiter keeps the gentle 0.03 rad/tick rate until its command has caught up with the policy's plan (executor default since 2026-09-19: prevents a full-rate plunge back to the grasp pose)")
    ap.add_argument("--arb-prefer-side", action="store_true", help="arbitration perception hint: prefer the arm on the fruit's side (sim: apple x; rig: overhead red-blob centroid)")
    ap.add_argument("--arb-hold", choices=["freeze", "rest"], default="rest", help="held arm: 'rest' = return to the rest pose (zero crowding; sim: same success, first grasp +6 s); 'freeze' = stop at the zone boundary (first grasp +3 s but crowding only halved)")
    ap.add_argument("--arb-zone", type=float, default=0.20, help="work-zone half width, m (gripper pad lateral distance from the rig centre line)")
    ap.add_argument("--arb-timeout", type=float, default=8.0, help="seconds an owner may sit in the zone without closing before it must yield")
    ap.add_argument("--tool-offset-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="CONSTANT translation of every commanded arm target in the gripper frame, mm (x across the pad width, + = away from the wrist camera; y closing axis; z along the fingers, - = toward the tips). For ABSOLUTE-action checkpoints; refused for relative ones (it would compound every replan) unless --tool-offset-force")
    ap.add_argument("--tool-offset-force", action="store_true")
    ap.add_argument("--preclose-advance-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="one-shot advance at close onset, mm in the gripper frame: when the policy starts to close a gripper it is held open for --preclose-ticks while the tool advances by this vector, then the close proceeds (sim 2026-09-19: the policy stops ~15-20 mm short along +x). Works for relative and absolute checkpoints")
    ap.add_argument("--preclose-ticks", type=int, default=6, help="ticks (at 30 Hz) over which the pre-close advance is made and the gripper is held")
    ap.add_argument("--plate-diameter", type=float, default=0.20, help="plate diameter, m (render + contact cylinder + rim). 0.20 = historical. With a smaller plate the per-episode apple offsets are scaled down so the apples still start inside the rim")
    ap.add_argument("--apple-diameter", type=float, default=0.075, help="apple diameter, m (render + contact sphere). 0.075 = historical; rig 2026-09-18: small apples ~0.055-0.068, the larger ones ~0.072-0.076 (from the gripper state while held)")
    ap.add_argument("--n-apples", type=int, default=1, help="apples on the plate (mujoco_contact only): free bodies with apple-apple and gripper-apple contact; extra apples are placed at random without overlap; the episode's stop clock starts when ALL apples have been lifted")
    ap.add_argument("--hide-wrist-cam-mesh", action="store_true", help="do not render the wrist-camera housings (needed when --wrist-offset moves the optical centre back into the housing mesh, e.g. the dataset-calibrated wrist pose)")
    ap.add_argument("--wrist-virtual-pitch", type=float, default=0.0, help="software re-aim of both wrist views by an image homography, deg, same sign as --wrist-pitch (+ = looks further ahead; sim 2026-09-19: +12 removes the ~13 mm forward overshoot at grasp). Deployable on the rig without touching the brackets")
    ap.add_argument("--wrist-mirror-px", type=float, default=0.0, help="per-arm MIRRORED wrist principal-point shift, px of the 640x400 frame: + = content shifted outward (left cam left, right cam right). Rig 2026-09-18: fingers ~+8..10 px outward vs the dataset")
    ap.add_argument("--grip-delay-ticks", type=int, default=0, help="executor-side: delay gripper CLOSING by this many ticks (opening stays immediate); tests the 'gripper fires before the descent completes' hypothesis")
    ap.add_argument("--grip-assist", type=float, default=0.0, help="executor-side extra closure added to the sent gripper command (ramps in over cmd 0.10..0.20), as yam_executor.py --grip-assist")
    ap.add_argument("--exposure-ms", type=float, default=0.0,
                    help="camera exposure (global shutter): motion blur by accumulating the render over --blur-poses poses spread across the "
                         "window ending at the observation (arm + apple interpolated between sim ticks). 0 = instantaneous (default); "
                         "rig: 8.3 ms max at 120 fps, up to 33 ms at 30 fps")
    ap.add_argument("--blur-poses", type=int, default=8)
    ap.add_argument("--camera-latency-ticks", type=int, default=0,
                    help="camera latency: the policy sees images rendered this many ticks BEFORE the state it is given (rig: ~6 avg before the "
                         "2026-09-21 buffering fix, ~1 after; the sim renders once per cycle either way, just earlier)")
    ap.add_argument("--latency-ticks", type=int, default=0, help="executor-style inference latency: a chunk observed at tick t is installed at t+D and entered at step D, "
                    "while the previous chunk keeps executing for those D ticks (real rig: 7-8). 0 = instantaneous inference (historical sim behaviour)")
    ap.add_argument("--rtc", action="store_true", help="real-time chunking through the policy server (needs --server and --latency-ticks > 0)")
    ap.add_argument("--rtc-horizon", type=int, default=20)
    ap.add_argument("--server", action="store_true", help="use the persistent policy server (prism_pi.policy.server of the prism-pi repo) instead of loading the model here; "
                    "the server must hold (or will hot-swap to) --ckpt. Per-episode --torch-seed is forwarded, so results match local runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="path-tracer sample seed (+ episode index); render noise only")
    ap.add_argument("--torch-seed", type=int, default=None, help="seed torch (+ episode index) at every episode start so the policy's flow-matching noise sequence is reproducible; "
                                                                  "paired conditions then share the same action noise (common random numbers). Default: unseeded, as in all runs before 2026-09-16")
    args = ap.parse_args()
    args.cam_distortion = (list(args.cam_distortion) + [0.0] * 5)[:5]
    args.wrist_distortion = (list(args.wrist_distortion) + [0.0] * 5)[:5]
    assert_no_hardware_modules()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else paths.RUNS / f"{ts}_{Path(args.ckpt).name}"
    out.mkdir(parents=True, exist_ok=True)
    is_follower = (Path(args.ckpt) / "FOLLOWER_ACTIONS").exists()
    try:
        is_follower |= "follower" in (Path(args.ckpt) / "train_config.json").read_text().lower()
    except OSError:
        pass
    from yamulator.control import sag_comp   # pure numpy + json
    if args.sag_comp == "on" and not sag_comp.available():
        raise SystemExit(f"--sag-comp on needs {sag_comp.TABLE} (see config/sag_table.example.json)")
    use_sag = args.sag_comp == "on" or (args.sag_comp == "auto" and not is_follower and sag_comp.available())
    if use_sag:
        sag = sag_comp.compensate14
    else:
        sag = lambda a, q: a
    print(f"ckpt {args.ckpt}: follower-action={is_follower} sag_comp={use_sag}", flush=True)

    global MAX_OPEN_GAP
    MAX_OPEN_GAP = args.max_open_gap
    if args.joint_offset_vec_deg is not None:
        for side in ("left", "right"):
            JOINT_OFFSET[side] = np.radians(np.array(args.joint_offset_vec_deg, float))
        print("joint encoder-offset hypothesis (deg, both arms):", np.array(args.joint_offset_vec_deg), flush=True)
    elif args.joint_offset_deg > 0:
        rng_off = np.random.default_rng(args.offset_seed)
        for side in ("left", "right"):
            JOINT_OFFSET[side] = np.radians(rng_off.uniform(-args.joint_offset_deg, args.joint_offset_deg, 6))
        print("joint zeroing offsets (deg): left", np.degrees(JOINT_OFFSET["left"]).round(2), "right", np.degrees(JOINT_OFFSET["right"]).round(2), flush=True)
    cfg = StationConfig(plate_diameter=args.plate_diameter, apple_diameter=args.apple_diameter, n_apples=args.n_apples, include_wrist_cameras=not args.hide_wrist_cam_mesh, plate_xy=tuple(args.plate_xy), cell_wall_albedo=args.wall_albedo, table_surface=args.table, light_shape=args.light_shape, light_disk_radius=args.light_radius, cell_walls=args.cell_walls, cell_wall_h=args.cell_wall_h, rear_texture=args.rear_texture,
                        arm_base_height=args.arm_height, arm_spacing=args.arm_spacing, arm_base_pitch_deg=args.base_pitch_deg,
                        **({"light_positions": tuple(zip(args.light_xy[0::2], args.light_xy[1::2]))} if args.light_xy else {}))
    st = YamStation(cfg, log_level=1)
    if args.dynamics == "mujoco_contact":
        from yamulator.sim.contact_sim import ContactGraspSim
        from yamulator.sim.mjworld import ContactConfig
        sim = ContactGraspSim(st, contact=ContactConfig(plate_rim_height=args.plate_rim, rolling=args.rolling, n_apples=args.n_apples, apple_radius=args.apple_diameter / 2), plate=not args.no_plate, plate_raise=args.plate_raise)
    else:
        sim = AppleGraspSim(st)
    arms = SimArms(st, sim, dynamics=args.dynamics)
    cams = {"high": OverheadCamera(), "left_wrist": WristCamera("left"), "right_wrist": WristCamera("right")}
    assert_no_hardware_modules()
    if args.server:
        from yamulator.policy.client import RemoteRunner   # no torch model in this process; never imports hardware modules

        class _ServedRunner:
            """Pi05Runner-compatible facade over the policy server; seed_next seeds the server's RNG once (episode start)."""
            def __init__(self, ckpt, dtype):
                self.rr = RemoteRunner(ckpt, dtype); self.seed_next = None
            def infer(self, frames, state14, task, rtc=None):
                chunks, _, _ = self.rr.infer_k(frames, state14, task, 1, seed=self.seed_next, rtc=rtc); self.seed_next = None
                return chunks[0]
        runner = _ServedRunner(args.ckpt, args.dtype)
        print(f"policy server in use: {runner.rr.info['ckpt']}", flush=True)
    else:
        runner = Pi05Runner(args.ckpt, args.dtype)
    stats = load_train_stats(args.ckpt)
    # Absolute target envelope. For a relative-action checkpoint the normalizer's action stats are DELTA ranges
    # (about +-1.1 rad), useless as an absolute clamp: use the follower STATE range instead (identical to the
    # follower-action range on the arm joints); grippers are absolute 0..1 either way.
    is_relative = bool(json.load(open(Path(args.ckpt) / "config.json")).get("use_relative_actions", False))
    env_key = "observation.state" if is_relative else "action"
    a_lo, a_hi = stats[f"{env_key}.min"][:14] - 0.10, stats[f"{env_key}.max"][:14] + 0.10
    a_lo[[6, 13]], a_hi[[6, 13]] = -0.10, 1.10
    print(f"action envelope from {env_key} stats (relative-action checkpoint: {is_relative})", flush=True)
    args.is_relative = is_relative
    if is_relative and any(v != 0 for v in args.tool_offset_mm) and not args.tool_offset_force:
        raise SystemExit("--tool-offset-mm is a constant command-side shift: with a relative-action checkpoint it becomes part of the observed state and is "
                         "re-applied at every replan (it compounds). Use --preclose-advance-mm instead, or pass --tool-offset-force to try it anyway.")
    print(f"policy loaded, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    results = []
    for ep in range(args.episode_start, args.episode_start + args.episodes):
        rng = np.random.default_rng(1000 + ep)   # same variation per episode index for every checkpoint
        if args.torch_seed is not None:
            torch.manual_seed(args.torch_seed + ep); torch.cuda.manual_seed_all(args.torch_seed + ep)
            if args.server:
                runner.seed_next = args.torch_seed + ep
        var = Variation.sample(rng, args.vary) if args.vary > 0 else Variation(apple_offset=tuple(args.apple_offset))
        if args.force_apple_offset:
            var = dataclasses.replace(var, apple_offset=tuple(args.apple_offset))
        if args.n_apples > 1:
            var = var.with_extra_apples(rng, args.n_apples - 1, r_apple=args.apple_diameter / 2)
        if args.plate_diameter < 0.20:   # keep the apples inside a smaller plate's rim
            k = max(0.0, min(1.0, (args.plate_diameter / 2 - args.apple_diameter / 2 - 0.010) / 0.06))
            var = dataclasses.replace(var, apple_offset=tuple(k * np.array(var.apple_offset)), extra_offsets=tuple(tuple(k * np.array(o)) for o in var.extra_offsets))
        ep_out = out / f"ep{ep:03d}" if (args.episodes > 1 or args.episode_start > 0) else out
        s = run_episode(args, ep, var, runner, st, sim, arms, cams, sag, a_lo, a_hi, ep_out)
        results.append(s)
        print(f"== ep{ep}: success={s['success']} placed={s['placed_in']} lift={s['max_lift_m']*100:.1f}cm grasp@{s['t_first_grasp']} hold={s['first_hold_seconds']:.1f}s closes_on_air={s['close_cmd_ticks_on_air']} ({s['wall_seconds']:.0f}s wall)", flush=True)
    agg = {"ckpt": args.ckpt, "episodes": len(results), "dynamics": args.dynamics, "success_rate": float(np.mean([r["success"] for r in results])),
           "place_rate": float(np.mean([r["placed_in"] is not None for r in results])),
           "grasp_rate": float(np.mean([r["t_first_grasp"] is not None for r in results])),
           "median_t_grasp": float(np.median([r["t_first_grasp"] for r in results if r["t_first_grasp"] is not None])) if any(r["t_first_grasp"] is not None for r in results) else None,
           "mean_closes_on_air_ticks": float(np.mean([r["close_cmd_ticks_on_air"] for r in results])),
           "mean_hold_s": float(np.mean([r["first_hold_seconds"] for r in results])), "per_episode": results}
    json.dump(agg, open(out / "aggregate.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_episode"}, indent=1), flush=True)
    print("->", out, flush=True)
    assert_no_hardware_modules()


if __name__ == "__main__":
    main()
