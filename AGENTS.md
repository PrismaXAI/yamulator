# AGENTS.md: yamulator

Guide for coding agents (Claude Code and others) setting up and operating this repository for a user.
Read it fully before running anything; every command below is meant to be run from the repository root.

## What this is

A simulator of a bimanual [I2RT YAM](https://github.com/i2rt-robotics/i2rt) station for evaluating pi0.5
(LeRobot) checkpoints closed-loop before they touch hardware:

- `native/`: an OptiX path tracer (Disney BRDF, area lights with NEE + MIS, glass, denoiser, motion blur) that
  renders the three policy cameras (overhead + two wrist cameras) at 640x400.
- `yamsim/`: the Python package: URDF kinematics, scene (`YamStation`), materials, the i2rt motor law on the
  MuJoCo arm model (`dynamics.py`), a physical gripper/fruit contact world (`mjworld.py`, `contact_sim.py`).
- `run_policy_sim.py`: runs a checkpoint in the loop (pick an apple from a plate, place it in a container) with
  the same control scheme as the real-rig executor in the companion repo `prism-pi`, and logs every tick.

It is sim only. Nothing here talks to a robot; `run_policy_sim.py` asserts at start and end that no hardware
module (`i2rt`, `can`, executor code) got imported. Keep it that way.

## What you need from the user before starting

1. A Linux x86_64 machine with an NVIDIA GPU. Tested: RTX 5090 laptop (driver 595.71, compute capability 12.0) and
   data-centre cards without RT cores (driver 610, capability 8.0; they work, about 5x slower per frame).
   OptiX 9.1 needs a recent driver; an old one fails when the renderer starts, see Troubleshooting.
2. The CUDA toolkit (`nvcc` on PATH or `NVCC=/path/to/nvcc`; 13.3 tested), `git`, and
   [`uv`](https://docs.astral.sh/uv/). No OptiX SDK install is needed.
3. Only for policy runs: a pi0.5 checkpoint in LeRobot format (see "Checkpoint conventions"), about 16 GB of free
   VRAM, and a Hugging Face account that has accepted the licence of the gated `google/paligemma-3b-pt-224`
   repo (only its tokenizer is fetched). No checkpoint ships with this repository: ask the user where theirs is.

The simulator-only part (steps 1-3) needs none of item 3 and is the right first milestone.

## Setup

### 1. Simulator environment (no torch)

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt -c constraints.txt

`constraints.txt` pins every package to the versions the code was validated with. Always pass it.

### 2. Build the renderer

    ./native/build.sh

The first run clones the OptiX headers (`github.com/NVIDIA/optix-dev`, tag v9.1.0) into `third_party/` and
compiles `build/device.ptx` + `build/libyamtrace.so` (under a minute; one unused-variable warning is normal).
The PTX target is chosen from the GPU's compute capability (capped at `compute_90`). Rebuild on every machine;
on a mixed-GPU machine build for the oldest card (`PTX_ARCH=compute_80 ./native/build.sh`).

Do not commit `third_party/` or `build/`: the OptiX headers are under an NVIDIA licence that does not allow
redistribution inside an open-source repository, which is why they are fetched at build time.

### 3. Verify without a policy

    .venv/bin/python render_test.py 64      # 5 test renders -> renders/*.png, about 0.3 s per frame
    .venv/bin/python grasp_demo.py 32       # scripted pick through the contact model -> renders/grasp_demo/

Expected: `scene built in ...: 46 instances`, three lines `rest / hover / grasp` with `lum mean` around 130 cd/m2,
and for the demo `success=True` with events `between_pads`, `grasp`, `lift`, `release`. Open
`renders/overhead_grasp.png`: a white work cell, the left arm reaching over a red apple on a white plate, two clear
containers. A black or noisy image means the build or the driver is wrong; do not continue to policy runs.

### 4. Policy environment

    uv venv --python 3.12 .venv-policy
    uv pip install --python .venv-policy/bin/python -r requirements-policy.txt -c constraints.txt

About 8 GB installed (torch 2.10.0 with CUDA 12.8, lerobot 0.5.1, transformers 5.3.0). These exact versions
matter: other LeRobot versions reject the checkpoint config or change preprocessing. Do not reuse a training
environment, and do not "upgrade to latest".

### 5. Tokenizer (gated, one-time download)

The policy's prompt tokenizer comes from `google/paligemma-3b-pt-224`. The user must accept its licence on
huggingface.co and provide a token (`hf auth login`, or `HF_TOKEN=...`); you cannot do that for them. Then:

    .venv-policy/bin/python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print(t('Pick and place various fruits into containers', add_special_tokens=True)['input_ids'][:4])"

It must print a list that starts with `2` (the `<bos>` token), e.g. `[2, 21585, 578, 2040]`. This check is
not a formality: a look-alike tokenizer copied from an ungated mirror had the right vocabulary but no `<bos>`,
and every rollout with it was silently off-distribution (the policy hesitated and closed on air). If the list
does not start with 2, fix the tokenizer in the HF cache before anything else. Afterwards run everything with
`HF_HUB_OFFLINE=1` so nothing is re-resolved at run time.

### 6. Checkpoint

Pass the checkpoint's `pretrained_model` directory as `--ckpt`. If loading fails on an unknown `config.json` key
(checkpoints saved by a newer or forked LeRobot), make a shim: a directory with a filtered `config.json` and
symlinks to the weights:

    .venv-policy/bin/python tools/make_ckpt_shim.py /path/to/checkpoints/004000/pretrained_model ckpt/my_ckpt

Add `--follower` if the checkpoint was trained on follower positions as actions (see `--sag-comp` below).

### 7. First policy rollout

    HF_HUB_OFFLINE=1 .venv-policy/bin/python run_policy_sim.py --ckpt ckpt/my_ckpt \
        --task "<task string, verbatim from the training data>" \
        --episodes 2 --vary 1 --duration 12 --auto-exposure 150 --torch-seed 0 --out runs/smoke

Expected: `policy loaded, VRAM 15.x GiB`, one `== ep0: success=... grasp@...` line per episode (a 12 s episode
takes about 6 s on an RTX 5090, about 30 s on a card without RT cores), then the aggregate JSON and `-> runs/smoke`.
With a fixed `--torch-seed` the run is reproducible bit for bit on the same machine and software. Add
`--video-every 2` to get `rollout.mp4` (three cameras side by side), or render it afterwards:

    .venv/bin/python render_episode.py runs/smoke/ep001 --every 2 --spp 24

## Checkpoint conventions the code assumes

The runner is written for one data convention (ours). A checkpoint trained on different conventions needs the
matching constants changed, not just a different `--ckpt`:

| item | convention | where it lives |
|---|---|---|
| policy | LeRobot `pi05`, chunk of 50 actions at 30 Hz | `Pi05Runner`, `CHUNK`, `CTRL_HZ` in `run_policy_sim.py` |
| images | `observation.images.{high,left_wrist,right_wrist}`, 640x400 RGB (16:10 sensor, downscaled) | `Pi05Runner.infer`; the checkpoint's own rename step maps them to the model's names |
| state (14) | `[L pan, lift, elbow, wr1, wr2, wr3, gripper, R ...]`, arm joints in rad (i2rt), gripper = raw motor rad of the data rig (closed about 6.35, open about 1.45) | `SimArms.state14`, `TRAIN_GRIPPER_CAL` in `yamsim/kinematics.py` |
| action (14) | same order; arm joints absolute rad (or deltas from the chunk-start state when `use_relative_actions` is set in `config.json`), gripper 0 = open .. 1 = closed | `split()` in `run_episode`; relative actions via `relative_actions_compat.py` |
| start pose | both arms at `[0, 0, 0, -0.08, 0, 0]`, grippers open | `REST_POSE` |
| action envelope | normalizer stats of the checkpoint +-0.10 rad (state stats for relative-action checkpoints) | `main()` |
| leader vs follower actions | `--sag-comp auto` adds the data rig's follower-sag table to leader-action checkpoints if `sag_table.json` exists; never for a `FOLLOWER_ACTIONS` marker or "follower" in `train_config.json` | `sag_comp.py`, `sag_table.example.json` |

The sag table is a property of the rig the training data was recorded on (its follower arms sagged below the
leader command). It is not installed by default; without `sag_table.json` no compensation is applied.

## Running evaluations

`run_policy_sim.py -h` lists every option. The ones that define a comparable experiment:

- `--episodes N --vary 1`: N scenes with seeded variation (apple within 6 cm of the plate centre, plate +-3 cm,
  camera +-2 cm / +-2 deg, exposure +-20 %). The scene seed is `1000 + episode index`, so every checkpoint and
  condition sees the same scenes; use `--episode-start K` for fresh scenes, never a different `--seed`
  (that only changes render noise). `--cam-jitter 0` when you test exact camera offsets.
- `--auto-exposure 150`: per-frame auto exposure to the training frames' mean gray. Use it always; the fixed
  exposure fallback leaves dark scenes two stops under, which a real camera would not do.
- `--torch-seed 0`: seeds the flow-matching noise per episode, so paired conditions share action noise.
- `--dynamics mujoco_contact` (default): motor law on the MuJoCo arm + physical contact. `lag` and `mujoco` use
  a simpler kinematic contact test that is too forgiving; do not report numbers from them.
- Appearance and geometry: `--table white|wood|wood_matte`, `--wall-albedo`, `--cell-walls all|rear|none`,
  `--light-shape disk --light-radius R`, `--light-xy ...`, `--rear-texture <file in assets/backdrops>`,
  `--arm-height`, `--arm-spacing`, `--base-pitch-deg`, `--cam-dx/dy/dz/tilt/yaw/roll/fov`, `--wrist-*`,
  `--cam-distortion`, `--plate-*`, `--apple-diameter`, `--n-apples`, `--exposure-ev`, `--exposure-ms` (motion blur).
- Executor emulation: `--latency-ticks 8` (inference latency as on the rig), `--arbitrate` (one arm at a time
  in the shared zone, `arm_arbiter.py`), `--joint-offset-vec-deg` (encoder-offset hypothesis),
  `--preclose-advance-mm`, `--grip-assist`, `--max-arm-step`.

Outputs per episode (`<out>/epNNN/`): `summary.json` (success = lifted > 5 cm, `placed_in`, grasp times,
air-close ticks, events, the full variation and camera settings) and `run.npz` (per-tick states, commands,
chunks, arm joints, apple pose, pad forces); `<out>/aggregate.json` for the run. Read results from these files,
not from the console.

Batches over several GPUs: write a jobs file (`examples/jobs_example.txt`) and run
`./launch_jobs.sh jobs.txt "0 1 2 3"` (detach it with `setsid nohup ... &` for long batches). Results land in
`runs/batch/<name>/`. Then:

    .venv/bin/python tools/compare_eval.py runs/batch          # one row per run directory + paired outcome grid
    .venv/bin/python tools/compare_sharded.py BIG              # merge <prefix>_*_s<k> shards, Wilson intervals
    .venv/bin/python tools/grasp_quality.py runs/batch/<name>  # antipodal-contact grasp quality, slips, lost holds
    .venv/bin/python tools/close_offset_stats.py runs/batch/<name>   # where closes land relative to the apple
    .venv/bin/python tools/crowding_stats.py runs/batch/<name>       # two-arm crowding over the plate

Sample size: 30 paired scenes resolve differences of about 5 episodes; anything closer needs 150 scenes
(5 shards of 30 with `--episode-start 0/30/60/90/120`). Held-out loss did not predict closed-loop ranking for us;
select checkpoints by rollouts.

### With the prism-pi policy server

`--server` uses a running policy server from the `prism-pi` repository (`python -m prism_pi.policy.server`
there) instead of loading the model here
(model stays in VRAM between runs, bf16 weights at about 8 GiB, optional TensorRT engines). Both repositories use
the same socket (`/tmp/prism-pi-policy-<uid>.sock`, override with `YAM_POLICY_SOCK`), so nothing needs
configuring. `--rtc` (real-time chunking) requires `--server` and `--latency-ticks > 0`. While the server holds
the model, a second local model load can run out of VRAM: use `--server` for everything, or stop the server.
Do not run sim batches through the server while it is serving a live robot session (inference slows down).

## Adapting the scene

The defaults model our station: arms 0.61 m apart with bases level with the work surface, plate 0.38 m ahead,
overhead camera on the base line 0.6 m up and 45 deg down (`--cam-tilt -10` reproduces our dataset's framing),
105 deg diagonal lenses. For another rig, measure and set the flags above, or edit `StationConfig`,
`OverheadCamera` and `WristCamera` in `yamsim/scene.py`. Things that mattered most in our ablations: arm base
height relative to the work surface (2 cm costs success, 5 cm breaks the policy), an overhead lens wider than the
training lens, wrist-camera position to a few millimetres, small hard light sources. Scene colours and textures
did not matter once exposure matched.

World frame: z up, origin on the work surface midway between the arm bases, arms face +y, +x is to the right
in the overhead image. Minimal API:

    from yamsim.scene import YamStation, StationConfig, OverheadCamera, WristCamera
    st = YamStation(StationConfig())                 # builds the scene on the GPU
    st.set_arm_state(left7, right7)                  # [q1..q6 rad, gripper rad] per arm
    st.set_camera(WristCamera("left"))
    rgb8, linear, aux = st.render(width=640, spp=24, denoise=True)

## Repository map

| path | purpose |
|---|---|
| `native/` | OptiX device program (`device.cu`), C API (`yamtrace.cpp`), `build.sh` |
| `yamsim/` | `kinematics.py`, `renderer.py` (ctypes), `materials.py`, `scene.py`, `dynamics.py`, `mjworld.py`, `contact_sim.py`, `physics.py` (kinematic contact, legacy), `tool_offset.py` |
| `assets/` | YAM URDF/MJCF + meshes (from the i2rt SDK), apple and wood textures (Poly Haven, CC0); see `THIRD_PARTY.md` |
| `run_policy_sim.py`, `policy_probe.py` | closed-loop runner; single-observation probe for cross-machine checks |
| `render_test.py`, `grasp_demo.py`, `render_episode.py` | smoke tests, offline video re-render |
| `tools/` | result tables, grasp-quality and offset statistics, `make_ckpt_shim.py` |
| `launch_jobs.sh`, `examples/` | multi-GPU batch launcher and a jobs file |
| `arm_arbiter.py`, `sag_comp.py`, `relative_actions_compat.py`, `policy_client.py`, `yamsim/kinematics.py`, `yamsim/tool_offset.py`, `tools/make_ckpt_shim.py`, `constraints.txt` | executor-side code the simulator runs in the loop. `prism-pi` carries the same code under `prism_pi/control/` and `prism_pi/policy/` (the copies differ only in imports and file locations). A change to the logic of one belongs in both, so that what the simulator validates is what the real executor runs; say so when you make one. |

Not tracked (see `.gitignore`): `build/`, `third_party/`, `.venv*/`, `runs/`, `renders/`, `ckpt/`,
`sag_table.json`, anything in `assets/backdrops/`.

## Troubleshooting

- **OptiX fails at start-up** (context creation or module errors mentioning an ABI or version): the driver is
  older than the OptiX 9.1 headers. Update the NVIDIA driver; do not edit the device code.
- **"minor Version ... newer than tool" or other module-load errors**: someone switched the build to OptiX-IR.
  The device code must be PTX (`native/build.sh` as shipped).
- **PTX compile fails with an unsupported `-arch`**: set `PTX_ARCH` to what your `nvcc` supports and the GPU can
  run (never above the GPU's capability).
- **`PI05Policy.from_pretrained` rejects a config key**: make a shim (step 6).
- **Loading fails on an unknown processor step `relative_actions_processor`**: `relative_actions_compat.py` was
  not imported; it registers that step name. `run_policy_sim.py` imports it; your own scripts must too.
- **CUDA out of memory at load**: another process holds the model (a policy server, a second runner). One fp32
  model is about 15.5 GiB.
- **Policy reaches but closes beside the fruit, or hedges**: check, in this order, the tokenizer (`[2, ...]`),
  that `--auto-exposure 150` is on, the camera framing versus the training frames, and the state convention
  (a constant shoulder-lift offset of a few degrees between the data rig and the model's kinematics is enough;
  `--joint-offset-vec-deg 0 -4 0 0 0 0` is the probe for it).

## Rules for agents working here

- Never add imports of `i2rt`, `can` or executor code to this repository.
- Long runs: launch detached, poll the `aggregate.json` files, and never edit a shell script while it is running.
- When killing by pattern, use patterns that cannot match your own shell (`pgrep -f "run_policy_si[m]"`), or
  kill by PID.
- Write results to new, uniquely named output directories; do not delete or overwrite earlier runs to make room.
- Compare numbers only within one protocol (same clamp, exposure mode, dynamics, scene seeds, duration).
- A README is maintained by hand by the authors; do not generate or overwrite it.
