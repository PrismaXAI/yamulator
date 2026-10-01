"""Make a LeRobot pi0.5 checkpoint loadable by the pinned inference stack (lerobot 0.5.1) without touching it.

A checkpoint saved by a newer or forked LeRobot can carry config.json keys that PI05Config 0.5.1 does not know
(ours had `pretrained_revision`), and `PI05Policy.from_pretrained` then refuses the whole config. The shim is a
directory with a filtered copy of config.json and symlinks to every other file of the checkpoint (weights are not
copied). Pass the shim directory wherever a --ckpt is expected.

  python tools/make_ckpt_shim.py <.../checkpoints/<step>/pretrained_model> ckpt/<name> [--follower] [--drop-key K ...]

Keys to drop are detected from the installed lerobot (anything PI05Config has no field for); without lerobot in the
interpreter only `pretrained_revision` and the --drop-key names are dropped.
--follower writes a FOLLOWER_ACTIONS marker: the checkpoint's actions are achievable follower positions, so the
data-rig sag compensation (sag_comp.py) must never be applied to it. Checkpoints whose train_config.json mentions
"follower" are detected without the marker.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_DROP = {"pretrained_revision"}


def unknown_keys(cfg: dict) -> set:
    try:
        import dataclasses
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except Exception as e:
        print(f"(lerobot not importable here: {e!r}; dropping only {sorted(DEFAULT_DROP)} + --drop-key)")
        return DEFAULT_DROP & set(cfg)
    known = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
    return set(cfg) - known


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="checkpoint directory holding config.json + model.safetensors (the pretrained_model dir)")
    ap.add_argument("dst", help="shim directory to create")
    ap.add_argument("--follower", action="store_true", help="mark as a follower-action checkpoint (no sag compensation)")
    ap.add_argument("--drop-key", action="append", default=[], help="additional config.json key to drop (repeatable)")
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst)
    if not (src / "config.json").exists() or not (src / "model.safetensors").exists():
        sys.exit(f"{src} does not look like a checkpoint (config.json + model.safetensors expected)")
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    drop = (unknown_keys(cfg) | set(a.drop_key)) & set(cfg)
    for k in sorted(drop):
        print(f"config.json: dropping {k!r} = {cfg.pop(k)!r}")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in sorted(src.iterdir()):
        if f.name == "config.json":
            continue
        link = dst / f.name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            sys.exit(f"{link} exists and is not a symlink: not overwriting")
        link.symlink_to(f)
    if a.follower:
        (dst / "FOLLOWER_ACTIONS").write_text("actions = follower positions: no sag compensation\n")
    print(f"shim {dst} -> {src} ({'follower-action' if a.follower else 'no follower marker'}, "
          f"relative actions: {bool(cfg.get('use_relative_actions', False))})")


if __name__ == "__main__":
    main()
