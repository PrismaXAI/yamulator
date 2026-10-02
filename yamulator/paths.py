"""Where things live. Everything hangs off the repository root, so a checkout is self-contained."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ASSETS = ROOT / "assets"                      # YAM URDF/MJCF + meshes, apple, wood textures, backdrops/
URDF = ASSETS / "yam" / "yam.urdf"
BUILD = ROOT / "build"                        # native/build.sh output: libyamtrace.so + device.ptx
SAG_TABLE = ROOT / "config" / "sag_table.json"   # optional data-rig sag offsets (copy config/sag_table.example.json)

# outputs (git-ignored)
RUNS = ROOT / "runs"
RENDERS = ROOT / "renders"
