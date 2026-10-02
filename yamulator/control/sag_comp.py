"""Feed-forward compensation for the data rig's follower sag (per arm).

In the training data the follower arm sat below the leader command under gravity: state - action
is a smooth function of shoulder lift (~+0.15 rad lift, -0.10 elbow, -0.03 wrist_1 at a typical
grasp), identical across sessions, slightly larger on the right arm. The policy emits leader-like
actions and expects follower-like states; our arms track stiffly, so identical actions leave the
fingers ~10 cm high. Add the dataset offset to each arm's 6 joint targets, indexed by that arm's
current shoulder-lift state. Validated 2026-09-13: compensated ep2900 replay put the fingertips on
the fruit at the dataset grasp onset.

The offsets are a property of the rig the TRAINING DATA was recorded on, so the table is not shipped as a default:
copy config/sag_table.example.json (the table measured for our dataset) to config/sag_table.json, or write your own
(median state - action per shoulder-lift bin, per arm). Without it the compensation is unavailable and
`--sag-comp auto` runs without it.
"""
import json

import numpy as np

from yamulator.paths import SAG_TABLE as TABLE


def available() -> bool:
    return TABLE.exists()


if available():
    _T = json.loads(TABLE.read_text())
    _C = np.array(_T["lift_centers"])
    _D = {"left": np.array(_T["left"]), "right": np.array(_T["right"])}  # (n, 5): lift, elbow, wr1, wr2, wr3


def sag_delta6(lift_state: float, side: str = "left") -> np.ndarray:
    """6-vector [pan, lift, elbow, wr1, wr2, wr3] to ADD to the commanded arm joints."""
    if not available():
        raise FileNotFoundError(f"{TABLE} not found: sag compensation needs a sag table (see config/sag_table.example.json)")
    d = np.array([np.interp(lift_state, _C, _D[side][:, j]) for j in range(5)])
    return np.array([0.0, d[0], d[1], d[2], d[3], d[4]])


def compensate14(target14: np.ndarray, state14: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Apply the sag offset to a 14-dim action, using each arm's measured shoulder lift."""
    out = np.array(target14, dtype=float, copy=True)
    out[0:6] += gain * sag_delta6(float(state14[1]), "left")
    out[7:13] += gain * sag_delta6(float(state14[8]), "right")
    return out
