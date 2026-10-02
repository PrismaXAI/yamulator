"""Client for the prism-pi policy server (prism_pi.policy.server). RemoteRunner mirrors the executor's Pi05Runner.infer() interface, so the control code
does not care whether the model is local or served. No torch import needed on the client side."""
import os
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

# one well-known per-user path, so the server (prism-pi) and any client (executor, web UI, simulator) agree without sharing a directory
SOCK = os.environ.get("YAM_POLICY_SOCK", f"/tmp/prism-pi-policy-{os.getuid()}.sock")
AUTH = b"yam-policy"


def server_info(timeout_ok: bool = True):
    """Return the server's info dict, or None if no server is reachable."""
    try:
        c = Client(address=SOCK, family="AF_UNIX", authkey=AUTH)
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return None
    try:
        c.send({"op": "info"}); return c.recv()
    finally:
        c.close()


class RemoteRunner:
    def __init__(self, ckpt: str, dtype: str = "bf16", allow_swap: bool = True):
        self.conn = Client(address=SOCK, family="AF_UNIX", authkey=AUTH)
        info = self._call({"op": "info"})
        want = str(Path(ckpt).resolve())
        if info.get("ckpt") != want or info.get("dtype") != dtype:
            if not allow_swap:
                raise RuntimeError(f"policy server holds {info.get('ckpt')} ({info.get('dtype')}), wanted {want} ({dtype})")
            print(f"== policy server: swapping checkpoint {info.get('ckpt')} -> {want} ({dtype}) ==", flush=True)
            info = self._call({"op": "load", "ckpt": want, "dtype": dtype})
        self.info = info
        self.last_costs = None

    def _call(self, req: dict) -> dict:
        self.conn.send(req)
        rep = self.conn.recv()
        if not rep.get("ok"):
            raise RuntimeError(f"policy server error: {rep.get('error')}\n{rep.get('traceback', '')}")
        return rep

    def infer_k(self, frames: dict, state14, task: str, n_samples: int = 1, seed=None, rtc=None):
        """-> (chunks (K,50,14) absolute, norm (K,50,14), latency_s). rtc = {"prev_abs": (H,14) absolute tail of the
        running plan starting at the observation tick, "delay": ticks until install, "horizon": guidance end}."""
        rep = self._call({"op": "infer", "frames": frames, "state": np.asarray(state14, np.float32), "task": task,
                          "n_samples": n_samples, "seed": seed, "rtc": rtc})
        return rep["chunks"], rep["norm"], rep["latency_s"]

    def infer(self, frames: dict, state14, task: str, leftover_norm=None, delay_steps: int = 0):
        """Drop-in for Pi05Runner.infer (RTC arguments are ignored: RTC needs the local runner)."""
        chunks, norm, _ = self.infer_k(frames, state14, task, 1)
        return norm[0], chunks[0]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def shutdown():
    """Ask the server to exit (used for checkpoint swaps: a fresh process per checkpoint)."""
    c = Client(address=SOCK, family="AF_UNIX", authkey=AUTH)
    try:
        c.send({"op": "shutdown"})
        try:
            c.recv()
        except EOFError:
            pass
    finally:
        c.close()


if __name__ == "__main__":  # tiny control CLI:  python -m yamulator.policy.client info | load <ckpt> [dtype] | shutdown
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "info":
        print(server_info() or "no policy server reachable")
    elif cmd == "load":
        rr = RemoteRunner(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "bf16"); print(rr.info); rr.close()
    elif cmd == "shutdown":
        c = Client(address=SOCK, family="AF_UNIX", authkey=AUTH); c.send({"op": "shutdown"})
        try:
            print(c.recv())
        except EOFError:
            print("server stopped")
    else:
        raise SystemExit("usage: python -m yamulator.policy.client info | load <ckpt> [dtype] | shutdown")
