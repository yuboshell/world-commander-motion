"""Shared helpers for the v0 experiment battery: the canonical command suite, a sim
driver with overridable world parameters, throughput timing, and plot/IO utilities.
Everything here is GPU-free (NumPy + matplotlib) per the repo contract."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")                   # headless
import matplotlib.pyplot as plt         # noqa: E402

from mca.commands import Command, targets_for   # noqa: E402
from mca.world import World                      # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

BOUNDS = 10.0
SIM_FPS = 30.0          # nominal playback rate: completion_time(steps) / SIM_FPS = seconds

# Representative canonical command suite (a 10x10 world). go_to/flank aim at a corner so
# the crowd has to travel; form_line spans the middle; regroup pulls to centre.
CANON = {
    "go_to":     Command("go_to",     {"point": [8.0, 8.0]}),
    "form_line": Command("form_line", {"start": [3.0, 5.0], "end": [7.0, 5.0]}),
    "regroup":   Command("regroup",   {"point": [5.0, 5.0]}),
    "disperse":  Command("disperse",  {}),
    "flank":     Command("flank",     {"point": [8.0, 5.0]}),   # open field: wings get room
}
ORDER = ["go_to", "form_line", "regroup", "disperse", "flank"]
COLORS = {"go_to": "#1f77b4", "form_line": "#ff7f0e", "regroup": "#2ca02c",
          "disperse": "#d62728", "flank": "#9467bd"}


def simulate(cmd: Command, n: int = 5, steps: int = 80, seed: int = 0, bounds: float = BOUNDS,
             **world_kw) -> dict:
    """Run L1 for one command and return a triple-shaped dict (trajectory/targets/init).
    `world_kw` overrides World fields (max_speed, sep_radius, sep_strength) for sweeps."""
    w = World.random_init(n, bounds=bounds, seed=seed)
    for k, v in world_kw.items():
        setattr(w, k, v)
    init = w.pos.copy()
    w.target = targets_for(cmd, w)
    traj = w.rollout(steps)
    return {"trajectory": traj, "targets": w.target.copy(), "init_state": init,
            "command_canonical": {"kind": cmd.kind, "params": cmd.params}}


def sim_throughput(cmd: Command, n: int, steps: int, seed: int = 0, repeats: int = 3) -> dict:
    """Measure L1 sim throughput: best-of-`repeats` steps/sec and agent-steps/sec."""
    best = float("inf")
    for _ in range(repeats):
        w = World.random_init(n, bounds=BOUNDS, seed=seed)
        w.target = targets_for(cmd, w)
        t0 = time.perf_counter()
        w.rollout(steps)
        best = min(best, time.perf_counter() - t0)
    steps_per_s = steps / best
    return {"n": n, "steps": steps, "sec": best, "steps_per_s": steps_per_s,
            "agent_steps_per_s": steps_per_s * n}


def pmap(fn, items, workers: int = 8):
    """Threaded map preserving order — for I/O-bound served-LLM calls."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def save_json(name: str, obj) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  wrote {p.relative_to(ROOT.parent)}")
    return p


def savefig(fig, name: str) -> Path:
    p = FIGS / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT.parent)}")
    return p
