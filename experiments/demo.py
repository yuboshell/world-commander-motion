"""End-to-end demo (the capstone): a free-form order -> served Qwen interpreter [A] ->
canonical command -> L1 sim -> rendered crowd motion. Runs the WHOLE v0 pipeline with the
real LLM and the coordinate guard. Prints what was recovered + grounding, and saves a GIF
per scenario plus a static montage.

Usage:
    python experiments/demo.py                      # built-in concrete + abstract orders
    python experiments/demo.py "swing around and hit them from both sides at 9,2"
"""
from __future__ import annotations

import sys

import numpy as np
from matplotlib import animation

from common import FIGS, savefig, plt
from mca.commands import Command, targets_for
from mca.interpret import LLMInterpreter
from mca.metrics import collision_rate, grounding
from mca.world import World

BASE, KEY, MODEL = "http://localhost:8000/v1", "EMPTY", "Qwen/Qwen3-14B-AWQ"
N, STEPS, SEED = 12, 90, 5
_NEEDED = {"go_to": ["point"], "regroup": ["point"], "flank": ["point"],
           "form_line": ["start", "end"], "disperse": []}
# canonical defaults used when an abstract order specifies no coordinates (the guard deferred them)
_DEFAULTS = {"go_to": {"point": [8.0, 8.0]}, "regroup": {"point": [5.0, 5.0]},
             "flank": {"point": [8.0, 5.0]}, "form_line": {"start": [3.0, 5.0], "end": [7.0, 5.0]},
             "disperse": {}}

ORDERS = [
    "everyone push hard to the top-right corner at 8,8",   # concrete go_to
    "fall back and regroup at the center",                 # abstract regroup
    "spread out and take cover, give yourselves room",     # abstract disperse
    "form a firing line across the middle",                # abstract form_line
    "swing wide and envelop them from both flanks",        # abstract flank
]


def fill(rec: Command) -> Command:
    params = dict(rec.params)
    for k in _NEEDED.get(rec.kind, []):
        params.setdefault(k, _DEFAULTS[rec.kind][k])
    return Command(rec.kind, params)


def run_order(interp, text):
    w = World.random_init(N, seed=SEED)
    rec = interp(text, world=w)
    cmd = fill(rec)
    w.target = targets_for(cmd, w)
    traj = w.rollout(STEPS)
    triple = {"trajectory": traj, "targets": w.target.copy()}
    return {"text": text, "kind": cmd.kind, "params": cmd.params, "traj": traj,
            "targets": w.target.copy(), "grounding": grounding(triple),
            "collision": collision_rate(triple), "latency": interp.last_latency}


def animate(r, idx):
    traj, tgt = r["traj"], r["targets"]
    cmap = plt.cm.viridis(np.linspace(0, 1, N))
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.scatter(tgt[:, 0], tgt[:, 1], marker="x", color="red", s=40, alpha=0.6)
    trails = [ax.plot([], [], "-", color=cmap[i], lw=1.0, alpha=0.5)[0] for i in range(N)]
    dots = ax.scatter(traj[0, :, 0], traj[0, :, 1], color=cmap, s=60, zorder=3)
    fig.suptitle(f'"{r["text"]}"', fontsize=10)

    def upd(f):
        dots.set_offsets(traj[f])
        for i, ln in enumerate(trails):
            ln.set_data(traj[max(0, f - 25):f + 1, i, 0], traj[max(0, f - 25):f + 1, i, 1])
        ax.set_title(f'Qwen -> {r["kind"]} {r["params"]}  (step {f})', fontsize=9)
        return [dots, *trails]

    anim = animation.FuncAnimation(fig, upd, frames=traj.shape[0], interval=50, blit=False)
    out = FIGS / f"demo_{idx}_{r['kind']}.gif"
    anim.save(out, writer=animation.PillowWriter(fps=20))
    plt.close(fig)
    print(f"  wrote experiments/figures/{out.name}")


def montage(results):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4.2))
    for ax, r in zip(np.atleast_1d(axes), results):
        traj, tgt = r["traj"], r["targets"]
        cmap = plt.cm.viridis(np.linspace(0, 1, N))
        for i in range(N):
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=cmap[i], lw=0.9, alpha=0.6)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], facecolors="none", edgecolors=cmap, s=40, lw=1.3)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color=cmap, s=40)
        ax.scatter(tgt[:, 0], tgt[:, 1], marker="x", color="red", s=35, alpha=0.7)
        ax.set(xlim=(0, 10), ylim=(0, 10), title=f"{r['kind']}  (grnd {r['grounding']:.2f})")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel(f'"{r["text"][:34]}…"', fontsize=8)
    fig.suptitle("End-to-end demo: free-form order -> Qwen interpreter -> crowd motion", fontsize=12)
    savefig(fig, "demo_montage.png")


def main():
    interp = LLMInterpreter(BASE, KEY, MODEL)        # guard_coords=True by default
    orders = [sys.argv[1]] if len(sys.argv) > 1 else ORDERS
    print("END-TO-END DEMO  (free-form text -> Qwen [A] -> sim -> motion)\n")
    print(f"{'recovered':28s} {'grnd':>5s} {'coll':>5s} {'lat':>6s}  order")
    results = []
    for text in orders:
        r = run_order(interp, text)
        results.append(r)
        rc = f"{r['kind']} {r['params']}"
        print(f"{rc:28.28s} {r['grounding']:>5.2f} {r['collision']:>5.2f} "
              f"{r['latency']:>5.2f}s  \"{text}\"")
    print()
    for i, r in enumerate(results):
        animate(r, i)
    montage(results)


if __name__ == "__main__":
    main()
