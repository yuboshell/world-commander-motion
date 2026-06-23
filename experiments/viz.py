"""Visualisation: turn L1 trajectories into something you can SEE. Produces a static trail
montage (one panel per canonical command) and an animated GIF per command. GPU-free
(matplotlib + Pillow/ffmpeg). This is the qualitative companion to the numeric metrics."""
from __future__ import annotations

import numpy as np
from matplotlib import animation

from common import CANON, ORDER, FIGS, simulate, savefig, plt

N_AGENTS = 12
STEPS = 90
SEED = 7


def _draw_panel(ax, kind, tr):
    traj, tgt = tr["trajectory"], tr["targets"]
    n = traj.shape[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n):
        ax.plot(traj[:, i, 0], traj[:, i, 1], "-", color=cmap[i], lw=1.0, alpha=0.6)
    ax.scatter(traj[0, :, 0], traj[0, :, 1], facecolors="none", edgecolors=cmap,
               s=55, lw=1.5, label="start")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color=cmap, s=55, label="end")
    ax.scatter(tgt[:, 0], tgt[:, 1], marker="x", color="red", s=45, alpha=0.7, label="target")
    ax.set(title=kind, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect("equal"); ax.grid(alpha=0.25)


def montage():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.ravel()
    for ax, kind in zip(axes, ORDER):
        _draw_panel(ax, kind, simulate(CANON[kind], n=N_AGENTS, steps=STEPS, seed=SEED))
    axes[2].legend(loc="upper right", fontsize=8)
    axes[-1].axis("off")
    axes[-1].text(0.5, 0.5, f"L1 crowd sim\n{N_AGENTS} agents, {STEPS} steps\n"
                  "hollow=start  filled=end\nred x = commanded target",
                  ha="center", va="center", fontsize=11)
    fig.suptitle("What the pipeline produces: canonical command -> coordinated trajectories",
                 fontsize=13)
    savefig(fig, "trail_montage.png")


def animate_command(kind, n=N_AGENTS, steps=STEPS, seed=SEED, fps=20):
    tr = simulate(CANON[kind], n=n, steps=steps, seed=seed)
    traj, tgt = tr["trajectory"], tr["targets"]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(0, 10), ylim=(0, 10), title=f"{kind}")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.scatter(tgt[:, 0], tgt[:, 1], marker="x", color="red", s=45, alpha=0.6)
    trails = [ax.plot([], [], "-", color=cmap[i], lw=1.0, alpha=0.5)[0] for i in range(n)]
    dots = ax.scatter(traj[0, :, 0], traj[0, :, 1], color=cmap, s=60, zorder=3)
    tail = 25

    def update(f):
        dots.set_offsets(traj[f])
        lo = max(0, f - tail)
        for i, ln in enumerate(trails):
            ln.set_data(traj[lo:f + 1, i, 0], traj[lo:f + 1, i, 1])
        ax.set_title(f"{kind}  (step {f}/{steps})")
        return [dots, *trails]

    anim = animation.FuncAnimation(fig, update, frames=traj.shape[0], interval=1000 / fps, blit=False)
    out = FIGS / f"anim_{kind}.gif"
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote experiments/figures/{out.name}")


if __name__ == "__main__":
    montage()
    for kind in ORDER:
        animate_command(kind)
