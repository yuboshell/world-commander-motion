"""Composite per-agent OmniControl motion into ONE coordinated crowd scene — the dots demo with
the dots replaced by walking characters.

OmniControl gives a realistic *walk cycle* but does not honour absolute world placement (it collapses
each agent near the path centre). So we use the model only for the body articulation and **re-root**
each character onto its TRUE L1 trajectory (scattered -> formation), facing its direction of travel.
Positions/coordination are therefore exact (from L1); the limbs are the generated motion. No
regeneration — this runs on the already-generated joints.

    python experiments/crowd_motion_demo.py form_line       # -> figures/crowd_form_line.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402
from mpl_toolkits.mplot3d import Axes3D               # noqa: F401,E402

CMD = sys.argv[1] if len(sys.argv) > 1 else "form_line"
SCALE = 0.6
OMNI = Path("/mnt/yubo/repos/OmniControl")
JDIR = OMNI / f"crowd_out/crowd_{CMD}"
PDIR = OMNI / f"crowd_paths/crowd_{CMD}"
FIG = Path(__file__).resolve().parent / "figures"
CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def _smooth(v, k=9):
    pad = np.pad(v, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(v.shape[1])], 1)


def reroot(J, world):
    """J (T,22,3) generated (x, y=up, z); world (T,2) true path [x,z] in metres.
    Keep limbs+height, replace horizontal root with the path, rotate to face travel direction."""
    T = J.shape[0]
    rel = J.copy()
    rel[:, :, 0] -= J[:, 0:1, 0]                        # pelvis-relative in x, z (keep height y)
    rel[:, :, 2] -= J[:, 0:1, 2]
    # facing: forward is perpendicular to the hip line (joint1 L-hip, joint2 R-hip)
    hip = rel[:, 2, [0, 2]] - rel[:, 1, [0, 2]]
    fwd = np.stack([hip[:, 1], -hip[:, 0]], 1)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-8
    vel = _smooth(np.gradient(world, axis=0))          # travel direction (smoothed)
    sp = np.linalg.norm(vel, axis=1, keepdims=True)
    vdir = np.where(sp > 1e-4, vel / (sp + 1e-8), fwd)  # hold facing when ~stationary
    ang = np.arctan2(vdir[:, 1], vdir[:, 0]) - np.arctan2(fwd[:, 1], fwd[:, 0])
    c, s = np.cos(ang), np.sin(ang)
    x, z = rel[:, :, 0].copy(), rel[:, :, 2].copy()
    rel[:, :, 0] = c[:, None] * x - s[:, None] * z
    rel[:, :, 2] = s[:, None] * x + c[:, None] * z
    out = rel
    out[:, :, 0] += world[:, 0:1]
    out[:, :, 2] += world[:, 1:2]
    out[:, :, 1] = J[:, :, 1]                           # original height (bob preserved)
    return out


def load_agents():
    agents, trails, i = [], [], 0
    while (JDIR / f"a{i}.npy").exists():
        J = np.load(JDIR / f"a{i}.npy").astype(float)
        path = np.load(PDIR / f"a{i}.npy").astype(float) * SCALE     # (121,2) world->m
        T = J.shape[0]
        w = np.stack([np.interp(np.linspace(0, len(path) - 1, T), np.arange(len(path)), path[:, k])
                      for k in (0, 1)], 1)                            # resample path to T
        agents.append(reroot(J, w)); trails.append(w); i += 1
    if not agents:
        raise SystemExit(f"no joints in {JDIR}")
    return agents, trails


def main():
    agents, trails = load_agents()
    T = agents[0].shape[0]
    tgt = np.load(PDIR / "targets.npy").astype(float) * SCALE
    cols = plt.cm.tab10(np.linspace(0, 1, 10))[:len(agents)]
    ext = 10 * SCALE
    gl = np.linspace(0, ext, 7)
    frames = list(range(0, T, 2)) + [T - 1] * 12

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    def draw(f):
        ax.clear()
        ax.view_init(elev=18, azim=-58)
        ax.set_xlim(0, ext); ax.set_ylim(0, ext); ax.set_zlim(0, 1.9)
        ax.set_box_aspect((1, 1, 0.4))
        for g in gl:
            ax.plot([g, g], [0, ext], [0, 0], color="0.85", lw=0.6, zorder=0)
            ax.plot([0, ext], [g, g], [0, 0], color="0.85", lw=0.6, zorder=0)
        ax.scatter(tgt[:, 0], tgt[:, 1], np.zeros(len(tgt)), marker="x", color="red", s=60, lw=2.2, zorder=2)
        for tr, c in zip(trails, cols):
            ax.plot(tr[:, 0], tr[:, 1], np.zeros(len(tr)), color=c, lw=1.0, alpha=0.4, zorder=1)
        for J, c in zip(agents, cols):
            x, y, z = J[f, :, 0], J[f, :, 1], J[f, :, 2]
            for ch in CHAIN:
                ax.plot([x[j] for j in ch], [z[j] for j in ch], [y[j] for j in ch],
                        color=c, lw=3.0, solid_capstyle="round", zorder=5)
            ax.scatter([x[0]], [z[0]], [0], color=c, s=16, alpha=0.5, zorder=3)
        ax.set_title(f"“{CMD}” — coordinated crowd motion  (step {min(f, T - 1)}/{T - 1})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=50)
    out = FIG / f"crowd_{CMD}.mp4"
    anim.save(out, writer=animation.FFMpegWriter(fps=20, bitrate=2400))
    plt.close(fig)
    print(f"wrote {out}  ({len(agents)} agents, {T} frames)")


if __name__ == "__main__":
    main()
