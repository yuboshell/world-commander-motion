"""E2 — Convergence. How fast does the crowd satisfy each command? We roll out once to a
long horizon and evaluate grounding / formation error at many intermediate horizons
(slicing the trajectory), averaged over inits. The knee of each curve is the command's
effective completion time — a coordination/real-time-budget quantity from the plan."""
from __future__ import annotations

import numpy as np

from common import CANON, ORDER, COLORS, SIM_FPS, simulate, save_json, savefig, plt
from mca.metrics import formation_error, grounding

N_AGENTS = 8
MAX_STEPS = 200
HORIZONS = list(range(5, MAX_STEPS + 1, 5))
VARIANTS = 16


def run():
    out = {}
    for kind in ORDER:
        cmd = CANON[kind]
        g_curves, f_curves = [], []
        for v in range(VARIANTS):
            tr = simulate(cmd, n=N_AGENTS, steps=MAX_STEPS, seed=v)
            traj = tr["trajectory"]
            g_row, f_row = [], []
            for h in HORIZONS:
                sub = {"trajectory": traj[:h + 1], "targets": tr["targets"]}
                g_row.append(grounding(sub)); f_row.append(formation_error(sub))
            g_curves.append(g_row); f_curves.append(f_row)
        out[kind] = {"grounding": np.mean(g_curves, axis=0).tolist(),
                     "formation_error": np.mean(f_curves, axis=0).tolist()}
    return out


def report(out):
    print(f"\nE2 CONVERGENCE  (n={N_AGENTS}, {VARIANTS} inits, horizons {HORIZONS[0]}..{HORIZONS[-1]})")
    print("steps-to-grounding>=0.9 (and seconds @30fps):")
    knees = {}
    for k in ORDER:
        g = out[k]["grounding"]
        idx = next((i for i, val in enumerate(g) if val >= 0.9), None)
        step = HORIZONS[idx] if idx is not None else None
        knees[k] = step
        s = f"{step} steps ({step / SIM_FPS:.2f}s)" if step else "not within horizon"
        print(f"  {k:10s} {s}")
    out["_knees_g0.9_steps"] = knees
    save_json("exp2_convergence.json", out)


def figure(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in ORDER:
        axes[0].plot(HORIZONS, out[k]["grounding"], label=k, color=COLORS[k], lw=2)
        axes[1].plot(HORIZONS, out[k]["formation_error"], label=k, color=COLORS[k], lw=2)
    axes[0].axhline(0.9, ls="--", c="gray", lw=1, alpha=0.7)
    axes[0].set(xlabel="rollout steps", ylabel="grounding (frac at target)",
                title="Grounding vs horizon", ylim=(0, 1.05))
    axes[1].set(xlabel="rollout steps", ylabel="formation error (units)",
                title="Formation error vs horizon")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"E2 — command convergence (n={N_AGENTS} agents)", fontsize=11)
    savefig(fig, "exp2_convergence.png")


if __name__ == "__main__":
    out = run()
    report(out)
    figure(out)
