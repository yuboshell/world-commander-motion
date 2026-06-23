"""E1 — Coverage. Run all five canonical commands through L1 and report the full v0
measurement layer (grounding, collision rate, formation error, completion time) as
mean +/- std over many random initialisations. This is the expanded baseline: the
original smoke run covered only 2 of the 5 commands and 2 of the 4 metrics."""
from __future__ import annotations

import numpy as np

from common import CANON, ORDER, COLORS, SIM_FPS, simulate, save_json, savefig, plt
from mca.metrics import collision_rate, completion_time, formation_error, grounding

N_AGENTS = 8
STEPS = 120
VARIANTS = 24


def run():
    rows = {}
    for kind in ORDER:
        cmd = CANON[kind]
        G, C, F, T = [], [], [], []
        for v in range(VARIANTS):
            tr = simulate(cmd, n=N_AGENTS, steps=STEPS, seed=v)
            G.append(grounding(tr)); C.append(collision_rate(tr))
            F.append(formation_error(tr)); T.append(completion_time(tr))
        rows[kind] = {
            "grounding": (np.mean(G), np.std(G)),
            "collision": (np.mean(C), np.std(C)),
            "formation_error": (np.mean(F), np.std(F)),
            "completion_steps": (np.mean(T), np.std(T)),
            "completion_sec": (np.mean(T) / SIM_FPS, np.std(T) / SIM_FPS),
        }
    return rows


def report(rows):
    print(f"\nE1 COVERAGE  (n={N_AGENTS} agents, {STEPS} steps, {VARIANTS} inits/command)")
    print(f"{'command':10s} {'grounding':>12s} {'collision':>12s} {'form.err':>12s} {'compl(s)':>12s}")
    for k in ORDER:
        r = rows[k]
        print(f"{k:10s} {r['grounding'][0]:>7.2f}±{r['grounding'][1]:<4.2f} "
              f"{r['collision'][0]:>7.2f}±{r['collision'][1]:<4.2f} "
              f"{r['formation_error'][0]:>7.2f}±{r['formation_error'][1]:<4.2f} "
              f"{r['completion_sec'][0]:>7.2f}±{r['completion_sec'][1]:<4.2f}")
    save_json("exp1_coverage.json", rows)


def figure(rows):
    metrics = [("grounding", "grounding (frac at target)", (0, 1.05)),
               ("collision", "collision rate", None),
               ("formation_error", "formation error (units)", None),
               ("completion_sec", "completion time (s @30fps)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (key, title, ylim) in zip(axes, metrics):
        means = [rows[k][key][0] for k in ORDER]
        errs = [rows[k][key][1] for k in ORDER]
        ax.bar(ORDER, means, yerr=errs, color=[COLORS[k] for k in ORDER], capsize=3)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"E1 — v0 measurement layer across all canonical commands "
                 f"(n={N_AGENTS}, {VARIANTS} inits)", fontsize=11)
    savefig(fig, "exp1_coverage.png")


if __name__ == "__main__":
    rows = run()
    report(rows)
    figure(rows)
