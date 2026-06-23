"""E3 — Scaling. Hold the world fixed and grow the crowd. Coordination metrics
(grounding, collision, formation error) are averaged over all commands and inits;
throughput (steps/sec) is timed on the sim itself. Because separation is an O(N^2)
all-pairs step, the throughput curve also exposes the real-time ceiling of the naive
v0 sim — a concrete v1 to-do (spatial hashing / neighbour lists)."""
from __future__ import annotations

import numpy as np

from common import CANON, ORDER, simulate, sim_throughput, save_json, savefig, plt
from mca.metrics import collision_rate, formation_error, grounding

NS = [3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200]
STEPS = 120
VARIANTS = 4


def run():
    res = {"N": NS, "grounding": [], "collision": [], "formation_error": [], "steps_per_s": []}
    for n in NS:
        G, C, F = [], [], []
        for kind in ORDER:
            for v in range(VARIANTS):
                tr = simulate(CANON[kind], n=n, steps=STEPS, seed=v)
                G.append(grounding(tr)); C.append(collision_rate(tr)); F.append(formation_error(tr))
        thr = sim_throughput(CANON["go_to"], n=n, steps=100, repeats=3)
        res["grounding"].append(float(np.mean(G)))
        res["collision"].append(float(np.mean(C)))
        res["formation_error"].append(float(np.mean(F)))
        res["steps_per_s"].append(thr["steps_per_s"])
        print(f"  N={n:4d}  grounding={np.mean(G):.2f}  collision={np.mean(C):.2f}  "
              f"form.err={np.mean(F):.2f}  {thr['steps_per_s']:.0f} steps/s "
              f"({thr['agent_steps_per_s']:.0f} agent-steps/s)")
    return res


def report(res):
    print(f"\nE3 SCALING  (avg over {len(ORDER)} commands x {VARIANTS} inits, {STEPS} steps)")
    save_json("exp3_scaling.json", res)


def figure(res):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].plot(res["N"], res["grounding"], "o-", color="#2ca02c")
    axes[0].set(title="grounding vs crowd size", xlabel="N agents",
                ylabel="grounding", ylim=(0, 1.05))
    axes[1].plot(res["N"], res["collision"], "o-", color="#d62728")
    axes[1].set(title="collision rate vs crowd size", xlabel="N agents", ylabel="collision rate")
    axes[2].plot(res["N"], res["formation_error"], "o-", color="#ff7f0e")
    axes[2].set(title="formation error vs crowd size", xlabel="N agents", ylabel="units")
    axes[3].loglog(res["N"], res["steps_per_s"], "o-", color="#1f77b4")
    axes[3].set(title="sim throughput (O(N^2))", xlabel="N agents", ylabel="steps / sec")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("E3 — scaling the crowd in a fixed 10x10 world", fontsize=11)
    savefig(fig, "exp3_scaling.png")


if __name__ == "__main__":
    res = run()
    report(res)
    figure(res)
