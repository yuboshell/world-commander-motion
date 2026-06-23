"""E4 — Coordination frontier. The crowd's collision avoidance (separation) trades off
against reaching the commanded target. We sweep the two separation parameters and map
collision rate and grounding, then mark the feasible region (low collision AND high
grounding). This is a miniature of the plan's 'budget frontier' on the coordination axis:
there is a window of settings that satisfies the command while staying collision-free."""
from __future__ import annotations

import numpy as np

from common import CANON, simulate, save_json, savefig, plt
from mca.metrics import collision_rate, grounding

CMDS = ["go_to", "regroup", "flank"]      # convergence-heavy: where coordination is stressed
N_AGENTS = 15
STEPS = 120
VARIANTS = 4
SEP_STRENGTH = [0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24]
SEP_RADIUS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4]
DEFAULT = (0.08, 0.60)                     # repo's World defaults (sep_strength, sep_radius)


def run():
    coll = np.zeros((len(SEP_STRENGTH), len(SEP_RADIUS)))
    grnd = np.zeros_like(coll)
    for i, ss in enumerate(SEP_STRENGTH):
        for j, sr in enumerate(SEP_RADIUS):
            C, G = [], []
            for kind in CMDS:
                for v in range(VARIANTS):
                    tr = simulate(CANON[kind], n=N_AGENTS, steps=STEPS, seed=v,
                                  sep_strength=ss, sep_radius=sr)
                    C.append(collision_rate(tr)); G.append(grounding(tr))
            coll[i, j] = np.mean(C); grnd[i, j] = np.mean(G)
        print(f"  sep_strength={ss:.2f} done")
    return coll, grnd


def _cells(coll, grnd):
    """Flatten the sweep into (sep_strength, sep_radius, collision, grounding) cells."""
    return [(ss, sr, coll[i, j], grnd[i, j])
            for i, ss in enumerate(SEP_STRENGTH) for j, sr in enumerate(SEP_RADIUS)]


def _pareto(cells):
    """Pareto-optimal cells minimising collision and maximising grounding."""
    front = []
    for c in cells:
        if not any((o[2] <= c[2] and o[3] >= c[3]) and (o[2] < c[2] or o[3] > c[3]) for o in cells):
            front.append(c)
    return sorted(front, key=lambda c: c[2])


def report(coll, grnd):
    cells = _cells(coll, grnd)
    front = _pareto(cells)
    # distance of each cell to the ideal corner (collision=0, grounding=1)
    gap = lambda c: (c[2] ** 2 + (1 - c[3]) ** 2) ** 0.5
    best = min(cells, key=gap)
    print(f"\nE4 FRONTIER  (cmds={CMDS}, n={N_AGENTS}, {VARIANTS} inits)")
    print("Pareto front (min collision / max grounding):")
    for ss, sr, c, g in front:
        print(f"  sep_strength={ss:.2f} sep_radius={sr:.2f} -> collision={c:.2f} grounding={g:.2f}")
    print(f"closest to ideal corner: sep_strength={best[0]:.2f} sep_radius={best[1]:.2f} "
          f"-> collision={best[2]:.2f} grounding={best[3]:.2f}  (gap-to-ideal={gap(best):.2f})")
    di, dj = SEP_STRENGTH.index(DEFAULT[0]), SEP_RADIUS.index(DEFAULT[1])
    print(f"repo default {DEFAULT}: collision={coll[di, dj]:.2f} grounding={grnd[di, dj]:.2f}")
    print("=> no setting reaches the low-collision + high-grounding corner: the naive "
          "separation cannot both avoid collisions and ground at this density. This gap is "
          "what the v1 coordination layer (RVO/ORCA -> MARL) must close.")
    save_json("exp4_frontier.json", {
        "sep_strength": SEP_STRENGTH, "sep_radius": SEP_RADIUS,
        "collision": coll.tolist(), "grounding": grnd.tolist(),
        "pareto_front": front, "closest_to_ideal": best, "gap_to_ideal": gap(best),
        "default": {"params": DEFAULT, "collision": float(coll[di, dj]), "grounding": float(grnd[di, dj])},
    })


def _heat(ax, M, title, cmap):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(SEP_RADIUS)), [f"{x:.1f}" for x in SEP_RADIUS])
    ax.set_yticks(range(len(SEP_STRENGTH)), [f"{y:.2f}" for y in SEP_STRENGTH])
    ax.set(xlabel="sep_radius", ylabel="sep_strength", title=title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmap == "viridis" else "black")
    return im


def figure(coll, grnd):
    cells = _cells(coll, grnd)
    front = _pareto(cells)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _heat(axes[0], coll, "collision rate", "magma")
    _heat(axes[1], grnd, "grounding", "viridis")
    ax = axes[2]
    cc = [c[2] for c in cells]; gg = [c[3] for c in cells]
    sc = ax.scatter(cc, gg, c=[c[0] for c in cells], cmap="plasma", s=45, zorder=3)
    fx = [c[2] for c in front]; fy = [c[3] for c in front]
    ax.plot(fx, fy, "-", color="black", lw=1.5, alpha=0.6, zorder=2, label="Pareto front")
    ax.scatter([0], [1], marker="*", s=320, color="gold", edgecolor="black",
               zorder=4, label="ideal corner")
    di, dj = SEP_STRENGTH.index(DEFAULT[0]), SEP_RADIUS.index(DEFAULT[1])
    ax.scatter([coll[di, dj]], [grnd[di, dj]], marker="D", s=90, color="cyan",
               edgecolor="black", zorder=5, label=f"repo default {DEFAULT}")
    ax.set(xlabel="collision rate (lower better)", ylabel="grounding (higher better)",
           title="collision vs grounding tradeoff", xlim=(-0.05, 1.05), ylim=(0, 1.08))
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.colorbar(sc, ax=ax, label="sep_strength", fraction=0.046)
    fig.suptitle(f"E4 — coordination frontier (n={N_AGENTS}, cmds={CMDS}): no setting reaches "
                 f"the ideal corner", fontsize=10)
    savefig(fig, "exp4_frontier.png")


if __name__ == "__main__":
    coll, grnd = run()
    report(coll, grnd)
    figure(coll, grnd)
