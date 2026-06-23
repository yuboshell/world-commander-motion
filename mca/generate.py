"""Triple generator: canonical command -> sim trajectories (L1) -> free-form text (L2)
-> motion (L3). Produces (command_text, command_canonical, init_state, targets,
trajectory, motion) triples — the synthetic training/eval data for the v0."""
from __future__ import annotations

from .commands import targets_for
from .world import World


def generate_triples(commands, paraphraser, renderer, *, n_agents=5, steps=60,
                     variants_per_command=4, bounds=10.0, seed=0):
    """Generate `variants_per_command` triples per command (different random init each)."""
    out = []
    for i, cmd in enumerate(commands):
        for v in range(variants_per_command):
            w = World.random_init(n_agents, bounds=bounds, seed=seed * 1000 + i * 17 + v)
            init = w.pos.copy()
            w.target = targets_for(cmd, w)
            traj = w.rollout(steps)
            out.append({
                "command_text": paraphraser(cmd),
                "command_canonical": {"kind": cmd.kind, "params": cmd.params},
                "init_state": init,
                "targets": w.target.copy(),
                "trajectory": traj,
                "motion": renderer(traj),
            })
    return out
