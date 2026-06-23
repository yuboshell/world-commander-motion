"""v0 metrics. For the sim, grounding is high by construction — the real test is when
a trained model is in the loop (on amax); these define the measurement layer now."""
from __future__ import annotations

import numpy as np


def grounding(triple, tol: float = 0.8) -> float:
    """Fraction of agents that end within `tol` of the command's target."""
    final = triple["trajectory"][-1]
    d = np.linalg.norm(final - triple["targets"], axis=1)
    return float((d < tol).mean())


def collision_rate(triple, radius: float = 0.3) -> float:
    """Fraction of timesteps with at least one agent-pair closer than `radius`."""
    traj = triple["trajectory"]                              # (T, N, 2)
    hits = 0
    for frame in traj:
        d = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        if (d < radius).any():
            hits += 1
    return hits / len(traj)
