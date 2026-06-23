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


def formation_error(triple) -> float:
    """RMS distance (world units) of agents from their assigned targets at the final
    frame — the plan's 'formation error'. Lower is a tighter realized formation."""
    final = triple["trajectory"][-1]
    d = np.linalg.norm(final - triple["targets"], axis=1)
    return float(np.sqrt((d ** 2).mean()))


def completion_time(triple, tol: float = 0.8, frac: float = 0.9):
    """First frame index at which at least `frac` of agents are within `tol` of their
    targets (the plan's 'command-completion time', in steps). Returns the sentinel
    len(trajectory) if the command never completes within the horizon."""
    traj = triple["trajectory"]
    targets = triple["targets"]
    for t, frame in enumerate(traj):
        d = np.linalg.norm(frame - targets, axis=1)
        if (d < tol).mean() >= frac:
            return t
    return len(traj)
