"""Motion renderer (data layer L3): per-agent trajectory -> full-body motion.

StubRenderer passes the path through (offline, no GPU) so the pipeline runs end to end;
RealRenderer wraps a pretrained controllable motion model (TLControl / CAMDM) on a GPU
and is filled in on amax."""
from __future__ import annotations

import numpy as np


class StubRenderer:
    """No motion model. Returns the trajectory plus a placeholder per-agent label so the
    pipeline is exercisable offline. Replace with RealRenderer on the GPU box."""

    def __call__(self, traj: np.ndarray) -> dict:
        # traj: (T, N, 2). Placeholder "motion" = the 2D path + a coarse walk/idle label.
        speed = np.linalg.norm(np.diff(traj, axis=0), axis=2)         # (T-1, N)
        labels = ["walk" if speed[:, i].max() > 1e-3 else "idle" for i in range(traj.shape[1])]
        return {"trajectory": traj, "per_agent_label": labels, "renderer": "stub"}


class RealRenderer:
    """Wrap a pretrained controllable motion model (e.g. TLControl / CAMDM).

    On amax: load the model, and for each agent map trajectory (+ optional style) ->
    full-body motion (joint rotations / SMPL). NOT implemented here — needs the GPU and
    model weights. See plan/research-plan.md, data layer L3."""

    def __init__(self, model_name: str = "TLControl"):
        raise NotImplementedError(
            "RealRenderer: load a controllable motion model (TLControl / CAMDM) on amax and "
            "map per-agent trajectory + style -> full-body motion. See the research plan, L3."
        )
