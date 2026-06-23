"""Canonical commands -> per-agent target positions (data layer)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Command:
    kind: str            # 'go_to' | 'form_line' | 'regroup' | 'disperse'
    params: dict


def targets_for(cmd: Command, world) -> np.ndarray:
    """Return (n, 2) per-agent targets for a canonical command, given the world state."""
    n = world.n
    if cmd.kind == "go_to":
        p = np.array(cmd.params["point"], dtype=float)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)   # small ring so agents don't stack
        return p[None, :] + 0.5 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if cmd.kind == "form_line":
        a = np.array(cmd.params["start"], dtype=float)
        b = np.array(cmd.params["end"], dtype=float)
        t = np.linspace(0, 1, n)[:, None]
        return a[None, :] * (1 - t) + b[None, :] * t
    if cmd.kind == "regroup":
        p = np.array(cmd.params["point"], dtype=float)
        return p[None, :] + world.rng.normal(0, 0.4, size=(n, 2))
    if cmd.kind == "disperse":
        c = world.pos.mean(axis=0)
        out = world.pos - c
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return np.clip(world.pos + out * (0.35 * world.bounds), 0.0, world.bounds)
    raise ValueError(f"unknown command kind: {cmd.kind}")
