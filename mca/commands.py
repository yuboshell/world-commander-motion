"""Canonical commands -> per-agent target positions (data layer)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Command:
    kind: str            # 'go_to' | 'form_line' | 'regroup' | 'disperse' | 'flank'
    params: dict


KINDS = ("go_to", "form_line", "regroup", "disperse", "flank")


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
    if cmd.kind == "flank":
        # Envelop a point from both sides: split the crowd into two wings offset
        # perpendicular to the crowd's approach axis, each spread into a short column.
        p = np.array(cmd.params["point"], dtype=float)
        dist = float(cmd.params.get("dist", 2.5))      # lateral offset of each wing
        spread = float(cmd.params.get("spread", 2.0))  # depth of each wing along the axis
        axis = p - world.pos.mean(axis=0)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        perp = np.array([-axis[1], axis[0]])           # 90 deg rotation
        idx = np.arange(n)
        half = (n + 1) // 2
        wing = np.where(idx < half, 1.0, -1.0)         # right wing / left wing
        within = np.where(idx < half, idx, idx - half).astype(float)
        wcount = np.where(idx < half, half, n - half)
        frac = (within - (wcount - 1) / 2.0) / np.maximum(wcount, 1)   # centered -0.5..0.5
        tgt = p[None, :] + wing[:, None] * dist * perp[None, :] + frac[:, None] * spread * axis[None, :]
        return np.clip(tgt, 0.0, world.bounds)
    raise ValueError(f"unknown command kind: {cmd.kind}")
