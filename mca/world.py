"""A minimal 2D crowd (data layer L1): agents move toward per-command targets with
simple separation, producing ground-truth coordinated trajectories. GPU-free."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class World:
    n: int
    pos: np.ndarray                       # (n, 2) current positions
    target: np.ndarray                    # (n, 2) per-agent targets
    bounds: float = 10.0
    max_speed: float = 0.15
    sep_radius: float = 0.6
    sep_strength: float = 0.08
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @classmethod
    def random_init(cls, n: int, bounds: float = 10.0, seed: int = 0) -> "World":
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0.15 * bounds, 0.85 * bounds, size=(n, 2))
        return cls(n=n, pos=pos.copy(), target=pos.copy(), bounds=bounds, rng=rng)

    def _separation(self) -> np.ndarray:
        d = self.pos[:, None, :] - self.pos[None, :, :]          # (n, n, 2)
        dist = np.linalg.norm(d, axis=2)
        mask = (dist < self.sep_radius) & (dist > 1e-6)
        rep = np.where(mask[..., None], d / (dist[..., None] + 1e-9), 0.0)
        return self.sep_strength * rep.sum(axis=1)

    def step(self) -> None:
        to_target = self.target - self.pos
        dist = np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-9
        desired = to_target / dist * np.minimum(dist, self.max_speed)
        vel = desired + self._separation()
        sp = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        vel = vel / sp * np.minimum(sp, self.max_speed)
        self.pos = np.clip(self.pos + vel, 0.0, self.bounds)

    def rollout(self, steps: int) -> np.ndarray:
        traj = np.empty((steps + 1, self.n, 2), dtype=np.float32)
        traj[0] = self.pos
        for t in range(steps):
            self.step()
            traj[t + 1] = self.pos
        return traj
