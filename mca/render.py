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
    """L3 on the GPU box: bridge to a pretrained, trajectory-controllable motion model
    (OmniControl) that turns a per-agent 2D path into full-body motion (HumanML3D joints).

    The motion model's heavy deps (torch + the model repo) live in a SEPARATE conda env, so
    this core package stays GPU-free per the repo contract — we invoke the model as a
    subprocess (`OmniControl/gen_crowd.py`) rather than importing it. Generation is offline and
    slow (diffusion + spatial guidance, ~150 s / 196-frame clip on a 2080 Ti) but tiny on VRAM
    (~0.6 GB); see experiments/REPORT.md (L3). Defaults target the amax41 setup."""

    def __init__(self, omni_dir: str = "/mnt/yubo/repos/OmniControl",
                 env_python: str = "/home/yubo/enter/envs/omnicontrol/bin/python",
                 gpu: str = "2", text: str = "a person walks", max_agents: int = 1):
        self.omni_dir = omni_dir
        self.env_python = env_python
        self.gpu = gpu
        self.text = text
        self.max_agents = max_agents

    def __call__(self, traj: np.ndarray) -> dict:
        # traj: (T, N, 2). Render up to max_agents agents (generation is slow — see docstring).
        import os
        import re
        import subprocess
        import tempfile

        n = min(traj.shape[1], self.max_agents)
        agents = []
        with tempfile.TemporaryDirectory() as td:
            for i in range(n):
                p = os.path.join(td, f"a{i}.npy")
                np.save(p, traj[:, i, :].astype(np.float32))
                r = subprocess.run(
                    [self.env_python, "gen_crowd.py", "--path", p, "--out", td, "--text", self.text],
                    cwd=self.omni_dir, capture_output=True, text=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu})
                m = re.search(r"gen ([\d.]+)s \(([\d.]+) fps\) \| peak VRAM ([\d.]+) GB", r.stdout)
                joints = np.load(p) if os.path.exists(p) else None   # gen_crowd overwrites with motion
                agents.append({
                    "agent": i, "joints": joints,
                    "gen_s": float(m.group(1)) if m else None,
                    "fps": float(m.group(2)) if m else None,
                    "peak_vram_gb": float(m.group(3)) if m else None,
                    "ok": m is not None,
                })
        return {"motion": agents, "renderer": "omnicontrol", "n_agents_rendered": n}
