#!/usr/bin/env python
"""v0 de-risk: generate synthetic (command -> coordinated motion) triples for two
commands and report pipeline sanity (grounding) + coordination (collisions).

Offline (MockParaphraser + StubRenderer) validates the pipeline structure with no GPU.
On amax: swap in RealParaphraser (served LLM) and RealRenderer (TLControl/CAMDM), then add
the model-in-the-loop grounding test + FPS / latency / VRAM (the budget frontier)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mca.commands import Command
from mca.generate import generate_triples
from mca.language import MockParaphraser
from mca.metrics import collision_rate, grounding
from mca.render import StubRenderer


def main() -> None:
    p = argparse.ArgumentParser(description="v0 synthetic-pipeline smoke run (offline)")
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--variants", type=int, default=8, help="triples per command")
    p.add_argument("--steps", type=int, default=80)
    args = p.parse_args()

    commands = [
        Command("form_line", {"start": [3.0, 5.0], "end": [7.0, 5.0]}),
        Command("go_to", {"point": [8.0, 8.0]}),
    ]
    triples = generate_triples(
        commands, MockParaphraser(), StubRenderer(),
        n_agents=args.agents, steps=args.steps, variants_per_command=args.variants,
    )

    print("[offline pipeline — MockParaphraser + StubRenderer]")
    print(f"generated {len(triples)} triples "
          f"({args.agents} agents, {args.steps} steps, {args.variants}/command)\n")
    print(f"{'command':12s} {'grounding':>10s} {'collision':>10s}  example paraphrase")
    for kind in sorted({t["command_canonical"]["kind"] for t in triples}):
        sub = [t for t in triples if t["command_canonical"]["kind"] == kind]
        g = sum(grounding(t) for t in sub) / len(sub)
        c = sum(collision_rate(t) for t in sub) / len(sub)
        print(f"{kind:12s} {g:>10.2f} {c:>10.2f}  \"{sub[0]['command_text']}\"")

    print("\nGrounding near 1.0 confirms the sim reaches the commanded target; the real "
          "test (on amax) is whether a model conditioned on command_text reproduces it.")
    print("NEXT (amax): RealParaphraser (served Qwen) + RealRenderer (TLControl/CAMDM) + "
          "model-in-the-loop grounding + FPS/latency/VRAM. See plan/research-plan.md.")


if __name__ == "__main__":
    main()
