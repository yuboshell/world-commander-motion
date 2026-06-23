import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mca.commands import Command, targets_for
from mca.generate import generate_triples
from mca.language import MockParaphraser
from mca.metrics import collision_rate, grounding
from mca.render import StubRenderer
from mca.world import World


def test_sim_moves_toward_target():
    w = World.random_init(5, seed=1)
    w.target = targets_for(Command("go_to", {"point": [5.0, 5.0]}), w)
    traj = w.rollout(120)
    assert np.linalg.norm(traj[-1] - np.array([5.0, 5.0]), axis=1).mean() < 2.0


def test_generate_triples_shape_and_text():
    triples = generate_triples(
        [Command("form_line", {"start": [3.0, 5.0], "end": [7.0, 5.0]})],
        MockParaphraser(), StubRenderer(),
        n_agents=4, steps=40, variants_per_command=3,
    )
    assert len(triples) == 3
    t = triples[0]
    assert t["trajectory"].shape == (41, 4, 2)
    assert t["targets"].shape == (4, 2)
    assert isinstance(t["command_text"], str) and t["command_text"]


def test_grounding_high_for_sim():
    triples = generate_triples(
        [Command("form_line", {"start": [3.0, 5.0], "end": [7.0, 5.0]})],
        MockParaphraser(), StubRenderer(),
        n_agents=5, steps=100, variants_per_command=4,
    )
    g = sum(grounding(t) for t in triples) / len(triples)
    assert g > 0.6


def test_collision_rate_in_range():
    triples = generate_triples(
        [Command("go_to", {"point": [8.0, 8.0]})],
        MockParaphraser(), StubRenderer(),
        n_agents=5, steps=80, variants_per_command=3,
    )
    c = collision_rate(triples[0])
    assert 0.0 <= c <= 1.0
