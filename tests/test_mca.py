import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mca.commands import KINDS, Command, targets_for
from mca.generate import generate_triples
from mca.interpret import LLMInterpreter, MockInterpreter
from mca.language import MockParaphraser
from mca.metrics import collision_rate, completion_time, formation_error, grounding
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


def test_flank_splits_into_two_wings():
    w = World.random_init(8, seed=2)
    w.target = targets_for(Command("flank", {"point": [5.0, 5.0]}), w)
    traj = w.rollout(150)
    final = traj[-1]
    # the approach axis runs centroid -> point; project finals onto the perpendicular
    axis = np.array([5.0, 5.0]) - traj[0].mean(axis=0)
    axis /= np.linalg.norm(axis) + 1e-9
    perp = np.array([-axis[1], axis[0]])
    side = (final - np.array([5.0, 5.0])) @ perp
    assert (side > 0).any() and (side < 0).any()        # agents on both flanks


def test_all_kinds_produce_valid_targets():
    w = World.random_init(6, seed=3)
    params = {
        "go_to": {"point": [5.0, 5.0]}, "regroup": {"point": [5.0, 5.0]},
        "flank": {"point": [8.0, 8.0]}, "disperse": {},
        "form_line": {"start": [2.0, 2.0], "end": [8.0, 8.0]},
    }
    for kind in KINDS:
        tgt = targets_for(Command(kind, params[kind]), w)
        assert tgt.shape == (6, 2)
        assert np.isfinite(tgt).all()


def test_formation_error_and_completion_time():
    triples = generate_triples(
        [Command("go_to", {"point": [5.0, 5.0]})],
        MockParaphraser(), StubRenderer(),
        n_agents=5, steps=200, variants_per_command=2,
    )
    fe = formation_error(triples[0])
    assert fe >= 0.0 and fe < 1.0                        # sim converges close to targets
    ct = completion_time(triples[0])
    assert 0 <= ct <= len(triples[0]["trajectory"])      # within horizon or sentinel


def test_completion_time_sentinel_when_unreached():
    triple = generate_triples(
        [Command("go_to", {"point": [5.0, 5.0]})],
        MockParaphraser(), StubRenderer(),
        n_agents=4, steps=2, variants_per_command=1,
    )[0]
    # 2 steps is far too few to converge -> sentinel == horizon length
    assert completion_time(triple) == len(triple["trajectory"])


def test_mock_interpreter_recovers_kinds():
    interp = MockInterpreter()
    assert interp("everyone fan out and spread").kind == "disperse"
    assert interp("form a line").kind == "form_line"
    assert interp("regroup on me").kind == "regroup"
    assert interp("flank them from both sides").kind == "flank"
    c = interp("move to (8.0, 3.0)")
    assert c.kind == "go_to" and c.params["point"] == [8.0, 3.0]


def test_llm_interpreter_parse_is_robust():
    # _parse must survive think-traces, prose, and bad kinds without raising
    raw = '<think>the user wants...</think> Sure: {"kind":"flank","params":{"point":[7,2]}}'
    cmd = LLMInterpreter._parse(raw)
    assert cmd.kind == "flank" and cmd.params["point"] == [7.0, 2.0]
    assert LLMInterpreter._parse("garbage, no json").kind == "go_to"
    assert LLMInterpreter._parse('{"kind":"nonsense","params":{}}').kind == "go_to"


def test_location_signal_detects_digits_and_numberwords():
    from mca.interpret import _has_location_signal
    assert _has_location_signal("move to 8, 8")
    assert _has_location_signal("advance to eight by eight")
    assert not _has_location_signal("advance to the objective and take cover")
    assert not _has_location_signal("flank them from both sides")
