"""Tests for MS-1 world generator (deterministic, GPU-free)."""

from __future__ import annotations

import numpy as np

from msl.data.ms1 import (
    ACTIONS,
    ATTRIBUTE_KEYS,
    MAX_ENTITIES,
    MODALITIES,
    MS1,
    RELATION_TYPES,
    TEST_RANGE,
    TRAIN_RANGE,
    VAL_RANGE,
    TextRenderer,
    sample_tasks,
    solve,
    split_for_seed,
)


def test_vocabulary_sizes_match_design():
    assert len(RELATION_TYPES) == 12
    assert len(ATTRIBUTE_KEYS) == 20
    assert len(ACTIONS) == 15
    assert len(MODALITIES) == 5
    assert MAX_ENTITIES == 8


def test_generation_is_deterministic_by_seed():
    g = MS1()
    s1 = g.generate(seed=42, k=16)
    s2 = g.generate(seed=42, k=16)
    assert s1 == s2
    assert s1.seed == 42


def test_difficulty_matches_k():
    g = MS1()
    for k in (2, 8, 32, 64):
        s = g.generate(seed=k, k=k)
        assert s.difficulty == k


def test_entities_scale_with_difficulty():
    g = MS1()
    big = g.generate(seed=999, k=64)
    small = g.generate(seed=1, k=2)
    assert len(big.entities) >= len(small.entities)
    assert len(big.entities) >= 2  # relations are possible


def test_relations_require_two_entities():
    g = MS1()
    # With many relations drawn, at least some relations exist at high k.
    s = g.generate(seed=7, k=64)
    assert len(s.relations) > 0


def test_splits_partition_seed_space():
    assert split_for_seed(0) == "train"
    assert split_for_seed(500_000) == "train"
    assert split_for_seed(1_000_000) == "val"
    assert split_for_seed(2_000_000) == "test"
    assert split_for_seed(3_000_000) == "test_comp"
    assert TRAIN_RANGE[1] == VAL_RANGE[0]
    assert TEST_RANGE[0] >= 2_000_000  # anti-leak convention


def test_sample_tasks_returns_all_kinds():
    g = MS1()
    s = g.generate(seed=42, k=32)
    rng = np.random.default_rng(0)
    tasks = sample_tasks(s, rng, n_per_kind=3)
    kinds = {t.kind for t in tasks}
    assert kinds == {"qa", "implication", "contradiction", "temps", "composition"}
    assert len(tasks) == 15


def test_solve_qa_returns_value_or_none():
    g = MS1()
    s = g.generate(seed=42, k=16)
    # Build a QA task for an attribute that exists.
    if s.attributes:
        a = s.attributes[0]
        from msl.data.ms1 import Task
        t = Task("qa", {"eid": a.eid, "key": a.key}, answer=None)
        assert solve(t, s) == a.value or solve(t, s) is None
    # A missing attribute returns None.
    from msl.data.ms1 import Task
    t = Task("qa", {"eid": s.entities[0], "key": "color"}, answer=None)
    val = solve(t, s)
    assert val is None or isinstance(val, int)


def test_solve_implication_is_bool():
    g = MS1()
    s = g.generate(seed=42, k=16)
    from msl.data.ms1 import Task
    t = Task("implication", {"src": s.entities[0], "dst": s.entities[-1]}, answer=None)
    assert isinstance(solve(t, s), bool)


def test_solve_contradiction_detects_conflict():
    from msl.data.ms1 import Attribute, State, Task, solve
    s = State(seed=0, entities=(0, 1), relations=(),
              attributes=(Attribute(0, "color", 1, "realized"),), events=())
    t_match = Task("contradiction", {"eid": 0, "key": "color", "value": 1}, answer=None)
    t_conflict = Task("contradiction", {"eid": 0, "key": "color", "value": 2}, answer=None)
    assert solve(t_match, s) is False
    assert solve(t_conflict, s) is True


def test_text_renderer_provides_distinct_views():
    g = MS1()
    s = g.generate(seed=42, k=16)
    r = TextRenderer()
    views = [r.render(s, v) for v in range(9)]
    assert len(views) == 9
    # Structured view (index 8) is JSON.
    assert views[8].startswith("{")
    # Natural views are non-empty and end with a period.
    assert all(v.endswith(".") for v in views[:8])
    # FR and EN views differ.
    assert views[0] != views[5]


def test_renderer_is_deterministic():
    g = MS1()
    s = g.generate(seed=42, k=16)
    r = TextRenderer()
    assert r.render(s, 0) == r.render(s, 0)


def test_test_seeds_are_isolated_from_train():
    # A test seed should never collide with a train seed value space.
    assert TEST_RANGE[0] > TRAIN_RANGE[1]
