"""MS-1: deterministic synthetic world generator.

A state is an attributed graph (entities, relations, attributes, events, modalities).
Generation is fully deterministic given a seed, and infinite (no fixed state set).
Ground truth for all tasks is COMPUTED by an interpreter, never annotated.

Conventions (see AGENTS.md and docs/02_experiences_falsification.md):
- Test states use seeds >= 2_000_000 (anti-leak).
- Difficulty k = #relations + #attributes + #events.
- Literals (names, numbers, dates) are DISABLED in the MVP (H5 is out of scope).

Vocabulary sizes (designed ~160 attribute atoms + relations + actions + modalities):
- 12 relation types, 20 attribute keys (~8 values each), 15 actions, 5 modalities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Vocabulary ---------------------------------------------------------------

RELATION_TYPES: tuple[str, ...] = (
    "cause", "before", "after", "contains", "part_of",
    "similar", "opposite", "uses", "produces", "prevents",
    "enables", "located_in",
)

# Each attribute key maps to a discrete value domain.
ATTRIBUTE_DOMAINS: dict[str, tuple[int, ...]] = {
    "color": (0, 1, 2, 3, 4, 5),
    "size": (0, 1, 2, 3),
    "weight": (0, 1, 2, 3, 4),
    "state": (0, 1, 2),          # open / closed / mixed
    "temperature": (0, 1, 2),    # cold / warm / hot
    "count": (0, 1, 2, 3, 4, 5, 6, 7),
    "position": (0, 1, 2, 3, 4),
    "level": (0, 1, 2, 3),
    "status": (0, 1),             # off / on
    "type": (0, 1, 2, 3, 4, 5),
    "speed": (0, 1, 2, 3),
    "age": (0, 1, 2, 3, 4),
    "height": (0, 1, 2, 3),
    "width": (0, 1, 2, 3),
    "depth": (0, 1, 2, 3),
    "volume": (0, 1, 2, 3, 4),
    "power": (0, 1, 2),
    "charge": (0, 1, 2, 3),
    "mode": (0, 1, 2),
    "phase": (0, 1, 2, 3),
}
ATTRIBUTE_KEYS: tuple[str, ...] = tuple(ATTRIBUTE_DOMAINS.keys())

ACTIONS: tuple[str, ...] = (
    "move", "heat", "cool", "open", "close", "fill", "empty", "start", "stop",
    "connect", "disconnect", "add", "remove", "transform", "reset",
)

# Each action modifies one attribute key.
ACTION_TARGET_KEY: dict[str, str] = {
    "move": "position", "heat": "temperature", "cool": "temperature",
    "open": "state", "close": "state", "fill": "level", "empty": "level",
    "start": "status", "stop": "status", "connect": "state",
    "disconnect": "state", "add": "count", "remove": "count",
    "transform": "phase", "reset": "state",
}

MODALITIES: tuple[str, ...] = ("realized", "envisioned", "denied", "uncertain", "reported")

MIN_ENTITIES = 1
MAX_ENTITIES = 8

# --- State dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class Relation:
    src: int
    rtype: str
    dst: int
    modality: str


@dataclass(frozen=True)
class Attribute:
    eid: int
    key: str
    value: int
    modality: str


@dataclass(frozen=True)
class Event:
    time: int            # ordinal timestamp (lower = earlier)
    eid: int
    action: str
    modality: str


@dataclass(frozen=True)
class State:
    seed: int
    entities: tuple[int, ...]
    relations: tuple[Relation, ...]
    attributes: tuple[Attribute, ...]
    events: tuple[Event, ...]

    @property
    def difficulty(self) -> int:
        return len(self.relations) + len(self.attributes) + len(self.events)

    def attribute_value(self, eid: int, key: str) -> int | None:
        """Return the value of an attribute, preferring 'realized' modality."""
        best: Attribute | None = None
        for a in self.attributes:
            if a.eid == eid and a.key == key:
                if a.modality == "realized":
                    return a.value
                if best is None:
                    best = a
        return best.value if best is not None else None

    def has_cause(self, src: int, dst: int) -> bool:
        return any(
            r.src == src and r.rtype == "cause" and r.dst == dst
            and r.modality == "realized"
            for r in self.relations
        )

    def is_before(self, a: int, b: int) -> bool:
        ta = [e.time for e in self.events if e.eid == a]
        tb = [e.time for e in self.events if e.eid == b]
        if not ta or not tb:
            return False
        return min(ta) < min(tb)


# --- Tasks --------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    kind: str            # qa | implication | contradiction | temps | composition
    payload: dict[str, Any]
    # Ground-truth answer. Types: int (qa), bool (implication/contradiction/temps), int|None (composition).
    answer: Any


# --- Generator ----------------------------------------------------------------


class MS1:
    """Deterministic infinite generator of MS-1 states."""

    def __init__(self, min_k: int = 2, max_k: int = 64) -> None:
        self.min_k = min_k
        self.max_k = max_k

    def generate(self, seed: int, k: int | None = None) -> State:
        rng = np.random.default_rng(seed)
        if k is None:
            k = int(rng.integers(self.min_k, self.max_k + 1))
        # Number of entities grows with difficulty so relations are possible.
        n_entities = min(MAX_ENTITIES, max(2, int(rng.integers(2, 3 + k // 8))))
        entities = tuple(range(n_entities))

        relations: list[Relation] = []
        attributes: list[Attribute] = []
        events: list[Event] = []
        # Distribute k atoms across the three categories.
        # Bias modalities toward "realized" so balanced tasks have queryable atoms.
        modality_choices = MODALITIES + ("realized",) * 4
        # Bias relation types so "cause" is common enough to balance implication.
        rtype_choices = RELATION_TYPES + ("cause",) * 3
        target = k
        while len(relations) + len(attributes) + len(events) < target:
            cat = rng.integers(0, 3)
            eid = int(rng.integers(0, n_entities))
            modality = str(rng.choice(modality_choices))
            if cat == 0 and n_entities >= 2:
                src = int(rng.integers(0, n_entities))
                dst = int(rng.integers(0, n_entities))
                if src == dst:
                    continue
                rtype = str(rng.choice(rtype_choices))
                relations.append(Relation(src, rtype, dst, modality))
            elif cat == 1:
                key = str(rng.choice(ATTRIBUTE_KEYS))
                val = int(rng.choice(ATTRIBUTE_DOMAINS[key]))
                attributes.append(Attribute(eid, key, val, modality))
            else:
                action = str(rng.choice(ACTIONS))
                time = int(rng.integers(0, 100))
                events.append(Event(time, eid, action, modality))

        return State(
            seed=seed,
            entities=entities,
            relations=tuple(relations),
            attributes=tuple(attributes),
            events=tuple(events),
        )

    # --- Sampling -------------------------------------------------------------


# --- Task interpreter ---------------------------------------------------------


def solve(task: Task, state: State) -> Any:
    """Compute ground truth by execution (used both for labels and eval)."""
    p = task.payload
    if task.kind == "qa":
        return state.attribute_value(int(p["eid"]), str(p["key"]))
    if task.kind == "implication":
        return state.has_cause(int(p["src"]), int(p["dst"]))
    if task.kind == "contradiction":
        # A probe proposition (eid, key, value) contradicts the state if the
        # state has a realized attribute on (eid, key) with a different value.
        cur = state.attribute_value(int(p["eid"]), str(p["key"]))
        if cur is None:
            return False
        return cur != int(p["value"])
    if task.kind == "temps":
        return state.is_before(int(p["a"]), int(p["b"]))
    if task.kind == "composition":
        # Apply a hypothetical event: action targets a key; new value is a
        # deterministic function of the action and target key domain.
        eid = int(p["eid"])
        key = ACTION_TARGET_KEY[str(p["action"])]
        domain = ATTRIBUTE_DOMAINS[key]
        # New value: pick a value different from current when possible.
        cur = state.attribute_value(eid, key)
        new = domain[(hash((p["action"],)) % len(domain))]
        if cur is not None and new == cur and len(domain) > 1:
            new = domain[(domain.index(cur) + 1) % len(domain)]
        return new
    raise ValueError(f"unknown task kind: {task.kind}")


def sample_tasks(state: State, rng: np.random.Generator, n_per_kind: int = 4) -> list[Task]:
    """Sample tasks for a state. Ground truth is filled in via solve().."""
    tasks: list[Task] = []
    ent = state.entities
    if not ent:
        return tasks

    def pick_entity() -> int:
        return int(rng.choice(list(ent)))

    def make(kind: str, payload: dict[str, Any]) -> Task:
        t = Task(kind=kind, payload=payload, answer=None)  # type: ignore[arg-type]
        return Task(kind=kind, payload=payload, answer=solve(t, state))

    for _ in range(n_per_kind):
        tasks.append(make("qa", {"eid": pick_entity(), "key": str(rng.choice(ATTRIBUTE_KEYS))}))
    for _ in range(n_per_kind):
        if len(ent) >= 2:
            src, dst = rng.choice(len(ent), 2, replace=False)
            tasks.append(make("implication", {"src": int(src), "dst": int(dst)}))
        else:
            tasks.append(make("implication", {"src": int(ent[0]), "dst": int(ent[0])}))
    for _ in range(n_per_kind):
        eid = pick_entity()
        key = str(rng.choice(ATTRIBUTE_KEYS))
        val = int(rng.choice(ATTRIBUTE_DOMAINS[key]))
        tasks.append(make("contradiction", {"eid": eid, "key": key, "value": val}))
    for _ in range(n_per_kind):
        if len(ent) >= 2:
            a, b = rng.choice(len(ent), 2, replace=False)
            tasks.append(make("temps", {"a": int(a), "b": int(b)}))
        else:
            tasks.append(make("temps", {"a": int(ent[0]), "b": int(ent[0])}))
    for _ in range(n_per_kind):
        tasks.append(make("composition", {"eid": pick_entity(), "action": str(rng.choice(ACTIONS))}))
    return tasks


def sample_balanced_tasks(state: State, rng: np.random.Generator, n_per_kind: int = 4) -> list[Task]:
    """Sample tasks whose answer depends on the STATE, not on question-type marginals.

    Each kind is constructed so the question-type-only baseline is near chance:
    - qa: ask only about (eid, key) where a realized attribute exists -> answer is a
      real value, roughly uniform over the key's domain (baseline ~1/|domain|).
    - implication: half the time use a real cause relation (yes), half a random pair (no).
    - contradiction: probe an existing realized attribute; half correct value (no), half wrong (yes).
    - temps: pick two events with distinct times, balance before/after.
    - composition: kept as-is (already low baseline); answer depends on action + state.
    If a state lacks the needed atoms, fall back to unbalanced for that item.
    """
    tasks: list[Task] = []
    ent = list(state.entities)
    if not ent:
        return tasks

    realized_attrs = {(a.eid, a.key): a.value for a in state.attributes if a.modality == "realized"}
    cause_pairs = [(r.src, r.dst) for r in state.relations if r.rtype == "cause" and r.modality == "realized"]
    events_by_eid: dict[int, list[int]] = {}
    for e in state.events:
        events_by_eid.setdefault(e.eid, []).append(e.time)

    def make(kind: str, payload: dict[str, Any]) -> Task:
        t = Task(kind=kind, payload=payload, answer=None)  # type: ignore[arg-type]
        return Task(kind=kind, payload=payload, answer=solve(t, state))

    def pick_entity() -> int:
        return int(rng.choice(ent))

    def pick_realized_attr() -> tuple[int, str, int]:
        items = list(realized_attrs.items())
        (eid, key), val = items[int(rng.integers(0, len(items)))]
        return int(eid), str(key), int(val)

    # qa: only existing realized attributes -> real value.
    for _ in range(n_per_kind):
        if realized_attrs:
            eid, key, _ = pick_realized_attr()
            tasks.append(make("qa", {"eid": eid, "key": key}))
        else:
            tasks.append(make("qa", {"eid": pick_entity(), "key": str(rng.choice(ATTRIBUTE_KEYS))}))

    # implication: 50% real cause (yes) / 50% random pair (no).
    for _ in range(n_per_kind):
        if cause_pairs and rng.random() < 0.5:
            src, dst = cause_pairs[int(rng.integers(0, len(cause_pairs)))]
        elif len(ent) >= 2:
            src, dst = rng.choice(len(ent), 2, replace=False)
            src, dst = int(src), int(dst)
        else:
            src = dst = int(ent[0])
        tasks.append(make("implication", {"src": src, "dst": dst}))

    # contradiction: probe existing realized attr; 50% correct (no) / 50% wrong (yes).
    for _ in range(n_per_kind):
        if realized_attrs:
            eid, key, true_val = pick_realized_attr()
            domain = ATTRIBUTE_DOMAINS[key]
            if rng.random() < 0.5 or len(domain) < 2:
                probe = true_val
            else:
                wrong = [v for v in domain if v != true_val]
                probe = int(rng.choice(wrong))
            tasks.append(make("contradiction", {"eid": eid, "key": key, "value": probe}))
        else:
            key = str(rng.choice(ATTRIBUTE_KEYS))
            tasks.append(make("contradiction", {"eid": pick_entity(), "key": key,
                                                "value": int(rng.choice(ATTRIBUTE_DOMAINS[key]))}))

    # temps: two entities with events; balance before/after.
    ent_with_events = [e for e, ts in events_by_eid.items() if len(ts) > 0]
    for _ in range(n_per_kind):
        if len(ent_with_events) >= 2:
            a, b = rng.choice(len(ent_with_events), 2, replace=False)
            a, b = int(ent_with_events[a]), int(ent_with_events[b])
        elif len(ent) >= 2:
            a, b = rng.choice(len(ent), 2, replace=False)
            a, b = int(a), int(b)
        else:
            a = b = int(ent[0])
        tasks.append(make("temps", {"a": a, "b": b}))

    # composition: kept as-is (answer depends on action + state).
    for _ in range(n_per_kind):
        tasks.append(make("composition", {"eid": pick_entity(), "action": str(rng.choice(ACTIONS))}))
    return tasks


# --- Text renderer ------------------------------------------------------------

# Bilingual templates. Paraphrases are selected deterministically by view_id.

_TEMPLATES_FR: dict[str, Any] = {
    "entity": ["l'objet {n}", "l'élément {n}", "l'entité {n}"],
    "attr": [
        "{e} a {key}={v}",
        "la valeur de {key} pour {e} est {v}",
        "{e} : {key} vaut {v}",
    ],
    "rel": [
        "{e1} {rtype} {e2}",
        "relation {rtype} de {e1} vers {e2}",
    ],
    "event": [
        "à t={t}, {e} {action}",
        "{e} {action} au temps {t}",
    ],
    "mod_prefix": {
        "realized": "",
        "envisioned": "(envisagé) ",
        "denied": "(nié) ",
        "uncertain": "(incertain) ",
        "reported": "(rapporté) ",
    },
}

_TEMPLATES_EN: dict[str, Any] = {
    "entity": ["object {n}", "element {n}", "entity {n}"],
    "attr": [
        "{e} has {key}={v}",
        "the {key} of {e} is {v}",
        "{e}: {key} equals {v}",
    ],
    "rel": [
        "{e1} {rtype} {e2}",
        "relation {rtype} from {e1} to {e2}",
    ],
    "event": [
        "at t={t}, {e} {action}",
        "{e} {action} at time {t}",
    ],
    "mod_prefix": {
        "realized": "",
        "envisioned": "(envisioned) ",
        "denied": "(denied) ",
        "uncertain": "(uncertain) ",
        "reported": "(reported) ",
    },
}


class TextRenderer:
    """Render a state to a textual view (FR/EN/structured) with paraphrases."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng

    def render(self, state: State, view_id: int = 0) -> str:
        # view_id % 9 selects among: 4 FR + 4 EN + 1 structured.
        mode = view_id % 9
        if mode == 8:
            return self._structured(state)
        lang = "fr" if mode < 4 else "en"
        return self._natural(state, lang, view_id)

    def _natural(self, state: State, lang: str, view_id: int) -> str:
        tpl = _TEMPLATES_FR if lang == "fr" else _TEMPLATES_EN
        rng = self.rng(state.seed * 1000 + view_id)
        parts: list[str] = []
        for e in state.entities:
            parts.append(rng.choice(tpl["entity"]).format(n=e))
        for a in state.attributes:
            mod = tpl["mod_prefix"][a.modality]
            parts.append(mod + rng.choice(tpl["attr"]).format(
                e=tpl["entity"][0].format(n=a.eid), key=a.key, v=a.value))
        for r in state.relations:
            mod = tpl["mod_prefix"][r.modality]
            parts.append(mod + rng.choice(tpl["rel"]).format(
                e1=tpl["entity"][0].format(n=r.src),
                e2=tpl["entity"][0].format(n=r.dst), rtype=r.rtype))
        for ev in state.events:
            mod = tpl["mod_prefix"][ev.modality]
            parts.append(mod + rng.choice(tpl["event"]).format(
                e=tpl["entity"][0].format(n=ev.eid), action=ev.action, t=ev.time))
        rng.shuffle(parts)
        return ". ".join(parts) + "."

    def _structured(self, state: State) -> str:
        d: dict[str, Any] = {
            "entities": list(state.entities),
            "relations": [r.__dict__ for r in state.relations],
            "attributes": [a.__dict__ for a in state.attributes],
            "events": [e.__dict__ for e in state.events],
        }
        return json.dumps(d, separators=(",", ":"))


# --- Splits -------------------------------------------------------------------

TRAIN_RANGE = (0, 1_000_000)
VAL_RANGE = (1_000_000, 1_010_000)
TEST_RANGE = (2_000_000, 2_010_000)
TEST_COMPOSITIONAL_RANGE = (3_000_000, 3_001_000)


def split_for_seed(seed: int) -> str:
    if TRAIN_RANGE[0] <= seed < TRAIN_RANGE[1]:
        return "train"
    if VAL_RANGE[0] <= seed < VAL_RANGE[1]:
        return "val"
    if TEST_RANGE[0] <= seed < TEST_RANGE[1]:
        return "test"
    if TEST_COMPOSITIONAL_RANGE[0] <= seed < TEST_COMPOSITIONAL_RANGE[1]:
        return "test_comp"
    raise ValueError(f"seed {seed} not in any defined split range")


__all__ = [
    "RELATION_TYPES", "ATTRIBUTE_KEYS", "ATTRIBUTE_DOMAINS", "ACTIONS",
    "ACTION_TARGET_KEY", "MODALITIES", "MAX_ENTITIES",
    "Relation", "Attribute", "Event", "State", "Task",
    "MS1", "solve", "sample_tasks", "sample_balanced_tasks", "TextRenderer",
    "TRAIN_RANGE", "VAL_RANGE", "TEST_RANGE", "TEST_COMPOSITIONAL_RANGE",
    "split_for_seed",
]
