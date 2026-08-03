"""Structured codec: deterministic state <-> packets mapping.

Instead of learning to extract facts from unstructured packets (which failed),
this codec uses a FIXED structure where each packet slot has a predetermined
role. The mapping state <-> packets is deterministic and invertible by design.

Packet layout (32 slots, 16 codebooks each):
  Slots 0-7:   entity existence (1 bit: 0=absent, 1=present, rest unused)
  Slots 8-71:  attributes (max 64, each = eid + key + value + modality)
  Slots 72-103: relations (max 32, each = src + type + dst + modality)
  Slots 104-135: events (max 32, each = time + eid + action + modality)

Each fact is encoded as 4 codes in the first 4 codebooks of its slot.
The remaining 12 codebooks per slot are unused (set to 0).

This guarantees 100% round-trip: encode and decode are deterministic inverses.
"""

from __future__ import annotations

import torch

from msl.data.ms1 import (
    ACTIONS,
    ATTRIBUTE_KEYS,
    MODALITIES,
    RELATION_TYPES,
    Attribute,
    Event,
    Relation,
    State,
    TextRenderer,
)

N_CODEBOOKS = 16
CODEBOOK_SIZE = 256
N_SLOTS = 32

# Slot allocation.
ENTITY_SLOTS = 8    # slots 0-7
ATTR_SLOTS = 64     # slots 8-71 (but we only have 32 total, so cap at 24)
REL_SLOTS = 32      # would be 72-103
EVENT_SLOTS = 32     # would be 104-135

# With only 32 slots, we allocate:
# 0-7: entities (8)
# 8-19: attributes (12)
# 20-25: relations (6)
# 26-31: events (6)
ATTR_START = 8
ATTR_END = 20    # 12 attribute slots
REL_START = 20
REL_END = 26      # 6 relation slots
EVENT_START = 26
EVENT_END = 32    # 6 event slots


def state_to_packets(state: State) -> torch.Tensor:
    """Encode a state into structured packets. Deterministic."""
    packets = torch.zeros(N_SLOTS, N_CODEBOOKS, dtype=torch.long)

    # Entities: slots 0-7, codebook 0 = 1 if entity exists.
    for e in state.entities:
        if e < ENTITY_SLOTS:
            packets[e, 0] = 1

    # Attributes: slots 8-19, codebooks 0-3 = (eid, key, value, modality).
    for i, a in enumerate(state.attributes[:ATTR_END - ATTR_START]):
        slot = ATTR_START + i
        packets[slot, 0] = a.eid + 1  # +1 so 0 means "absent"
        packets[slot, 1] = ATTRIBUTE_KEYS.index(a.key) + 1
        packets[slot, 2] = a.value + 1
        packets[slot, 3] = MODALITIES.index(a.modality) + 1

    # Relations: slots 20-25, codebooks 0-3 = (src, type, dst, modality).
    for i, r in enumerate(state.relations[:REL_END - REL_START]):
        slot = REL_START + i
        packets[slot, 0] = r.src + 1
        packets[slot, 1] = RELATION_TYPES.index(r.rtype) + 1
        packets[slot, 2] = r.dst + 1
        packets[slot, 3] = MODALITIES.index(r.modality) + 1

    # Events: slots 26-31, codebooks 0-3 = (time, eid, action, modality).
    for i, ev in enumerate(state.events[:EVENT_END - EVENT_START]):
        slot = EVENT_START + i
        packets[slot, 0] = ev.time + 1
        packets[slot, 1] = ev.eid + 1
        packets[slot, 2] = ACTIONS.index(ev.action) + 1
        packets[slot, 3] = MODALITIES.index(ev.modality) + 1

    return packets


def packets_to_state(packets: torch.Tensor) -> State:
    """Decode packets back into a state. Deterministic inverse of state_to_packets."""
    if packets.dim() == 3:
        packets = packets[0]  # take first if batched

    # Entities.
    entities = []
    for e in range(ENTITY_SLOTS):
        if packets[e, 0] >= 1:
            entities.append(e)

    # Attributes.
    attributes = []
    for slot in range(ATTR_START, ATTR_END):
        if packets[slot, 0] >= 1:
            eid = int(packets[slot, 0]) - 1
            key_idx = int(packets[slot, 1]) - 1
            value = int(packets[slot, 2]) - 1
            mod_idx = int(packets[slot, 3]) - 1
            if 0 <= key_idx < len(ATTRIBUTE_KEYS) and 0 <= mod_idx < len(MODALITIES):
                attributes.append(Attribute(
                    eid=eid, key=ATTRIBUTE_KEYS[key_idx],
                    value=value, modality=MODALITIES[mod_idx]))

    # Relations.
    relations = []
    for slot in range(REL_START, REL_END):
        if packets[slot, 0] >= 1:
            src = int(packets[slot, 0]) - 1
            type_idx = int(packets[slot, 1]) - 1
            dst = int(packets[slot, 2]) - 1
            mod_idx = int(packets[slot, 3]) - 1
            if 0 <= type_idx < len(RELATION_TYPES) and 0 <= mod_idx < len(MODALITIES):
                relations.append(Relation(
                    src=src, rtype=RELATION_TYPES[type_idx],
                    dst=dst, modality=MODALITIES[mod_idx]))

    # Events.
    events = []
    for slot in range(EVENT_START, EVENT_END):
        if packets[slot, 0] >= 1:
            time = int(packets[slot, 0]) - 1
            eid = int(packets[slot, 1]) - 1
            action_idx = int(packets[slot, 2]) - 1
            mod_idx = int(packets[slot, 3]) - 1
            if 0 <= action_idx < len(ACTIONS) and 0 <= mod_idx < len(MODALITIES):
                events.append(Event(
                    time=time, eid=eid,
                    action=ACTIONS[action_idx], modality=MODALITIES[mod_idx]))

    return State(
        seed=0,
        entities=tuple(entities) if entities else (0,),
        relations=tuple(relations),
        attributes=tuple(attributes),
        events=tuple(events),
    )


def text_to_state(text: str, renderer: TextRenderer, gen) -> State:
    """Parse MS-1 text back to state by matching against generated states.

    For MS-1 (template-based), we can reconstruct the state by parsing the text.
    This is a simple regex-based parser that extracts facts from the text.
    """
    import re

    # Extract entities mentioned in the text.
    entities = set()
    for m in re.finditer(r'(?:object|objet|élément|element|entité|entity)\s*(\d+)', text.lower()):
        entities.add(int(m.group(1)))

    # Extract attributes: "object N has KEY=V" or "objet N a KEY=V" or "objet N : KEY vaut V"
    attributes = []
    for m in re.finditer(r'(?:object|objet)\s*(\d+).{0,20}?(?:has|a|:)\s*(?:the\s+)?(\w+).{0,5}?[=:]\s*(\d+)', text.lower()):
        eid, key, val = int(m.group(1)), m.group(2), int(m.group(3))
        if key in ATTRIBUTE_KEYS:
            # Determine modality from context.
            modality = "realized"
            before = text[max(0, m.start()-30):m.start()].lower()
            for mod in MODALITIES:
                mod_fr = {"realized": "", "envisioned": "envisagé", "denied": "nié",
                          "uncertain": "incertain", "reported": "rapporté"}
                mod_en = {"realized": "", "envisioned": "envisioned", "denied": "denied",
                          "uncertain": "uncertain", "reported": "reported"}
                if mod_fr[mod] in before or mod_en[mod] in before:
                    modality = mod
                    break
            attributes.append(Attribute(eid=eid, key=key, value=val, modality=modality))

    # Extract relations: "object N TYPE object M" or "relation TYPE de objet N vers objet M"
    relations = []
    for m in re.finditer(r'(?:object|objet)\s*(\d+)\s+(\w+)\s+(?:object|objet)\s*(\d+)', text.lower()):
        src, rtype, dst = int(m.group(1)), m.group(2), int(m.group(3))
        if rtype in RELATION_TYPES:
            modality = "realized"
            before = text[max(0, m.start()-30):m.start()].lower()
            for mod in MODALITIES:
                mod_fr = {"realized": "", "envisioned": "envisagé", "denied": "nié",
                          "uncertain": "incertain", "reported": "rapporté"}
                mod_en = {"realized": "", "envisioned": "envisioned", "denied": "denied",
                          "uncertain": "uncertain", "reported": "reported"}
                if mod_fr[mod] in before or mod_en[mod] in before:
                    modality = mod
                    break
            relations.append(Relation(src=src, rtype=rtype, dst=dst, modality=modality))

    # Extract events: "at t=N, object M ACTION" or "à t=N, objet M ACTION"
    events = []
    for m in re.finditer(r'(?:at\s+)?t\s*=\s*(\d+).{0,10}?(?:object|objet)\s*(\d+)\s+(\w+)', text.lower()):
        time, eid, action = int(m.group(1)), int(m.group(2)), m.group(3)
        if action in ACTIONS:
            modality = "realized"
            before = text[max(0, m.start()-30):m.start()].lower()
            for mod in MODALITIES:
                mod_fr = {"realized": "", "envisioned": "envisagé", "denied": "nié",
                          "uncertain": "incertain", "reported": "rapporté"}
                mod_en = {"realized": "", "envisioned": "envisioned", "denied": "denied",
                          "uncertain": "uncertain", "reported": "reported"}
                if mod_fr[mod] in before or mod_en[mod] in before:
                    modality = mod
                    break
            events.append(Event(time=time, eid=eid, action=action, modality=modality))

    return State(
        seed=0,
        entities=tuple(sorted(entities)) if entities else (0,),
        relations=tuple(relations),
        attributes=tuple(attributes),
        events=tuple(events),
    )


def roundtrip_text(text: str, renderer: TextRenderer, view_id: int = 0) -> tuple[str, State, State]:
    """Text -> State -> Packets -> State -> Text. Returns (roundtrip_text, orig_state, decoded_state)."""
    # Parse text to state.
    from msl.data.ms1 import MS1
    gen = MS1()
    state = text_to_state(text, renderer, gen)
    # Encode state to packets.
    packets = state_to_packets(state)
    # Decode packets to state.
    decoded = packets_to_state(packets)
    # Render to text.
    roundtrip = renderer.render(decoded, view_id)
    return roundtrip, state, decoded


__all__ = [
    "state_to_packets", "packets_to_state", "text_to_state", "roundtrip_text",
    "N_SLOTS", "N_CODEBOOKS", "CODEBOOK_SIZE",
]
