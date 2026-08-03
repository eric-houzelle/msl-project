"""StateDecoder: extracts structured facts from packets, then renders to text.

Instead of generating text token-by-token (which compounds errors), this decoder
PREDICTS the structured state (entities, attributes, relations, events) from the
packets via classification heads. The state is then rendered to text by the
deterministic TextRenderer.

This guarantees faithful round-trip: if the facts are correct, the rendered
text is identical to the original (same renderer, same facts = same text).

State structure:
  - n_entities: 1-8 (classification)
  - attributes: up to 64 slots, each predicting (eid, key, value, modality, is_present)
  - relations: up to 32 slots, each predicting (src, rtype, dst, modality, is_present)
  - events: up to 32 slots, each predicting (time, eid, action, modality, is_present)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from msl.data.ms1 import (
    ACTIONS,
    ATTRIBUTE_DOMAINS,
    ATTRIBUTE_KEYS,
    MAX_ENTITIES,
    MODALITIES,
    RELATION_TYPES,
    Attribute,
    Event,
    Relation,
    State,
    TextRenderer,
)


class StateDecoder(nn.Module):
    """Extracts structured facts from packets via classification heads."""

    def __init__(
        self,
        d_z: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_attributes: int = 64,
        max_relations: int = 32,
        max_events: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.d_model = d_model
        self.max_attributes = max_attributes
        self.max_relations = max_relations
        self.max_events = max_events
        self.renderer = TextRenderer()

        self.packet_proj = nn.Linear(d_z, d_model)

        # Shared encoder for the packet sequence.
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           batch_first=True, activation="gelu", dropout=dropout)
        self.packet_encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        # Entity prediction: which of 8 entities exist.
        self.entity_head = nn.Linear(d_model, MAX_ENTITIES)

        # Attribute heads: for each of max_attributes slots.
        n_attr_keys = len(ATTRIBUTE_KEYS)
        n_attr_values = max(len(v) for v in ATTRIBUTE_DOMAINS.values())
        n_modalities = len(MODALITIES)
        self.attr_present = nn.Linear(d_model, 2)
        self.attr_eid = nn.Linear(d_model, MAX_ENTITIES)
        self.attr_key = nn.Linear(d_model, n_attr_keys)
        self.attr_value = nn.Linear(d_model, n_attr_values)
        self.attr_modality = nn.Linear(d_model, n_modalities)

        # Relation heads.
        n_rel_types = len(RELATION_TYPES)
        self.rel_present = nn.Linear(d_model, 2)
        self.rel_src = nn.Linear(d_model, MAX_ENTITIES)
        self.rel_type = nn.Linear(d_model, n_rel_types)
        self.rel_dst = nn.Linear(d_model, MAX_ENTITIES)
        self.rel_modality = nn.Linear(d_model, n_modalities)

        # Event heads.
        n_actions = len(ACTIONS)
        self.event_present = nn.Linear(d_model, 2)
        self.event_time = nn.Linear(d_model, 100)  # 0-99
        self.event_eid = nn.Linear(d_model, MAX_ENTITIES)
        self.event_action = nn.Linear(d_model, n_actions)
        self.event_modality = nn.Linear(d_model, n_modalities)

        # Slot queries (learnable): each slot attends to packets to predict one fact.
        self.attr_queries = nn.Parameter(torch.randn(max_attributes, d_model) * 0.02)
        self.rel_queries = nn.Parameter(torch.randn(max_relations, d_model) * 0.02)
        self.event_queries = nn.Parameter(torch.randn(max_events, d_model) * 0.02)

        # Cross-attention: slot queries attend to packets.
        self.attr_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.rel_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.event_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

    def forward(self, z_q: torch.Tensor) -> dict[str, Any]:
        """z_q: (B, n_slots, d_z) -> predicted state logits."""
        B, n_slots, _ = z_q.shape
        packets = self.packet_proj(z_q)  # (B, n_slots, d_model)
        packet_mem = self.packet_encoder(packets)  # (B, n_slots, d_model)

        # Entity prediction from mean of packets.
        pooled = packet_mem.mean(dim=1)  # (B, d_model)
        entity_logits = self.entity_head(pooled)  # (B, 8)

        # Attribute slots: cross-attention from queries to packets.
        attr_q = self.attr_queries.unsqueeze(0).expand(B, -1, -1)  # (B, max_attr, d_model)
        attr_h, _ = self.attr_attn(attr_q, packet_mem, packet_mem)  # (B, max_attr, d_model)
        attr_out: dict[str, torch.Tensor] = {
            "present": self.attr_present(attr_h),  # (B, max_attr, 2)
            "eid": self.attr_eid(attr_h),          # (B, max_attr, 8)
            "key": self.attr_key(attr_h),          # (B, max_attr, 20)
            "value": self.attr_value(attr_h),      # (B, max_attr, 8)
            "modality": self.attr_modality(attr_h),# (B, max_attr, 5)
        }

        # Relation slots.
        rel_q = self.rel_queries.unsqueeze(0).expand(B, -1, -1)
        rel_h, _ = self.rel_attn(rel_q, packet_mem, packet_mem)
        rel_out: dict[str, torch.Tensor] = {
            "present": self.rel_present(rel_h),
            "src": self.rel_src(rel_h),
            "type": self.rel_type(rel_h),
            "dst": self.rel_dst(rel_h),
            "modality": self.rel_modality(rel_h),
        }

        # Event slots.
        event_q = self.event_queries.unsqueeze(0).expand(B, -1, -1)
        event_h, _ = self.event_attn(event_q, packet_mem, packet_mem)
        event_out: dict[str, torch.Tensor] = {
            "present": self.event_present(event_h),
            "time": self.event_time(event_h),
            "eid": self.event_eid(event_h),
            "action": self.event_action(event_h),
            "modality": self.event_modality(event_h),
        }

        return {"entities": entity_logits, "attrs": attr_out,
                "rels": rel_out, "events": event_out}

    def state_loss(self, z_q: torch.Tensor, state: State) -> dict[str, torch.Tensor]:
        """Compute classification loss against the true state."""
        B = z_q.shape[0]
        device = z_q.device
        out = self.forward(z_q)

        losses = []

        # Entity loss: multi-label (which entities exist).
        entity_target = torch.zeros(B, MAX_ENTITIES, dtype=torch.float, device=device)
        for i, s in enumerate([state] if not isinstance(state, list) else state):
            for e in s.entities:
                entity_target[i, e] = 1.0
        ent_logits = out["entities"]
        assert isinstance(ent_logits, torch.Tensor)
        ent_loss = F.binary_cross_entropy_with_logits(ent_logits, entity_target)
        losses.append(ent_loss)

        # Attribute loss: match each true attribute to the best slot.
        # For simplicity: assign true attrs to the first N slots, rest are "absent".
        attr_present_target = torch.zeros(B, self.max_attributes, dtype=torch.long, device=device)
        attr_eid_target = torch.zeros(B, self.max_attributes, dtype=torch.long, device=device)
        attr_key_target = torch.zeros(B, self.max_attributes, dtype=torch.long, device=device)
        attr_value_target = torch.zeros(B, self.max_attributes, dtype=torch.long, device=device)
        attr_mod_target = torch.zeros(B, self.max_attributes, dtype=torch.long, device=device)

        states = state if isinstance(state, list) else [state]
        for i, s in enumerate(states):
            for j, a in enumerate(s.attributes[:self.max_attributes]):
                attr_present_target[i, j] = 1
                attr_eid_target[i, j] = a.eid
                attr_key_target[i, j] = ATTRIBUTE_KEYS.index(a.key)
                attr_value_target[i, j] = a.value
                attr_mod_target[i, j] = MODALITIES.index(a.modality)

        ao: dict[str, torch.Tensor] = out["attrs"]  # type: ignore[assignment]
        attr_loss = (F.cross_entropy(ao["present"].reshape(-1, 2), attr_present_target.reshape(-1))
                     + F.cross_entropy(ao["eid"].reshape(-1, MAX_ENTITIES), attr_eid_target.reshape(-1))
                     + F.cross_entropy(ao["key"].reshape(-1, len(ATTRIBUTE_KEYS)), attr_key_target.reshape(-1))
                     + F.cross_entropy(ao["value"].reshape(-1, ao["value"].size(-1)), attr_value_target.reshape(-1))
                     + F.cross_entropy(ao["modality"].reshape(-1, len(MODALITIES)), attr_mod_target.reshape(-1)))
        losses.append(attr_loss)

        # Relation loss.
        rel_present_target = torch.zeros(B, self.max_relations, dtype=torch.long, device=device)
        rel_src_target = torch.zeros(B, self.max_relations, dtype=torch.long, device=device)
        rel_type_target = torch.zeros(B, self.max_relations, dtype=torch.long, device=device)
        rel_dst_target = torch.zeros(B, self.max_relations, dtype=torch.long, device=device)
        rel_mod_target = torch.zeros(B, self.max_relations, dtype=torch.long, device=device)

        for i, s in enumerate(states):
            for j, r in enumerate(s.relations[:self.max_relations]):
                rel_present_target[i, j] = 1
                rel_src_target[i, j] = r.src
                rel_type_target[i, j] = RELATION_TYPES.index(r.rtype)
                rel_dst_target[i, j] = r.dst
                rel_mod_target[i, j] = MODALITIES.index(r.modality)

        ro: dict[str, torch.Tensor] = out["rels"]  # type: ignore[assignment]
        rel_loss = (F.cross_entropy(ro["present"].reshape(-1, 2), rel_present_target.reshape(-1))
                    + F.cross_entropy(ro["src"].reshape(-1, MAX_ENTITIES), rel_src_target.reshape(-1))
                    + F.cross_entropy(ro["type"].reshape(-1, len(RELATION_TYPES)), rel_type_target.reshape(-1))
                    + F.cross_entropy(ro["dst"].reshape(-1, MAX_ENTITIES), rel_dst_target.reshape(-1))
                    + F.cross_entropy(ro["modality"].reshape(-1, len(MODALITIES)), rel_mod_target.reshape(-1)))
        losses.append(rel_loss)

        # Event loss.
        ev_present_target = torch.zeros(B, self.max_events, dtype=torch.long, device=device)
        ev_time_target = torch.zeros(B, self.max_events, dtype=torch.long, device=device)
        ev_eid_target = torch.zeros(B, self.max_events, dtype=torch.long, device=device)
        ev_action_target = torch.zeros(B, self.max_events, dtype=torch.long, device=device)
        ev_mod_target = torch.zeros(B, self.max_events, dtype=torch.long, device=device)

        for i, s in enumerate(states):
            for j, e in enumerate(s.events[:self.max_events]):
                ev_present_target[i, j] = 1
                ev_time_target[i, j] = e.time
                ev_eid_target[i, j] = e.eid
                ev_action_target[i, j] = ACTIONS.index(e.action)
                ev_mod_target[i, j] = MODALITIES.index(e.modality)

        eo: dict[str, torch.Tensor] = out["events"]  # type: ignore[assignment]
        ev_loss = (F.cross_entropy(eo["present"].reshape(-1, 2), ev_present_target.reshape(-1))
                   + F.cross_entropy(eo["time"].reshape(-1, 100), ev_time_target.reshape(-1))
                   + F.cross_entropy(eo["eid"].reshape(-1, MAX_ENTITIES), ev_eid_target.reshape(-1))
                   + F.cross_entropy(eo["action"].reshape(-1, len(ACTIONS)), ev_action_target.reshape(-1))
                   + F.cross_entropy(eo["modality"].reshape(-1, len(MODALITIES)), ev_mod_target.reshape(-1)))
        losses.append(ev_loss)

        total = torch.stack(losses).sum()
        return {"state_loss": total, "ent_loss": ent_loss, "attr_loss": attr_loss,
                "rel_loss": rel_loss, "ev_loss": ev_loss}

    @torch.no_grad()
    def decode_state(self, z_q: torch.Tensor) -> list[State]:
        """Decode packets -> State objects (deterministic extraction)."""
        out = self.forward(z_q)
        B = z_q.shape[0]
        states = []
        for i in range(B):
            # Entities.
            ent_pred = (torch.sigmoid(out["entities"][i]) > 0.5).nonzero().squeeze(-1).tolist()
            entities = tuple(ent_pred) if ent_pred else (0,)

            # Attributes.
            attrs = []
            ao = out["attrs"]
            for j in range(self.max_attributes):
                if ao["present"][i, j].argmax() == 1:
                    eid = ao["eid"][i, j].argmax().item()
                    key = ATTRIBUTE_KEYS[ao["key"][i, j].argmax().item()]
                    value = ao["value"][i, j].argmax().item()
                    mod = MODALITIES[ao["modality"][i, j].argmax().item()]
                    attrs.append(Attribute(eid=eid, key=key, value=value, modality=mod))

            # Relations.
            rels = []
            ro = out["rels"]
            for j in range(self.max_relations):
                if ro["present"][i, j].argmax() == 1:
                    src = ro["src"][i, j].argmax().item()
                    rtype = RELATION_TYPES[ro["type"][i, j].argmax().item()]
                    dst = ro["dst"][i, j].argmax().item()
                    mod = MODALITIES[ro["modality"][i, j].argmax().item()]
                    rels.append(Relation(src=src, rtype=rtype, dst=dst, modality=mod))

            # Events.
            events = []
            eo: dict[str, torch.Tensor] = out["events"]  # type: ignore[assignment]
            for j in range(self.max_events):
                if eo["present"][i, j].argmax() == 1:
                    time = int(eo["time"][i, j].argmax().item())
                    eid = int(eo["eid"][i, j].argmax().item())
                    action = ACTIONS[int(eo["action"][i, j].argmax().item())]
                    mod = MODALITIES[int(eo["modality"][i, j].argmax().item())]
                    events.append(Event(time=time, eid=eid, action=action, modality=mod))

            states.append(State(
                seed=0, entities=entities, relations=tuple(rels),
                attributes=tuple(attrs), events=tuple(events),
            ))
        return states

    @torch.no_grad()
    def decode_to_text(self, z_q: torch.Tensor, view_id: int = 0) -> list[str]:
        """Decode packets -> State -> text (deterministic rendering)."""
        states = self.decode_state(z_q)
        return [self.renderer.render(s, view_id) for s in states]


__all__ = ["StateDecoder"]
