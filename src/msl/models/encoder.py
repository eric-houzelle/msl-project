"""Encoder E: text views -> n latent vectors (one per packet slot).

Per the resolved design (audit §5.1): the message = a sequence of n packets;
each packet is one quantization of one latent vector. The encoder produces n
latent vectors via learnable slot queries that cross-attend to the text.
Sweeping n is how H1 measures minimal length L(s).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_z: int = 64,
        n_slots: int = 4,
        pad_id: int = 0,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_z = d_z
        self.n_slots = n_slots
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
            enable_nested_tensor=False,
        )
        # Learnable slot queries (cross-attend to text hidden states).
        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.to_z = nn.Linear(d_model, d_z)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_ids: (B, T) -> latents (B, n_slots, d_z)."""
        B, T = token_ids.shape
        h = self.embed(token_ids) + self.pos(torch.arange(T, device=token_ids.device)).unsqueeze(0)
        key_padding_mask = mask.bool() if mask is not None else None
        memory = self.text_encoder(h, src_key_padding_mask=key_padding_mask)   # (B, T, d_model)
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)                    # (B, n_slots, d_model)
        slot_h, _ = self.cross_attn(q, memory, memory, key_padding_mask=key_padding_mask)
        z = self.to_z(slot_h)                                                   # (B, n_slots, d_z)
        return z


class PacketDecoder(nn.Module):
    """Decoder D: packet embeddings + task query -> answer + text reconstruction.

    Two heads:
    - task_head: classifies the answer (qa/composition: value+unknown; bool tasks: yes/no).
    - text_head: LM over vocab for reconstruction (auxiliary, L_reconstruction).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_z: int = 64,
        n_answer_tokens: int = 11,   # unknown, yes, no, 0..7
        pad_id: int = 0,
        max_len: int = 256,
        dropout: float = 0.0,
        n_task_kinds: int = 5,
        n_langs: int = 3,   # 0=fr, 1=en, 2=structured
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.packet_proj = nn.Linear(d_z, d_model)
        self.task_kind_embed = nn.Embedding(n_task_kinds, d_model)
        self.lang_embed = nn.Embedding(n_langs, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                        batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
        )
        # Cross-attention from task query to packets.
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        # Embedding for reconstruction target tokens (separate from task kind).
        self.recon_embed = nn.Embedding(vocab_size, d_model)
        # Task head: predict one of n_answer_tokens.
        self.task_head = nn.Linear(d_model, n_answer_tokens)
        # Text reconstruction head (LM over full vocab).
        self.text_head = nn.Linear(d_model, vocab_size)

    def forward(self, z_q: torch.Tensor, task_kind_id: torch.Tensor) -> dict[str, torch.Tensor]:
        """z_q: (B, n_slots, d_z), task_kind_id: (B,) -> logits.

        The decoder sees ONLY the task kind (qa/implication/etc.) and the packets.
        It cannot see the question text, so it MUST extract the answer from the packets.
        """
        B, n_slots, _ = z_q.shape
        packet_mem = self.packet_proj(z_q)                                    # (B, n_slots, d_model)
        kind_h = self.task_kind_embed(task_kind_id)                           # (B, d_model)
        # The decoder cross-attends FROM the kind query TO the packets.
        attended, _ = self.cross_attn(kind_h.unsqueeze(1), packet_mem, packet_mem)  # (B, 1, d_model)
        task_logits = self.task_head(attended[:, 0])                           # (B, n_answer)
        return {"task_logits": task_logits}

    def reconstruct(self, z_q: torch.Tensor, target_ids: torch.Tensor,
                    lang_id: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct a text view (teacher forcing). Returns logits (B, T, vocab).

        lang_id: (B,) optional language selector. If provided, conditions generation.
        """
        B, n_slots, _ = z_q.shape
        packet_mem = self.packet_proj(z_q)
        T = target_ids.size(1)
        tgt = self.recon_embed(target_ids) + self.pos(torch.arange(T, device=target_ids.device)).unsqueeze(0)
        if lang_id is not None:
            # Prepend language condition to every position.
            tgt = tgt + self.lang_embed(lang_id).unsqueeze(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=z_q.device), diagonal=1)
        dec = self.decoder(tgt, packet_mem, tgt_mask=causal)
        return self.text_head(dec)

    @torch.no_grad()
    def generate(self, z_q: torch.Tensor, lang_id: torch.Tensor, bos_id: int,
                 eos_id: int, max_len: int = 96) -> torch.Tensor:
        """Autoregressive generation: packets -> text in the chosen language.

        No teacher forcing — each token is predicted then fed back.
        """
        B, n_slots, _ = z_q.shape
        device = z_q.device
        packet_mem = self.packet_proj(z_q)
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        lang_h = self.lang_embed(lang_id)  # (B, d_model)
        for _ in range(max_len):
            T = tokens.size(1)
            tgt = self.recon_embed(tokens) + self.pos(torch.arange(T, device=device)).unsqueeze(0)
            tgt = tgt + lang_h.unsqueeze(1)
            causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            dec = self.decoder(tgt, packet_mem, tgt_mask=causal)
            next_logits = self.text_head(dec[:, -1])  # (B, vocab)
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return tokens


__all__ = ["Encoder", "PacketDecoder"]
