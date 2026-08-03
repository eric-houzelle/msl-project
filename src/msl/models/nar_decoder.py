"""Non-autoregressive decoder: predicts ALL tokens at once from packets.

This solves the exposure bias problem: instead of generating token-by-token
(each error compounds), the decoder sees the full packets and predicts the
entire output sequence in one pass. No causal mask, no error accumulation.

Architecture:
  packets (N_slots, d_z) -> project to (N_slots, d_model)
  + learnable position queries for the output (T, d_model)
  -> cross-attention (output queries attend to packets)
  -> FFN
  -> token logits (T, vocab)

The decoder also takes a language embedding (FR/EN) to condition the output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonAutoregressiveDecoder(nn.Module):
    """Predicts all output tokens simultaneously from packets + language ID."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_z: int = 128,
        max_out_len: int = 96,
        n_langs: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Project packets to model dimension.
        self.packet_proj = nn.Linear(d_z, d_model)

        # Learnable output position embeddings (one per output position).
        self.output_pos = nn.Embedding(max_out_len, d_model)

        # Length embedding: condition the decoder on the target length.
        self.len_embed = nn.Embedding(max_out_len, d_model)

        # Language embedding.
        self.lang_embed = nn.Embedding(n_langs, d_model)

        # Transformer decoder: output queries cross-attend to packets.
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                        batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
        )

        # Token prediction head.
        self.token_head = nn.Linear(d_model, vocab_size)

        # Task answering head (separate, for QA).
        self.task_head = nn.Linear(d_model, 11)  # unknown, yes, no, 0..7
        self.n_task_kinds = 5
        self.task_kind_embed = nn.Embedding(self.n_task_kinds, d_model)

    def forward(self, z_q: torch.Tensor, lang_id: torch.Tensor,
                out_len: int | None = None,
                target_len: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Predict all output tokens at once.

        z_q: (B, n_slots, d_z) — the packets
        lang_id: (B,) — target language (0=fr, 1=en, 2=structured)
        out_len: number of output positions to predict (default: max)
        target_len: (B,) optional actual length of each target (for length conditioning)
        Returns logits (B, out_len, vocab_size).
        """
        B, n_slots, _ = z_q.shape

        if out_len is None:
            out_len = self.output_pos.num_embeddings

        # Packet memory.
        packet_mem = self.packet_proj(z_q)  # (B, n_slots, d_model)

        # Output queries: explicit position embeddings + language embedding.
        pos_ids = torch.arange(out_len, device=z_q.device)
        queries = self.output_pos(pos_ids).unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)
        queries = queries + self.lang_embed(lang_id).unsqueeze(1)  # add language condition
        # Add length conditioning if provided.
        if target_len is not None:
            queries = queries + self.len_embed(target_len.clamp(max=self.len_embed.num_embeddings-1)).unsqueeze(1)

        # Cross-attention: output queries attend to packets.
        dec = self.decoder(queries, packet_mem)  # (B, T, d_model)

        # Token logits.
        token_logits = self.token_head(dec)  # (B, T, vocab_size)
        return {"token_logits": token_logits, "hidden": dec}

    def reconstruct_loss(self, z_q: torch.Tensor, target_ids: torch.Tensor,
                         lang_id: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute reconstruction loss (all tokens predicted in parallel)."""
        T = target_ids.size(1)
        # Compute actual length of each target (number of non-pad tokens).
        target_len = (target_ids != 0).sum(dim=1)  # (B,)
        out = self.forward(z_q, lang_id, out_len=T, target_len=target_len)
        logits = out["token_logits"]  # (B, T, vocab)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=0,
        )
        pred = logits.argmax(dim=-1)
        non_pad = (target_ids != 0)
        acc = (pred == target_ids)[non_pad].float().mean() if non_pad.any() else torch.tensor(0.0, device=z_q.device)
        return {"recon_loss": loss, "recon_acc": acc}

    def answer_task(self, z_q: torch.Tensor, task_kind_id: torch.Tensor) -> dict[str, torch.Tensor]:
        """Answer a task question from the packets (separate head)."""
        # Use the mean of packet representations as a summary.
        packet_summary = self.packet_proj(z_q).mean(dim=1)  # (B, d_model)
        # Add task kind embedding.
        h = packet_summary + self.task_kind_embed(task_kind_id)
        logits = self.task_head(h)  # (B, 11)
        return {"task_logits": logits}

    @torch.no_grad()
    def generate(self, z_q: torch.Tensor, lang_id: torch.Tensor,
                 out_len: int = 96) -> torch.Tensor:
        """Generate text (non-autoregressive — just one forward pass!)."""
        out = self.forward(z_q, lang_id, out_len=out_len)
        return out["token_logits"].argmax(dim=-1)  # (B, out_len)


__all__ = ["NonAutoregressiveDecoder"]
