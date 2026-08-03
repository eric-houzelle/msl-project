"""Text baseline (B-text): reads the text directly and answers the task.

This is the reference model for comparing MSL against. It has the same backbone
as the MSL codec (Transformer encoder + task head), the same size, the same
training data — but it reads the TEXT directly instead of going through the
quantized packet bottleneck.

If MSL (reading packets) answers tasks as well as B-text (reading text), then
the packets carry the same information as the words, but more compactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextBaseline(nn.Module):
    """Reads text -> Transformer encoder -> task head. No quantizer, no packets."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_answer_tokens: int = 11,
        pad_id: int = 0,
        max_len: int = 96,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
            enable_nested_tensor=False,
        )
        # Task head uses a learnable [TASK] query that cross-attends to the text.
        self.task_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.task_head = nn.Linear(d_model, n_answer_tokens)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                task_ids: torch.Tensor, task_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """text_ids: (B, T_text), task_ids: (B, T_task) -> logits (B, n_answer)."""
        B, T = text_ids.shape
        h = self.embed(text_ids) + self.pos(torch.arange(T, device=text_ids.device)).unsqueeze(0)
        key_padding_mask = text_mask.bool()
        memory = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)
        # Cross-attend from a task query to the encoded text.
        q = self.task_query.unsqueeze(0).expand(B, -1, -1)  # (B, 1, d_model)
        # Use the task text as part of the query context by also encoding it.
        task_h = self.embed(task_ids) + self.pos(torch.arange(task_ids.size(1), device=task_ids.device)).unsqueeze(0)
        q = q.expand(B, task_ids.size(1), -1) + task_h * 0.0  # keep task text available
        attended, _ = self.cross_attn(task_h, memory, memory, key_padding_mask=key_padding_mask)
        # Pool the attended representation and predict.
        logits = self.task_head(attended[:, 0])  # first position
        return {"task_logits": logits}

    def forward_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        out = self.forward(batch["text_ids"], batch["text_mask"],
                          batch["task_ids"], batch.get("task_mask"))
        loss = F.cross_entropy(out["task_logits"], batch["task_cls"])
        acc = (out["task_logits"].argmax(-1) == batch["task_cls"]).float().mean()
        return {"loss": loss, "task_acc": acc.detach()}


__all__ = ["TextBaseline"]
