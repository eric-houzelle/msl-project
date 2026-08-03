"""Native MSL Language Model: predicts the next packet in a sequence.

This is the core of the project: a Transformer that thinks ONLY in MSL packets.
It never sees text. It predicts the next packet (8 codes in parallel, one step
instead of 8 autoregressive steps) from the previous packets.

For comparison, a text LM predicts one token at a time. MSL predicts a whole
packet (the "word" of the machine language) in a single step — that's where the
efficiency gain should come from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeMSLLM(nn.Module):
    """Transformer that operates on packet sequences. Predicts next packet.

    Each packet = (c1, ..., c8) codes from 8 codebooks. The model:
    1. Embeds each packet into a single vector (sum of codebook embeddings).
    2. Runs a Transformer over the packet sequence.
    3. Predicts the next packet via 8 parallel heads (one per codebook).
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        codebook_size: int = 256,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.d_model = d_model

        # One embedding table per codebook (the "vocabulary" of MSL).
        self.code_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(n_codebooks)
        ])
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
            enable_nested_tensor=False,
        )
        # 8 parallel prediction heads (one per codebook) — single step, not 8.
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(n_codebooks)
        ])

    def forward(self, packets: torch.Tensor) -> dict[str, torch.Tensor]:
        """packets: (B, S, n_codebooks) -> logits (B, S, n_codebooks, codebook_size).

        Predicts the NEXT packet at each position (shifted by 1).
        """
        B, S, C = packets.shape
        assert C == self.n_codebooks
        # Embed each packet: sum of codebook embeddings.
        h = torch.zeros(B, S, self.d_model, device=packets.device)
        for i in range(self.n_codebooks):
            h = h + self.code_embeds[i](packets[:, :, i])
        h = h + self.pos(torch.arange(S, device=packets.device)).unsqueeze(0)
        # Causal attention: predict next from previous.
        causal = torch.triu(torch.full((S, S), float("-inf"), device=packets.device), diagonal=1)
        h = self.transformer(h, mask=causal)
        # 8 parallel heads.
        all_logits = []
        for i in range(self.n_codebooks):
            all_logits.append(self.heads[i](h))  # (B, S, codebook_size)
        logits = torch.stack(all_logits, dim=2)  # (B, S, n_codebooks, codebook_size)
        return {"logits": logits}

    def loss(self, packets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Cross-entropy loss: predict next packet from current."""
        out = self.forward(packets)
        logits = out["logits"][:, :-1]  # (B, S-1, C, V)
        targets = packets[:, 1:]        # (B, S-1, C)
        losses = []
        for i in range(self.n_codebooks):
            losses.append(F.cross_entropy(
                logits[:, :, i, :].reshape(-1, self.codebook_size),
                targets[:, :, i].reshape(-1),
            ))
        total_loss = torch.stack(losses).mean()
        # Accuracy: fraction of packets where ALL 8 codes are correct.
        pred = logits.argmax(dim=-1)  # (B, S-1, C)
        exact = (pred == targets).all(dim=-1).float().mean()
        per_code = (pred == targets).float().mean()
        return {"loss": total_loss, "exact_acc": exact, "per_code_acc": per_code}

    @torch.no_grad()
    def generate_constrained(self, prompt: torch.Tensor, n_steps: int,
                             valid_ranges: dict[int, tuple[int, int]] | None = None,
                             temperature: float = 1.0) -> torch.Tensor:
        """Generate packets autoregressively with domain constraints.

        prompt: (B, S_prompt, n_codebooks) the context packets.
        n_steps: how many more packets to generate.
        valid_ranges: optional dict {codebook_index: (min, max)} to constrain
                      each code to a valid range. Values outside are masked to -inf.
        Returns: (B, S_prompt + n_steps, n_codebooks)
        """
        B = prompt.shape[0]
        current = prompt
        for _ in range(n_steps):
            out = self.forward(current)
            next_logits = out["logits"][:, -1]  # (B, n_codebooks, codebook_size)
            if temperature != 1.0:
                next_logits = next_logits / temperature
            # Apply validity constraints.
            if valid_ranges:
                for cb_idx, (vmin, vmax) in valid_ranges.items():
                    mask = torch.zeros_like(next_logits[:, cb_idx])
                    mask[:, :vmin] = float("-inf")
                    mask[:, vmax + 1:] = float("-inf")
                    next_logits[:, cb_idx] = next_logits[:, cb_idx] + mask
            next_pkt = next_logits.argmax(dim=-1)  # (B, n_codebooks)
            current = torch.cat([current, next_pkt.unsqueeze(1)], dim=1)
        return current


__all__ = ["NativeMSLLM"]
