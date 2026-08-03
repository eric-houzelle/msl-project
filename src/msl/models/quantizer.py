"""Discrete quantizers: the MSL bottleneck.

Three variants (brief §5.2, MVP §3.3):
- PQ  (Product Quantization): split z into B sub-vectors, quantize each in its own codebook.
- RVQ (Residual Vector Quantization): B codebooks applied successively to the residual.
- FSQ (Finite Scalar Quantization): quantize each of B dims to L levels (no codebook vectors).

All quantizers share an interface:
    forward(z: [N, d_z]) -> QuantizerOutput(codes [N, B], z_q [N, d_z], commit_loss, usage)

Straight-through estimator passes gradients to the encoder. PQ/RVQ use EMA codebook
updates (robust to collapse, a named risk in MVP §12). FSQ has no codebook vectors,
so it cannot collapse — it serves as a lower-bound control on codebook width.

Bits per packet = B * log2(V) for PQ/RVQ, sum(log2(L_i)) for FSQ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    codes: torch.Tensor        # (N, B) long
    z_q: torch.Tensor         # (N, d_z)
    commit_loss: torch.Tensor  # scalar
    bits: float               # bits per packet
    usage: dict[str, torch.Tensor]


class Quantizer(nn.Module):
    """Base class. Subclasses implement _quantize(z) -> (codes, z_q_hard, commit_loss, stats)."""

    def __init__(self, d_z: int, n_codebooks: int) -> None:
        super().__init__()
        self.d_z = d_z
        self.n_codebooks = n_codebooks

    @property
    def bits_per_packet(self) -> float:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        codes, z_q_hard, commit_loss, stats = self._quantize(z)
        # Straight-through: gradient flows through z, forward uses quantized value.
        z_q = z + (z_q_hard - z).detach()
        return QuantizerOutput(
            codes=codes,
            z_q=z_q,
            commit_loss=commit_loss,
            bits=self.bits_per_packet,
            usage=stats,
        )

    def _quantize(self, z: torch.Tensor) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _codebook_usage(counts: torch.Tensor) -> dict[str, torch.Tensor]:
        total = counts.sum().clamp(min=1)
        probs = counts.float() / total
        active = (counts > 0).float().mean()
        entropy = -(probs * (probs + 1e-10).log()).sum()
        perplexity = torch.exp(entropy)
        return {"active_fraction": active, "perplexity": perplexity}


class _EMACodebook(nn.Module):
    """A single codebook with EMA updates + dead code restart.

    Dead codes (codes rarely used) are periodically replaced with random samples
    from the current batch, preventing codebook collapse.
    """

    dim: int
    n_codes: int
    decay: float
    restart_threshold: float
    embed: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor
    usage: torch.Tensor

    def __init__(self, dim: int, n_codes: int, decay: float = 0.99,
                 restart_threshold: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.decay = decay
        self.restart_threshold = restart_threshold  # EMA usage below this = dead
        self.register_buffer("embed", torch.randn(n_codes, dim) * 0.02)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("usage", torch.zeros(n_codes))

    @torch.no_grad()
    def _ema_update(self, onehot: torch.Tensor, flat_z: torch.Tensor) -> None:
        # onehot: (M, n_codes), flat_z: (M, dim)
        counts = onehot.sum(0)                          # (n_codes,)
        sums = onehot.T @ flat_z                        # (n_codes, dim)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + 1e-5) / (n + self.n_codes * 1e-5) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1).clamp(min=1e-5))
        # Track usage for dead code detection.
        self.usage.mul_(0.99).add_(counts.clamp(max=1.0), alpha=0.01)

    @torch.no_grad()
    def restart_dead(self, flat_z: torch.Tensor) -> int:
        """Replace dead codes with random samples from the batch. Returns n restarted."""
        if flat_z.shape[0] == 0:
            return 0
        dead = self.usage < self.restart_threshold
        n_dead: int = int(dead.sum().item())
        if n_dead == 0:
            return 0
        # Sample replacement vectors from the batch.
        rand_idx = torch.randint(0, flat_z.shape[0], (n_dead,), device=flat_z.device)
        replacements = flat_z[rand_idx]
        self.embed[dead] = replacements
        self.embed_avg[dead] = replacements
        self.cluster_size[dead] = float(self.cluster_size.mean().item())
        self.usage[dead] = self.restart_threshold  # give them a chance
        return n_dead

    def nearest(self, flat_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (M, dim) -> (M,) indices, (M, dim) embeddings
        dist = torch.cdist(flat_z.unsqueeze(0), self.embed.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        return idx, self.embed[idx]

    def entropy_loss(self, idx: torch.Tensor) -> torch.Tensor:
        """Entropy bonus: rewards using many codes uniformly. Penalizes collapse."""
        probs = torch.bincount(idx, minlength=self.n_codes).float()
        probs = probs / (probs.sum() + 1e-8)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        max_entropy = math.log(self.n_codes)
        return max_entropy - entropy  # 0 = perfect, higher = more collapse


class PQQuantizer(Quantizer):
    """Product Quantization: B independent codebooks over sub-vectors of z."""

    def __init__(self, d_z: int, n_codebooks: int, codebook_size: int, decay: float = 0.99) -> None:
        super().__init__(d_z, n_codebooks)
        assert d_z % n_codebooks == 0, "d_z must be divisible by n_codebooks for PQ"
        self.sub_dim = d_z // n_codebooks
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [_EMACodebook(self.sub_dim, codebook_size, decay) for _ in range(n_codebooks)]
        )
        self._commit_beta = 0.25

    @property
    def bits_per_packet(self) -> float:
        return self.n_codebooks * math.log2(self.codebook_size)

    def _quantize(self, z: torch.Tensor) -> tuple:
        z = F.normalize(z, dim=-1)  # bound magnitude, stabilize cdist + commit
        N = z.shape[0]
        codes = torch.zeros(N, self.n_codebooks, dtype=torch.long, device=z.device)
        parts = torch.zeros(N, self.d_z, device=z.device)
        commit = torch.zeros((), device=z.device)
        entropy = torch.zeros((), device=z.device)
        usages = []
        for i, cb in enumerate(self.codebooks):
            cb = cast(_EMACodebook, cb)
            sub = z[:, i * self.sub_dim:(i + 1) * self.sub_dim]
            idx, emb = cb.nearest(sub)
            codes[:, i] = idx
            parts[:, i * self.sub_dim:(i + 1) * self.sub_dim] = emb
            commit = commit + F.mse_loss(sub, emb.detach())
            entropy = entropy + cb.entropy_loss(idx)
            if self.training:
                onehot = F.one_hot(idx, cb.n_codes).float()
                cb._ema_update(onehot, sub)
                cb.restart_dead(sub)
            usages.append(self._codebook_usage(torch.bincount(idx, minlength=cb.n_codes)))
        usage = {
            "active_fraction": torch.stack([u["active_fraction"] for u in usages]).mean(),
            "perplexity": torch.stack([u["perplexity"] for u in usages]).mean(),
            "entropy_loss": entropy / self.n_codebooks,
        }
        return codes, parts, self._commit_beta * commit, usage


class RVQQuantizer(Quantizer):
    """Residual Vector Quantization: B codebooks applied to successive residuals."""

    def __init__(self, d_z: int, n_codebooks: int, codebook_size: int, decay: float = 0.99) -> None:
        super().__init__(d_z, n_codebooks)
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [_EMACodebook(d_z, codebook_size, decay) for _ in range(n_codebooks)]
        )
        self._commit_beta = 0.25

    @property
    def bits_per_packet(self) -> float:
        return self.n_codebooks * math.log2(self.codebook_size)

    def _quantize(self, z: torch.Tensor) -> tuple:
        z = F.normalize(z, dim=-1)  # bound magnitude, stabilize cdist + commit
        N = z.shape[0]
        codes = torch.zeros(N, self.n_codebooks, dtype=torch.long, device=z.device)
        quantized = torch.zeros_like(z)
        residual = z.clone()
        commit = torch.zeros((), device=z.device)
        entropy = torch.zeros((), device=z.device)
        usages = []
        for i, cb in enumerate(self.codebooks):
            cb = cast(_EMACodebook, cb)
            idx, emb = cb.nearest(residual)
            codes[:, i] = idx
            quantized = quantized + emb
            commit = commit + F.mse_loss(residual, emb.detach())
            entropy = entropy + cb.entropy_loss(idx)
            residual = residual - emb.detach()
            if self.training:
                onehot = F.one_hot(idx, cb.n_codes).float()
                cb._ema_update(onehot, residual + emb.detach())
                cb.restart_dead(residual + emb.detach())
            usages.append(self._codebook_usage(torch.bincount(idx, minlength=cb.n_codes)))
        usage = {
            "active_fraction": torch.stack([u["active_fraction"] for u in usages]).mean(),
            "perplexity": torch.stack([u["perplexity"] for u in usages]).mean(),
            "entropy_loss": entropy / self.n_codebooks,
        }
        return codes, quantized, self._commit_beta * commit, usage


class FSQQuantizer(Quantizer):
    """Finite Scalar Quantization: quantize each of B dims to L levels.

    No codebook vectors: quantization is deterministic rounding of bounded values.
    Cannot collapse — serves as a lower-bound control. To match the d_z interface,
    we project z -> B dims, quantize, then project back -> d_z (learnable, differentiable).
    """

    def __init__(self, d_z: int, n_codebooks: int, levels: int) -> None:
        super().__init__(d_z, n_codebooks)
        self.levels = levels
        self.project_down = nn.Linear(d_z, n_codebooks)
        self.project_up = nn.Linear(n_codebooks, d_z)

    @property
    def bits_per_packet(self) -> float:
        return self.n_codebooks * math.log2(self.levels)

    def _quantize(self, z: torch.Tensor) -> tuple:
        h = self.project_down(z)                       # (N, B)
        # Bound to [-1, 1] then scale to [0, L-1] and round.
        h_bounded = torch.tanh(h)
        scaled = (h_bounded + 1) / 2 * (self.levels - 1)
        codes_float = scaled.detach().round().clamp(0, self.levels - 1)
        codes = codes_float.long()                      # (N, B)
        # Quantized continuous value (straight-through applied to scaled).
        quantized_scaled = codes_float / (self.levels - 1) * 2 - 1   # back to [-1,1]
        h_q = h + (quantized_scaled - h).detach()
        z_q = self.project_up(h_q)                      # (N, d_z)
        # No commitment loss for FSQ (no codebook vectors).
        usages = [self._codebook_usage(torch.bincount(codes[:, i], minlength=self.levels))
                  for i in range(self.n_codebooks)]
        usage = {
            "active_fraction": torch.stack([u["active_fraction"] for u in usages]).mean(),
            "perplexity": torch.stack([u["perplexity"] for u in usages]).mean(),
        }
        return codes, z_q, torch.zeros((), device=z.device), usage


def build_quantizer(kind: str, d_z: int, n_codebooks: int, codebook_size: int = 1024,
                    levels: int = 5) -> Quantizer:
    if kind == "pq":
        return PQQuantizer(d_z, n_codebooks, codebook_size)
    if kind == "rvq":
        return RVQQuantizer(d_z, n_codebooks, codebook_size)
    if kind == "fsq":
        return FSQQuantizer(d_z, n_codebooks, levels)
    raise ValueError(f"unknown quantizer kind: {kind}")


__all__ = [
    "QuantizerOutput", "Quantizer", "PQQuantizer", "RVQQuantizer", "FSQQuantizer",
    "build_quantizer",
]
