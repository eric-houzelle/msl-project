"""Codec v2: bigger capacity + autoregressive decoder + translation.

The decoder is autoregressive (generates token by token, like a text LM) but
cross-attends to the packets at every step. This is the proven architecture
from v1, but with 4x more capacity:
  - d_z=128 (2x)
  - 16 codebooks (2x)
  - 32 slots (2x)
  - Total capacity: 4096 bits (4x more than v1's 1024)

For generation, we use top-k sampling instead of greedy to avoid the
repetition problem seen in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .quantizer import Quantizer, build_quantizer


@dataclass
class CodecV2Config:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_z: int = 128
    n_slots: int = 32
    quantizer_kind: str = "pq"
    n_codebooks: int = 16
    codebook_size: int = 256
    levels: int = 5
    pad_id: int = 0
    dropout: float = 0.0
    max_out_len: int = 96
    n_langs: int = 3
    w_reconstruction: float = 2.0
    w_task: float = 0.1
    w_alignment: float = 0.2
    w_commit: float = 1.0
    w_entropy: float = 0.1
    w_bits: float = 1e-4


class AutoregressiveDecoderV2(nn.Module):
    """Autoregressive decoder with cross-attention to packets + language conditioning.

    Generates text token by token, but each token cross-attends to the packets.
    Language embedding conditions which language to generate (FR/EN/structured).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_z: int = 128,
        max_len: int = 96,
        n_langs: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.packet_proj = nn.Linear(d_z, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.lang_embed = nn.Embedding(n_langs, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                        batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
        )
        self.text_head = nn.Linear(d_model, vocab_size)

        # Task head (separate).
        self.task_head = nn.Linear(d_model, 11)
        self.task_kind_embed = nn.Embedding(5, d_model)

    def reconstruct(self, z_q: torch.Tensor, target_ids: torch.Tensor,
                    lang_id: torch.Tensor, mask_id: int = 4,
                    mask_prob: float = 0.4,
                    content_token_ids: set | None = None) -> dict[str, torch.Tensor]:
        """Teacher-forced reconstruction with content masking.

        During training, content tokens (digits, attribute keys) are replaced
        with  mask with probability mask_prob. This forces the decoder to read
        the PACKETS to recover the content, instead of cheating from context.
        """
        B, n_slots, _ = z_q.shape
        T = target_ids.size(1)
        packet_mem = self.packet_proj(z_q)

        input_ids = target_ids.clone()
        if self.training and mask_prob > 0:
            # Mask content tokens (digits + attribute names) randomly.
            mask = torch.rand(B, T, device=z_q.device) < mask_prob
            if content_token_ids:
                content_mask = torch.zeros(B, T, dtype=torch.bool, device=z_q.device)
                for tid in content_token_ids:
                    content_mask |= (target_ids == tid)
                mask = mask & content_mask
            else:
                mask = mask & (target_ids > 4)  # above special tokens
            input_ids[mask] = mask_id

        tgt = self.token_embed(input_ids) + self.pos(torch.arange(T, device=z_q.device)).unsqueeze(0)
        tgt = tgt + self.lang_embed(lang_id).unsqueeze(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=z_q.device), diagonal=1)
        dec = self.decoder(tgt, packet_mem, tgt_mask=causal)
        logits = self.text_head(dec)

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            target_ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        pred = logits[:, :-1].argmax(dim=-1)
        targets = target_ids[:, 1:]
        non_pad = (targets != 0)
        acc = (pred == targets)[non_pad].float().mean() if non_pad.any() else torch.tensor(0.0, device=z_q.device)
        return {"recon_loss": loss, "recon_acc": acc}

    def answer_task(self, z_q: torch.Tensor, task_kind_id: torch.Tensor) -> dict[str, torch.Tensor]:
        packet_summary = self.packet_proj(z_q).mean(dim=1)
        h = packet_summary + self.task_kind_embed(task_kind_id)
        return {"task_logits": self.task_head(h)}

    @torch.no_grad()
    def generate(self, z_q: torch.Tensor, lang_id: torch.Tensor, bos_id: int,
                 eos_id: int, max_len: int = 96, top_k: int = 10,
                 temperature: float = 1.0) -> torch.Tensor:
        """Generate text with top-k sampling (avoids greedy repetition)."""
        B = z_q.shape[0]
        device = z_q.device
        packet_mem = self.packet_proj(z_q)
        lang_h = self.lang_embed(lang_id)  # (B, d_model)

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            T = tokens.size(1)
            tgt = self.token_embed(tokens) + self.pos(torch.arange(T, device=device)).unsqueeze(0)
            tgt = tgt + lang_h.unsqueeze(1)
            causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            dec = self.decoder(tgt, packet_mem, tgt_mask=causal)
            logits = self.text_head(dec[:, -1]) / temperature  # (B, vocab)

            # Top-k sampling.
            if top_k > 0:
                top_vals, top_idx = logits.topk(top_k, dim=-1)
                probs = F.softmax(top_vals, dim=-1)
                sampled = torch.multinomial(probs, 1)  # (B, 1)
                next_token = top_idx.gather(1, sampled)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return tokens


class CodecV2(nn.Module):
    def __init__(self, cfg: CodecV2Config, content_token_ids: set | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.content_token_ids = content_token_ids or set()
        self.encoder = Encoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, d_z=cfg.d_z, n_slots=cfg.n_slots, pad_id=cfg.pad_id,
            dropout=cfg.dropout,
        )
        self.quantizer: Quantizer = build_quantizer(
            cfg.quantizer_kind, cfg.d_z, cfg.n_codebooks, cfg.codebook_size, cfg.levels,
        )
        self.decoder = AutoregressiveDecoderV2(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, d_z=cfg.d_z, max_len=cfg.max_out_len,
            n_langs=cfg.n_langs, dropout=cfg.dropout,
        )

    def encode_quantize(self, text_ids, text_mask):
        z = self.encoder(text_ids, text_mask)
        B, n_slots, d_z = z.shape
        z_flat = z.reshape(B * n_slots, d_z)
        out = self.quantizer(z_flat)
        codes = out.codes.reshape(B, n_slots, -1)
        z_q = out.z_q.reshape(B, n_slots, d_z)
        entropy_loss = out.usage.get("entropy_loss", torch.zeros((), device=z.device))
        return codes, z_q, out.commit_loss, entropy_loss

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        text_ids = batch["text_ids"]
        text_mask = batch["text_mask"]
        codes, z_q, commit_loss, entropy_loss = self.encode_quantize(text_ids, text_mask)

        loss = commit_loss * self.cfg.w_commit + entropy_loss * self.cfg.w_entropy
        logs: dict[str, torch.Tensor] = {
            "commit_loss": commit_loss.detach(), "entropy_loss": entropy_loss.detach()}

        if "recon_ids" in batch:
            recon_lang = batch.get("recon_lang_id")
            if recon_lang is None:
                recon_lang = torch.zeros(z_q.shape[0], dtype=torch.long, device=z_q.device)
            dec_out = self.decoder.reconstruct(
                z_q, batch["recon_ids"], recon_lang,
                mask_id=4, mask_prob=0.4 if self.training else 0.0,
                content_token_ids=self.content_token_ids,
            )
            loss = loss + self.cfg.w_reconstruction * dec_out["recon_loss"]
            logs["l_reconstruction"] = dec_out["recon_loss"].detach()
            logs["recon_acc"] = dec_out["recon_acc"].detach()

        if "task_kind_id" in batch and self.cfg.w_task > 0:
            task_out = self.decoder.answer_task(z_q, batch["task_kind_id"])
            l_task = F.cross_entropy(task_out["task_logits"], batch["task_cls"])
            loss = loss + self.cfg.w_task * l_task
            logs["l_task"] = l_task.detach()
            logs["task_acc"] = (task_out["task_logits"].argmax(-1) == batch["task_cls"]).float().mean().detach()

        if "text_ids_b" in batch:
            _, z_q_b, _, _ = self.encode_quantize(batch["text_ids_b"], batch["text_mask_b"])
            l_align = 1.0 - F.cosine_similarity(
                z_q.flatten(1), z_q_b.flatten(1), dim=-1, eps=1e-6).mean()
            loss = loss + self.cfg.w_alignment * l_align
            logs["l_alignment"] = l_align.detach()

        total_bits = self.cfg.n_slots * self.quantizer.bits_per_packet
        loss = loss + self.cfg.w_bits * total_bits
        logs["total_bits"] = torch.tensor(float(total_bits))
        logs["loss"] = loss.detach()
        return {"loss": loss, "codes": codes, "logs": logs}


__all__ = ["CodecV2", "CodecV2Config", "AutoregressiveDecoderV2"]
