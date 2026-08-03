"""Codec: joint encoder + quantizer + decoder, with the MVP loss terms.

Wires E, Q, D and computes the loss terms from MVP §6:
    L_total = L_semantique + alpha*L_reconstruction + beta*L_alignement_multivue
            + gamma*L_taches + delta*L_compositionnalite + lambda*cout_bits

Answer-token mapping (matches tokenizer ANSWER_TOKENS order):
    unknown, yes, no, 0, 1, 2, 3, 4, 5, 6, 7   -> head class indices 0..10
    None -> 0 (unknown), True -> 1 (yes), False -> 2 (no), int v -> 3 + v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, PacketDecoder
from .quantizer import Quantizer, build_quantizer

UNK_CLS, YES_CLS, NO_CLS = 0, 1, 2


def answer_to_cls(answer) -> int:
    if answer is None:
        return UNK_CLS
    if isinstance(answer, bool):
        return YES_CLS if answer else NO_CLS
    return 3 + int(answer)


@dataclass
class CodecConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_z: int = 64
    n_slots: int = 4
    quantizer_kind: str = "pq"
    n_codebooks: int = 8
    codebook_size: int = 1024
    levels: int = 5
    pad_id: int = 0
    dropout: float = 0.0
    # loss weights
    w_task: float = 1.0          # 0 in phase 1 (reconstruction only), 1 in phase 2
    w_reconstruction: float = 0.3
    w_alignment: float = 0.2
    w_bits: float = 1e-4
    w_commit: float = 1.0
    w_entropy: float = 0.1       # anti-collapse: rewards code diversity


class Codec(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, d_z=cfg.d_z, n_slots=cfg.n_slots, pad_id=cfg.pad_id,
            dropout=cfg.dropout,
        )
        self.quantizer: Quantizer = build_quantizer(
            cfg.quantizer_kind, cfg.d_z, cfg.n_codebooks, cfg.codebook_size, cfg.levels,
        )
        self.decoder = PacketDecoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, d_z=cfg.d_z, n_answer_tokens=11, pad_id=cfg.pad_id,
            dropout=cfg.dropout,
        )

    def encode_quantize(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(text_ids, text_mask)             # (B, n_slots, d_z)
        B, n_slots, d_z = z.shape
        z_flat = z.reshape(B * n_slots, d_z)
        out = self.quantizer(z_flat)
        codes = out.codes.reshape(B, n_slots, -1)
        z_q = out.z_q.reshape(B, n_slots, d_z)
        entropy_loss = out.usage.get("entropy_loss", torch.zeros((), device=z.device))
        return codes, z_q, out.commit_loss, entropy_loss

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        text_ids = batch["text_ids"]
        text_mask = batch["text_mask"]
        codes, z_q, commit_loss, entropy_loss = self.encode_quantize(text_ids, text_mask)

        loss = commit_loss * self.cfg.w_commit + entropy_loss * self.cfg.w_entropy
        logs: dict[str, torch.Tensor] = {"commit_loss": commit_loss.detach(),
                                         "entropy_loss": entropy_loss.detach()}

        # L_semantique: task answering from the packets.
        # The decoder sees ONLY the task kind (qa/implication/...), NOT the question text.
        # It MUST extract the answer from the packets — no shortcut via the question.
        # w_task=0 in phase 1 (reconstruction only); >0 in phase 2.
        if "task_kind_id" in batch and self.cfg.w_task > 0:
            dec = self.decoder(z_q, batch["task_kind_id"])
            task_logits = dec["task_logits"]               # (B, n_answer)
            task_targets = batch["task_cls"]               # (B,)
            l_sem = F.cross_entropy(task_logits, task_targets)
            loss = loss + self.cfg.w_task * l_sem
            logs["l_semantique"] = l_sem.detach()
            logs["task_acc"] = (task_logits.argmax(-1) == task_targets).float().mean().detach()

        # L_reconstruction: teacher-forced text view reconstruction (auxiliary).
        # Conditioned on the target language so the decoder learns FR and EN separately.
        if "recon_ids" in batch:
            recon_lang = batch.get("recon_lang_id")
            recon_logits = self.decoder.reconstruct(z_q, batch["recon_ids"], recon_lang)
            recon_targets = batch["recon_ids"][:, 1:]      # predict next token
            recon_loss = F.cross_entropy(
                recon_logits[:, :-1].reshape(-1, recon_logits.size(-1)),
                recon_targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )
            loss = loss + self.cfg.w_reconstruction * recon_loss
            logs["l_reconstruction"] = recon_loss.detach()

        # L_alignement_multivue: cosine between latents of two views of the same state.
        if "text_ids_b" in batch:
            _, z_q_b, _, _ = self.encode_quantize(batch["text_ids_b"], batch["text_mask_b"])
            l_align = 1.0 - F.cosine_similarity(
                z_q.flatten(1), z_q_b.flatten(1), dim=-1, eps=1e-6
            ).mean()
            loss = loss + self.cfg.w_alignment * l_align
            logs["l_alignment"] = l_align.detach()

        # cout_bits: penalize total bits (n_slots * bits_per_packet).
        total_bits = self.cfg.n_slots * self.quantizer.bits_per_packet
        l_bits = self.cfg.w_bits * total_bits
        loss = loss + l_bits
        logs["total_bits"] = torch.tensor(float(total_bits))
        logs["loss"] = loss.detach()
        return {"loss": loss, "codes": codes, "logs": logs}


__all__ = ["Codec", "CodecConfig", "answer_to_cls"]
