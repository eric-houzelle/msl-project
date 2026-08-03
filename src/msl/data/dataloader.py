"""On-the-fly and precomputed MS-1 datasets for codec training.

Each sample is a (state, task) pair carrying two textual views of the same
state (for L_alignement_multivue) plus a tokenized question and its answer class.

Two variants:
- MS1Dataset: on-the-fly generation per __getitem__ (fresh states each epoch).
- PrecomputedMS1Dataset: pre-generates everything to padded tensors at __init__.
  Bit-identical data (same seeds), but __getitem__ is pure indexing → much faster
  and trivially parallelizable via DataLoader num_workers (picklable tensors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from msl.data.ms1 import MS1, TextRenderer, sample_balanced_tasks
from msl.models.codec import answer_to_cls
from msl.models.tokenizer import Tokenizer


@dataclass
class MS1Config:
    min_k: int = 2
    max_k: int = 32
    tasks_per_state: int = 1
    n_views: int = 2          # views rendered per state (view A and B for alignment)


class MS1Dataset(Dataset):
    def __init__(self, tok: Tokenizer, cfg: MS1Config, size: int, seed_floor: int = 0) -> None:
        self.tok = tok
        self.cfg = cfg
        self.size = size
        self.seed_floor = seed_floor
        self.gen = MS1(min_k=cfg.min_k, max_k=cfg.max_k)
        self.renderer = TextRenderer()

    def __len__(self) -> int:
        return self.size

    def _state_seed(self, idx: int, epoch: int) -> int:
        # Deterministic, non-overlapping with test range.
        return self.seed_floor + (idx * 7919 + epoch * 104729) % 1_000_000

    def _build_sample(self, seed: int) -> dict:
        k = int(np.random.default_rng(seed).integers(self.cfg.min_k, self.cfg.max_k + 1))
        s = self.gen.generate(seed=seed, k=k)
        view_a = self.renderer.render(s, int(np.random.default_rng(seed).integers(0, 8)))
        view_b = self.renderer.render(s, int(np.random.default_rng(seed + 7).integers(0, 8)))
        tks = sample_balanced_tasks(s, np.random.default_rng(seed + 3), self.cfg.tasks_per_state)
        t = tks[0]
        return {
            "text": view_a,
            "text_b": view_b,
            "task_text": f"{t.kind} {t.payload}",
            "task_cls": answer_to_cls(t.answer),
            "k": k,
        }

    def __getitem__(self, idx: int, epoch: int = 0) -> dict:
        seed = self._state_seed(idx, epoch)
        return self._build_sample(seed)


class PrecomputedMS1Dataset(Dataset):
    """Pre-generates all samples to padded tensors at __init__.

    Bit-identical to MS1Dataset (same seed derivation), but __getitem__ is pure
    indexing into pre-padded tensors. Much faster in training loops and trivially
    parallelizable via DataLoader(num_workers>0) since the cached tensors are picklable.
    """

    def __init__(self, tok: Tokenizer, cfg: MS1Config, size: int, seed_floor: int = 0,
                 max_len: int = 96, cross_lang: bool = False) -> None:
        """cross_lang: if True, recon_ids = other language (translation). If False, same language."""
        self.tok = tok
        self.cfg = cfg
        self.size = size
        self.seed_floor = seed_floor
        self.max_len = max_len
        self.cross_lang = cross_lang
        gen = MS1(min_k=cfg.min_k, max_k=cfg.max_k)
        renderer = TextRenderer()
        # Pre-allocate padded tensors.
        pad = tok.pad_id
        self.text_ids = torch.full((size, max_len), pad, dtype=torch.long)
        self.text_mask = torch.ones((size, max_len), dtype=torch.long)
        self.text_ids_b = torch.full((size, max_len), pad, dtype=torch.long)
        self.text_mask_b = torch.ones((size, max_len), dtype=torch.long)
        self.task_ids = torch.full((size, max_len), pad, dtype=torch.long)
        self.task_mask = torch.ones((size, max_len), dtype=torch.long)
        self.task_cls = torch.zeros(size, dtype=torch.long)
        self.task_kind_id = torch.zeros(size, dtype=torch.long)  # 0..4, no state content
        self.recon_lang_id = torch.zeros(size, dtype=torch.long)  # 0=fr, 1=en, 2=structured
        self.k = torch.zeros(size, dtype=torch.long)
        KIND_IDS = {"qa": 0, "implication": 1, "contradiction": 2, "temps": 3, "composition": 4}
        for idx in range(size):
            seed = self.seed_floor + (idx * 7919) % 1_000_000
            k = int(np.random.default_rng(seed).integers(cfg.min_k, cfg.max_k + 1))
            s = gen.generate(seed=seed, k=k)
            # View A = encoder input (random language), View B = reconstruction target (other language).
            rng_a = np.random.default_rng(seed)
            a_is_fr = rng_a.random() < 0.5
            if a_is_fr:
                view_a = renderer.render(s, int(rng_a.integers(0, 4)))    # FR views 0-3
                view_b = renderer.render(s, int(rng_a.integers(4, 8)))    # EN views 4-7
                recon_lang = 1  # EN
            else:
                view_a = renderer.render(s, int(rng_a.integers(4, 8)))   # EN views 4-7
                view_b = renderer.render(s, int(rng_a.integers(0, 4)))    # FR views 0-3
                recon_lang = 0  # FR
            tks = sample_balanced_tasks(s, np.random.default_rng(seed + 3), cfg.tasks_per_state)
            t = tks[0]
            self._write(self.text_ids, self.text_mask, idx, tok.encode(view_a, True, True))
            self._write(self.text_ids_b, self.text_mask_b, idx, tok.encode(view_b, True, True))
            self._write(self.task_ids, self.task_mask, idx, tok.encode(f"{t.kind} {t.payload}", True, True))
            self.task_cls[idx] = answer_to_cls(t.answer)
            self.task_kind_id[idx] = KIND_IDS.get(t.kind, 0)
            self.recon_lang_id[idx] = recon_lang
            self.k[idx] = k

    @staticmethod
    def _write(ids: torch.Tensor, mask: torch.Tensor, idx: int, seq: list[int]) -> None:
        n = min(len(seq), ids.size(1))
        ids[idx, :n] = torch.tensor(seq[:n], dtype=torch.long)
        mask[idx, :n] = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        if self.cross_lang:
            recon_ids = self.text_ids_b[idx]   # reconstruct in OTHER language
        else:
            recon_ids = self.text_ids[idx]     # reconstruct in SAME language
            # In same-lang mode, recon_lang_id should match the input language.
            # We stored the target lang; for same-lang, use the input lang instead.
        return {
            "text_ids": self.text_ids[idx],
            "text_mask": self.text_mask[idx],
            "text_ids_b": self.text_ids_b[idx],
            "text_mask_b": self.text_mask_b[idx],
            "task_ids": self.task_ids[idx],
            "task_mask": self.task_mask[idx],
            "task_cls": self.task_cls[idx],
            "task_kind_id": self.task_kind_id[idx],
            "recon_ids": recon_ids,
            "recon_lang_id": self.recon_lang_id[idx] if self.cross_lang else (1 - self.recon_lang_id[idx]),
            "k": self.k[idx],
        }


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    m = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    mask = torch.ones((len(seqs), m))
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 0
    return ids, mask


def make_collate(tok: Tokenizer):
    def collate(batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        texts_b = [b["text_b"] for b in batch]
        tasks = [b["task_text"] for b in batch]
        text_ids, text_mask = _pad([tok.encode(t, True, True) for t in texts], tok.pad_id)
        text_ids_b, text_mask_b = _pad([tok.encode(t, True, True) for t in texts_b], tok.pad_id)
        task_ids, task_mask = _pad([tok.encode(t, True, True) for t in tasks], tok.pad_id)
        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "text_ids_b": text_ids_b,
            "text_mask_b": text_mask_b,
            "task_ids": task_ids,
            "task_mask": task_mask,
            "task_cls": torch.tensor([b["task_cls"] for b in batch], dtype=torch.long),
            "recon_ids": text_ids,
            "k": torch.tensor([b["k"] for b in batch], dtype=torch.long),
        }

    return collate


def default_collate(batch: list[dict]) -> dict:
    """Collate for PrecomputedMS1Dataset: stacks pre-padded tensors along batch dim."""
    return {
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "text_ids_b": torch.stack([b["text_ids_b"] for b in batch]),
        "text_mask_b": torch.stack([b["text_mask_b"] for b in batch]),
        "task_ids": torch.stack([b["task_ids"] for b in batch]),
        "task_mask": torch.stack([b["task_mask"] for b in batch]),
        "task_cls": torch.stack([b["task_cls"] for b in batch]),
        "task_kind_id": torch.stack([b["task_kind_id"] for b in batch]),
        "recon_ids": torch.stack([b["recon_ids"] for b in batch]),
        "recon_lang_id": torch.stack([b["recon_lang_id"] for b in batch]),
        "k": torch.stack([b["k"] for b in batch]),
    }


__all__ = ["MS1Config", "MS1Dataset", "PrecomputedMS1Dataset", "make_collate", "default_collate"]
