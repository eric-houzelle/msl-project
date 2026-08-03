"""End-to-end codec tests: forward, backward, and overfit on a tiny batch (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from msl.data.ms1 import MS1, TextRenderer, sample_tasks
from msl.models.codec import Codec, CodecConfig, answer_to_cls
from msl.models.tokenizer import Tokenizer


def _build_batch(tokenizer, states, tasks_per_state=2, seed=0):
    """Build a minimal batch from MS-1 states for a smoke test."""
    r = TextRenderer()
    rng = np.random.default_rng(seed)
    text_ids_list, task_ids_list, task_cls_list, recon_ids_list = [], [], [], []
    for s in states:
        tks = sample_tasks(s, np.random.default_rng(s.seed), tasks_per_state)
        view = r.render(s, int(rng.integers(0, 8)))
        ids = tokenizer.encode(view, add_bos=True, add_eos=True)
        text_ids_list.append(ids)
        t = tks[0]
        q_text = f"{t.kind} {t.payload}"
        q_ids = tokenizer.encode(q_text, add_bos=True, add_eos=True)
        task_ids_list.append(q_ids)
        task_cls_list.append(answer_to_cls(t.answer))
        recon_ids_list.append(ids)
    # Pad to common length.
    def pad(seqs):
        m = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), m), tokenizer.pad_id, dtype=torch.long)
        mask = torch.ones((len(seqs), m))
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 0
        return ids, mask
    text_ids, text_mask = pad(text_ids_list)
    task_ids, task_mask = pad(task_ids_list)
    recon_ids, _ = pad(recon_ids_list)
    return {
        "text_ids": text_ids, "text_mask": text_mask,
        "task_ids": task_ids, "task_mask": task_mask,
        "task_cls": torch.tensor(task_cls_list, dtype=torch.long),
        "recon_ids": recon_ids,
    }


def _tiny_codec(vocab_size, kind="fsq"):
    cfg = CodecConfig(
        vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4, d_z=16,
        n_slots=2, quantizer_kind=kind, n_codebooks=4, codebook_size=64, levels=5,
    )
    return Codec(cfg)


def test_codec_forward_shapes():
    tok = Tokenizer().build(n_states=200)
    codec = _tiny_codec(len(tok.itos))
    states = [MS1().generate(seed=10 + i, k=8) for i in range(4)]
    batch = _build_batch(tok, states)
    out = codec(batch)
    assert out["loss"].dim() == 0
    assert out["codes"].shape[0] == 4
    assert out["codes"].shape[1] == 2  # n_slots


def test_codec_backward_runs():
    tok = Tokenizer().build(n_states=200)
    codec = _tiny_codec(len(tok.itos))
    states = [MS1().generate(seed=10 + i, k=8) for i in range(4)]
    batch = _build_batch(tok, states)
    out = codec(batch)
    out["loss"].backward()
    # Encoder has learnable params; gradients must flow.
    enc_params = [p for p in codec.encoder.parameters() if p.requires_grad]
    assert any(p.grad is not None for p in enc_params)
    assert all(torch.isfinite(p.grad).all() for p in enc_params if p.grad is not None)


def test_codec_overfits_a_single_state():
    """A codec with enough capacity should drive task loss toward zero on one state."""
    torch.manual_seed(0)
    tok = Tokenizer().build(n_states=200)
    codec = _tiny_codec(len(tok.itos), kind="fsq")
    codec.train()
    states = [MS1().generate(seed=11, k=4)]
    batch = _build_batch(tok, states, seed=0)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-3)
    initial_loss = codec(batch)["loss"].item()
    for _ in range(200):
        opt.zero_grad()
        out = codec(batch)
        out["loss"].backward()
        opt.step()
    final_loss = codec(batch)["loss"].item()
    assert final_loss < initial_loss
    assert final_loss < 1.0
