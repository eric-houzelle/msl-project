"""End-to-end cost comparison: MSL pipeline vs text pipeline.

MSL pipeline: text -> encoder -> packets -> LLM MSL -> packets -> decoder -> text
Text pipeline: text -> LLM text -> text

This is the test that decides whether MSL is economically viable: does the
LLM speedup survive the codec translation cost?

Usage:
    python -u -m msl.eval.end_to_end
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.native_lm import NativeMSLLM
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec import build_codec
from msl.train.train_text_lm import TextLM
from msl.utils.seeding import default_device, seed_everything


def time_fn(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.empty(1, device=next(iter(fn.__self__.parameters() if hasattr(fn, '__self__') else [])).device
                if hasattr(fn, '__self__') else "cpu")
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


def main() -> None:
    seed_everything(0)
    device = default_device()
    print(f"device: {device}")
    tok = default_tokenizer()

    # Load all models
    codec_ckpt = torch.load("runs/phase2_task_ext_0.pt", map_location=device, weights_only=False)
    codec = build_codec(codec_ckpt["cfg"], len(tok.itos)).to(device)
    codec.load_state_dict(codec_ckpt["codec"])
    codec.eval()

    msl_lm_ckpt = torch.load("runs/native_lm_0.pt", map_location=device, weights_only=False)
    msl_lm = NativeMSLLM(n_codebooks=8, codebook_size=256, d_model=256, n_layers=6,
                         n_heads=8, max_seq_len=17).to(device)
    msl_lm.load_state_dict(msl_lm_ckpt["model"])
    msl_lm.eval()

    text_lm_ckpt = torch.load("runs/text_lm_0.pt", map_location=device, weights_only=False)
    text_lm = TextLM(vocab_size=len(tok.itos), d_model=256, n_layers=6, n_heads=8,
                     max_len=96).to(device)
    text_lm.load_state_dict(text_lm_ckpt["model"])
    text_lm.eval()

    # Prepare a batch
    ms1_cfg = MS1Config(min_k=2, max_k=32, tasks_per_state=1, n_views=2)
    ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=1024, seed_floor=0)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=default_collate)
    batch = next(iter(loader))
    text_ids = batch["text_ids"].to(device)
    text_mask = batch["text_mask"].to(device)
    B = text_ids.shape[0]

    print(f"\nbatch size: {B}")
    print(f"text sequence length: {text_ids.shape[1]}")

    # --- Measure each component ---
    print("\n=== COMPONENT COSTS (batch=64) ===")

    # 1. Encoder + quantizer: text -> packets
    def encode_step():
        with torch.no_grad():
            codes, z_q, _, _ = codec.encode_quantize(text_ids, text_mask)
            return codes

    encode_ms = time_fn(encode_step)
    print(f"1. Encoder+Quantizer: {encode_ms:.1f} ms (text -> 16 packets)")

    # 2. LLM MSL: generate 16 packets autoregressively
    # In practice: start from context packets, predict next ones
    packets = encode_step()  # (B, 16, 8)
    def msl_lm_step():
        with torch.no_grad():
            return msl_lm(packets)

    msl_step_ms = time_fn(msl_lm_step)
    msl_total_ms = 16 * msl_step_ms  # 16 autoregressive steps
    print(f"2. LLM MSL: {msl_step_ms:.1f} ms/step x 16 steps = {msl_total_ms:.0f} ms")

    # 3. Decoder: packets -> text (reconstruct)
    def decode_step():
        with torch.no_grad():
            _, z_q_q, _, _ = codec.encode_quantize(text_ids, text_mask)
            return codec.decoder.reconstruct(z_q_q, text_ids)

    decode_ms = time_fn(decode_step)
    print(f"3. Decoder: {decode_ms:.1f} ms (packets -> text)")

    # 4. Text LM: generate 40 tokens autoregressively
    text_step_ms = time_fn(lambda: text_lm(text_ids))
    text_total_ms = 40 * text_step_ms  # 40 autoregressive steps
    print(f"4. Text LM: {text_step_ms:.1f} ms/step x 40 steps = {text_total_ms:.0f} ms")

    # --- End-to-end totals ---
    msl_e2e = encode_ms + msl_total_ms + decode_ms
    text_e2e = text_total_ms
    print("\n=== END-TO-END TOTALS ===")
    print(f"MSL pipeline:  {encode_ms:.0f} + {msl_total_ms:.0f} + {decode_ms:.0f} = {msl_e2e:.0f} ms")
    print(f"Text pipeline: {text_e2e:.0f} ms")
    ratio = text_e2e / msl_e2e
    if ratio > 1:
        print(f"MSL is {ratio:.1f}x FASTER end-to-end")
    else:
        print(f"MSL is {1/ratio:.1f}x SLOWER end-to-end")

    # --- Params ---
    codec_params = sum(p.numel() for p in codec.parameters())
    msl_lm_params = sum(p.numel() for p in msl_lm.parameters())
    text_lm_params = sum(p.numel() for p in text_lm.parameters())
    msl_total_params = codec_params + msl_lm_params
    print("\n=== TOTAL PARAMS ===")
    print(f"MSL: codec {codec_params:,} + LLM {msl_lm_params:,} = {msl_total_params:,}")
    print(f"Text: LLM {text_lm_params:,}")

    # --- Breakdown ---
    print("\n=== MSL COST BREAKDOWN ===")
    print(f"Encoder:  {encode_ms/msl_e2e*100:.0f}%")
    print(f"LLM:      {msl_total_ms/msl_e2e*100:.0f}%")
    print(f"Decoder:  {decode_ms/msl_e2e*100:.0f}%")


if __name__ == "__main__":
    main()
