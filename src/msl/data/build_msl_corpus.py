"""Precompute a corpus of MSL packets from MS-1 states, using a frozen codec.

This is the "translation" step: we take textual states, run them through the
frozen encoder+quantizer, and store the resulting packets. The native LLM will
then train ONLY on these packets — it never sees text.

Usage:
    python -u -m msl.data.build_msl_corpus --codec runs/mvp_n16_0.pt --size 16384 --out runs/msl_corpus.pt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec import build_codec
from msl.utils.seeding import default_device, seed_everything


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="runs/mvp_n16_0.pt")
    ap.add_argument("--size", type=int, default=16384)
    ap.add_argument("--out", default="runs/msl_corpus.pt")
    args = ap.parse_args()

    seed_everything(0)
    device = default_device()
    tok = default_tokenizer()

    ckpt = torch.load(args.codec, map_location=device, weights_only=False)
    codec = build_codec(ckpt["cfg"], len(tok.itos)).to(device)
    codec.load_state_dict(ckpt["codec"])
    codec.eval()
    n_slots = ckpt["cfg"]["codec"]["n_slots"]
    n_codebooks = ckpt["cfg"]["codec"]["n_codebooks"]
    print(f"frozen codec: n_slots={n_slots} n_codebooks={n_codebooks} ({ckpt['n_params']:,} params)")

    ms1_cfg = MS1Config(min_k=2, max_k=32, tasks_per_state=1, n_views=2)
    ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=args.size, seed_floor=0)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=default_collate, num_workers=0)

    all_codes = []
    all_k = []
    for batch in loader:
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        codes, _, _, _ = codec.encode_quantize(text_ids, text_mask)  # (B, n_slots, n_codebooks)
        all_codes.append(codes.cpu())
        all_k.append(batch["k"])

    corpus_codes = torch.cat(all_codes, dim=0)  # (N, n_slots, n_codebooks)
    corpus_k = torch.cat(all_k, dim=0)          # (N,)
    print(f"corpus: {corpus_codes.shape} packets, {corpus_k.shape} difficulties")
    print(f"sample packet[0]: {corpus_codes[0].tolist()}")

    torch.save({
        "codes": corpus_codes,
        "k": corpus_k,
        "n_slots": n_slots,
        "n_codebooks": n_codebooks,
        "codec_cfg": ckpt["cfg"]["codec"],
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
