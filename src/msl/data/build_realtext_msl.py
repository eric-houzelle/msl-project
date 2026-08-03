"""Build MSL corpus for native LLM from real text corpus.

Takes the real text corpus (sentences + embeddings) and:
1. Re-trains a quantizer on the embeddings
2. Quantizes each sentence to a packet (48 codes)
3. Groups sentences into sequences (10 sentences per sequence)
4. Saves in the format expected by train_native_lm.py

The LLM will learn to predict the next sentence's packet from previous
sentences' packets — thinking in MSL at the sentence level.

Usage:
    python -u -m msl.data.build_realtext_msl --corpus runs/big_corpus.pt --out runs/msl_corpus_real.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import default_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/big_corpus.pt")
    ap.add_argument("--out", default="runs/msl_corpus_real.pt")
    ap.add_argument("--seq_len", type=int, default=10, help="sentences per sequence")
    ap.add_argument("--quantizer_steps", type=int, default=2000)
    args = ap.parse_args()

    seed_everything(0)
    device = default_device()
    print(f"device: {device}")

    # 1. Load corpus.
    corpus = torch.load(args.corpus, weights_only=False)
    sentences = corpus["sentences"]
    embeddings = corpus["embeddings"].to(device)
    n_codebooks = corpus.get("n_codebooks", 48)
    codebook_size = corpus.get("codebook_size", 256)
    emb_dim = embeddings.shape[1]
    print(f"corpus: {len(sentences)} sentences, emb_dim={emb_dim}")
    print(f"quantizer: {n_codebooks} codebooks x {codebook_size} = {n_codebooks*8} bits")

    # 2. Train quantizer.
    print(f"training quantizer ({args.quantizer_steps} steps)...")
    quantizer = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
    quantizer.train()
    for step in range(args.quantizer_steps):
        idx = np.random.randint(0, len(embeddings), 256)
        with torch.no_grad():
            quantizer(embeddings[idx])
        if step % 500 == 0:
            print(f"  step {step}", flush=True)
    quantizer.eval()

    # 3. Quantize all sentences.
    print("quantizing all sentences...")
    packets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(embeddings), 512):
            batch = embeddings[i:i+512]
            out = quantizer(batch)
            packets_list.append(out.codes.cpu())
    all_packets = torch.cat(packets_list, dim=0)  # (N, n_codebooks)
    print(f"packets: {all_packets.shape}")
    print(f"unique packets: {all_packets.unique(dim=0).shape[0]}")

    # 4. Group into sequences for the LLM.
    # Each sequence = seq_len consecutive sentences' packets.
    seq_len = args.seq_len
    n_sequences = len(all_packets) // seq_len
    sequences = all_packets[:n_sequences * seq_len].reshape(n_sequences, seq_len, n_codebooks)
    print(f"sequences: {sequences.shape} ({n_sequences} sequences of {seq_len} packets)")

    # 5. Save in format expected by train_native_lm.py.
    torch.save({
        "codes": sequences,          # (N_seq, seq_len, n_codebooks)
        "n_slots": seq_len,          # sequence length
        "n_codebooks": n_codebooks,
        "codebook_size": codebook_size,
        "codec_cfg": {"codebook_size": codebook_size},
        "sentences": sentences[:n_sequences * seq_len],
        "emb_dim": emb_dim,
    }, args.out)
    print(f"saved {args.out}")
    print(f"  {n_sequences} sequences x {seq_len} packets x {n_codebooks} codes")
    print(f"  {n_sequences * seq_len * n_codebooks * 8} bits total")


if __name__ == "__main__":
    main()
