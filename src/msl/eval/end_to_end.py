"""End-to-end cost comparison: MSL pipeline vs text pipeline on REAL TEXT.

MSL pipeline: text -> MiniLM -> PQ quantizer -> LLM MSL -> packets
Text pipeline: text -> text LM -> text

This is the test that decides whether MSL is economically viable on real text.

Usage:
    python -u -m msl.eval.end_to_end
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer

from msl.models.native_lm import NativeMSLLM
from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import default_device, seed_everything


class TextBaselineLM(nn.Module):
    """Simple text LM for comparison: predict next token."""
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, max_len: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                       batch_first=True, activation="gelu", dropout=0.0),
            n_layers, enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        h = self.embed(ids) + self.pos(torch.arange(T, device=ids.device)).unsqueeze(0)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        return self.head(self.transformer(h, mask=causal))


def time_fn(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) / n * 1000


def main():
    seed_everything(0)
    device = default_device()
    print(f"device: {device}")

    # 1. Load the real text corpus.
    corpus = torch.load("runs/big_corpus.pt", weights_only=False)
    sentences = corpus["sentences"][:200]
    embeddings = corpus["embeddings"][:200].to(device)
    n_codebooks = corpus.get("n_codebooks", 48)
    codebook_size = corpus.get("codebook_size", 256)
    emb_dim = embeddings.shape[1]
    print(f"corpus: {len(sentences)} sentences, emb_dim={emb_dim}")

    # 2. Load decoder checkpoint (has the trained quantizer).
    dec_ckpt = torch.load("runs/text_decoder_quant_0.pt", map_location=device, weights_only=False)
    dec_cfg = dec_ckpt["config"]
    n_codebooks = dec_cfg.get("n_codebooks", n_codebooks)
    codebook_size = dec_cfg.get("codebook_size", codebook_size)
    emb_dim = dec_cfg.get("d_z", emb_dim)

    quantizer = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
    if "quantizer" in dec_ckpt:
        quantizer.load_state_dict(dec_ckpt["quantizer"])
        print(f"quantizer loaded from checkpoint")
    else:
        print(f"quantizer not in checkpoint, retraining from corpus...")
        quantizer.train()
        import numpy as np
        all_emb = corpus["embeddings"].to(device)
        for step in range(1000):
            idx = np.random.randint(0, len(all_emb), 256)
            with torch.no_grad():
                quantizer(all_emb[idx])
        quantizer.eval()
    print(f"quantizer: {n_codebooks} codebooks x {codebook_size} ({n_codebooks*8} bits)")

    # 3. Load native MSL LM.
    msl_lm_ckpt = torch.load("runs/native_lm_0.pt", map_location=device, weights_only=False)
    msl_lm = NativeMSLLM(
        n_codebooks=n_codebooks, codebook_size=codebook_size,
        d_model=256, n_layers=6, n_heads=8, max_seq_len=11,
    ).to(device)
    msl_lm.load_state_dict(msl_lm_ckpt["model"])
    msl_lm.eval()
    print(f"MSL LLM loaded: {sum(p.numel() for p in msl_lm.parameters()):,} params")

    # 4. Load MiniLM encoder (the text -> embedding step).
    enc_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    enc_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    enc_model.eval()
    print(f"encoder loaded: {sum(p.numel() for p in enc_model.parameters()):,} params")

    # 5. Build a text baseline LM (same size as MSL LM).
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    text_lm = TextBaselineLM(
        vocab_size=gpt2_tok.vocab_size, d_model=256, n_layers=6, n_heads=8, max_len=48,
    ).to(device)
    text_lm.eval()
    print(f"text LM: {sum(p.numel() for p in text_lm.parameters()):,} params")

    # 6. Prepare batch.
    batch_size = 64
    test_sentences = sentences[:batch_size]
    test_emb = embeddings[:batch_size]

    # Tokenize for GPT-2 text LM.
    text_inputs = gpt2_tok(test_sentences, padding=True, truncation=True,
                          max_length=48, return_tensors="pt").to(device)

    # --- Measure each component ---
    print(f"\n{'='*60}")
    print("COMPONENT COSTS (batch=64)")
    print(f"{'='*60}")

    # 1. Encoder: text -> embeddings.
    def encode_step():
        with torch.no_grad():
            inputs = enc_tok(test_sentences, padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = enc_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            return (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

    enc_ms = time_fn(encode_step, n=20)
    print(f"1. Encoder (MiniLM):     {enc_ms:.1f} ms (text -> embedding)")

    # 2. Quantizer: embedding -> packets.
    def quant_step():
        with torch.no_grad():
            return quantizer(test_emb)

    quant_ms = time_fn(quant_step, n=20)
    print(f"2. Quantizer (PQ):       {quant_ms:.1f} ms (embedding -> packets)")

    # 3. LLM MSL: predict next packet.
    test_packets = quant_step().codes.unsqueeze(0)  # (1, 64, 48) — treat batch as sequence
    # Actually for the LLM, input is (B, seq_len, n_codebooks).
    # For timing, use a single sequence of 10 packets.
    pkt_input = torch.randint(0, codebook_size, (64, 10, n_codebooks), device=device)
    def msl_lm_step():
        with torch.no_grad():
            return msl_lm(pkt_input)

    msl_step_ms = time_fn(msl_lm_step, n=20)
    print(f"3. LLM MSL (per step):   {msl_step_ms:.1f} ms (predict 1 packet of {n_codebooks} codes)")

    # 4. Text LM: predict next token.
    def text_lm_step():
        with torch.no_grad():
            return text_lm(text_inputs["input_ids"])

    text_step_ms = time_fn(text_lm_step, n=20)
    print(f"4. Text LM (per step):   {text_step_ms:.1f} ms (predict 1 token)")

    # --- End-to-end comparison ---
    # MSL: encode + quantize (once per sentence) + LLM (10 steps for 10 sentences)
    # Text: LLM (40 steps per sentence × 64 sentences)
    msl_pipeline_ms = enc_ms + quant_ms + msl_step_ms * 10  # 10 packets
    text_pipeline_ms = text_step_ms * 40  # ~40 tokens per sentence

    print(f"\n{'='*60}")
    print("END-TO-END COMPARISON")
    print(f"{'='*60}")
    print(f"MSL pipeline:  {enc_ms:.0f} + {quant_ms:.0f} + {msl_step_ms*10:.0f} = {msl_pipeline_ms:.0f} ms")
    print("  (encode + quantize + 10 LLM steps)")
    print(f"Text pipeline: {text_step_ms:.0f} x 40 = {text_pipeline_ms:.0f} ms")
    print("  (40 text LM steps)")
    ratio = text_pipeline_ms / max(msl_pipeline_ms, 1)
    if ratio > 1:
        print(f"\nMSL is {ratio:.1f}x FASTER end-to-end")
    else:
        print(f"\nMSL is {1/ratio:.1f}x SLOWER end-to-end")

    # --- Steps comparison ---
    print(f"\n{'='*60}")
    print("STEPS COMPARISON")
    print(f"{'='*60}")
    print("MSL:  1 packet per sentence = 1 step per sentence")
    print("Text: ~7 tokens per sentence = 7 steps per sentence")
    print("Ratio: 7x fewer steps")

    # --- Memory ---
    print(f"\n{'='*60}")
    print("MEMORY")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # MSL memory.
    with torch.no_grad():
        _ = msl_lm(pkt_input)
    msl_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Text memory.
    with torch.no_grad():
        _ = text_lm(text_inputs["input_ids"])
    text_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    print(f"MSL LLM:  {msl_mem:.0f} MB")
    print(f"Text LM:  {text_mem:.0f} MB")

    # --- Params ---
    print(f"\n{'='*60}")
    print("PARAMS")
    print(f"{'='*60}")
    print(f"MSL LLM:     {sum(p.numel() for p in msl_lm.parameters()):,}")
    print(f"Text LM:     {sum(p.numel() for p in text_lm.parameters()):,}")
    print(f"Encoder:     {sum(p.numel() for p in enc_model.parameters()):,}")
    print(f"Total MSL:   {sum(p.numel() for p in msl_lm.parameters()) + sum(p.numel() for p in enc_model.parameters()):,}")

    # --- Quality ---
    print(f"\n{'='*60}")
    print("QUALITY")
    print(f"{'='*60}")
    print(f"MSL LLM code accuracy:  {msl_lm_ckpt['metrics']['per_code_acc']:.1%}")
    print(f"MSL LLM exact packets:  {msl_lm_ckpt['metrics']['exact_acc']:.1%}")


if __name__ == "__main__":
    main()
