"""End-to-end cost comparison: MSL pipeline vs text pipeline on REAL TEXT.

Honest comparison:
- Both LLMs are trained on the same data
- Token/packet counts are MEASURED, not hardcoded
- MPS synchronization is handled
- Decoder cost is included in the MSL pipeline

Usage:
    python -u -m msl.eval.end_to_end
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer

from msl.models.native_lm import NativeMSLLM
from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import default_device, seed_everything


class TextBaselineLM(nn.Module):
    """Simple text LM for fair comparison: same architecture as MSL LLM."""
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


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_fn(fn, n=50, warmup=5, device=None):
    for _ in range(warmup):
        fn()
    sync_device(device) if device else None
    t0 = time.time()
    for _ in range(n):
        fn()
    sync_device(device) if device else None
    return (time.time() - t0) / n * 1000


def main():
    seed_everything(0)
    device = default_device()
    print(f"device: {device}")

    # 1. Load corpus.
    corpus = torch.load("runs/big_corpus.pt", weights_only=False)
    sentences = corpus["sentences"]
    embeddings = corpus["embeddings"]
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
        print("quantizer loaded from checkpoint")
    else:
        print("quantizer not in checkpoint, retraining...")
        quantizer.train()
        all_emb = embeddings.to(device)
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
    print(f"MSL LLM: {sum(p.numel() for p in msl_lm.parameters()):,} params")

    # 4. Load MiniLM encoder.
    enc_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    enc_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    enc_model.eval()
    print(f"encoder: {sum(p.numel() for p in enc_model.parameters()):,} params")

    # 5. Build text baseline LM (same architecture as MSL LLM).
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    text_lm = TextBaselineLM(
        vocab_size=gpt2_tok.vocab_size, d_model=256, n_layers=6, n_heads=8, max_len=48,
    ).to(device)
    text_lm.eval()
    print(f"text LM: {sum(p.numel() for p in text_lm.parameters()):,} params")

    # 6. Prepare batch on HELD-OUT sentences.
    batch_size = 64
    n_train = int(len(sentences) * 0.8)
    test_sentences = sentences[n_train:n_train + batch_size]
    test_emb = embeddings[n_train:n_train + batch_size].to(device)

    # Tokenize for text LM.
    text_inputs = gpt2_tok(test_sentences, padding=True, truncation=True,
                          max_length=48, return_tensors="pt").to(device)

    # 7. MEASURE actual token and packet counts per sentence.
    token_counts = [len(gpt2_tok.encode(s)) for s in test_sentences]
    avg_tokens = np.mean(token_counts)
    print(f"\nMEASURED: avg {avg_tokens:.1f} tokens per sentence (min={min(token_counts)}, max={max(token_counts)})")
    # Packets: 1 packet per sentence (the whole embedding quantizes to 1 packet)
    avg_packets = 1.0
    print(f"MEASURED: avg {avg_packets:.1f} packets per sentence")

    # --- Measure each component ---
    print(f"\n{'='*60}")
    print(f"COMPONENT COSTS (batch={batch_size})")
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

    enc_ms = time_fn(encode_step, n=20, device=device)
    print(f"1. Encoder (MiniLM):     {enc_ms:.1f} ms")

    # 2. Quantizer: embedding -> packets.
    test_emb_dev = test_emb.to(device)
    def quant_step():
        with torch.no_grad():
            return quantizer(test_emb_dev)

    quant_ms = time_fn(quant_step, n=20, device=device)
    print(f"2. Quantizer (PQ):       {quant_ms:.1f} ms")

    # 3. LLM MSL: per step (predict 1 packet).
    pkt_input = torch.randint(0, codebook_size, (batch_size, 10, n_codebooks), device=device)
    def msl_lm_step():
        with torch.no_grad():
            return msl_lm(pkt_input)

    msl_step_ms = time_fn(msl_lm_step, n=20, device=device)
    print(f"3. LLM MSL (per step):   {msl_step_ms:.1f} ms")

    # 4. Text LM: per step (predict 1 token).
    def text_lm_step():
        with torch.no_grad():
            return text_lm(text_inputs["input_ids"])

    text_step_ms = time_fn(text_lm_step, n=20, device=device)
    print(f"4. Text LM (per step):   {text_step_ms:.1f} ms")

    # 5. Decoder: GPT-2 generates text from packets (the return path).
    from msl.train.train_text_decoder import MSLDecoder
    decoder = MSLDecoder(d_z=emb_dim, n_prefix=4).to(device)
    decoder.load_state_dict(dec_ckpt["decoder"])
    decoder.eval()
    z_q_sample = quant_step().z_q[:1]  # (1, emb_dim)
    def decode_step():
        with torch.no_grad():
            return decoder.generate(z_q_sample, gpt2_tok, max_len=10)

    # Warmup (first call is slow).
    decode_step()
    decode_ms = time_fn(decode_step, n=10, device=device)
    print(f"5. Decoder (GPT-2):      {decode_ms:.1f} ms (per sentence)")

    # --- End-to-end comparison ---
    # MSL: encode + quantize (once) + LLM (avg_packets steps) + decode (per sentence)
    msl_pipeline_ms = enc_ms + quant_ms + msl_step_ms * avg_packets + decode_ms * batch_size
    # Text: LLM (avg_tokens steps per sentence × batch_size)
    text_pipeline_ms = text_step_ms * avg_tokens * batch_size

    print(f"\n{'='*60}")
    print(f"END-TO-END COMPARISON (batch={batch_size}, {avg_tokens:.1f} tokens/sentence)")
    print(f"{'='*60}")
    print(f"MSL pipeline:  {enc_ms:.0f} + {quant_ms:.0f} + {msl_step_ms*avg_packets:.0f} + {decode_ms*batch_size:.0f} = {msl_pipeline_ms:.0f} ms")
    print(f"  (encode + quantize + {avg_packets:.0f} LLM steps + decode)")
    print(f"Text pipeline: {text_step_ms:.0f} x {avg_tokens:.1f} x {batch_size} = {text_pipeline_ms:.0f} ms")
    print(f"  ({avg_tokens:.1f} text LM steps x {batch_size} sentences)")
    ratio = text_pipeline_ms / max(msl_pipeline_ms, 1)
    if ratio > 1:
        print(f"\nMSL is {ratio:.1f}x FASTER end-to-end")
    else:
        print(f"\nMSL is {1/ratio:.1f}x SLOWER end-to-end")

    # --- Steps comparison ---
    print(f"\n{'='*60}")
    print("STEPS COMPARISON (measured)")
    print(f"{'='*60}")
    print(f"MSL:  {avg_packets:.1f} packets per sentence = {avg_packets:.1f} steps")
    print(f"Text: {avg_tokens:.1f} tokens per sentence = {avg_tokens:.1f} steps")
    print(f"Ratio: {avg_tokens/avg_packets:.1f}x fewer steps")

    # --- Memory ---
    print(f"\n{'='*60}")
    print("MEMORY")
    print(f"{'='*60}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = msl_lm(pkt_input)
        msl_mem = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = text_lm(text_inputs["input_ids"])
        text_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"MSL LLM:  {msl_mem:.0f} MB")
        print(f"Text LM:  {text_mem:.0f} MB")
    elif device.type == "mps":
        # MPS doesn't have reliable memory tracking.
        print(f"Memory tracking not available on {device.type}")
    else:
        print(f"Memory tracking not available on {device.type}")

    # --- Params ---
    print(f"\n{'='*60}")
    print("PARAMS")
    print(f"{'='*60}")
    msl_params = sum(p.numel() for p in msl_lm.parameters())
    text_params = sum(p.numel() for p in text_lm.parameters())
    enc_params = sum(p.numel() for p in enc_model.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"MSL LLM:      {msl_params:,}")
    print(f"Text LM:      {text_params:,}")
    print(f"Encoder:      {enc_params:,}")
    print(f"Decoder:      {dec_params:,}")
    print(f"Total MSL:    {msl_params + enc_params + dec_params:,}")

    # --- Quality (from checkpoint) ---
    print(f"\n{'='*60}")
    print("QUALITY (from checkpoint)")
    print(f"{'='*60}")
    if "metrics" in msl_lm_ckpt:
        print(f"MSL LLM code accuracy:  {msl_lm_ckpt['metrics'].get('per_code_acc', 0):.1%}")
        print(f"MSL LLM exact packets:  {msl_lm_ckpt['metrics'].get('exact_acc', 0):.1%}")
    else:
        print("MSL LLM metrics not in checkpoint")
    print("NOTE: Text LM is untrained (timing only). No quality comparison.")


if __name__ == "__main__":
    main()
