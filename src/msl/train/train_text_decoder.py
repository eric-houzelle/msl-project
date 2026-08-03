"""MSL decoder: generates text from MSL packets using a pretrained GPT-2.

The decoder takes the quantized embedding (z_q, 384-dim) and conditions GPT-2
to generate the original sentence. GPT-2 already knows how to generate text;
we just learn a projection from z_q to GPT-2's embedding space.

Architecture:
  z_q (384-dim) -> projection -> prefix embeddings (K tokens, 768-dim)
  GPT-2 generates text conditioned on these prefix embeddings

Training:
  1. Encode sentence -> z_q (frozen quantizer)
  2. Project z_q -> prefix embeddings
  3. GPT-2 generates the original sentence (teacher forcing)
  4. Loss: cross-entropy on the generated tokens

Usage:
    python -u -m msl.train.train_text_decoder --steps 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from msl.utils.seeding import default_device, seed_everything


class MSLDecoder(nn.Module):
    """GPT-2 conditioned on MSL packets via prefix embeddings."""

    def __init__(self, d_z: int = 384, n_prefix: int = 4, gpt2_name: str = "gpt2") -> None:
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.d_model = self.gpt2.config.n_embd  # 768 for GPT-2
        self.n_prefix = n_prefix

        # Projection: z_q -> n_prefix prefix embeddings
        self.proj = nn.Sequential(
            nn.Linear(d_z, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * n_prefix),
        )
        # Initialize prefix to near-zero so GPT-2 starts from its pretrained state.
        nn.init.normal_(self.proj[-1].weight, std=0.02)  # type: ignore
        nn.init.zeros_(self.proj[-1].bias)  # type: ignore

    def forward(self, z_q: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """z_q: (B, d_z), input_ids: (B, T) -> loss.

        GPT-2 generates the text conditioned on prefix embeddings from z_q.
        """
        B = z_q.shape[0]
        T = input_ids.shape[1]

        # Project z_q to prefix embeddings.
        prefix = self.proj(z_q)  # (B, d_model * n_prefix)
        prefix = prefix.view(B, self.n_prefix, self.d_model)  # (B, n_prefix, d_model)

        # Get GPT-2 token embeddings.
        token_embeds = self.gpt2.transformer.wte(input_ids)  # (B, T, d_model)

        # Prepend prefix embeddings.
        full_embeds = torch.cat([prefix, token_embeds], dim=1)  # (B, n_prefix + T, d_model)

        # Forward through GPT-2.
        outputs = self.gpt2.transformer(inputs_embeds=full_embeds)
        hidden = outputs.last_hidden_state  # (B, n_prefix + T, d_model)

        # LM head: the LAST prefix position predicts the FIRST token,
        # then each token position predicts the NEXT.
        # hidden[:, n_prefix-1] -> predict input_ids[:, 0]
        # hidden[:, n_prefix]   -> predict input_ids[:, 1]
        # etc.
        logits = self.gpt2.lm_head(hidden[:, self.n_prefix - 1:])  # (B, T+1, vocab)

        # Loss: predict ALL tokens (including the first from the prefix).
        shift_logits = logits[:, :-1].contiguous()  # (B, T, vocab) - predicts all tokens
        shift_labels = input_ids.contiguous()        # (B, T) - all target tokens

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, z_q: torch.Tensor, tokenizer: GPT2Tokenizer,
                 max_len: int = 50) -> list[str]:
        """Generate text from packets. No start token — first token comes from prefix."""
        B = z_q.shape[0]
        device = z_q.device
        prefix = self.proj(z_q).view(B, self.n_prefix, self.d_model)

        # Generate first token from prefix alone (no start token needed).
        outputs = self.gpt2.transformer(inputs_embeds=prefix)
        first_logits = self.gpt2.lm_head(outputs.last_hidden_state[:, -1])
        first_token = first_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
        tokens = first_token

        for _ in range(max_len - 1):
            token_embeds = self.gpt2.transformer.wte(tokens)
            full_embeds = torch.cat([prefix, token_embeds], dim=1)
            outputs = self.gpt2.transformer(inputs_embeds=full_embeds)
            next_logits = self.gpt2.lm_head(outputs.last_hidden_state[:, -1])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token == tokenizer.eos_token_id).all():
                break

        return [str(tokenizer.decode(t)) for t in tokens]


class TextDataset(Dataset):
    """Dataset of (embedding, token_ids) pairs. Quantization happens in the training loop."""

    def __init__(self, corpus_path: str, tokenizer: GPT2Tokenizer, max_len: int = 48) -> None:
        corpus = torch.load(corpus_path, weights_only=False)
        self.sentences = corpus["sentences"]
        self.embeddings = corpus["embeddings"]
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Pre-tokenize all sentences.
        self.token_ids = []
        for s in self.sentences:
            ids = tokenizer.encode(s, truncation=True, max_length=max_len)
            ids = ids + [tokenizer.eos_token_id] * (max_len - len(ids))
            self.token_ids.append(ids[:max_len])

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        return {
            "embedding": self.embeddings[idx],
            "token_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/big_corpus.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Detect embedding dim and quantizer config from corpus.
    import torch as _torch
    corpus_meta = _torch.load(args.corpus, weights_only=False, map_location="cpu")
    emb_dim = corpus_meta.get("embeddings", _torch.zeros(1, 384)).shape[1]
    n_codebooks = corpus_meta.get("n_codebooks", 48)
    codebook_size = corpus_meta.get("codebook_size", 256)
    del corpus_meta
    print(f"corpus: emb_dim={emb_dim}, {n_codebooks} codebooks x {codebook_size}")

    # Build decoder (adapts to embedding dim).
    decoder = MSLDecoder(d_z=emb_dim, n_prefix=4).to(device)
    # Freeze GPT-2 except the last 2 layers (let them adapt to quantization noise).
    n_layers = len(decoder.gpt2.transformer.h)
    for i, block in enumerate(decoder.gpt2.transformer.h):
        for p in block.parameters():
            p.requires_grad = i >= n_layers - 2
    # Also unfreeze the LM head.
    for p in decoder.gpt2.lm_head.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"decoder: GPT-2 last 2 layers + LM head + projection trainable: {n_trainable:,} params")

    # Build quantizer (trained inline with the decoder).
    from msl.models.quantizer import PQQuantizer
    quantizer = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
    quantizer.train()
    print(f"quantizer: {n_codebooks} codebooks x {codebook_size} ({n_codebooks*8} bits)")

    # Dataset.
    ds = TextDataset(args.corpus, tok, max_len=48)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                       collate_fn=lambda b: {
                           "embedding": torch.stack([x["embedding"] for x in b]),
                           "token_ids": torch.stack([x["token_ids"] for x in b]),
                       })

    opt = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)

    print(f"training {args.steps} steps (batch=16, quantizer in loop)")
    decoder.train()
    t0 = time.time()
    step = 0
    running_loss, running_n = 0.0, 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            emb = batch["embedding"].to(device)
            token_ids = batch["token_ids"].to(device)
            # Quantize: embedding -> z_q (discrete, with noise)
            with torch.no_grad():
                q_out = quantizer(emb)
                z_q = q_out.z_q  # quantized version (has noise from discretization)
            # Train decoder to reconstruct from QUANTIZED embeddings
            opt.zero_grad()
            out = decoder(z_q, token_ids)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(decoder.proj.parameters(), 1.0)
            opt.step()
            running_loss += out["loss"].item()
            running_n += 1
            step += 1
            if step % 500 == 0:
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running_loss/running_n:.4f} | {el*1000:.0f}ms/step", flush=True)
                running_loss, running_n = 0.0, 0
    print(f"done in {time.time()-t0:.1f}s")

    # Test: generate from QUANTIZED embeddings (real MSL packets).
    print("\n=== GENERATION TEST (with quantization) ===")
    decoder.eval()
    quantizer.eval()
    test_sentences = ds.sentences[:5]
    for i, s in enumerate(test_sentences):
        emb = ds.embeddings[i:i+1].to(device)
        with torch.no_grad():
            q_out = quantizer(emb)
            z_q = q_out.z_q  # quantized (discrete packets)
        generated = decoder.generate(z_q, tok, max_len=40)
        print(f"\n--- Example {i+1} ---")
        print(f"  Original:  {s}")
        print(f"  Generated:  {generated[0]}")

    # Also test with continuous (no quantization) for comparison.
    print("\n=== GENERATION TEST (continuous, no quantization) ===")
    for i, s in enumerate(test_sentences):
        emb = ds.embeddings[i:i+1].to(device)
        generated = decoder.generate(emb, tok, max_len=40)
        print(f"  {s[:50]} -> {generated[0][:50]}")

    out_dir = Path("runs")
    torch.save({
        "decoder": decoder.state_dict(),
        "quantizer": quantizer.state_dict(),
        "config": {"d_z": emb_dim, "n_prefix": 4, "n_codebooks": n_codebooks, "codebook_size": codebook_size},
    }, out_dir / "text_decoder_quant_0.pt")
    print(f"\nsaved {out_dir / 'text_decoder_quant_0.pt'}")


if __name__ == "__main__":
    main()
