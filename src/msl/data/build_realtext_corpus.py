"""Build a real-text MSL corpus: encode Tatoeba sentences with a pretrained encoder,
then learn a quantizer on the embeddings.

Pipeline:
  1. Load Tatoeba English sentences (real text)
  2. Encode with all-MiniLM-L6-v2 (pretrained, 22M params) -> 384-dim embeddings
  3. Learn PQ quantizer on embeddings -> 16 codes per sentence (MSL packets)
  4. Save: (sentences, embeddings, packets)

The LLM will then train on the packet sequences, and the decoder will learn to
reconstruct sentences from packets.
"""

from __future__ import annotations

import bz2
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import seed_everything


def load_tatoeba(max_sentences: int = 10000, min_len: int = 20, max_len: int = 120):
    """Load English sentences from Tatoeba."""
    local_path = Path("/tmp/tatoeba_sentences.tar.bz2")
    if not local_path.exists():
        print("downloading Tatoeba...")
        urllib.request.urlretrieve(
            "https://downloads.tatoeba.org/exports/sentences.tar.bz2", local_path
        )
    with bz2.open(local_path) as f:
        data = f.read().decode("utf-8")
    sentences = []
    for line in data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "eng":
            s = parts[2].strip()
            if min_len <= len(s) <= max_len:
                sentences.append(s)
    return sentences[:max_sentences]


def main():
    seed_everything(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load sentences
    print("loading Tatoeba sentences...")
    sentences = load_tatoeba(max_sentences=10000)
    print(f"sentences: {len(sentences)}")
    for s in sentences[:5]:
        print(f"  {s}")

    # 2. Encode with pretrained encoder
    print("\nencoding with all-MiniLM-L6-v2...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    embeddings: list[torch.Tensor] = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            embeddings.append(emb.cpu())
    all_embeddings: torch.Tensor = torch.cat(embeddings, dim=0)  # (N, 384)
    print(f"embeddings: {all_embeddings.shape}")

    # 3. Learn PQ quantizer on embeddings
    print("\nlearning PQ quantizer (EMA updates, no optimizer needed)...")
    d_z = 384
    n_codebooks = 16
    codebook_size = 256
    quantizer = PQQuantizer(d_z, n_codebooks, codebook_size)
    quantizer.train()

    # PQ uses EMA updates (no learnable params). Just feed data through it.
    for step in range(500):
        idx = np.random.randint(0, len(all_embeddings), 256)
        batch = all_embeddings[idx]
        with torch.no_grad():
            out = quantizer(batch)
        if step % 100 == 0:
            print(f"  quantizer step {step}: commit_loss={out.commit_loss.item():.4f}, "
                  f"active={out.usage['active_fraction'].item():.3f}")

    # 4. Quantize all embeddings
    quantizer.eval()
    with torch.no_grad():
        out = quantizer(all_embeddings)
        packets = out.codes  # (N, 16)
    print(f"\npackets: {packets.shape}")
    print(f"unique packets: {packets.unique(dim=0).shape[0]}")
    print(f"bits per sentence: {n_codebooks * np.log2(codebook_size):.0f}")

    # 5. Save
    torch.save({
        "sentences": sentences,
        "embeddings": all_embeddings,
        "packets": packets,
        "n_codebooks": n_codebooks,
        "codebook_size": codebook_size,
    }, "runs/realtext_corpus.pt")
    print("saved runs/realtext_corpus.pt")


if __name__ == "__main__":
    main()
