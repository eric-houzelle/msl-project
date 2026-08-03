"""Download large datasets from HuggingFace for MSL training.

Downloads and prepares 100k+ sentences from HuggingFace datasets:
- Tatoeba (multilingual sentence pairs)
- Wikipedia Simple English (real articles)
- OpenWebText (web text)

Usage:
    python -u -m msl.data.download_hf --size 100000 --out runs/big_corpus.pt
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def download_tatoeba(size: int = 50000) -> list[str]:
    """Download Tatoeba English sentences via the Tatoeba dump."""
    print("downloading Tatoeba sentences...")
    local = Path("/tmp/tatoeba_sentences.tar.bz2")
    if not local.exists():
        urllib.request.urlretrieve(
            "https://downloads.tatoeba.org/exports/sentences.tar.bz2", local
        )
    import bz2
    with bz2.open(local) as f:
        data = f.read().decode("utf-8")
    sentences = []
    for line in data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "eng":
            s = parts[2].strip()
            if 15 <= len(s) <= 200:
                sentences.append(s)
    print(f"  Tatoeba: {len(sentences)} English sentences available")
    return sentences[:size]


def download_wikipedia_simple(size: int = 50000) -> list[str]:
    """Download Simple English Wikipedia via HuggingFace parquet API."""
    print("downloading Simple English Wikipedia...")
    # HuggingFace hosts the data as parquet. We download and parse it.
    # The dataset is small (one file) so this is feasible.
    url = "https://huggingface.co/datasets/wikipedia/20220301.simple/resolve/main/data/train-00000-of-00001.parquet"
    try:
        import pandas as pd
        local = Path("/tmp/wiki_simple.parquet")
        if not local.exists():
            urllib.request.urlretrieve(url, local)
        df = pd.read_parquet(local)
        # The 'text' column contains full articles. Split into sentences.
        sentences = []
        for text in df["text"]:
            # Simple sentence splitting.
            for sent in text.split(". "):
                sent = sent.strip()
                if 15 <= len(sent) <= 200:
                    sentences.append(sent + ".")
                    if len(sentences) >= size:
                        break
            if len(sentences) >= size:
                break
        print(f"  Wikipedia Simple: {len(sentences)} sentences")
        return sentences[:size]
    except Exception as e:
        print(f"  Wikipedia Simple: failed ({e}), skipping")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100000)
    ap.add_argument("--out", default="runs/big_corpus.pt")
    args = ap.parse_args()

    from msl.utils.seeding import seed_everything
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # 1. Download sentences.
    tatoeba = download_tatoeba(size=args.size // 2)
    wiki = download_wikipedia_simple(size=args.size // 2)
    sentences: list[str] = tatoeba + wiki
    # Deduplicate.
    sentences = list(dict.fromkeys(sentences))
    print(f"total sentences: {len(sentences)} (after dedup)")

    # 2. Encode with all-MiniLM-L6-v2.
    print("encoding with all-MiniLM-L6-v2...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    embeddings = []
    batch_size = 128 if device.type == "cuda" else 64
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            embeddings.append(emb.cpu())
            if i % (batch_size * 100) == 0:
                print(f"  encoded {i}/{len(sentences)}", flush=True)
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"embeddings: {all_embeddings.shape}")

    # 3. Learn PQ quantizer (48 codebooks x 256 = 384 bits).
    print("learning PQ quantizer...")
    from msl.models.quantizer import PQQuantizer
    quantizer = PQQuantizer(384, 48, 256).to(device)
    quantizer.train()
    for step in range(1000):
        idx = np.random.randint(0, len(all_embeddings), 256)
        with torch.no_grad():
            quantizer(all_embeddings[idx].to(device))
        if step % 200 == 0:
            print(f"  quantizer step {step}", flush=True)
    quantizer.eval()

    # 4. Quantize all.
    with torch.no_grad():
        packets_list: list[torch.Tensor] = []
        for i in range(0, len(all_embeddings), 512):
            batch_emb = all_embeddings[i:i+512].to(device)
            out = quantizer(batch_emb)
            packets_list.append(out.codes.cpu())
        all_packets: torch.Tensor = torch.cat(packets_list, dim=0)
    print(f"packets: {all_packets.shape}")
    print(f"unique packets: {all_packets.unique(dim=0).shape[0]}")

    # 5. Save.
    torch.save({
        "sentences": sentences,
        "embeddings": all_embeddings,
        "packets": all_packets,
        "n_codebooks": 48,
        "codebook_size": 256,
    }, args.out)
    print(f"saved {args.out} ({len(sentences)} sentences, {all_packets.shape[1]*8} bits/sentence)")


if __name__ == "__main__":
    main()
