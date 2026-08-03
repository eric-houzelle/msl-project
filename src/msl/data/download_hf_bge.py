"""Build MSL corpus with BGE-M3 encoder (568M params, 1024-dim, multilingual).

BGE-M3 is a major upgrade over MiniLM:
- 1024-dim embeddings (vs 384) → captures more semantic detail
- 568M params (vs 22M) → better understanding
- Multilingual (100+ languages) → works for FR and EN
- Supports long sequences (8192 tokens)

Usage:
    python -u -m msl.data.download_hf_bge --size 100000 --out runs/bge_corpus.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def download_tatoeba(size: int = 50000) -> list[str]:
    """Download Tatoeba English + French sentences."""
    import bz2
    import urllib.request
    from pathlib import Path

    print("downloading Tatoeba sentences...")
    local = Path("/tmp/tatoeba_sentences.tar.bz2")
    if not local.exists():
        urllib.request.urlretrieve(
            "https://downloads.tatoeba.org/exports/sentences.tar.bz2", local
        )
    with bz2.open(local) as f:
        data = f.read().decode("utf-8")
    sentences = []
    for line in data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] in ("eng", "fra"):
            s = parts[2].strip()
            if 15 <= len(s) <= 200:
                sentences.append(s)
    print(f"  Tatoeba: {len(sentences)} English + French sentences available")
    return sentences[:size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100000)
    ap.add_argument("--out", default="runs/bge_corpus.pt")
    args = ap.parse_args()

    from msl.utils.seeding import default_device, seed_everything
    seed_everything(0)
    device = default_device()
    print(f"device: {device}")

    # 1. Download sentences.
    sentences = download_tatoeba(size=args.size)
    sentences = list(dict.fromkeys(sentences))  # dedup
    print(f"total sentences: {len(sentences)} (after dedup)")

    # 2. Encode with BGE-M3.
    print("encoding with BGE-M3 (BAAI/bge-m3)...")
    model_name = "BAAI/bge-m3"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    emb_dim = model.config.hidden_size
    print(f"BGE-M3: {sum(p.numel() for p in model.parameters()):,} params, dim={emb_dim}")

    embeddings: list[torch.Tensor] = []
    batch_size = 64 if device.type == "cuda" else 32
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=128,
                        return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            # L2 normalize (BGE recommends this)
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu())
            if i % (batch_size * 200) == 0:
                print(f"  encoded {i}/{len(sentences)}", flush=True)
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"embeddings: {all_embeddings.shape}")

    # 3. Learn PQ quantizer (adapted for 1024-dim).
    # 1024 is divisible by: 1,2,4,8,16,32,64,128,256,512,1024
    # Use 64 codebooks x 256 = 512 bits (more capacity than MiniLM's 384)
    print("learning PQ quantizer (64 codebooks x 256 = 512 bits)...")
    from msl.models.quantizer import PQQuantizer
    n_codebooks = 64
    codebook_size = 256
    quantizer = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
    quantizer.train()
    for step in range(2000):
        idx = np.random.randint(0, len(all_embeddings), 256)
        with torch.no_grad():
            quantizer(all_embeddings[idx].to(device))
        if step % 500 == 0:
            print(f"  quantizer step {step}", flush=True)
    quantizer.eval()

    # 4. Quantize all.
    print("quantizing all sentences...")
    packets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(all_embeddings), 512):
            batch_emb = all_embeddings[i:i+512].to(device)
            out = quantizer(batch_emb)
            packets_list.append(out.codes.cpu())
        all_packets: torch.Tensor = torch.cat(packets_list, dim=0)
    print(f"packets: {all_packets.shape}")
    print(f"unique packets: {all_packets.unique(dim=0).shape[0]}")

    # 5. Semantic resolution test.
    print("\n=== SEMANTIC RESOLUTION ===")
    emb = all_embeddings[:500]
    pkt = all_packets[:500]
    emb_n = emb / emb.norm(dim=1, keepdim=True)
    sims = emb_n @ emb_n.T
    sims.fill_diagonal_(-1)
    vals, idxs = sims.view(-1).topk(100)
    sim_d = [(pkt[i//500] != pkt[i%500]).sum().item() for i in idxs]
    rand_d = [(pkt[a] != pkt[b]).sum().item()
              for a, b in np.random.randint(0, 500, (100, 2))]
    gap = np.mean(rand_d) - np.mean(sim_d)
    print(f"  Similar pairs: {np.mean(sim_d):.1f}/{n_codebooks} distance")
    print(f"  Random pairs:  {np.mean(rand_d):.1f}/{n_codebooks} distance")
    print(f"  Gap: {gap:.1f} codes (BGE-M3 vs MiniLM was 8.0)")

    # 6. Show examples.
    print("\n=== NEAREST NEIGHBOR EXAMPLES ===")
    pkt_dist = torch.cdist(pkt.float(), pkt.float(), p=1)
    pkt_dist.fill_diagonal_(float("inf"))
    for i in [0, 1, 42, 500]:
        nn_idx = int(pkt_dist[i].argmin().item())
        print(f'  Q: "{sentences[i][:60]}"')
        print(f'  NN: "{sentences[nn_idx][:60]}"')
        print(f'  dist={pkt_dist[i, nn_idx].item():.0f}/{n_codebooks}')

    # 7. Save.
    torch.save({
        "sentences": sentences,
        "embeddings": all_embeddings,
        "packets": all_packets,
        "n_codebooks": n_codebooks,
        "codebook_size": codebook_size,
        "encoder": model_name,
        "emb_dim": emb_dim,
    }, args.out)
    print(f"\nsaved {args.out} ({len(sentences)} sentences, "
          f"{all_packets.shape[1]*8} bits/sentence)")


if __name__ == "__main__":
    main()
