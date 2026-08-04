"""Train a semantic quantizer on real-text embeddings.

The key innovation: a SEMANTIC PRESERVATION LOSS. Standard PQ only minimizes
reconstruction error. We add a loss that explicitly preserves cosine similarity:
  cos(z_i, z_j) ≈ cos(z_q_i, z_q_j)

This forces the quantizer to keep semantically similar sentences close in packet
space, not just geometrically close in embedding space.

Pipeline:
  1. Load pretrained embeddings (all-MiniLM-L6-v2)
  2. Add a learnable projection: 384 -> 384 (prepares embeddings for quantization)
  3. Train PQ quantizer + projection with:
     - Reconstruction loss (z -> z_q should be close)
     - Semantic loss (cosine similarity should be preserved)
  4. Quantize all sentences -> packets
  5. Test: nearest neighbor retrieval in packet space vs embedding space
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import default_device, seed_everything


class SemanticQuantizer(nn.Module):
    """Projection + PQ quantizer with semantic preservation loss.

    The projection layer learns to map embeddings to a space where PQ
    quantization preserves semantic similarity.
    """

    def __init__(self, d_in: int = 384, d_out: int = 384, n_codebooks: int = 48,
                 codebook_size: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.quantizer = PQQuantizer(d_out, n_codebooks, codebook_size)
        self.n_codebooks = n_codebooks

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z_proj = self.proj(z)
        out = self.quantizer(z_proj)
        return {
            "codes": out.codes,
            "z_q": out.z_q,
            "commit_loss": out.commit_loss,
            "z_proj": z_proj,
        }

    def semantic_loss(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        """Preserve pairwise cosine similarity through quantization.

        For a batch of N sentences, compute the N×N cosine similarity matrix
        of the original embeddings and the quantized embeddings. Minimize
        the difference.
        """
        z_n = F.normalize(z, dim=-1)
        zq_n = F.normalize(z_q, dim=-1)
        sim_z = z_n @ z_n.T  # (N, N)
        sim_zq = zq_n @ zq_n.T
        return F.mse_loss(sim_zq, sim_z)


def main():
    seed_everything(0)
    device = default_device()
    print(f"device: {device}")

    # Load corpus
    corpus = torch.load("runs/realtext_corpus.pt", weights_only=False)
    embeddings = corpus["embeddings"].to(device)
    sentences = corpus["sentences"]
    print(f"corpus: {len(sentences)} sentences, embeddings {embeddings.shape}")

    # Build semantic quantizer
    model = SemanticQuantizer(d_in=384, d_out=384, n_codebooks=48, codebook_size=256).to(device)
    # The projection layer has learnable params; the PQ codebooks use EMA.
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training: 5000 steps with reconstruction + semantic loss
    model.train()
    print("training semantic quantizer (5000 steps)...")
    for step in range(5000):
        idx = np.random.randint(0, len(embeddings), 128)
        batch = embeddings[idx]

        out = model(batch)
        # Reconstruction loss: quantized should be close to projected (both normalized)
        z_proj_n = F.normalize(out["z_proj"], dim=-1)
        recon_loss = F.mse_loss(out["z_q"], z_proj_n.detach())
        # Semantic loss: preserve cosine similarity
        sem_loss = model.semantic_loss(batch, out["z_q"])
        # Commit loss from quantizer
        commit_loss = out["commit_loss"]
        # Total: reconstruction is primary, semantic is secondary
        loss = recon_loss * 10.0 + sem_loss * 0.1 + commit_loss * 0.5

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            from msl.models.quantizer import _EMACodebook
            cb0 = model.quantizer.codebooks[0]
            assert isinstance(cb0, _EMACodebook)
            print(f"  step {step:4d} | recon={recon_loss.item():.4f} | "
                  f"sem={sem_loss.item():.4f} | commit={commit_loss.item():.4f} | "
                  f"active={cb0.usage.mean().item():.2f}", flush=True)

    # Quantize all sentences
    model.eval()
    with torch.no_grad():
        out = model(embeddings)
        packets = out["codes"].cpu()
        z_q = out["z_q"].cpu()

    print(f"\npackets: {packets.shape}")
    print(f"unique packets: {packets.unique(dim=0).shape[0]}")

    # Test semantic resolution
    print("\n=== TEST SEMANTIC RESOLUTION ===")
    emb = embeddings.cpu()[:500]
    pkt = packets[:500]
    emb_n = emb / emb.norm(dim=1, keepdim=True)
    sims = emb_n @ emb_n.T
    mask = torch.eye(500).bool()
    sims[mask] = -1

    for n_top in [50]:
        vals, idxs = sims.view(-1).topk(n_top)
        sim_dists = []
        for i in idxs:
            a, b = i // 500, i % 500
            sim_dists.append((pkt[a] != pkt[b]).sum().item())
        rand_dists = []
        for _ in range(n_top):
            a, b = np.random.randint(0, 500, 2)
            rand_dists.append((pkt[a] != pkt[b]).sum().item())
        print(f"  Similar pairs: {np.mean(sim_dists):.1f}/{48} distance")
        print(f"  Random pairs:  {np.mean(rand_dists):.1f}/{48} distance")
        print(f"  Gap: {np.mean(rand_dists) - np.mean(sim_dists):.1f} codes")

    # Test: nearest neighbor retrieval
    print("\n=== TEST NEAREST NEIGHBOR RETRIEVAL ===")
    # For each sentence, find the nearest in packet space (excluding self)
    # and check if it's semantically related
    pkt_dist = torch.cdist(pkt.float(), pkt.float(), p=1)  # L1 = hamming for ints
    pkt_dist.fill_diagonal_(float("inf"))
    emb_dist = 1 - sims  # convert sim to distance
    emb_dist.fill_diagonal_(float("inf"))

    n_retrieval_correct = 0
    for i in range(20):
        # Nearest in embedding space (ground truth semantic neighbor)
        emb_nn = emb_dist[i].argmin().item()
        # Nearest in packet space
        pkt_nn = pkt_dist[i].argmin().item()
        # Check if packet NN is in top-10 of embedding NN
        emb_top10 = emb_dist[i].argsort()[:10].tolist()
        if pkt_nn in emb_top10:
            n_retrieval_correct += 1
        if i < 5:
            print(f"  Q: \"{sentences[i][:60]}\"")
            print(f"    Emb NN: \"{sentences[emb_nn][:60]}\"")
            print(f"    Pkt NN: \"{sentences[pkt_nn][:60]}\"")
            print(f"    Match: {pkt_nn in emb_top10}")

    print(f"\n  Retrieval in top-10: {n_retrieval_correct}/20 ({n_retrieval_correct/20*100:.0f}%)")

    # Save
    torch.save({
        "sentences": sentences,
        "embeddings": embeddings.cpu(),
        "packets": packets,
        "z_q": z_q,
        "n_codebooks": 48,
        "codebook_size": 256,
        "model_state": model.state_dict(),
    }, "runs/realtext_semantic.pt")
    print("saved runs/realtext_semantic.pt")


if __name__ == "__main__":
    main()
