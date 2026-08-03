"""H2: Standard inter-modeles — le test qui decide tout.

Test 1 (Emergence): Deux codecs independents (seeds differentes) produisent-ils
des paquets similaires pour les memes phrases ? Si oui, le langage emerge naturellement.

Test 2 (Adoption): Un nouvel encodeur peut-il apprendre a utiliser un codec fige ?
Si oui, le standard est adoptable.

Usage:
    python -u -m msl.eval.test_h2
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msl.models.quantizer import PQQuantizer
from msl.utils.seeding import seed_everything


def test_emergence():
    """Test 1: Deux codecs independents convergent-ils vers le meme langage ?"""
    print("=" * 70)
    print("TEST 1: EMERGENCE")
    print("Deux codecs independents (seeds differentes) produisent-ils")
    print("des paquets similaires pour les memes phrases ?")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    corpus = torch.load("runs/realtext_corpus.pt", weights_only=False)
    embeddings = corpus["embeddings"][:2000].to(device)
    sentences = corpus["sentences"][:2000]

    # Train two independent quantizers.
    quantizers = []
    for seed in [0, 42]:
        seed_everything(seed)
        q = PQQuantizer(384, 48, 256).to(device)
        q.train()
        for step in range(1000):
            idx = np.random.randint(0, len(embeddings), 256)
            with torch.no_grad():
                q(embeddings[idx])
        q.eval()
        quantizers.append(q)
        print(f"  codec {seed}: trained 1000 steps")

    # Quantize the same sentences with both codecs.
    with torch.no_grad():
        pkt_a = quantizers[0](embeddings).codes.cpu()  # (N, 48)
        pkt_b = quantizers[1](embeddings).codes.cpu()  # (N, 48)

    # Question 1: Do the same sentences get the same codes?
    # (They won't — different codebook init means different code indices.)
    exact_match = (pkt_a == pkt_b).all(dim=1).float().mean().item()
    code_match = (pkt_a == pkt_b).float().mean().item()
    print(f"\n  Code-level match: {code_match*100:.1f}% (exact: {exact_match*100:.1f}%)")
    print("  (Expected ~0.4% by chance for 256-entry codebooks)")

    # Question 2: Do SIMILAR sentences get SIMILAR code PATTERNS in both codecs?
    # Even if the code indices differ, the STRUCTURE should be similar.
    # Test: for each codec, compute pairwise Hamming distances.
    # Then check: are the distances correlated across codecs?
    emb_n = embeddings.cpu() / embeddings.cpu().norm(dim=1, keepdim=True)
    sim_emb = emb_n @ emb_n.T
    sim_emb.fill_diagonal_(-1)

    # For each codec, find the nearest neighbor of each sentence.
    dist_a = (pkt_a.unsqueeze(1) != pkt_a.unsqueeze(0)).sum(dim=2)  # (N, N)
    dist_b = (pkt_b.unsqueeze(1) != pkt_b.unsqueeze(0)).sum(dim=2)  # (N, N)
    dist_a.fill_diagonal_(999999)
    dist_b.fill_diagonal_(999999)

    nn_a = dist_a.argmin(dim=1)  # nearest neighbor in codec A
    nn_b = dist_b.argmin(dim=1)  # nearest neighbor in codec B

    # Do they find the SAME neighbors?
    same_nn = (nn_a == nn_b).float().mean().item()
    # Is codec B's NN in codec A's top-10?
    top10_a = dist_a.argsort(dim=1)[:, :10]
    in_top10 = 0.0
    for i in range(len(nn_b)):
        if nn_b[i] in top10_a[i]:
            in_top10 += 1
    in_top10 /= len(nn_b)

    print(f"\n  Same nearest neighbor: {same_nn*100:.1f}%")
    print(f"  Codec B's NN in codec A's top-10: {in_top10*100:.1f}%")
    print(f"  (Chance baseline: {10/2000*100:.1f}%)")

    # Question 3: Correlation of distance matrices.
    # If the codecs agree on similarity, their distance matrices should correlate.
    dist_a_flat = dist_a[~torch.eye(2000, dtype=torch.bool)].numpy()
    dist_b_flat = dist_b[~torch.eye(2000, dtype=torch.bool)].numpy()
    corr = np.corrcoef(dist_a_flat, dist_b_flat)[0, 1]
    print(f"\n  Distance matrix correlation: {corr:.3f}")
    print("  (1.0 = perfect agreement, 0.0 = no agreement)")

    # Show examples.
    print("\n  Examples:")
    for i in [0, 1, 42, 500]:
        print(f"    Q: \"{sentences[i][:60]}\"")
        print(f"      Codec A NN: \"{sentences[nn_a[i].item()][:60]}\"")
        print(f"      Codec B NN: \"{sentences[nn_b[i].item()][:60]}\"")

    return {"exact_match": exact_match, "same_nn": same_nn, "in_top10": in_top10, "corr": corr}


def test_adoption():
    """Test 2: Un nouvel encodeur peut-il apprendre un codec fige ?"""
    print("\n" + "=" * 70)
    print("TEST 2: ADOPTION")
    print("Un nouvel encodeur (init aleatoire) peut-il apprendre a produire")
    print("des paquets compatibles avec un codec fige (le standard) ?")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    corpus = torch.load("runs/realtext_corpus.pt", weights_only=False)
    embeddings = corpus["embeddings"][:5000].to(device)
    sentences = corpus["sentences"][:5000]

    # Step 1: Train the "standard" codec (MSL v0).
    seed_everything(100)
    std_quantizer = PQQuantizer(384, 48, 256).to(device)
    std_quantizer.train()
    for step in range(1000):
        idx = np.random.randint(0, len(embeddings), 256)
        with torch.no_grad():
            std_quantizer(embeddings[idx])
    std_quantizer.eval()

    # Get the "standard" packets for all sentences.
    with torch.no_grad():
        std_out = std_quantizer(embeddings)
        std_packets = std_out.codes  # (N, 48) — the standard
        std_z_q = std_out.z_q       # (N, 384) — the standard quantized embeddings

    print("  Standard (MSL v0): 48 codebooks x 256, trained 1000 steps")
    print(f"  Standard packets: {std_packets.shape}")

    # Step 2: Train a NEW projection to map embeddings -> standard's codebooks.
    # The new encoder must learn to produce embeddings that, when quantized by
    # the FROZEN standard, produce the same packets.
    seed_everything(999)
    new_proj = nn.Linear(384, 384).to(device)
    opt = torch.optim.Adam(new_proj.parameters(), lr=3e-4)

    print("  New encoder: random init, training 3000 steps...")
    for step in range(3000):
        idx = np.random.randint(0, len(embeddings), 128)
        batch = embeddings[idx]
        # Project with new encoder.
        z_new = new_proj(batch)
        # Quantize with FROZEN standard.
        with torch.no_grad():
            q_out = std_quantizer(z_new)
            new_packets = q_out.codes
            new_z_q = q_out.z_q
        # Loss: the new encoder's packets should match the standard's packets.
        # Also: the quantized embedding should be close to the standard's.
        target_packets = std_packets[idx]
        target_z_q = std_z_q[idx]
        # Project with new encoder.
        z_new = new_proj(batch)
        # Quantize with FROZEN standard (but keep gradient via straight-through).
        with torch.no_grad():
            q_out = std_quantizer(z_new)
            new_packets = q_out.codes
            new_z_q_hard = q_out.z_q
        # Straight-through: gradient flows through z_new, forward uses quantized.
        new_z_q = z_new + (new_z_q_hard - z_new).detach()
        # Loss: quantized embedding should match the standard's quantized embedding.
        recon_loss = F.mse_loss(new_z_q, target_z_q)
        loss = recon_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            # Measure packet agreement.
            with torch.no_grad():
                q_test = std_quantizer(new_proj(embeddings[:500]))
                agree = (q_test.codes == std_packets[:500]).float().mean().item()
            print(f"    step {step:4d} | loss={loss.item():.4f} | packet agreement={agree*100:.1f}%")

    # Final test: how well does the new encoder match the standard?
    with torch.no_grad():
        new_z = new_proj(embeddings[:1000])
        new_q = std_quantizer(new_z)
        new_pkt = new_q.codes
        std_pkt = std_packets[:1000]

    agreement = (new_pkt == std_pkt).float().mean().item()
    exact_match = (new_pkt == std_pkt).all(dim=1).float().mean().item()

    # Does the new encoder produce RECONSTRUCTABLE text?
    # Test: use the standard decoder with the new encoder's packets.
    print(f"\n  Packet agreement: {agreement*100:.1f}% (chance: 0.4%)")
    print(f"  Exact sentence match: {exact_match*100:.1f}%")

    # Semantic test: do the new encoder's nearest neighbors match the standard's?
    dist_new = (new_pkt.unsqueeze(1) != new_pkt.unsqueeze(0)).sum(dim=2)
    dist_std = (std_pkt.unsqueeze(1) != std_pkt.unsqueeze(0)).sum(dim=2)
    dist_new.fill_diagonal_(999999)
    dist_std.fill_diagonal_(999999)
    nn_new = dist_new.argmin(dim=1)
    nn_std = dist_std.argmin(dim=1)
    same_nn = (nn_new == nn_std).float().mean().item()
    print(f"\n  Same nearest neighbor as standard: {same_nn*100:.1f}%")

    # Show examples.
    print("\n  Examples:")
    for i in [0, 1, 42]:
        print(f"    Q: \"{sentences[i][:60]}\"")
        print(f"      Standard NN: \"{sentences[nn_std[i].item()][:60]}\"")
        print(f"      New enc NN:  \"{sentences[nn_new[i].item()][:60]}\"")

    return {"agreement": agreement, "exact_match": exact_match, "same_nn": same_nn}


def main():
    seed_everything(0)
    print("H2: STANDARD INTER-MODELES")
    print("Le test qui decide si MSL est un langage partage ou un code prive.\n")

    r1 = test_emergence()
    r2 = test_adoption()

    print("\n" + "=" * 70)
    print("VERDICT H2")
    print("=" * 70)
    print("\n  EMERGENCE:")
    print(f"    Same NN: {r1['same_nn']*100:.1f}% (chance: 0.05%)")
    print(f"    In top-10: {r1['in_top10']*100:.1f}% (chance: 0.5%)")
    print(f"    Distance correlation: {r1['corr']:.3f}")
    print("\n  ADOPTION:")
    print(f"    Packet agreement: {r2['agreement']*100:.1f}% (chance: 0.4%)")
    print(f"    Same NN as standard: {r2['same_nn']*100:.1f}%")

    print("\n  INTERPRETATION:")
    if r1["corr"] > 0.5:
        print("  EMERGENCE: Les codecs convergent vers des structures similaires.")
        print("  → Le langage emerge naturellement (standard fort).")
    elif r1["corr"] > 0.2:
        print("  EMERGENCE: Convergence partielle. Le langage emerge mais avec des variations.")
        print("  → Standard possible mais necessite un alignement.")
    else:
        print("  EMERGENCE: Pas de convergence naturelle.")
        print("  → Le langage doit etre standardise (fige et enseigne).")

    if r2["agreement"] > 0.5:
        print("  ADOPTION: Un nouvel encodeur peut apprendre le standard.")
        print("  → MSL est adoptable comme standard.")
    elif r2["agreement"] > 0.1:
        print("  ADOPTION: Apprentissage partiel. Possible mais difficile.")
    else:
        print("  ADOPTION: Le standard est trop difficile a apprendre.")


if __name__ == "__main__":
    main()
