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
from msl.utils.seeding import default_device, seed_everything


def test_emergence():
    """Test 1: Deux codecs independents convergent-ils vers le meme langage ?"""
    print("=" * 70)
    print("TEST 1: EMERGENCE")
    print("Deux codecs independents (seeds differentes) produisent-ils")
    print("des paquets similaires pour les memes phrases ?")
    print("=" * 70)

    device = default_device()
    corpus = torch.load("runs/big_corpus.pt", weights_only=False)
    emb_dim = corpus["embeddings"].shape[1]
    n_codebooks = corpus.get("n_codebooks", 48)
    codebook_size = corpus.get("codebook_size", 256)
    # Split: train on first 80%, test on last 20%.
    n_total = len(corpus["sentences"])
    n_train = int(n_total * 0.8)
    train_emb = corpus["embeddings"][:n_train].to(device)
    test_emb = corpus["embeddings"][n_train:n_train + 500].to(device)
    test_sentences = corpus["sentences"][n_train:n_train + 500]
    print(f"train: {n_train} sentences, test: {len(test_sentences)} sentences")

    # Train two independent quantizers on TRAIN set only.
    quantizers = []
    for seed in [0, 42]:
        seed_everything(seed)
        q = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
        q.train()
        for step in range(1000):
            idx = np.random.randint(0, len(train_emb), 256)
            with torch.no_grad():
                q(train_emb[idx])
        q.eval()
        quantizers.append(q)
        print(f"  codec {seed}: trained 1000 steps on {n_train} sentences")

    # Quantize the TEST sentences with both codecs.
    with torch.no_grad():
        pkt_a = quantizers[0](test_emb).codes.cpu()
        pkt_b = quantizers[1](test_emb).codes.cpu()

    # Question 1: Do the same sentences get the same codes?
    exact_match = (pkt_a == pkt_b).all(dim=1).float().mean().item()
    code_match = (pkt_a == pkt_b).float().mean().item()
    print(f"\n  Code-level match: {code_match*100:.1f}% (exact: {exact_match*100:.1f}%)")
    print(f"  (Expected ~{1/codebook_size*100:.1f}% by chance)")

    # Question 2: Do SIMILAR sentences get SIMILAR code PATTERNS in both codecs?
    emb_n = test_emb.cpu() / test_emb.cpu().norm(dim=1, keepdim=True)
    sim_emb = emb_n @ emb_n.T
    sim_emb.fill_diagonal_(-1)

    n_test = len(test_sentences)
    dist_a = (pkt_a.unsqueeze(1) != pkt_a.unsqueeze(0)).sum(dim=2)
    dist_b = (pkt_b.unsqueeze(1) != pkt_b.unsqueeze(0)).sum(dim=2)
    dist_a.fill_diagonal_(999999)
    dist_b.fill_diagonal_(999999)

    nn_a = dist_a.argmin(dim=1)
    nn_b = dist_b.argmin(dim=1)

    same_nn = (nn_a == nn_b).float().mean().item()
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
    dist_a_flat = dist_a[~torch.eye(n_test, dtype=torch.bool)].numpy()
    dist_b_flat = dist_b[~torch.eye(n_test, dtype=torch.bool)].numpy()
    corr = np.corrcoef(dist_a_flat, dist_b_flat)[0, 1]
    print(f"\n  Distance matrix correlation: {corr:.3f}")
    print("  (1.0 = perfect agreement, 0.0 = no agreement)")

    # Show examples from test set.
    print("\n  Examples:")
    for i in [0, 1, 42, min(100, n_test - 1)]:
        print(f"    Q: \"{test_sentences[i][:60]}\"")
        print(f"      Codec A NN: \"{test_sentences[nn_a[i].item()][:60]}\"")
        print(f"      Codec B NN: \"{test_sentences[nn_b[i].item()][:60]}\"")

    return {"exact_match": exact_match, "same_nn": same_nn, "in_top10": in_top10, "corr": corr}


def test_adoption():
    """Test 2: Un nouvel encodeur peut-il apprendre un codec fige ?"""
    print("\n" + "=" * 70)
    print("TEST 2: ADOPTION")
    print("Un nouvel encodeur (init aleatoire) peut-il apprendre a produire")
    print("des paquets compatibles avec un codec fige (le standard) ?")
    print("=" * 70)

    device = default_device()
    corpus = torch.load("runs/big_corpus.pt", weights_only=False)
    emb_dim = corpus["embeddings"].shape[1]
    n_codebooks = corpus.get("n_codebooks", 48)
    codebook_size = corpus.get("codebook_size", 256)
    # Split: train on first 80%, test on last 20%.
    n_total = len(corpus["sentences"])
    n_train = int(n_total * 0.8)
    train_emb = corpus["embeddings"][:n_train].to(device)
    test_emb = corpus["embeddings"][n_train:n_train + 1000].to(device)
    test_sentences = corpus["sentences"][n_train:n_train + 1000]
    print(f"train: {n_train}, test: {len(test_sentences)}")

    # Step 1: Train the "standard" codec on TRAIN set only.
    seed_everything(100)
    std_quantizer = PQQuantizer(emb_dim, n_codebooks, codebook_size).to(device)
    std_quantizer.train()
    for step in range(1000):
        idx = np.random.randint(0, len(train_emb), 256)
        with torch.no_grad():
            std_quantizer(train_emb[idx])
    std_quantizer.eval()

    # Get the "standard" packets for TRAIN set.
    with torch.no_grad():
        std_out = std_quantizer(train_emb)
        std_packets = std_out.codes
        std_z_q = std_out.z_q

    print(f"  Standard (MSL v0): {n_codebooks} codebooks x {codebook_size}, trained on {n_train} sentences")
    print(f"  Standard packets: {std_packets.shape}")

    # Step 2: Train a NEW projection to map embeddings -> standard's codebooks.
    # The new encoder must learn to produce embeddings that, when quantized by
    # the FROZEN standard, produce the same packets.
    seed_everything(999)
    new_proj = nn.Linear(emb_dim, emb_dim).to(device)
    opt = torch.optim.Adam(new_proj.parameters(), lr=3e-4)

    print("  New encoder: random init, training 3000 steps on TRAIN set...")
    for step in range(3000):
        idx = np.random.randint(0, len(train_emb), 128)
        batch = train_emb[idx]
        z_new = new_proj(batch)
        with torch.no_grad():
            q_out = std_quantizer(z_new)
            new_z_q_hard = q_out.z_q
        new_z_q = z_new + (new_z_q_hard - z_new).detach()
        target_z_q = std_z_q[idx]
        recon_loss = F.mse_loss(new_z_q, target_z_q)
        loss = recon_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                q_test = std_quantizer(new_proj(train_emb[:500]))
                agree = (q_test.codes == std_packets[:500]).float().mean().item()
            print(f"    step {step:4d} | loss={loss.item():.4f} | packet agreement={agree*100:.1f}%")

    # Final test: evaluate on HELD-OUT test set.
    with torch.no_grad():
        std_test_out = std_quantizer(test_emb)
        std_test_pkt = std_test_out.codes
        new_z = new_proj(test_emb)
        new_q = std_quantizer(new_z)
        new_pkt = new_q.codes

    agreement = (new_pkt == std_test_pkt).float().mean().item()
    exact_match = (new_pkt == std_test_pkt).all(dim=1).float().mean().item()

    print(f"\n  Packet agreement (TEST): {agreement*100:.1f}% (chance: {1/codebook_size*100:.1f}%)")
    print(f"  Exact sentence match:    {exact_match*100:.1f}%")

    dist_new = (new_pkt.unsqueeze(1) != new_pkt.unsqueeze(0)).sum(dim=2)
    dist_std = (std_test_pkt.unsqueeze(1) != std_test_pkt.unsqueeze(0)).sum(dim=2)
    dist_new.fill_diagonal_(999999)
    dist_std.fill_diagonal_(999999)
    nn_new = dist_new.argmin(dim=1)
    nn_std = dist_std.argmin(dim=1)
    same_nn = (nn_new == nn_std).float().mean().item()
    print(f"\n  Same nearest neighbor as standard (TEST): {same_nn*100:.1f}%")

    print("\n  Examples:")
    for i in [0, 1, 42]:
        print(f"    Q: \"{test_sentences[i][:60]}\"")
        print(f"      Standard NN: \"{test_sentences[nn_std[i].item()][:60]}\"")
        print(f"      New enc NN:  \"{test_sentences[nn_new[i].item()][:60]}\"")

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
