"""Round-trip test on real text: text -> MSL -> text.

Encodes a sentence, quantizes it to MSL packets, decodes it back to text,
and measures semantic similarity between original and reconstruction.

Usage:
    python -u -m msl.eval.test_roundtrip_realtext
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer

from msl.models.quantizer import PQQuantizer
from msl.train.train_text_decoder import MSLDecoder
from msl.utils.seeding import seed_everything


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # Load decoder.
    dec_ckpt = torch.load("runs/text_decoder_quant_0.pt", weights_only=False)
    decoder = MSLDecoder(d_z=384, n_prefix=4).to(device)
    decoder.load_state_dict(dec_ckpt["decoder"])
    decoder.eval()

    # Load corpus.
    corpus = torch.load("runs/big_corpus.pt", weights_only=False)
    sentences = corpus["sentences"]
    embeddings = corpus["embeddings"].to(device)

    # Build quantizer (must match training).
    quantizer = PQQuantizer(384, 48, 256).to(device)
    quantizer.eval()
    quantizer.train()
    import numpy as np
    for _ in range(1000):
        idx = np.random.randint(0, len(embeddings), 256)
        with torch.no_grad():
            quantizer(embeddings[idx])
    quantizer.eval()

    # Load MiniLM for semantic similarity.
    tok_enc = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model_enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model_enc.eval()

    def embed(text: str) -> torch.Tensor:
        inp = tok_enc([text], padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model_enc(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).float()
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

    # Test on last 20 sentences (unseen during training if we split).
    test_sentences = sentences[-20:]
    test_embeddings = embeddings[-20:]

    print("\n" + "=" * 70)
    print("ROUND-TRIP TEST: text -> MSL packets -> text")
    print("=" * 70)

    similarities = []
    for i, s in enumerate(test_sentences):
        emb = test_embeddings[i:i+1]
        with torch.no_grad():
            q_out = quantizer(emb)
            z_q = q_out.z_q
        generated = decoder.generate(z_q, GPT2Tokenizer.from_pretrained("gpt2"), max_len=40)[0]

        emb_orig = embed(s)
        emb_gen = embed(generated)
        sim = F.cosine_similarity(emb_orig, emb_gen).item()
        similarities.append(sim)

        print(f"\n--- Test {i+1} ---")
        print(f"  Original:  {s[:100]}")
        print(f"  MSL round: {generated[:100]}")
        print(f"  Sim: {sim:.3f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Mean semantic similarity: {sum(similarities)/len(similarities):.3f}")
    print(f"  Min: {min(similarities):.3f}")
    print(f"  Max: {max(similarities):.3f}")
    n_good = sum(1 for s in similarities if s > 0.5)
    print(f"  Sentences with sim > 0.5: {n_good}/{len(similarities)} ({n_good/len(similarities)*100:.0f}%)")


if __name__ == "__main__":
    main()
