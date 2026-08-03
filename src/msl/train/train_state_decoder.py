"""Train the StateDecoder: packets -> facts.

This is the decoder that actually WORKS: instead of generating text (which
compounds errors), it predicts the structured state via classification heads,
then renders deterministically.

Usage:
    python -u -m msl.train.train_state_decoder --codec runs/v2_p1_recon_0.pt --steps 10000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from msl.data.ms1 import MS1
from msl.models.state_decoder import StateDecoder
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec_v2 import build_codec_v2
from msl.utils.seeding import default_device, seed_everything


class StateDataset(Dataset):
    """Generates (text_ids, state) pairs for training the state decoder."""

    def __init__(self, size: int, seed_floor: int = 0, min_k: int = 2, max_k: int = 32) -> None:
        self.gen = MS1(min_k=min_k, max_k=max_k)
        self.renderer = MS1  # just the class
        self.size = size
        self.seed_floor = seed_floor
        self.min_k = min_k
        self.max_k = max_k
        # Pre-generate states.
        self.states = []
        for idx in range(size):
            seed = seed_floor + (idx * 7919) % 1_000_000
            import numpy as np
            k = int(np.random.default_rng(seed).integers(min_k, max_k + 1))
            self.states.append(self.gen.generate(seed=seed, k=k))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.states[idx]


def collate_states(batch):
    return batch  # list of State objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="runs/v2_p1_recon_0.pt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    tok = default_tokenizer()

    # Load frozen codec (encoder + quantizer only).
    codec_ckpt = torch.load(args.codec, map_location=device, weights_only=False)
    content_ids = set()
    from msl.data.ms1 import ACTIONS, ATTRIBUTE_KEYS, MODALITIES, RELATION_TYPES
    from msl.models.tokenizer import tokenize
    for word in list(ATTRIBUTE_KEYS) + list(RELATION_TYPES) + list(ACTIONS) + list(MODALITIES):
        for tok_str in tokenize(word):
            if tok_str in tok.stoi:
                content_ids.add(tok.stoi[tok_str])
    for d in range(8):
        if str(d) in tok.stoi:
            content_ids.add(tok.stoi[str(d)])

    codec = build_codec_v2(codec_ckpt["cfg"], len(tok.itos), content_token_ids=content_ids).to(device)
    codec.load_state_dict(codec_ckpt["codec"], strict=False)
    codec.eval()
    # Freeze codec.
    for p in codec.parameters():
        p.requires_grad = False
    print(f"frozen codec loaded ({codec_ckpt['n_params']:,} params)")

    # State decoder.
    cc = codec_ckpt["cfg"]["codec"]
    state_dec = StateDecoder(
        d_z=cc["d_z"], d_model=cc["d_model"], n_heads=cc["n_heads"],
        n_layers=4, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in state_dec.parameters())
    print(f"state decoder params: {n_params:,}")

    # Dataset.
    from msl.data.ms1 import TextRenderer
    renderer = TextRenderer()
    ds = StateDataset(size=16384, seed_floor=0, min_k=2, max_k=32)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_states, num_workers=0)

    opt = torch.optim.AdamW(state_dec.parameters(), lr=3e-4, weight_decay=0.01)

    import math
    base_lr = 3e-4
    warmup = 500
    total = args.steps

    print(f"training {total} steps")
    state_dec.train()
    t0 = time.time()
    step = 0
    running = {"loss": 0.0, "ent": 0.0, "attr": 0.0, "rel": 0.0, "ev": 0.0, "n": 0}
    while step < total:
        for batch_states in loader:
            if step >= total:
                break
            # Encode each state's text to packets.
            texts = [renderer.render(s, 0) for s in batch_states]
            text_ids_list = [tok.encode(t, True, True) for t in texts]
            max_len = max(len(t) for t in text_ids_list)
            text_ids = torch.full((len(batch_states), max_len), tok.pad_id, dtype=torch.long, device=device)
            text_mask = torch.ones((len(batch_states), max_len), dtype=torch.long, device=device)
            for i, ids in enumerate(text_ids_list):
                text_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                text_mask[i, :len(ids)] = 0

            with torch.no_grad():
                _, z_q, _, _ = codec.encode_quantize(text_ids, text_mask)

            lr = base_lr * (step / warmup if step < warmup else
                            0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = state_dec.state_loss(z_q, batch_states)
            out["state_loss"].backward()
            torch.nn.utils.clip_grad_norm_(state_dec.parameters(), 1.0)
            opt.step()

            running["loss"] += out["state_loss"].item()
            running["ent"] += out["ent_loss"].item()
            running["attr"] += out["attr_loss"].item()
            running["rel"] += out["rel_loss"].item()
            running["ev"] += out["ev_loss"].item()
            running["n"] += 1
            step += 1
            if step % 500 == 0:
                n = running["n"]
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running['loss']/n:.3f} | "
                      f"ent {running['ent']/n:.3f} attr {running['attr']/n:.3f} "
                      f"rel {running['rel']/n:.3f} ev {running['ev']/n:.3f} | "
                      f"lr {lr:.2e} | {el*1000:.0f}ms/step", flush=True)
                running = {k: 0.0 for k in running}
    print(f"done in {time.time()-t0:.1f}s")

    # Test: round-trip fidelity.
    print("\n=== ROUND-TRIP TEST ===")
    state_dec.eval()
    gen = MS1(min_k=4, max_k=12)
    n_correct_facts = 0
    n_total_facts = 0
    n_exact_states = 0
    n_test = 20

    for i in range(n_test):
        s = gen.generate(seed=2_000_000 + i, k=8)
        # Encode.
        text = renderer.render(s, 0)
        enc_ids = torch.tensor([tok.encode(text, True, True)], dtype=torch.long, device=device)
        enc_mask = (enc_ids != tok.pad_id).long()
        with torch.no_grad():
            _, z_q, _, _ = codec.encode_quantize(enc_ids, enc_mask)
            pred_states = state_dec.decode_state(z_q)
        pred_s = pred_states[0]

        # Compare facts.
        src_facts: set[tuple] = set()
        for a in s.attributes:
            src_facts.add(("attr", a.eid, a.key, a.value, a.modality))
        for r in s.relations:
            src_facts.add(("rel", r.src, r.rtype, r.dst, r.modality))
        for e in s.events:
            src_facts.add(("ev", e.time, e.eid, e.action, e.modality))

        pred_facts: set[tuple] = set()
        for a in pred_s.attributes:
            pred_facts.add(("attr", a.eid, a.key, a.value, a.modality))
        for r in pred_s.relations:
            pred_facts.add(("rel", r.src, r.rtype, r.dst, r.modality))
        for e in pred_s.events:
            pred_facts.add(("ev", e.time, e.eid, e.action, e.modality))

        overlap = len(src_facts & pred_facts)
        n_correct_facts += overlap
        n_total_facts += len(src_facts)
        if src_facts == pred_facts:
            n_exact_states += 1

        if i < 4:
            print(f"\n--- Test {i+1} (k={s.difficulty}) ---")
            print(f"  Source:  {text[:150]}")
            pred_text = renderer.render(pred_s, 0)
            print(f"  Round:   {pred_text[:150]}")
            print(f"  Facts: {overlap}/{len(src_facts)} correct")

    print("\n=== VERDICT ===")
    print(f"  Facts correct: {n_correct_facts}/{n_total_facts} ({n_correct_facts/max(n_total_facts,1)*100:.0f}%)")
    print(f"  Exact states: {n_exact_states}/{n_test}")

    out_dir = Path("runs")
    ckpt = out_dir / "state_decoder_0.pt"
    torch.save({"state_decoder": state_dec.state_dict(), "n_params": n_params,
                "codec_path": args.codec}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
