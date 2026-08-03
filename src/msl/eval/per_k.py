"""Per-difficulty (per-k) evaluation of a trained codec.

Loads a checkpoint and evaluates the noise-ablation gap on states of varying
difficulty k. This produces the real H1 curve: gap vs k for different n_slots.
If the gap grows with k, H1 is supported (harder states need more packet info).
If flat, H1 is in difficulty at this scale.

Usage:
    python -u -m msl.eval.per_k --checkpoint runs/h1_signal_0.pt --ks 2,4,8,16,32,64
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.codec import Codec
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec import build_codec, evaluate
from msl.utils.seeding import default_device, seed_everything


@torch.no_grad()
def evaluate_per_k(codec: Codec, tok, device: torch.device, k: int,
                   n_samples: int = 512, seed_floor: int = 2_000_000) -> dict:
    """Evaluate gap noise on states of a fixed difficulty k."""
    cfg = MS1Config(min_k=k, max_k=k, tasks_per_state=1)
    ds = PrecomputedMS1Dataset(tok, cfg, size=n_samples, seed_floor=seed_floor + k * 1000)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=default_collate, num_workers=0)
    return evaluate(codec, loader, device, n_batches=20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/h1_signal_0.pt")
    ap.add_argument("--ks", default="2,4,8,16,32,64")
    ap.add_argument("--n_samples", type=int, default=512)
    args = ap.parse_args()

    seed_everything(0)
    device = default_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    tok = default_tokenizer()
    codec = build_codec(cfg_dict, len(tok.itos)).to(device)
    codec.load_state_dict(ckpt["codec"])
    codec.eval()
    print(f"loaded {args.checkpoint} ({ckpt['n_params']:,} params)")
    print(f"n_slots={cfg_dict['codec']['n_slots']} quantizer={cfg_dict['codec']['quantizer_kind']}")
    print()
    print("k  test_acc  noise   gap")
    print("-" * 30)
    for k in [int(x) for x in args.ks.split(",")]:
        m = evaluate_per_k(codec, tok, device, k, args.n_samples)
        gap = m["task_acc"] - m["task_acc_noise"]
        print(f"{k:>2}  {m['task_acc']:.3f}   {m['task_acc_noise']:.3f}   {gap:+.3f}")


if __name__ == "__main__":
    main()
