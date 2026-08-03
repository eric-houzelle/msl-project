"""Codec training loop for Phase 1A (H1).

Usage:
    python -m msl.train.train_codec --config configs/mvp_smoke.yaml

Logs loss terms and task accuracy to stdout. Saves the final checkpoint.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from msl.data.dataloader import (
    MS1Config,
    MS1Dataset,
    PrecomputedMS1Dataset,
    default_collate,
    make_collate,
)
from msl.models.codec import Codec, CodecConfig
from msl.models.tokenizer import default_tokenizer
from msl.utils.seeding import default_device, seed_everything


def load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def build_codec(cfg: dict, vocab_size: int) -> Codec:
    c = cfg.get("codec", {})
    cc = CodecConfig(vocab_size=vocab_size, **{k: v for k, v in c.items() if k in CodecConfig.__dataclass_fields__ and k != "vocab_size"})
    return Codec(cc)


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(codec: Codec, loader: DataLoader, device: torch.device, n_batches: int = 5) -> dict[str, float]:
    codec.eval()
    acc, losses, acc_noise = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = codec(batch)
        losses.append(out["loss"].item())
        if "task_acc" in out["logs"]:
            acc.append(out["logs"]["task_acc"].item())
        # Noise ablation: replace packets with random noise. If the model truly
        # uses the packets, accuracy must drop sharply (H1 diagnostic).
        codes, z_q, _, _ = codec.encode_quantize(batch["text_ids"], batch["text_mask"])
        noise = torch.randn_like(z_q)
        dec = codec.decoder(noise, batch["task_kind_id"])
        pred = dec["task_logits"].argmax(-1)
        acc_noise.append((pred == batch["task_cls"]).float().mean().item())
    codec.train()
    return {
        "eval_loss": sum(losses) / max(len(losses), 1),
        "task_acc": sum(acc) / max(len(acc), 1) if acc else 0.0,
        "task_acc_noise": sum(acc_noise) / max(len(acc_noise), 1) if acc_noise else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override total steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to init from (phase 2)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    total = args.steps or cfg["train"]["steps"]
    result = train_one_run(cfg, args.seed, total, verbose=True, init_ckpt=args.init)

    out_dir = Path(cfg.get("output", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{cfg['name']}_{args.seed}.pt"
    torch.save({
        "codec": result["codec_state"], "cfg": cfg, "n_params": result["n_params"],
        "train_metrics": result["train_metrics"], "test_metrics": result["test_metrics"],
    }, ckpt)
    print(f"saved {ckpt}")


def train_one_run(
    cfg: dict,
    seed: int,
    steps: int,
    verbose: bool = False,
    log_fn: Any = None,
    init_ckpt: str | None = None,
) -> dict:
    """Train one codec run and evaluate on train + test-unseen splits.

    Returns dict with: codec_state, n_params, train_metrics, test_metrics.
    `log_fn(step, logs)` if provided is called every log_every steps (for sweep).
    `init_ckpt` if provided loads weights from a previous checkpoint (phase 2).
    """
    seed_everything(seed)
    device = default_device()
    if verbose:
        print(f"device: {device}")

    tok = default_tokenizer()
    if verbose:
        print(f"vocab_size: {len(tok.itos)}")

    ms1_cfg = MS1Config(**cfg.get("data", {}))
    use_precomputed = cfg["train"].get("precompute", True)
    collate: Any
    if use_precomputed:
        ds: torch.utils.data.Dataset = PrecomputedMS1Dataset(
            tok, ms1_cfg, size=cfg["train"]["dataset_size"], seed_floor=0)
        collate = default_collate
    else:
        ds = MS1Dataset(tok, ms1_cfg, size=cfg["train"]["dataset_size"], seed_floor=0)
        collate = make_collate(tok)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        collate_fn=collate, num_workers=0)

    codec = build_codec(cfg, len(tok.itos)).to(device)
    n_params = sum(p.numel() for p in codec.parameters())
    if init_ckpt is not None:
        init = torch.load(init_ckpt, map_location=device, weights_only=False)
        codec.load_state_dict(init["codec"], strict=False)
        if verbose:
            print(f"loaded init from {init_ckpt}")
    if verbose:
        print(f"codec params: {n_params:,}")

    warmup = cfg["train"].get("warmup", 1000)
    base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(codec.parameters(), lr=base_lr, weight_decay=0.01)

    if verbose:
        print(f"training {steps} steps, batch {cfg['train']['batch_size']}, lr {base_lr}")
    codec.train()
    t0 = time.time()
    step = 0
    log_every = cfg["train"].get("log_every", 50)
    running = {"loss": 0.0, "task_acc": 0.0, "n": 0}
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            lr = cosine_lr(step, warmup, steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = codec(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            running["loss"] += out["loss"].item()
            running["task_acc"] += out["logs"].get("task_acc", torch.tensor(0.0)).item()
            running["n"] += 1
            step += 1
            if step % log_every == 0:
                n = running["n"]
                logs = {
                    "step": step, "loss": running["loss"] / n,
                    "task_acc": running["task_acc"] / n,
                    "lr": lr, "ms_per_step": (time.time() - t0) / step * 1000,
                }
                if verbose:
                    el = (time.time() - t0) / step
                    print(f"step {step:5d} | loss {logs['loss']:.4f} | "
                          f"acc {logs['task_acc']:.3f} | lr {lr:.2e} | {el*1000:.0f}ms/step")
                if log_fn is not None:
                    log_fn(step, logs)
                running = {k: 0.0 for k in running}
    if verbose:
        print(f"done in {time.time()-t0:.1f}s")

    train_metrics = evaluate(codec, loader, device)
    if verbose:
        print(f"eval[train]: {train_metrics}")

    test_size = cfg["train"].get("test_dataset_size", 1024)
    test_ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=test_size, seed_floor=2_000_000)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             collate_fn=default_collate, num_workers=0)
    test_metrics = evaluate(codec, test_loader, device, n_batches=10)
    if verbose:
        print(f"eval[test unseen]: {test_metrics}")

    return {
        "codec_state": codec.state_dict(),
        "n_params": n_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    main()
