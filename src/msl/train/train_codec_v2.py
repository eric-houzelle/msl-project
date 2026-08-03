"""Train CodecV2: non-autoregressive decoder, translation-focused.

Two phases:
  Phase 1: same-language reconstruction (learn to encode/decode faithfully)
  Phase 2: cross-language reconstruction (learn to translate FR<->EN)

Usage:
  python -u -m msl.train.train_codec_v2 --config configs/codec_v2_p1.yaml --steps 10000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.codec_v2 import CodecV2, CodecV2Config
from msl.models.tokenizer import default_tokenizer
from msl.utils.seeding import default_device, seed_everything


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_codec_v2(cfg: dict, vocab_size: int, content_token_ids: set | None = None) -> CodecV2:
    c = cfg.get("codec", {})
    kwargs = {k: v for k, v in c.items() if k in CodecV2Config.__dataclass_fields__}
    return CodecV2(CodecV2Config(vocab_size=vocab_size, **kwargs),
                    content_token_ids=content_token_ids)


def cosine_lr(step, warmup, total, base):
    import math
    if step < warmup:
        return base * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(codec: CodecV2, loader: DataLoader, device: torch.device, n_batches: int = 10):
    codec.eval()
    recon_accs, losses = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = codec(batch)
        losses.append(out["loss"].item())
        if "recon_acc" in out["logs"]:
            recon_accs.append(out["logs"]["recon_acc"].item())
    codec.train()
    return {
        "eval_loss": sum(losses) / max(len(losses), 1),
        "recon_acc": sum(recon_accs) / max(len(recon_accs), 1) if recon_accs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to init from")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    tok = default_tokenizer()
    print(f"vocab_size: {len(tok.itos)}")

    ms1_cfg = MS1Config(**cfg.get("data", {}))
    cross_lang = cfg.get("train", {}).get("cross_lang", False)
    ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=cfg["train"]["dataset_size"], seed_floor=0,
                               cross_lang=cross_lang)
    print(f"cross_lang: {cross_lang}")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       collate_fn=default_collate, num_workers=0)

    # Build the set of content token IDs (digits + attribute keys + relation types + actions).
    # These are the "variable content" tokens that get masked during training.
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
    print(f"content tokens to mask: {len(content_ids)}")

    codec = build_codec_v2(cfg, len(tok.itos), content_token_ids=content_ids).to(device)
    n_params = sum(p.numel() for p in codec.parameters())
    if args.init:
        init = torch.load(args.init, map_location=device, weights_only=False)
        codec.load_state_dict(init["codec"], strict=False)
        print(f"loaded init from {args.init}")
    print(f"codec v2 params: {n_params:,}")

    total = args.steps or cfg["train"]["steps"]
    warmup = cfg["train"].get("warmup", 500)
    base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(codec.parameters(), lr=base_lr, weight_decay=0.01)

    print(f"training {total} steps, batch {cfg['train']['batch_size']}, lr {base_lr}")
    codec.train()
    t0 = time.time()
    step = 0
    running = {"loss": 0.0, "recon_acc": 0.0, "l_recon": 0.0, "l_task": 0.0, "n": 0}
    while step < total:
        for batch in loader:
            if step >= total:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            lr = cosine_lr(step, warmup, total, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = codec(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            running["loss"] += out["loss"].item()
            running["recon_acc"] += out["logs"].get("recon_acc", torch.tensor(0.0)).item()
            running["l_recon"] += out["logs"].get("l_reconstruction", torch.tensor(0.0)).item()
            running["l_task"] += out["logs"].get("l_task", torch.tensor(0.0)).item()
            running["n"] += 1
            step += 1
            if step % cfg["train"].get("log_every", 500) == 0:
                n = running["n"]
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running['loss']/n:.4f} | "
                      f"recon {running['l_recon']/n:.3f} | recon_acc {running['recon_acc']/n:.3f} | "
                      f"task {running['l_task']/n:.3f} | lr {lr:.2e} | {el*1000:.0f}ms/step", flush=True)
                running = {k: 0.0 for k in running}
    print(f"done in {time.time()-t0:.1f}s")

    # Eval
    train_m = evaluate(codec, loader, device)
    print(f"eval[train]: {train_m}")
    test_ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=1024, seed_floor=2_000_000)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=default_collate)
    test_m = evaluate(codec, test_loader, device, n_batches=16)
    print(f"eval[test]: {test_m}")

    out_dir = Path(cfg.get("output", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{cfg['name']}_{args.seed}.pt"
    torch.save({"codec": codec.state_dict(), "cfg": cfg, "n_params": n_params,
                "train_metrics": train_m, "test_metrics": test_m}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
