"""Training loop for the text baseline (B-text).

Same data, same size, same training as the MSL codec — but no quantizer.
This gives us the reference point: how well can a model read the TEXT directly?

Usage:
    python -u -m msl.train.train_text_baseline --config configs/mvp_text.yaml --steps 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.text_baseline import TextBaseline
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec import cosine_lr
from msl.utils.seeding import default_device, seed_everything


def evaluate_baseline(model: TextBaseline, loader: DataLoader, device: torch.device,
                      n_batches: int = 10) -> dict[str, float]:
    """Evaluate: accuracy (no noise ablation — there are no packets to ablate)."""
    model.eval()
    acc, losses = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model.forward_loss(batch)
            losses.append(out["loss"].item())
            acc.append(out["task_acc"].item())
    model.train()
    return {"eval_loss": sum(losses) / max(len(losses), 1),
            "task_acc": sum(acc) / max(len(acc), 1) if acc else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    tok = default_tokenizer()
    print(f"vocab_size: {len(tok.itos)}")

    ms1_cfg = MS1Config(**cfg.get("data", {}))
    ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=cfg["train"]["dataset_size"], seed_floor=0)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       collate_fn=default_collate, num_workers=0)

    mc = cfg["model"]
    model = TextBaseline(
        vocab_size=len(tok.itos),
        d_model=mc["d_model"], n_layers=mc["n_layers"], n_heads=mc["n_heads"],
        dropout=mc.get("dropout", 0.0), pad_id=tok.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"B-text params: {n_params:,}")

    total = args.steps or cfg["train"]["steps"]
    warmup = cfg["train"].get("warmup", 500)
    base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    print(f"training {total} steps, batch {cfg['train']['batch_size']}, lr {base_lr}")
    model.train()
    t0 = time.time()
    step = 0
    running_loss, running_acc, running_n = 0.0, 0.0, 0
    while step < total:
        for batch in loader:
            if step >= total:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            lr = cosine_lr(step, warmup, total, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = model.forward_loss(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += out["loss"].item()
            running_acc += out["task_acc"].item()
            running_n += 1
            step += 1
            if step % cfg["train"].get("log_every", 500) == 0:
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running_loss/running_n:.4f} | "
                      f"acc {running_acc/running_n:.3f} | lr {lr:.2e} | {el*1000:.0f}ms/step",
                      flush=True)
                running_loss, running_acc, running_n = 0.0, 0.0, 0
    print(f"done in {time.time()-t0:.1f}s")

    train_metrics = evaluate_baseline(model, loader, device)
    print(f"eval[train]: {train_metrics}")

    test_ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=1024, seed_floor=2_000_000)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            collate_fn=default_collate, num_workers=0)
    test_metrics = evaluate_baseline(model, test_loader, device, n_batches=16)
    print(f"eval[test unseen]: {test_metrics}")

    out_dir = Path(cfg.get("output", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{cfg['name']}_{args.seed}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "n_params": n_params,
                "train_metrics": train_metrics, "test_metrics": test_metrics}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
