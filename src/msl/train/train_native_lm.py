"""Train the native MSL Language Model on packet sequences.

The LLM learns to predict the next packet in a sequence — like a text LM
predicts the next word, but each "word" is a whole packet (8 codes in parallel).

Usage:
    python -u -m msl.train.train_native_lm --corpus runs/msl_corpus_final.pt --steps 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from msl.models.native_lm import NativeMSLLM
from msl.utils.seeding import default_device, seed_everything


class PacketSequenceDataset(Dataset):
    """Treats each state's 16 packets as a sequence to predict autoregressively."""

    def __init__(self, corpus: dict, split: str = "train") -> None:
        codes = corpus["codes"]  # (N, n_slots, n_codebooks)
        self.codes = codes.long()
        self.n = len(self.codes)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.codes[idx]  # (n_slots, n_codebooks)


def collate_packets(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)  # (B, n_slots, n_codebooks)


@torch.no_grad()
def evaluate_lm(model: NativeMSLLM, loader: DataLoader, device: torch.device,
                n_batches: int = 10) -> dict[str, float]:
    model.eval()
    losses, exacts, per_code = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = batch.to(device)
        out = model.loss(batch)
        losses.append(out["loss"].item())
        exacts.append(out["exact_acc"].item())
        per_code.append(out["per_code_acc"].item())
    model.train()
    return {"loss": sum(losses) / max(len(losses), 1),
            "exact_acc": sum(exacts) / max(len(exacts), 1),
            "per_code_acc": sum(per_code) / max(len(per_code), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/msl_corpus_final.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    corpus = torch.load(args.corpus, weights_only=False)
    n_codebooks = corpus["n_codebooks"]
    codebook_size = corpus["codec_cfg"]["codebook_size"]
    n_slots = corpus["n_slots"]
    print(f"corpus: {corpus['codes'].shape}, {n_codebooks} codebooks, size {codebook_size}, {n_slots} slots")

    ds = PacketSequenceDataset(corpus)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_packets, num_workers=0)
    test_loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_packets, num_workers=0)

    model = NativeMSLLM(
        n_codebooks=n_codebooks, codebook_size=codebook_size,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=n_slots + 1, dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"native MSL LM params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    base_lr = 3e-4
    warmup = 500
    total = args.steps

    def lr_at(step):
        if step < warmup:
            return base_lr * step / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    print(f"training {total} steps")
    model.train()
    t0 = time.time()
    step = 0
    running_loss, running_acc, running_n = 0.0, 0.0, 0
    while step < total:
        for batch in loader:
            if step >= total:
                break
            batch = batch.to(device)
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = model.loss(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += out["loss"].item()
            running_acc += out["per_code_acc"].item()
            running_n += 1
            step += 1
            if step % 500 == 0:
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running_loss/running_n:.4f} | "
                      f"code_acc {running_acc/running_n:.3f} | lr {lr:.2e} | {el*1000:.0f}ms/step",
                      flush=True)
                running_loss, running_acc, running_n = 0.0, 0.0, 0
    print(f"done in {time.time()-t0:.1f}s")

    metrics = evaluate_lm(model, test_loader, device, n_batches=20)
    print(f"eval: {metrics}")

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "native_lm_0.pt"
    torch.save({"model": model.state_dict(), "n_params": n_params, "metrics": metrics,
                "corpus_path": args.corpus}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
