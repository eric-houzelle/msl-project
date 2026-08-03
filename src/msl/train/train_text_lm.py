"""Text LM jumeau: same task, same size, but predicts text tokens instead of packets.

For a fair comparison with the native MSL LM:
- Same Transformer backbone (6 layers, d_model=256, 8 heads)
- Same training data (MS-1 states, same 16384 samples)
- Same training (5000 steps, same lr)
- But predicts next TEXT TOKEN instead of next PACKET

The key difference: MSL predicts 8 codes in 1 step (1 autoregressive step per packet),
while text predicts 1 token per step (many steps per sentence).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from msl.data.dataloader import MS1Config, PrecomputedMS1Dataset, default_collate
from msl.models.tokenizer import default_tokenizer
from msl.utils.seeding import default_device, seed_everything


class TextLM(nn.Module):
    """Standard text language model: predict next token."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, max_len: int = 96, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       batch_first=True, activation="gelu", dropout=dropout),
            n_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        h = self.embed(token_ids) + self.pos(torch.arange(T, device=token_ids.device)).unsqueeze(0)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=token_ids.device), diagonal=1)
        h = self.transformer(h, mask=causal)
        return self.head(h)

    def loss(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(token_ids)[:, :-1]
        targets = token_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        acc = (logits.argmax(-1) == targets).float().mean()
        return {"loss": loss, "token_acc": acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = default_device()
    print(f"device: {device}")

    tok = default_tokenizer()
    ms1_cfg = MS1Config(min_k=2, max_k=32, tasks_per_state=1, n_views=2)
    ds = PrecomputedMS1Dataset(tok, ms1_cfg, size=16384, seed_floor=0)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=default_collate, num_workers=0)

    model = TextLM(vocab_size=len(tok.itos), d_model=256, n_layers=6, n_heads=8,
                   max_len=96, dropout=0.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"text LM params: {n_params:,}")

    base_lr = 3e-4
    warmup = 500
    total = args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    print(f"training {total} steps")
    model.train()
    t0 = time.time()
    step = 0
    running_loss, running_acc, running_n = 0.0, 0.0, 0
    while step < total:
        for batch in loader:
            if step >= total:
                break
            token_ids = batch["text_ids"].to(device)
            if step < warmup:
                lr = base_lr * step / warmup
            else:
                progress = (step - warmup) / max(total - warmup, 1)
                lr = base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            out = model.loss(token_ids)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += out["loss"].item()
            running_acc += out["token_acc"].item()
            running_n += 1
            step += 1
            if step % 500 == 0:
                el = (time.time() - t0) / step
                print(f"step {step:5d} | loss {running_loss/running_n:.4f} | "
                      f"token_acc {running_acc/running_n:.3f} | lr {lr:.2e} | {el*1000:.0f}ms/step",
                      flush=True)
                running_loss, running_acc, running_n = 0.0, 0.0, 0
    print(f"done in {time.time()-t0:.1f}s")

    # Final eval
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 20:
                break
            token_ids = batch["text_ids"].to(device)
            out = model.loss(token_ids)
            losses.append(out["loss"].item())
            accs.append(out["token_acc"].item())
    print(f"eval: loss={sum(losses)/len(losses):.4f} token_acc={sum(accs)/len(accs):.3f}")

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "text_lm_0.pt"
    torch.save({"model": model.state_dict(), "n_params": n_params}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
