"""H1 sweep: train codecs across n_slots × quantizers × seeds.

Measures how task accuracy and the noise-ablation gap vary with the number of
packet slots (the L(s) proxy). Results are written incrementally to a CSV so
the sweep is interrupt-safe.

Usage:
    python -u -m msl.eval.sweep_h1 --steps 3000 --out runs/sweep_h1.csv

The CSV columns:
    n_slots, quantizer, seed, bits_per_packet,
    train_acc, train_acc_noise, train_gap,
    test_acc, test_acc_noise, test_gap, test_loss, n_params, runtime_s
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

from msl.train.train_codec import train_one_run


def base_cfg(n_slots: int, quantizer: str, n_codebooks: int = 8,
             codebook_size: int = 1024, levels: int = 5) -> dict:
    return {
        "name": f"sweep_h1_{quantizer}_n{n_slots}",
        "output": "runs",
        "data": {"min_k": 2, "max_k": 32, "tasks_per_state": 1, "n_views": 2},
        "codec": {
            "d_model": 128, "n_layers": 4, "n_heads": 4, "d_z": 64,
            "n_slots": n_slots,
            "quantizer_kind": quantizer,
            "n_codebooks": n_codebooks,
            "codebook_size": codebook_size,
            "levels": levels,
            "w_reconstruction": 0.3, "w_alignment": 0.2,
            "w_bits": 0.0001, "w_commit": 1.0, "dropout": 0.0,
        },
        "train": {
            "dataset_size": 16384, "batch_size": 64, "warmup": 300,
            "lr": 0.0003, "log_every": 500, "precompute": True,
            "test_dataset_size": 1024,
        },
    }


def bits_for(quantizer: str, n_codebooks: int, codebook_size: int, levels: int) -> float:
    if quantizer in ("pq", "rvq"):
        return n_codebooks * math.log2(codebook_size)
    return n_codebooks * math.log2(levels)


CSV_FIELDS = [
    "n_slots", "quantizer", "seed", "bits_per_packet",
    "train_acc", "train_acc_noise", "train_gap",
    "test_acc", "test_acc_noise", "test_gap", "test_loss",
    "n_params", "runtime_s",
]


def already_done(rows: list[dict], n_slots: int, quantizer: str, seed: int) -> bool:
    return any(int(r["n_slots"]) == n_slots and r["quantizer"] == quantizer
               and int(r["seed"]) == seed for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="runs/sweep_h1.csv")
    ap.add_argument("--n_slots", default="1,2,4,8")
    ap.add_argument("--quantizers", default="pq,rvq,fsq")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()

    n_slots_list = [int(x) for x in args.n_slots.split(",")]
    quantizers = args.quantizers.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing_rows = list(csv.DictReader(f))
    total = len(n_slots_list) * len(quantizers) * len(seeds)
    done = 0
    print(f"sweep H1: {total} runs planned, {len(existing_rows)} already done")
    write_header = not os.path.exists(args.out) or len(existing_rows) == 0

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        for n_slots in n_slots_list:
            for quantizer in quantizers:
                qb = quantizer.strip()
                for seed in seeds:
                    if already_done(existing_rows, n_slots, qb, seed):
                        done += 1
                        print(f"[{done}/{total}] skip n_slots={n_slots} {qb} seed={seed} (done)")
                        continue
                    cfg = base_cfg(n_slots, qb)
                    bits = bits_for(qb, 8, 1024, 5)
                    t0 = time.time()
                    print(f"[{done+1}/{total}] n_slots={n_slots} {qb} seed={seed} "
                          f"({bits:.1f} bits/pkt) ...", flush=True)
                    result = train_one_run(cfg, seed, args.steps, verbose=False)
                    runtime = time.time() - t0
                    tm, tr = result["test_metrics"], result["train_metrics"]
                    row = {
                        "n_slots": n_slots, "quantizer": qb, "seed": seed,
                        "bits_per_packet": f"{bits:.2f}",
                        "train_acc": f"{tr['task_acc']:.4f}",
                        "train_acc_noise": f"{tr['task_acc_noise']:.4f}",
                        "train_gap": f"{tr['task_acc'] - tr['task_acc_noise']:.4f}",
                        "test_acc": f"{tm['task_acc']:.4f}",
                        "test_acc_noise": f"{tm['task_acc_noise']:.4f}",
                        "test_gap": f"{tm['task_acc'] - tm['task_acc_noise']:.4f}",
                        "test_loss": f"{tm['eval_loss']:.4f}",
                        "n_params": result["n_params"],
                        "runtime_s": f"{runtime:.1f}",
                    }
                    writer.writerow(row)
                    f.flush()
                    done += 1
                    print(f"  -> test_acc={tm['task_acc']:.3f} "
                          f"test_noise={tm['task_acc_noise']:.3f} "
                          f"gap={tm['task_acc']-tm['task_acc_noise']:+.3f} "
                          f"({runtime:.0f}s)", flush=True)
    print(f"sweep done: {done}/{total} runs, results in {args.out}")


if __name__ == "__main__":
    main()
