"""Seeding utilities for reproducibility.

All random number generation in the project MUST go through this module so that
runs are reproducible and tests are deterministic. MS-1 test states use seeds
>= 2_000_000 (anti-leak convention, see AGENTS.md).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

TEST_SEED_FLOOR = 2_000_000


@dataclass(frozen=True)
class SeedBundle:
    """Container holding independent seeds for each RNG backend."""

    root: int
    python: int
    numpy: int
    torch: int
    torch_cuda: int

    @classmethod
    def from_root(cls, root: int) -> SeedBundle:
        # Derive independent sub-seeds deterministically from the root.
        rng = np.random.default_rng(root)
        py, np_, t, tc = (int(rng.integers(0, 2**63)) for _ in range(4))
        return cls(root=root, python=py, numpy=np_, torch=t, torch_cuda=tc)


def seed_everything(seed: int) -> SeedBundle:
    """Seed python, numpy and torch RNGs. Returns the derived bundle."""
    bundle = SeedBundle.from_root(seed)
    random.seed(bundle.python)
    np.random.seed(bundle.numpy & (2**32 - 1))  # noqa: NPY002
    torch.manual_seed(bundle.torch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bundle.torch_cuda)
    os.environ["PYTHONHASHSEED"] = str(bundle.python)
    return bundle


def default_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
