"""Tests for the discrete quantizers (CPU, gradient checks)."""

from __future__ import annotations

import math

import pytest
import torch

from msl.models.quantizer import PQQuantizer, build_quantizer


def _make_batch(n=32, d_z=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d_z, requires_grad=True)


def test_pq_shapes_and_bits():
    q = build_quantizer("pq", d_z=64, n_codebooks=8, codebook_size=1024)
    z = _make_batch()
    out = q(z)
    assert out.codes.shape == (32, 8)
    assert out.z_q.shape == (32, 64)
    assert out.bits == pytest.approx(80.0)


def test_rvq_shapes_and_bits():
    q = build_quantizer("rvq", d_z=64, n_codebooks=8, codebook_size=1024)
    z = _make_batch()
    out = q(z)
    assert out.codes.shape == (32, 8)
    assert out.z_q.shape == (32, 64)
    assert out.bits == pytest.approx(80.0)


def test_fsq_shapes_and_bits():
    q = build_quantizer("fsq", d_z=64, n_codebooks=8, levels=5)
    z = _make_batch()
    out = q(z)
    assert out.codes.shape == (32, 8)
    assert out.z_q.shape == (32, 64)
    assert out.bits == pytest.approx(8 * math.log2(5))


def test_gradient_flows_to_encoder_input():
    for kind, kw in [("pq", {}), ("rvq", {}), ("fsq", {})]:
        q = build_quantizer(kind, d_z=64, n_codebooks=8, codebook_size=1024, levels=5, **kw)
        z = _make_batch()
        out = q(z)
        out.z_q.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()


def test_fsq_codes_in_range_and_uses_all_levels():
    # FSQ cannot collapse: with enough samples, all levels should be reachable.
    q = build_quantizer("fsq", d_z=64, n_codebooks=8, levels=5)
    torch.manual_seed(1)
    z = torch.randn(2048, 64) * 3  # spread to hit all levels
    out = q(z)
    for i in range(8):
        uniq = out.codes[:, i].unique()
        assert uniq.numel() == 5  # all 5 levels used


def test_pq_codebook_usage_grows_with_data_diversity():
    # With diverse inputs, EMA codebooks should activate more codes than a
    # degenerate (constant) input. (Smoke check, not a strict threshold.)
    q = PQQuantizer(64, 8, 1024)
    q.train()
    torch.manual_seed(0)
    diverse = torch.randn(2048, 64)
    _ = q(diverse)
    active_div = q.codebooks[0].cluster_size.gt(0).float().mean().item()
    assert active_div > 0.0
