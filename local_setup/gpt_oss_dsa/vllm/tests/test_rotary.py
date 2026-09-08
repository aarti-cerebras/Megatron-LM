"""Rotary tables: the YaRN blend, the concentration factor, and the dtype of the application."""

import math

import pytest
import torch

from gpt_oss_dsa.config import IndexerRope
from gpt_oss_dsa.rotary import MegatronRotary, apply_rope, concentration_factor, inverse_frequencies

GPT_OSS = dict(dim=64, theta=150000.0, scaling_factor=32.0, original_max_position_embeddings=4096)


def test_yarn_at_factor_one_is_plain_rope():
    plain = inverse_frequencies(IndexerRope(type="rope", dim=64, theta=150000.0))
    yarn = inverse_frequencies(IndexerRope(type="yarn", dim=64, theta=150000.0, scaling_factor=1.0))
    # The blend `inter * (1 - m) + extra * m` with inter == extra is equal up to fp32 rounding,
    # in megatron as here.
    assert torch.allclose(plain, yarn, rtol=1e-6, atol=0)
    assert concentration_factor(IndexerRope(type="yarn", dim=64, theta=150000.0, scaling_factor=1.0)) == 1.0


def test_gpt_oss_concentration_factor():
    rope = IndexerRope(type="yarn", **GPT_OSS)
    assert concentration_factor(rope) == pytest.approx(0.1 * math.log(32.0) + 1.0)


def test_yarn_blend_is_between_extrapolation_and_interpolation():
    rope = IndexerRope(type="yarn", **GPT_OSS)
    extra = inverse_frequencies(IndexerRope(type="rope", dim=64, theta=150000.0))
    blended = inverse_frequencies(rope)
    assert torch.all(blended <= extra + 1e-12)
    assert torch.all(blended >= extra / 32.0 - 1e-12)
    # High frequencies extrapolate untouched, the lowest interpolate fully.
    assert blended[0] == extra[0]
    assert blended[-1] == pytest.approx(float(extra[-1]) / 32.0, rel=1e-6)


def test_round_to_int_changes_the_ramp():
    rounded = inverse_frequencies(IndexerRope(type="yarn", correction_range_round_to_int=True, **GPT_OSS))
    exact = inverse_frequencies(IndexerRope(type="yarn", correction_range_round_to_int=False, **GPT_OSS))
    assert not torch.equal(rounded, exact)


def test_tables_survive_a_module_dtype_cast():
    """`.to(dtype=bfloat16)` on the owning module must not round the frequencies; at 1024
    positions that rounding moves cos by up to 1.9, which is how the first parity run failed."""
    rope = IndexerRope(type="yarn", **GPT_OSS)
    positions = torch.arange(1024)
    reference, _ = MegatronRotary(rope)(positions)
    cast, _ = MegatronRotary(rope).to(dtype=torch.bfloat16)(positions)
    assert cast.dtype == torch.float32
    assert torch.equal(reference, cast)


def test_apply_rope_computes_in_activation_dtype_and_preserves_norm():
    rope = IndexerRope(type="yarn", **GPT_OSS)
    rotary = MegatronRotary(rope)
    positions = torch.arange(37)
    cos, sin = rotary(positions)
    assert cos.dtype == torch.float32 and cos.shape == (37, 64)
    x = torch.randn(37, 3, 64).to(torch.bfloat16)
    y = apply_rope(x, cos, sin)
    assert y.dtype == torch.bfloat16
    # A rotation scaled by the concentration factor scales every vector's norm by it.
    ratio = y.float().norm(dim=-1) / x.float().norm(dim=-1)
    assert torch.allclose(ratio, torch.full_like(ratio, concentration_factor(rope)), rtol=2e-2)
