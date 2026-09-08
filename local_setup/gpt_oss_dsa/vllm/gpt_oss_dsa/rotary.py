"""The indexer's rotary embedding, computed the way megatron.core computes it.

Tables follow `RotaryEmbedding` / `YarnRotaryEmbedding.get_emb` (fp32 inverse frequencies, fp32
outer product, the concentration factor folded into cos and sin) and application follows
`_apply_rotary_pos_emb_bshd` (cos and sin cast to the activation dtype, NeoX half rotation, the
product taken in the activation dtype). The indexer trains in bf16, so keeping those casts is
what makes the serving scores agree with training rather than merely approximate it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import IndexerRope


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(rope: IndexerRope) -> tuple[float, float]:
    low = _yarn_correction_dim(
        rope.beta_fast, rope.dim, rope.theta, rope.original_max_position_embeddings
    )
    high = _yarn_correction_dim(
        rope.beta_slow, rope.dim, rope.theta, rope.original_max_position_embeddings
    )
    if rope.correction_range_round_to_int:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, rope.dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def _yarn_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def concentration_factor(rope: IndexerRope) -> float:
    """Factor multiplied into cos and sin; megatron calls it the concentration factor or mscale."""
    if rope.type != "yarn":
        return 1.0
    return _yarn_mscale(rope.scaling_factor, rope.mscale) / _yarn_mscale(
        rope.scaling_factor, rope.mscale_all_dim
    )


def inverse_frequencies(rope: IndexerRope) -> torch.Tensor:
    """fp32 [dim // 2] inverse frequencies, YaRN-blended when `rope.type == "yarn"`."""
    exponents = torch.arange(0, rope.dim, 2, dtype=torch.float32) / rope.dim
    inv_freq_extra = 1.0 / (rope.theta**exponents)
    if rope.type != "yarn":
        return inv_freq_extra
    inv_freq_inter = 1.0 / (rope.scaling_factor * rope.theta**exponents)
    low, high = _yarn_correction_range(rope)
    inv_freq_mask = 1.0 - _yarn_linear_ramp(low, high, rope.dim // 2)
    return inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask


class MegatronRotary(nn.Module):
    """cos/sin tables for arbitrary positions, matching megatron's cached tables entry for entry.

    The frequencies are a plain attribute, not a buffer: `module.to(dtype=torch.bfloat16)` casts
    every floating buffer, and a bf16 frequency table is wrong by whole periods at long positions.
    """

    def __init__(self, rope: IndexerRope) -> None:
        super().__init__()
        self.rope = rope
        self.mscale = concentration_factor(rope)
        self._inv_freq = inverse_frequencies(rope)
        self._inv_freq_by_device: dict[torch.device, torch.Tensor] = {}

    def inv_freq(self, device: torch.device) -> torch.Tensor:
        """The fp32 [dim // 2] frequencies on `device`, copied there once."""
        table = self._inv_freq_by_device.get(device)
        if table is None:
            table = self._inv_freq.to(device)
            self._inv_freq_by_device[device] = table
        return table

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [T] -> (cos, sin), each fp32 [T, dim] with the concentration factor applied."""
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq(positions.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb) * self.mscale, torch.sin(emb) * self.mscale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t [T, heads, dim] in its own dtype; cos/sin [T, dim] fp32. Computes in t's dtype."""
    cos_ = cos.to(t.dtype).unsqueeze(1)
    sin_ = sin.to(t.dtype).unsqueeze(1)
    return (t * cos_) + (rotate_half(t) * sin_)
