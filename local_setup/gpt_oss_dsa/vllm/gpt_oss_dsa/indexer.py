"""Serving twin of megatron.core's `DSAIndexer` as built by the `dsa_gqa` module spec.

Per layer: `wq` projects the layer's normalized input to `n_heads x head_dim` queries, `wk` to one
`head_dim` key normalized by a LayerNorm, both roped over the whole head, both Hadamard-rotated,
both quantized to row-wise E4M3 with UE8M0 scales; `weights_proj` gives per-head gates, scaled by
`n_heads**-0.5 * head_dim**-0.5`. The score of key `s` for query `t` is
`sum_h w[t, h] * relu(q[t, h] . k[s])` and the attend reads the top `index_topk` keys.

`project` and `torch_scores` follow the training module operation for operation, in bf16 where it
computes in bf16, so they double as the parity reference. `forward` hands the same q, k and gates
to vLLM's DeepSeek-V3.2 indexer kernels, which need `head_dim == 128` and a head count in
{32, 64, 128}: q and k are zero-padded 64 -> 128 and the heads 16 -> 32 with zero gates. The pad is
exact: appended zeros change neither the dot product nor the row absmax that sets the UE8M0 scale.

Imports vLLM only inside `_build_runtime_op`, so this module also loads inside the Megatron
container for the parity check against the training indexer.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSAServingConfig
from .rotary import MegatronRotary, apply_rope

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
KERNEL_HEAD_DIM = 128
KERNEL_QUANT_BLOCK = 128
KERNEL_HEAD_COUNTS = (32, 64, 128)
KERNEL_SCALE_FMT = "ue8m0"

# Resolved at import: an `import` inside a Dynamo-traced region is an unconditional graph break.
try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_cuda
except ImportError:  # pragma: no cover - environment dependent
    _hadamard_cuda = None

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8 as _vllm_group_quant,
    )
except ImportError:  # pragma: no cover - environment dependent
    _vllm_group_quant = None

HADAMARD_IMPL = "fast_hadamard_transform" if _hadamard_cuda is not None else "torch_fwht"


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized fast Walsh-Hadamard transform over a power-of-two last dim."""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard transform needs a power-of-two last dim, got {n}")
    y = x.clone()
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.cat([a + b, a - b], dim=-1).reshape(*x.shape[:-1], n)
        h *= 2
    return y


@torch.library.custom_op("gpt_oss_dsa::rotate_activation", mutates_args=())
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard rotation of the last dim, megatron's `rotate_activation`.

    A custom op so Dynamo treats it as opaque: the CUDA kernel is an autograd.Function over a C
    extension that Dynamo cannot trace but a CUDA graph can record. The kernel and the torch
    fallback agree only up to summation order, and the rotation feeds fp8 quantization, so which
    one runs is part of the train/serve contract; training uses the kernel.
    """
    scale = x.shape[-1] ** -0.5
    if x.is_cuda and _hadamard_cuda is not None:
        with torch.no_grad():
            return _hadamard_cuda(x.to(torch.bfloat16), scale=scale).to(x.dtype).clone()
    return (_fwht(x.float()) * scale).to(x.dtype).clone()


@rotate_activation.register_fake
def _rotate_activation_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def fake_quant_fp8(x: torch.Tensor, use_ue8m0: bool = True) -> torch.Tensor:
    """Row-wise E4M3 round trip in x's dtype, the forward value of megatron's `_fake_quant_fp8`."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    normalized = (x / scale).clamp(-FP8_MAX, FP8_MAX)
    return (normalized.to(FP8_DTYPE).float() * scale).to(x.dtype)


def quant_fp8_rows(x: torch.Tensor, use_ue8m0: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3 quantization -> (fp8 values, fp32 scales). vLLM's kernel when importable."""
    if _vllm_group_quant is not None and x.is_cuda:
        x_fp8, scale = _vllm_group_quant(
            x, x.shape[-1], column_major_scales=False, use_ue8m0=use_ue8m0
        )
        return x_fp8, scale.reshape(x.shape[0])
    x = x.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / FP8_MAX
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale.reshape(x.shape[0])


class GptOssDSAServingIndexer(nn.Module):
    """One layer's lightning indexer. Parameter names match the exported checkpoint 1:1."""

    def __init__(
        self,
        cfg: DSAServingConfig,
        hidden_size: int,
        *,
        vllm_config=None,
        cache_config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
        allow_torch_hadamard: bool | None = None,
    ) -> None:
        super().__init__()
        if not cfg.fp8:
            raise RuntimeError(
                "the serving indexer kernels are fp8; a checkpoint trained with "
                "dsa_indexer_fp8=False would select differently at serve time"
            )
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.padded_n_heads = next(h for h in KERNEL_HEAD_COUNTS if h >= cfg.n_heads)
        self.wq = nn.Linear(hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, cfg.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(cfg.head_dim, eps=cfg.k_norm_eps)
        self.weights_proj = nn.Linear(hidden_size, cfg.n_heads, bias=False)
        self.rotary = MegatronRotary(cfg.rope)

        if allow_torch_hadamard is None:
            allow_torch_hadamard = os.environ.get("GPT_OSS_DSA_ALLOW_TORCH_HADAMARD") == "1"
        if cfg.rotate_activation and _hadamard_cuda is None and not allow_torch_hadamard:
            raise RuntimeError(
                "fast_hadamard_transform is not importable, but the indexer was trained with it "
                "(dsa_indexer_rotate_activation=True). The torch fallback perturbs the fp8 "
                "quantization of q and k and therefore the selected keys. Install it into the "
                "serving venv (README.md), or set GPT_OSS_DSA_ALLOW_TORCH_HADAMARD=1 to accept "
                "the drift deliberately."
            )

        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = cfg.index_topk
        self.indexer_op = None
        self.k_cache = None
        if vllm_config is not None:
            self._build_runtime_op(vllm_config, cache_config, topk_indices_buffer, prefix)

    def _build_runtime_op(self, vllm_config, cache_config, topk_indices_buffer, prefix) -> None:
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        if topk_indices_buffer is None:
            raise ValueError("the serve path needs the shared top-k buffer")
        # fp8 side cache for k: 128 value bytes plus one 4-byte UE8M0 scale per 128-block.
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=KERNEL_HEAD_DIM + KERNEL_HEAD_DIM // KERNEL_QUANT_BLOCK * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            KERNEL_QUANT_BLOCK,
            KERNEL_SCALE_FMT,
            self.cfg.index_topk,
            KERNEL_HEAD_DIM,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
        )

    def project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[T, hidden] -> (q [T, n_heads, d], k [T, d], gates [T, n_heads]), all in the model dtype.

        Same operations, order and dtypes as `DSAIndexer.forward_before_topk`: project, k-norm,
        rope in the activation dtype, Hadamard, and gates scaled by two separate multiplies.
        """
        x = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = x.shape[0]
        cos, sin = self.rotary(positions.reshape(-1))
        q = apply_rope(self.wq(x).view(num_tokens, self.n_heads, self.head_dim), cos, sin)
        k = self.k_norm(self.wk(x)).view(num_tokens, 1, self.head_dim)
        k = apply_rope(k, cos, sin).view(num_tokens, self.head_dim)
        if self.cfg.rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)
        gates = self.weights_proj(x) * (self.n_heads**-0.5) * (self.head_dim**-0.5)
        return q, k, gates

    def _pad_for_kernel(
        self, q: torch.Tensor, k: torch.Tensor, gates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_d = KERNEL_HEAD_DIM - self.head_dim
        if pad_d:
            q = F.pad(q, (0, pad_d))
            k = F.pad(k, (0, pad_d))
        pad_h = self.padded_n_heads - self.n_heads
        if pad_h:
            q = F.pad(q, (0, 0, 0, pad_h))
            gates = F.pad(gates, (0, pad_h))
        return q, k, gates

    @torch.no_grad()
    def torch_scores(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[T, T] fp32 causal index scores for one sequence, computed as training computes them.

        Needs no engine, KV cache or DeepGEMM; this is what the kernels are checked against.
        """
        q, k, gates = self.project(hidden_states, positions)
        if self.cfg.fp8:
            q = fake_quant_fp8(q, self.cfg.fp8_ue8m0)
            k = fake_quant_fp8(k, self.cfg.fp8_ue8m0)
        dots = torch.einsum("qhd,kd->qhk", q.float(), k.float())
        if self.cfg.scoring_relu:
            dots = torch.relu(dots)
        scores = torch.einsum("qhk,qh->qk", dots, gates.float())
        pos = positions.reshape(-1)
        return scores.masked_fill(pos.unsqueeze(0) > pos.unsqueeze(1), float("-inf"))

    @staticmethod
    def torch_topk(scores: torch.Tensor, index_topk: int) -> torch.Tensor:
        """Training's top-k: [T, min(k, T)] int64 key indices, -1 where the score is -inf."""
        k = min(index_topk, scores.shape[-1])
        values, indices = scores.topk(k, dim=-1)
        return indices.masked_fill(values == float("-inf"), -1)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Serve path: writes the selected key indices into the shared top-k buffer."""
        if self.indexer_op is None:
            raise RuntimeError(
                "serve path needs a live vLLM engine (construct with vllm_config=...); "
                "engine-free callers use torch_scores()"
            )
        q, k, gates = self.project(hidden_states, positions)
        q, k, gates = self._pad_for_kernel(q, k, gates)
        q_fp8, q_scale = quant_fp8_rows(
            q.reshape(-1, KERNEL_HEAD_DIM).contiguous(), self.cfg.fp8_ue8m0
        )
        q_fp8 = q_fp8.view(-1, self.padded_n_heads, KERNEL_HEAD_DIM)
        q_scale = q_scale.view(-1, self.padded_n_heads)
        # The kernel dequantizes k with the scale stored beside it in the side cache; q's per-row
        # scale is a positive scalar per (token, head) and pulls straight out of the ReLU, so it
        # folds into the gate. The gate already carries n_heads**-0.5 * head_dim**-0.5.
        fused_gates = (gates.float() * q_scale).contiguous()
        return self.indexer_op(hidden_states, q_fp8, k.contiguous(), fused_gates)
