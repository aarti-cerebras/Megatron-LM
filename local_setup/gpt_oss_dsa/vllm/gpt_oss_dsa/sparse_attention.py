"""Token-granular sparse attention for GQA with attention sinks, as a vLLM CUSTOM backend.

The attend is one FA3 varlen call over a page-size-1 view of the standard KV cache, the way vLLM's
own MLA sparse backend does it (`v1/attention/backends/mla/flashattn_mla_sparse.py`): every query
token is its own length-1 sequence, the indexer's selected keys are its block table, and
`seqused_k` truncates each row to the keys the indexer actually filled. GQA is the symmetric path
(no `q_v`), and the GPT-OSS sink enters through FA3's `s_aux`, exactly as stock GptOss attention
passes it. Causality comes from the selection, not from the kernel mask.

Registered into `AttentionBackendEnum.CUSTOM`, the slot for out-of-tree backends, which lets
`Attention(..., attn_backend=...)` reuse vLLM's KV-cache allocation, forward-context registration,
cache-write op and cudagraph plumbing unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec

# The indexer's paged-logits kernel and the top-k -> slot conversion assume this block size.
DSA_KERNEL_BLOCK_SIZE = 64

# GPT_OSS_DSA_DEBUG_SELECTION=N prints, for the first N attention calls, how many keys each query
# actually reads. Forces a device sync per call, so it needs --enforce-eager.
_DEBUG_SELECTION = int(os.environ.get("GPT_OSS_DSA_DEBUG_SELECTION", "0"))
# GPT_OSS_DSA_CHECK_SELECTION=1 raises if any query attends over fewer keys than
# min(position + 1, top-k), which the indexer must always select. Syncs per call; eager only.
_CHECK_SELECTION = os.environ.get("GPT_OSS_DSA_CHECK_SELECTION") == "1"


def page_size_1_view(cache: torch.Tensor) -> torch.Tensor:
    """[num_blocks, block_size, H_kv, D] -> [num_blocks * block_size, 1, H_kv, D], no copy.

    vLLM packs K and V into the trailing dim, so the split halves have a head stride of `2 * D`
    and `.view()` cannot merge the first two dims; `as_strided` expresses the same
    reinterpretation, and FA3 accepts it.
    """
    nb, bs, h, d = cache.shape
    s = cache.stride()
    return torch.as_strided(cache, (nb * bs, 1, h, d), (s[1], s[1], s[2], s[3]))


class GptOssDSASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    # fp8 KV under a page-size-1 view is untested and would fail as wrong numbers, not an error.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16", "float16"]

    @staticmethod
    def get_name() -> str:
        # `Attention.__init__` resolves `AttentionBackendEnum[get_name()]`.
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [DSA_KERNEL_BLOCK_SIZE]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128]

    @staticmethod
    def get_impl_cls() -> type["GptOssDSASparseImpl"]:
        return GptOssDSASparseImpl

    @staticmethod
    def get_builder_cls() -> type["GptOssDSASparseMetadataBuilder"]:
        return GptOssDSASparseMetadataBuilder

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FA3 with sinks is Hopper-only here; Blackwell's sparse path goes through other kernels.
        return capability.major == 9

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Identical to FlashAttentionBackend: K and V packed into the content dim.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        if cache_layout == "NHD":
            return (0, 2, 1, 3)
        if cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 0, 3, 4)
        if cache_layout == "HND":
            return (0, 1, 2, 3)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")


register_backend(AttentionBackendEnum.CUSTOM, f"{__name__}.GptOssDSASparseBackend")


@dataclass
class GptOssDSASparseMetadata(AttentionMetadata):
    num_actual_tokens: int
    block_table: torch.Tensor  # int32 [num_reqs, max_blocks_per_req]
    req_id_per_token: torch.Tensor  # int32 [num_tokens]
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor  # int32 [num_reqs], including this step's query tokens
    query_start_loc: torch.Tensor  # int32 [num_reqs + 1]
    block_size: int = DSA_KERNEL_BLOCK_SIZE


class GptOssDSASparseMetadataBuilder(AttentionMetadataBuilder[GptOssDSASparseMetadata]):
    """Every token is one length-1 query sequence, so there is no prefill/decode split."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        if self.block_size != DSA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                f"GPT-OSS DSA requires --block-size {DSA_KERNEL_BLOCK_SIZE}, got {self.block_size}"
            )
        # Persistent, filled per step, so `build` allocates nothing on the hot path.
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _req_id_per_token(self, common: CommonAttentionMetadata) -> torch.Tensor:
        starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
        seg = np.diff(starts)
        rep = np.repeat(np.arange(seg.shape[0], dtype=np.int32), seg)
        n = rep.shape[0]
        self.req_id_per_token_buffer[:n].copy_(
            torch.from_numpy(rep).pin_memory(), non_blocking=True
        )
        return self.req_id_per_token_buffer[: common.num_actual_tokens]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GptOssDSASparseMetadata:
        return GptOssDSASparseMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=self._req_id_per_token(common_attn_metadata),
            slot_mapping=common_attn_metadata.slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            block_size=self.block_size,
        )


class GptOssDSASparseImpl(AttentionImpl):
    """FA3 varlen over the indexer's selected tokens, with the layer's sink logits.

    The selection arrives through `topk_indices_buffer`, a preallocated tensor the indexer writes
    and this impl reads: on CUDA vLLM wraps attention in an opaque custom op, so the model cannot
    pass the indices as an argument.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "GptOssDSASparseImpl does not support alibi, sliding window or logit soft cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"Unsupported attention type {attn_type}")
        if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
            raise NotImplementedError(
                f"GptOssDSASparseImpl supports only bf16/fp16 KV cache, got {kv_cache_dtype!r}"
            )
        if topk_indices_buffer is None:
            raise ValueError(
                "GptOssDSASparseImpl requires topk_indices_buffer: without the indexer's "
                "selection there is nothing sparse to attend over, and falling back to dense "
                "would serve a different model than the config describes"
            )
        if sinks is not None and sinks.shape[0] != num_heads:
            raise ValueError(f"sinks has {sinks.shape[0]} entries for {num_heads} heads")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = topk_indices_buffer.shape[-1]

    _dbg_calls: int = 0

    @torch.no_grad()
    def _log_selection(self, topk_indices, topk_slots, valid_counts, md) -> None:
        vc = valid_counts.to(torch.int64)
        sel = (topk_indices >= 0).sum(dim=1)
        seq_max = int(md.seq_lens.max().item()) if md.seq_lens is not None else -1
        print(
            f"[GptOssDSA-SELECT] tokens={topk_indices.shape[0]} width={topk_indices.shape[1]} "
            f"seqused_k(min/mean/max)={int(vc.min())}/{float(vc.float().mean()):.1f}/{int(vc.max())} "
            f"indexer_selected(min/max)={int(sel.min())}/{int(sel.max())} max_seq_len={seq_max} "
            f"density={int(vc.max()) / max(seq_max, 1):.3f} "
            f"buffer_all_negative={bool((topk_indices < 0).all().item())}",
            flush=True,
        )

    @torch.no_grad()
    def _check_selection(self, valid_counts, md, num_actual_tokens: int) -> None:
        """Every query has min(position + 1, top-k) legal keys and the indexer selects all of them."""
        starts = md.query_start_loc.to(torch.int64)
        seq_lens = md.seq_lens.to(torch.int64)
        req = md.req_id_per_token[:num_actual_tokens].to(torch.int64)
        query_lens = starts[1:] - starts[:-1]
        token = torch.arange(num_actual_tokens, device=req.device)
        positions = seq_lens[req] - query_lens[req] + (token - starts[req])
        expected = torch.clamp(positions + 1, max=self.topk_tokens)
        got = valid_counts[:num_actual_tokens].to(torch.int64)
        bad = (got != expected).nonzero().flatten()
        if bad.numel():
            head = bad[:8]
            raise RuntimeError(
                f"[GptOssDSA] {bad.numel()} of {num_actual_tokens} queries attend over the wrong "
                f"number of keys: tokens {head.tolist()} expected {expected[head].tolist()} got "
                f"{got[head].tolist()} (positions {positions[head].tolist()}, top-k {self.topk_tokens})"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # (num_blocks, num_kv_heads, block_size, 2 * head_size) logical
        attn_metadata: GptOssDSASparseMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads * head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported")
        if attn_metadata is None:
            return output.fill_(0)  # profiling / dummy run

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # Per-request token indices from the indexer -> global cache slots, plus each row's valid
        # count. The kernel is page-size generic; nothing about it is MLA-specific.
        topk_indices = self.topk_indices_buffer[:num_actual_tokens]
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        if _DEBUG_SELECTION and self._dbg_calls < _DEBUG_SELECTION:
            self._dbg_calls += 1
            self._log_selection(topk_indices, topk_slots, valid_counts, attn_metadata)
        if _CHECK_SELECTION:
            self._check_selection(valid_counts, attn_metadata, num_actual_tokens)

        cu_seqlens_q = torch.arange(
            0, num_actual_tokens + 1, dtype=torch.int32, device=query.device
        )
        out_view = output.view(-1, self.num_heads, self.head_size)
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=page_size_1_view(key_cache),
            v=page_size_1_view(value_cache),
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_slots,
            softmax_scale=self.scale,
            # One query per sequence: bottom-right causal alignment lets it see all `seqused_k`
            # selected keys. Causality is already in the selection.
            causal=True,
            fa_version=3,
            s_aux=self.sinks,
            out=out_view[:num_actual_tokens],
        )
        return output
