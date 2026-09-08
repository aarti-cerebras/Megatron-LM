"""`GptOssDSAForCausalLM`: stock vLLM GPT-OSS with a DSA indexer and sparse attend on the layers
`config.dsa_sparse_layer_ids` names. Everything else, including the sliding-window layers, the MoE
blocks, the norms, the embeddings and the weight loaders, is the stock GptOss code by inheritance.

Per sparse layer the replaced region is exactly: normalized hidden states in -> attention output.
The indexer runs first on the same normalized input the training indexer saw and writes the
shared top-k buffer; the attend, built with the CUSTOM sparse backend and the layer's own sink
parameter, reads it inside vLLM's opaque attention op.

A dense `Attention` is never built and then replaced: `Attention.__init__` registers itself in the
forward context, and a discarded one would leave vLLM allocating KV cache for a layer that no
longer exists. The sparse layers therefore build their `Attention` inside `_build_attention`, the
hook stock GptOss exposes for exactly this.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import os
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    OAIAttention,
    TransformerBlock,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from .config import DSAServingConfig
from .indexer import GptOssDSAServingIndexer
from .sparse_attention import DSA_KERNEL_BLOCK_SIZE, GptOssDSASparseBackend

INDEXER_LEAVES = ("wq.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight")


class _FlashAttentionBlock64(FlashAttentionBackend):
    """Stock FA3 with its kernel block pinned to the DSA block size.

    vLLM sizes a sliding-window layer's KV blocks from the backend's smallest supported block
    (16 for FlashAttention) and relies on page-size unification to scale it up later. With the
    hybrid KV manager disabled, which this model requires, the sliding-window specs are instead
    converted to full-attention specs before that scaling and must already share the block size of
    every other layer. Pinning the kernel block to 64 gives them that; FA3 runs block 64 unchanged.
    """

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [DSA_KERNEL_BLOCK_SIZE]


@dataclasses.dataclass(frozen=True)
class _BuildContext:
    cfg: DSAServingConfig
    topk_indices_buffer: torch.Tensor


# Set by GptOssDSAModel around its layer construction. Stock TransformerBlock instantiates
# `attention_cls(config, prefix=, quant_config=, cache_config=)` with a fixed signature, so the
# per-model state reaches the attention the same way vLLM's own config does: through a context.
_BUILD: contextvars.ContextVar[_BuildContext | None] = contextvars.ContextVar(
    "gpt_oss_dsa_build", default=None
)


@contextlib.contextmanager
def _building(ctx: _BuildContext):
    token = _BUILD.set(ctx)
    try:
        yield
    finally:
        _BUILD.reset(token)


def _build_context() -> _BuildContext:
    ctx = _BUILD.get()
    if ctx is None:
        raise RuntimeError("GptOssDSAAttention must be constructed by GptOssDSAModel")
    return ctx


class GptOssDSAAttention(OAIAttention):
    """OAIAttention whose full-attention layers, when listed as sparse, attend over the indexer's
    top-k with the sink. Sliding-window layers and unlisted layers are stock."""

    def _build_attention(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> Attention:
        ctx = _build_context()
        self.dsa_sparse = self.layer_idx in ctx.cfg.sparse_layer_ids
        if not self.dsa_sparse:
            # Stock GptOss attention (sliding window on even layers), on FA3 with the kernel
            # block pinned so every layer's KV spec shares one block size.
            return Attention(
                self.num_local_attention_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_local_key_value_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=config.sliding_window if self.layer_idx % 2 == 0 else None,
                attn_type=AttentionType.DECODER,
                prefix=f"{prefix}.attn",
                sinks=self.sinks,
                attn_backend=_FlashAttentionBlock64,
            )
        # Stock GptOss puts the sliding window on even layers; a sparse layer must not be one.
        if self.layer_idx % 2 == 0:
            raise ValueError(f"layer {self.layer_idx} is a sliding-window layer; it cannot be sparse")
        return Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
            attn_backend=GptOssDSASparseBackend,
            topk_indices_buffer=ctx.topk_indices_buffer,
        )

    def __init__(self, config, quant_config=None, cache_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, cache_config=cache_config, prefix=prefix)
        if self.dsa_sparse:
            ctx = _build_context()
            self.indexer = GptOssDSAServingIndexer(
                ctx.cfg,
                config.hidden_size,
                vllm_config=get_current_vllm_config(),
                cache_config=cache_config,
                topk_indices_buffer=ctx.topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.dsa_sparse:
            # Selection first: it writes the buffer self.attn reads inside the opaque attention op.
            self.indexer(hidden_states, positions)
        return super().forward(hidden_states, positions)


class GptOssDSATransformerBlock(TransformerBlock):
    attention_cls = GptOssDSAAttention


@support_torch_compile
class GptOssDSAModel(GptOssModel):
    block_cls = GptOssDSATransformerBlock

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        hf_config = vllm_config.model_config.hf_config
        cfg = DSAServingConfig.from_hf_config(hf_config)
        if cfg.training_phase == 1 and os.environ.get("GPT_OSS_DSA_ALLOW_PHASE1") != "1":
            raise RuntimeError(
                "this serving dir was built from a Phase-1 (dense-warmup) checkpoint, whose "
                "backbone never trained against sparse attention. Serving it sparse is a "
                "plumbing control (with index_topk >= the sequence length it must reproduce "
                "stock GPT-OSS), not an evaluable model. Set GPT_OSS_DSA_ALLOW_PHASE1=1 to serve "
                "it anyway."
            )
        if vllm_config.cache_config.block_size != DSA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                f"GPT-OSS DSA needs --block-size {DSA_KERNEL_BLOCK_SIZE}, got "
                f"{vllm_config.cache_config.block_size}"
            )
        # The indexer's fp8 side cache has a 132-byte token slot against the attention layers'
        # 2048, and DeepSeek's indexer backend declares its pages non-paddable. vLLM tolerates
        # per-layer page sizes only when every layer needs the same number of token slots, which
        # GPT-OSS's sliding-window layers break. Disabling the hybrid manager converts them to
        # full-length KV (the kernel still applies the window), which is the same one-group path
        # DeepSeek-V3.2 takes. Cost: the 12 sliding-window layers hold full-context KV.
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is not True:
            raise ValueError(
                "GPT-OSS DSA needs --disable-hybrid-kv-cache-manager: the indexer cache page "
                "cannot be unified with the sliding-window layers' pages otherwise"
            )
        # One row per query token, int32 key indices, -1 padded: the shape stock vLLM allocates
        # for DeepSeek-V3.2. A plain attribute, so it stays out of the state dict.
        buf = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            cfg.index_topk,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        with _building(_BuildContext(cfg, buf)):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.dsa_config = cfg
        self.topk_indices_buffer = buf
        # Layers owned by other pipeline ranks are PPMissingLayer placeholders without `.attn`.
        n_sparse = sum(
            1 for layer in self.layers if getattr(getattr(layer, "attn", None), "dsa_sparse", False)
        )
        print(
            f"[GptOssDSA] phase {cfg.training_phase} checkpoint, {n_sparse}/{hf_config.num_hidden_layers} "
            f"layers sparse on this rank, index_topk={cfg.index_topk}, "
            f"indexer {cfg.n_heads}x{cfg.head_dim}, source={cfg.source or 'unknown'}",
            flush=True,
        )


class GptOssDSAForCausalLM(GptOssForCausalLM):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # GptOssForCausalLM.__init__ instantiates GptOssModel directly, so it is replicated here
        # with GptOssDSAModel; the mixins, mappers and loaders are inherited.
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = GptOssDSAModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stock loading, then a hard check that every local sparse layer's indexer was loaded.

        The stock loaders end in `if name not in params_dict: continue`, which is how an indexer
        branch can vanish in silence and leave a fluent, subtly wrong model.
        """
        loaded = super().load_weights(weights)
        local_layers = range(self.model.start_layer, self.model.end_layer)
        expected = {
            f"model.layers.{i}.attn.indexer.{leaf}"
            for i in self.model.dsa_config.sparse_layer_ids
            if i in local_layers
            for leaf in INDEXER_LEAVES
        }
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} indexer tensors were not loaded, e.g. {missing[:4]}. The serving "
                "dir is incomplete or its layer set disagrees with the checkpoint."
            )
        return loaded
