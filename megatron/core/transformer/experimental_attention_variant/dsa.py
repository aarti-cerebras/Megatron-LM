# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_indexer_loss,
    dsa_kernels,
    dsa_layout,
    dsa_masking,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result."""
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-indexed and positive, got {layer_number}.")
    if skip_topk_offset < 0:
        raise ValueError(f"skip_topk_offset must be non-negative, got {skip_topk_offset}.")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    # Layers are 1-indexed, so the default offset 0 must still start at layer 1.
    skip_topk_offset = max(skip_topk_offset, 1)
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    skip_topk_offset = max(skip_topk_offset, 1)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


def _unfused_absorbed_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unfused absorbed-MLA attention: output stays [sq, b, np, v_channels]."""
    sq, b, np, hn = query.size()
    skv = key.size(0)
    assert key.size(2) == 1, "Absorbed DSA expects MQA key head dimension = 1"
    assert key.size(-1) >= v_channels, "key last dim must contain latent value channels"
    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    # [sq,b,np,hn] -> [b,np,sq,hn]
    q = query.permute(1, 2, 0, 3)
    # [skv,b,1,hn] -> [b,1,hn,skv]
    k = key.permute(1, 2, 3, 0)
    attention_scores = torch.matmul(q.float(), k.float()) * softmax_scale

    # Sparse + causal/varlen validity mask.
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)
    index_mask = dsa_masking.apply_sparse_validity_to_index_mask(
        index_mask,
        row_mask=row_mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    attention_scores = attention_scores + index_mask.unsqueeze(1)
    valid_index_mask = torch.isfinite(index_mask)
    attention_scores = dsa_masking.masked_softmax(
        attention_scores.float(), valid_index_mask.unsqueeze(1).expand(b, np, sq, skv), dim=-1
    )

    # Latent value is the first v_channels slice of absorbed key cache.
    value = key[..., :v_channels].permute(1, 2, 0, 3)  # [b,1,skv,v]
    output = torch.matmul(attention_scores.to(value.dtype), value)  # [b,np,sq,v]
    return output.permute(2, 0, 1, 3).contiguous()


def _run_sparse_attention(
    *,
    absorbed_mla: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    up_v_weight: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    softmax_scale: float,
    config: TransformerConfig,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run sparse attention for absorbed and non-absorbed MLA paths."""
    if absorbed_mla:
        latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
        if latent_v_channels <= 0:
            raise RuntimeError(
                "Invalid kv_lora_rank for absorbed-MLA DSAttention sparse attention."
            )
        if up_v_weight is None:
            raise RuntimeError(
                "Absorbed DSAttention requires up_v_weight for latent-to-value projection."
            )
        if value is not None:
            raise RuntimeError(
                "Absorbed DSAttention expects value=None (latent path). "
                "Received absorbed layout with explicit value tensor."
            )
        output = None
        if dsa_kernels.use_fused_dsa_kernels(config):
            output = dsa_kernels.run_fused_absorbed_sparse_attention(
                config,
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                topk_length=topk_length,
            )
        # Fused backends may decline unsupported shapes or layouts by returning
        # None, so keep the absorbed PyTorch path as the authoritative fallback.
        if output is None:
            output = _unfused_absorbed_dsa_fn(
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )
        assert output is not None
        output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        return output

    return unfused_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )


def _normalize_dsattention_output_rank(output: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Normalize DSAttention output rank to match caller hidden-state rank."""
    if target_ndim not in (2, 3):
        raise RuntimeError(f"DSAttention expected x.ndim in (2, 3), got {target_ndim}")

    if output.ndim == 4:
        output = output.reshape(output.size(0), output.size(1), -1)
    elif output.ndim not in (2, 3):
        raise RuntimeError(
            f"DSAttention produced unexpected output rank {output.ndim}; expected 2D/3D/4D."
        )

    if target_ndim == 3 and output.ndim == 2:
        output = output.unsqueeze(1)
    elif target_ndim == 2 and output.ndim == 3:
        if output.size(1) != 1:
            raise RuntimeError(
                "DSAttention cannot squeeze non-singleton batch dim for packed output: "
                f"shape={tuple(output.shape)}"
            )
        output = output.squeeze(1)

    if output.ndim != target_ndim:
        raise RuntimeError(
            "DSAttention output rank mismatch after normalization: "
            f"target_ndim={target_ndim}, output_shape={tuple(output.shape)}"
        )
    return output


def _validate_nonpacked_cp_uniform_length(
    sq: int,
    skv: int,
    cp_size: int,
    cp_group: Optional[torch.distributed.ProcessGroup],
    device: torch.device,
) -> None:
    """Validate the uniform-length precondition for non-packed allgather CP."""
    expected_skv = sq * cp_size
    if (
        cp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and get_pg_size(cp_group) == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        all_lens = torch.cat(all_lens)
        if not torch.all(all_lens == sq):
            raise RuntimeError(
                "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
                f"got per-rank lengths {all_lens.tolist()}."
            )
        expected_skv = int(all_lens.sum().item())

    if skv != sq and skv != expected_skv:
        raise RuntimeError(
            "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
            f"got local query length {sq} and key length {skv} for cp_size={cp_size}."
        )


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def _fake_quant_fp8(x: torch.Tensor, use_ue8m0: bool = True) -> torch.Tensor:
    """Fake-quantize indexer activations to row-wise E4M3 with an STE.

    The serving indexer quantizes one head row at a time. UE8M0 rounds each
    scale up to a power of two. The returned values match the dequantized FP8
    forward while the straight-through estimator preserves gradients into the
    indexer projections.

    Args:
        x: Indexer query or key activations.
        use_ue8m0: Whether to use serving-compatible power-of-two scales.

    Returns:
        Dequantized E4M3 values with identity gradients with respect to ``x``.
    """
    with torch.no_grad():
        amax = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / _FP8_MAX
        if use_ue8m0:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        normalized = (x.detach() / scale).clamp(-_FP8_MAX, _FP8_MAX)
        quantized = (normalized.to(_FP8_DTYPE).float() * scale).to(dtype=x.dtype)
    return x + (quantized - x).detach()


_DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES = {
    "topk_recall": "indexer topk recall",
    "indexer_attention_mass": "attention mass captured by indexer",
    "teacher_topk_attention_mass": "attention mass captured by teacher top-k",
    "topk_intersection_attention_mass": "attention mass captured by top-k intersection",
    "attention_score_recall": "attention score recall",
}


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer training diagnostics."""

    tracker = {}

    @staticmethod
    def _initialize_tracker(num_layers: int) -> None:
        """Allocate per-layer metric state on the current CUDA device."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            return
        tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["kl_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["active_layers"] = torch.zeros(
            num_layers, dtype=torch.int32, device=torch.cuda.current_device()
        )
        for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
            tracker[f"{metric_name}_sum"] = torch.zeros(
                num_layers, device=torch.cuda.current_device()
            )
            tracker[f"{metric_name}_count"] = torch.zeros(
                num_layers, device=torch.cuda.current_device()
            )

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        loss_coeff: float,
        quality_metrics: Optional[dict] = None,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            loss_coeff: Coefficient already applied to ``loss``.
            quality_metrics: Optional detached numerator/denominator statistics.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        DSAIndexerLossLoggingHelper._initialize_tracker(num_layers)
        layer_index = layer_number - 1
        tracker["values"][layer_index] += loss.detach()
        tracker["kl_values"][layer_index] += loss.detach() / loss_coeff
        # Track participation explicitly: a valid DSA layer can have exactly zero KL and must
        # still contribute to the active-layer denominator.
        tracker["active_layers"][layer_index] = 1
        if quality_metrics is not None:
            for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
                tracker[f"{metric_name}_sum"][layer_index] += quality_metrics[
                    f"{metric_name}_sum"
                ].detach()
                tracker[f"{metric_name}_count"][layer_index] += quality_metrics[
                    f"{metric_name}_count"
                ].detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
            tracker["kl_values"].zero_()
            tracker["active_layers"].zero_()
            for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
                tracker[f"{metric_name}_sum"].zero_()
                tracker[f"{metric_name}_count"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker(
        pipeline_group: torch.distributed.ProcessGroup | None,
        data_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        """Collect and reduce indexer diagnostics across explicit process groups."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        if pipeline_group is None or data_parallel_group is None:
            raise ValueError(
                "DSA indexer metric reduction requires pipeline and data-parallel groups"
            )
        values = tracker["values"]
        kl_values = tracker["kl_values"]
        active_layers = tracker["active_layers"]

        torch.distributed.all_reduce(values, group=pipeline_group)
        torch.distributed.all_reduce(kl_values, group=pipeline_group)
        # Pipeline stages own disjoint layer ranges, so union their active-layer masks while
        # summing the corresponding loss slots.
        torch.distributed.all_reduce(
            active_layers, group=pipeline_group, op=torch.distributed.ReduceOp.MAX
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            torch.distributed.all_reduce(kl_values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
            torch.distributed.all_reduce(
                kl_values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values, group=data_parallel_group, op=torch.distributed.ReduceOp.AVG
        )
        torch.distributed.all_reduce(
            kl_values, group=data_parallel_group, op=torch.distributed.ReduceOp.AVG
        )

        # Quality metrics are ratios of global sums. Pipeline stages own disjoint layers and
        # context/data-parallel ranks own distinct rows, so sum both numerator and denominator.
        metric_groups = (
            pipeline_group,
            tracker.get('reduce_group') or tracker.get('avg_group'),
            data_parallel_group,
        )
        quality_metric_tensors = [
            tracker[f"{metric_name}_{statistic}"]
            for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES
            for statistic in ("sum", "count")
        ]
        packed_quality_metrics = torch.stack(quality_metric_tensors)
        for group in metric_groups:
            if group is not None:
                torch.distributed.all_reduce(packed_quality_metrics, group=group)
        for metric_tensor, reduced_metric in zip(
            quality_metric_tensors, packed_quality_metrics.unbind()
        ):
            metric_tensor.copy_(reduced_metric)

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
        num_layers: int | None = None,
        loss_coeff: float | None = None,
        pipeline_group: torch.distributed.ProcessGroup | None = None,
        data_parallel_group: torch.distributed.ProcessGroup | None = None,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
            num_layers: Total transformer layers. Providing this lets pipeline stages with no DSA
                layers initialize empty metric state and participate in cross-stage reductions.
            loss_coeff: Configured KL coefficient. Passed explicitly because the logging pipeline
                rank may own no DSA layers and therefore may not observe it during forward.
            pipeline_group: Pipeline-parallel group used to combine disjoint layer metrics.
            data_parallel_group: Data-parallel group used to average loss metrics and sum quality
                metric numerators and denominators.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            if num_layers is None:
                return
            DSAIndexerLossLoggingHelper._initialize_tracker(num_layers)
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            pipeline_group=pipeline_group, data_parallel_group=data_parallel_group
        )

        indexer_loss_values = tracker["values"] * loss_scale
        indexer_kl_values = tracker["kl_values"] * loss_scale
        active_layer_count = tracker["active_layers"].sum().clamp_min(1)

        # Standard attention layers have no indexer objective. Average only over DSA layers that
        # produced a loss so the metric is comparable across mixed layer patterns.
        avg_indexer_loss = indexer_loss_values.sum() / active_layer_count
        avg_indexer_kl = indexer_kl_values.sum() / active_layer_count

        quality_values = {}
        for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
            metric_sum = tracker[f"{metric_name}_sum"].sum()
            metric_count = tracker[f"{metric_name}_count"].sum()
            if metric_count > 0:
                quality_values[metric_name] = metric_sum / metric_count

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss
            total_loss_dict["indexer kl"] = total_loss_dict.get("indexer kl", 0) + avg_indexer_kl
            if loss_coeff is not None:
                total_loss_dict["indexer loss coefficient"] = total_loss_dict.get(
                    "indexer loss coefficient", 0
                ) + avg_indexer_loss.new_tensor(loss_coeff)
            for metric_name, metric_value in quality_values.items():
                display_name = _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES[metric_name]
                total_loss_dict[display_name] = total_loss_dict.get(display_name, 0) + metric_value

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)
            writer.add_scalar("indexer kl", avg_indexer_kl, iteration)
            if loss_coeff is not None:
                writer.add_scalar("indexer loss coefficient", loss_coeff, iteration)
            for metric_name, metric_value in quality_values.items():
                writer.add_scalar(
                    _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES[metric_name], metric_value, iteration
                )
            if per_layer_logging:
                for layer_index in tracker["active_layers"].nonzero().flatten().tolist():
                    writer.add_scalar(
                        f"indexer kl/layer {layer_index + 1}",
                        indexer_kl_values[layer_index],
                        iteration,
                    )
                    for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
                        metric_count = tracker[f"{metric_name}_count"][layer_index]
                        if metric_count > 0:
                            writer.add_scalar(
                                f"{_DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES[metric_name]}"
                                f"/layer {layer_index + 1}",
                                tracker[f"{metric_name}_sum"][layer_index] / metric_count,
                                iteration,
                            )

        if wandb_writer is not None:
            wandb_metrics = {
                "indexer loss": avg_indexer_loss,
                "indexer kl": avg_indexer_kl,
                **{
                    _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES[metric_name]: metric_value
                    for metric_name, metric_value in quality_values.items()
                },
            }
            if loss_coeff is not None:
                wandb_metrics["indexer loss coefficient"] = loss_coeff
            if per_layer_logging:
                for layer_index in tracker["active_layers"].nonzero().flatten().tolist():
                    wandb_metrics[f"indexer kl/layer {layer_index + 1}"] = indexer_kl_values[
                        layer_index
                    ]
                    for metric_name in _DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES:
                        metric_count = tracker[f"{metric_name}_count"][layer_index]
                        if metric_count > 0:
                            wandb_metrics[
                                f"{_DSA_INDEXER_QUALITY_METRIC_DISPLAY_NAMES[metric_name]}"
                                f"/layer {layer_index + 1}"
                            ] = (tracker[f"{metric_name}_sum"][layer_index] / metric_count)
            wandb_writer.log(wandb_metrics, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def _compute_grouped_attention_scores(
    query: torch.Tensor, key: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Compute per-query-head scores without expanding GQA keys to query-head count."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    sq, b, num_query_heads, head_dim = query.shape
    sk, key_batch, num_kv_heads, key_head_dim = key.shape
    if key_batch != b or key_head_dim != head_dim:
        raise ValueError(
            "DSA teacher query/key shapes are incompatible: "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}."
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "DSA teacher requires query heads to be divisible by KV heads, got "
            f"{num_query_heads} and {num_kv_heads}."
        )

    groups_per_kv = num_query_heads // num_kv_heads
    query_grouped = query.permute(1, 2, 0, 3).reshape(b, num_kv_heads, groups_per_kv, sq, head_dim)
    key_grouped = key.permute(1, 2, 3, 0)
    scores = torch.einsum("bngqd,bndk->bngqk", query_grouped.float(), key_grouped.float())
    return scores.reshape(b, num_query_heads, sq, sk) * softmax_scale


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.
        mask: Optional additive attention mask. Supports shape [sq, sk] or [b, sq, sk].
            Invalid positions should be -inf.
        varlen_starts: Optional row-wise key start bounds [sq] for packed THD.
        varlen_ends: Optional row-wise key end bounds [sq] for packed THD.
        key_positions: Optional global key positions [sk] for packed THD.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=index_scores.device
    )

    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )

    attention_scores = _compute_grouped_attention_scores(query, key, softmax_scale)
    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores += index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = dsa_masking.masked_softmax(
        attention_scores.float(), attention_valid_mask.unsqueeze(1).expand(b, np, sq, sk), dim=-1
    )
    # [b, sq, sk] -> [b, sq, sk]
    index_log_scores = dsa_masking.masked_log_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # The target is already non-negative because it is a sum of softmax probabilities.
    attention_scores = dsa_indexer_loss.normalize_indexer_target(attention_scores)
    return dsa_indexer_loss.indexer_loss_from_target(
        attention_scores,
        index_log_scores,
        loss_coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor, use_relu: bool = True
) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Optionally apply ReLU activation (DeepSeek V3.2 only; disabled for GLM5);
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Optionally apply ReLU activation (used by DeepSeek V3.2, not GLM5).
    if use_relu:
        index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    use_relu: bool = True,
):
    """Naive implementation of QK Topk."""
    sk = k.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )
    if varlen_starts is not None:
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
    elif mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, sk)
    if topk_k > 0:
        topk_scores, topk_indices = index_scores.topk(topk_k, dim=-1)
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    else:
        topk_indices = torch.empty(
            index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device
        )

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk,
    softmax_scale,
    loss_coeff,
    mask,
    sparse_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(
        q,
        k,
        weights,
        topk,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        use_relu=use_relu,
    )

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    mask,
    grad_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
):
    """Naive implementation of backward pass for indexer loss."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=query.device
    )

    attention_scores = _compute_grouped_attention_scores(query, key, softmax_scale)
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=attention_scores.device,
    )

    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both.
    if sparse_loss:
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask
    attention_scores_softmax = dsa_masking.masked_softmax(
        attention_scores.float(), attention_valid_mask.unsqueeze(1).expand(b, np, sq, sk), dim=-1
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = dsa_masking.masked_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize. Fully masked packed/varlen rows can have zero summed
    # attention mass; clamp the denominator so those rows stay finite and are
    # later zeroed by the row-valid loss mask.
    attention_scores_normalized = dsa_indexer_loss.normalize_indexer_target(attention_scores_sum)
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    if calculate_per_token_loss:
        grad_kl_per_row = grad_kl_div
    else:
        valid_row_count = (
            query_valid_rows.sum().to(
                dtype=torch.float32, device=attention_scores_normalized.device
            )
            if query_valid_rows is not None
            else torch.tensor(
                float(b * sq), dtype=torch.float32, device=attention_scores_normalized.device
            )
        ).clamp_min(1.0)
        grad_kl_per_row = grad_kl_div / valid_row_count  # scalar value for each real row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)
    if query_valid_rows is not None:
        grad_kl_per_element = grad_kl_per_element * query_valid_rows.unsqueeze(-1).to(
            dtype=grad_kl_per_element.dtype
        )

    # For KL(target || softmax(logits)), the exact logit gradient is predict - target.
    # Computing it through -target / (predict + eps) incorrectly suppresses gradients when
    # valid predicted probabilities are smaller than eps.
    grad_index_scores_logits = (
        index_scores_softmax - attention_scores_normalized
    ) * grad_kl_per_element
    del index_scores_softmax, attention_scores_normalized

    # Zero out gradients for masked positions.
    if sparse_loss:
        # Also apply index mask - only topk positions are valid.
        del index_mask
        valid_mask = base_valid_mask & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = base_valid_mask  # [b, sq, sk]
    del base_valid_mask
    if query_valid_rows is not None:
        valid_mask = valid_mask & query_valid_rows.unsqueeze(-1)

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]

    # Backward through multiplication by weights (with optional ReLU).
    if use_relu:
        scores_for_weights = torch.relu(scores)
        relu_mask = scores > 0
    else:
        scores_for_weights = scores
        relu_mask = None
    del scores

    # ∂L/∂weights = grad * scores_for_weights (sum over sk)
    grad_weights = (grad_weighted_scores * scores_for_weights).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂scores = grad * weights
    grad_scores = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_for_weights

    # Backward through ReLU (skip when use_relu=False)
    if use_relu:
        grad_scores = grad_scores * relu_mask.float()
        del relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


def _indexer_loss_block_mask(
    *,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    b: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return validity and additive bias for one query/key loss block."""
    if varlen_starts is not None:
        valid = dsa_masking.build_valid_mask_from_starts_ends(
            varlen_starts[q_start:q_end], varlen_ends[q_start:q_end], key_positions[k_start:k_end]
        )
        return valid.unsqueeze(0).expand(b, -1, -1), None

    if mask is None:
        query_positions = torch.arange(q_start, q_end, dtype=torch.int64, device=device)
        block_key_positions = torch.arange(k_start, k_end, dtype=torch.int64, device=device)
        valid = block_key_positions.unsqueeze(0) <= query_positions.unsqueeze(-1)
        return valid.unsqueeze(0).expand(b, -1, -1), None

    bias = (
        mask[q_start:q_end, k_start:k_end]
        if mask.ndim == 2
        else mask[:, q_start:q_end, k_start:k_end]
    )
    if bias.ndim == 2:
        bias = bias.unsqueeze(0)
    valid = torch.isfinite(bias).expand(b, -1, -1)
    return valid, bias


def _masked_block_logits(
    logits: torch.Tensor, valid_mask: torch.Tensor, additive_bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply a broadcastable additive bias and validity mask to block logits."""
    if additive_bias is not None:
        logits = logits + (additive_bias.unsqueeze(1) if logits.ndim == 4 else additive_bias)
    block_valid = valid_mask.unsqueeze(1) if logits.ndim == 4 else valid_mask
    return logits.masked_fill(~block_valid, float("-inf"))


def _block_probabilities(
    logits: torch.Tensor, log_normalizer: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Recover probabilities for a block, including fully masked rows."""
    block_valid = valid_mask.unsqueeze(1) if logits.ndim == 4 else valid_mask
    probabilities = torch.exp(logits - log_normalizer.unsqueeze(-1))
    return probabilities.masked_fill(~block_valid, 0.0)


def _build_sparse_support(topk_indices: torch.Tensor, sk: int) -> torch.Tensor:
    """Build boolean top-k support for one query block."""
    support = torch.zeros(
        (*topk_indices.shape[:2], sk + 1), dtype=torch.bool, device=topk_indices.device
    )
    valid = topk_indices >= 0
    safe_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, sk))
    support.scatter_(dim=-1, index=safe_indices, src=valid)
    return support[..., :sk]


def _blockwise_indexer_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    index_topk: int,
    *,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    block_size: int,
    use_relu: bool,
) -> torch.Tensor:
    """Select indexer top-k while bounding score workspace by two block axes."""
    sq, b = q.shape[:2]
    sk = k.size(0)
    topk_k = min(index_topk, sk)
    if topk_k <= 0:
        return torch.empty((b, sq, 0), dtype=torch.int64, device=q.device)

    topk_indices = torch.empty((b, sq, topk_k), dtype=torch.int64, device=q.device)
    for q_start in range(0, sq, block_size):
        q_end = min(q_start + block_size, sq)
        block_scores = None
        block_indices = None
        for k_start in range(0, sk, block_size):
            k_end = min(k_start + block_size, sk)
            scores = _compute_index_scores(
                q[q_start:q_end], weights[q_start:q_end], k[k_start:k_end], use_relu=use_relu
            )
            valid, bias = _indexer_loss_block_mask(
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                q_start=q_start,
                q_end=q_end,
                k_start=k_start,
                k_end=k_end,
                b=b,
                device=q.device,
            )
            scores = _masked_block_logits(scores, valid, bias)
            indices = torch.arange(k_start, k_end, dtype=torch.int64, device=q.device)
            indices = indices.view(1, 1, -1).expand(b, q_end - q_start, -1)
            if block_scores is not None:
                scores = torch.cat((block_scores, scores), dim=-1)
                indices = torch.cat((block_indices, indices), dim=-1)
            keep = min(topk_k, scores.size(-1))
            block_scores, order = scores.topk(keep, dim=-1)
            block_indices = torch.gather(indices, dim=-1, index=order)

        block_indices = block_indices.masked_fill(block_scores == float("-inf"), -1)
        topk_indices[:, q_start:q_end] = block_indices
    return topk_indices


def _blockwise_loss_normalizers(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    *,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    sparse_support: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    block_size: int,
    use_relu: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute teacher and student log-normalizers without full score tensors."""
    b = q.size(1)
    sk = k.size(0)
    np = query.size(2)
    q_rows = q_end - q_start
    teacher_norm = torch.full((b, np, q_rows), float("-inf"), dtype=torch.float32, device=q.device)
    student_norm = torch.full((b, q_rows), float("-inf"), dtype=torch.float32, device=q.device)

    for k_start in range(0, sk, block_size):
        k_end = min(k_start + block_size, sk)
        valid, bias = _indexer_loss_block_mask(
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            q_start=q_start,
            q_end=q_end,
            k_start=k_start,
            k_end=k_end,
            b=b,
            device=q.device,
        )
        if sparse_support is not None:
            valid = valid & sparse_support[..., k_start:k_end]
        teacher_logits = _compute_grouped_attention_scores(
            query[q_start:q_end], key[k_start:k_end], softmax_scale
        )
        teacher_logits = _masked_block_logits(teacher_logits, valid, bias)
        student_logits = _compute_index_scores(
            q[q_start:q_end], weights[q_start:q_end], k[k_start:k_end], use_relu=use_relu
        )
        student_logits = _masked_block_logits(student_logits, valid, bias)
        teacher_norm = torch.logaddexp(teacher_norm, torch.logsumexp(teacher_logits, dim=-1))
        student_norm = torch.logaddexp(student_norm, torch.logsumexp(student_logits, dim=-1))
    return teacher_norm, student_norm


def _blockwise_teacher_target(
    teacher_logits: torch.Tensor,
    teacher_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """Average a teacher-probability block over global query heads."""
    target = _block_probabilities(teacher_logits, teacher_norm, valid_mask).sum(dim=1)
    if pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(target, group=pg_collection.tp)
    return target / (teacher_logits.size(1) * pg_collection.tp.size())


def fwd_blockwise_indexer_loss(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    topk: int,
    softmax_scale: float,
    loss_coeff: float,
    mask: Optional[torch.Tensor],
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    block_size: int = 256,
    compute_topk: bool = True,
    metrics_out: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k and Design-A KL with bounded query/key score blocks."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    sq, b = q.shape[:2]
    sk = k.size(0)
    block_size = max(1, int(block_size))
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=q.device
    )
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=q.device,
    )
    if sparse_loss and not compute_topk:
        raise ValueError("Sparse indexer loss requires top-k computation.")
    if compute_topk:
        topk_indices = _blockwise_indexer_topk(
            q,
            weights,
            k,
            topk,
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            block_size=block_size,
            use_relu=use_relu,
        )
    else:
        topk_indices = torch.empty((b, sq, 0), dtype=torch.int64, device=q.device)

    collect_quality_metrics = metrics_out is not None and compute_topk and not sparse_loss
    topk_recall_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    topk_recall_count = torch.zeros((), dtype=torch.float32, device=q.device)
    indexer_attention_mass_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    indexer_attention_mass_count = torch.zeros((), dtype=torch.float32, device=q.device)
    teacher_topk_attention_mass_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    teacher_topk_attention_mass_count = torch.zeros((), dtype=torch.float32, device=q.device)
    topk_intersection_attention_mass_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    topk_intersection_attention_mass_count = torch.zeros((), dtype=torch.float32, device=q.device)
    attention_score_recall_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    attention_score_recall_count = torch.zeros((), dtype=torch.float32, device=q.device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    for q_start in range(0, sq, block_size):
        q_end = min(q_start + block_size, sq)
        sparse_support = (
            _build_sparse_support(topk_indices[:, q_start:q_end], sk) if sparse_loss else None
        )
        teacher_norm, student_norm = _blockwise_loss_normalizers(
            q,
            weights,
            k,
            query,
            key,
            softmax_scale,
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            sparse_support=sparse_support,
            q_start=q_start,
            q_end=q_end,
            block_size=block_size,
            use_relu=use_relu,
        )
        row_kl = torch.zeros((b, q_end - q_start), dtype=torch.float32, device=q.device)
        teacher_topk_scores = None
        teacher_topk_indices = None
        attention_mass_per_row = torch.zeros_like(row_kl)
        indexer_support = (
            _build_sparse_support(topk_indices[:, q_start:q_end], sk)
            if collect_quality_metrics
            else None
        )
        for k_start in range(0, sk, block_size):
            k_end = min(k_start + block_size, sk)
            valid, bias = _indexer_loss_block_mask(
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                q_start=q_start,
                q_end=q_end,
                k_start=k_start,
                k_end=k_end,
                b=b,
                device=q.device,
            )
            if sparse_support is not None:
                valid = valid & sparse_support[..., k_start:k_end]
            teacher_logits = _compute_grouped_attention_scores(
                query[q_start:q_end], key[k_start:k_end], softmax_scale
            )
            teacher_logits = _masked_block_logits(teacher_logits, valid, bias)
            target = _blockwise_teacher_target(teacher_logits, teacher_norm, valid, pg_collection)
            if collect_quality_metrics:
                attention_mass_per_row += (
                    target * indexer_support[..., k_start:k_end].to(dtype=target.dtype)
                ).sum(dim=-1)
                teacher_scores = target.masked_fill(~valid, float("-inf"))
                teacher_indices = torch.arange(
                    k_start, k_end, dtype=torch.int64, device=q.device
                ).view(1, 1, -1)
                teacher_indices = teacher_indices.expand(b, q_end - q_start, -1)
                if teacher_topk_scores is not None:
                    teacher_scores = torch.cat((teacher_topk_scores, teacher_scores), dim=-1)
                    teacher_indices = torch.cat((teacher_topk_indices, teacher_indices), dim=-1)
                keep = min(topk_indices.size(-1), teacher_scores.size(-1))
                teacher_topk_scores, teacher_order = teacher_scores.topk(keep, dim=-1)
                teacher_topk_indices = torch.gather(teacher_indices, dim=-1, index=teacher_order)
            student_logits = _compute_index_scores(
                q[q_start:q_end], weights[q_start:q_end], k[k_start:k_end], use_relu=use_relu
            )
            student_logits = _masked_block_logits(student_logits, valid, bias)
            student_log_probs = (student_logits - student_norm.unsqueeze(-1)).masked_fill(
                ~valid, 0.0
            )
            row_kl += dsa_indexer_loss.indexer_kl_per_row(target, student_log_probs, valid)
        if query_valid_rows is not None:
            row_kl *= query_valid_rows[:, q_start:q_end].to(dtype=row_kl.dtype)
        kl_sum += row_kl.sum()

        if collect_quality_metrics:
            teacher_topk_indices = teacher_topk_indices.masked_fill(
                teacher_topk_scores == float("-inf"), -1
            )
            teacher_topk_valid = teacher_topk_indices >= 0
            row_valid = teacher_topk_valid.any(dim=-1)
            if query_valid_rows is not None:
                row_valid &= query_valid_rows[:, q_start:q_end]
            safe_teacher_topk_indices = teacher_topk_indices.clamp_min(0)
            teacher_topk_matched = (
                torch.gather(indexer_support, dim=-1, index=safe_teacher_topk_indices)
                & teacher_topk_valid
            )
            overlap_count = teacher_topk_matched.sum(dim=-1).float()
            teacher_count = teacher_topk_valid.sum(dim=-1).float()
            teacher_topk_probabilities = teacher_topk_scores.masked_fill(~teacher_topk_valid, 0.0)
            teacher_topk_mass_per_row = teacher_topk_probabilities.sum(dim=-1)
            intersection_mass_per_row = (teacher_topk_probabilities * teacher_topk_matched).sum(
                dim=-1
            )
            quality_valid_row_count = row_valid.sum()
            topk_recall_sum += (overlap_count * row_valid).sum()
            topk_recall_count += (teacher_count * row_valid).sum()
            indexer_attention_mass_sum += (attention_mass_per_row * row_valid).sum()
            indexer_attention_mass_count += quality_valid_row_count
            teacher_topk_attention_mass_sum += (teacher_topk_mass_per_row * row_valid).sum()
            teacher_topk_attention_mass_count += quality_valid_row_count
            topk_intersection_attention_mass_sum += (intersection_mass_per_row * row_valid).sum()
            topk_intersection_attention_mass_count += quality_valid_row_count
            # Score-weighted recall of the teacher's optimal K-key support. This differs from
            # indexer_attention_mass: keys selected outside teacher top-K contribute to the
            # latter, but not to this numerator.
            attention_score_recall_sum += (intersection_mass_per_row * row_valid).sum()
            attention_score_recall_count += (teacher_topk_mass_per_row * row_valid).sum()

    valid_row_count = query_valid_rows.sum() if query_valid_rows is not None else None
    loss = dsa_indexer_loss.reduce_indexer_kl_sum(
        kl_sum,
        num_rows=b * sq,
        calculate_per_token_loss=calculate_per_token_loss,
        valid_row_count=valid_row_count,
    )
    if metrics_out is not None:
        metrics_out.update(
            {
                "topk_recall_sum": topk_recall_sum,
                "topk_recall_count": topk_recall_count,
                "indexer_attention_mass_sum": indexer_attention_mass_sum,
                "indexer_attention_mass_count": indexer_attention_mass_count,
                "teacher_topk_attention_mass_sum": teacher_topk_attention_mass_sum,
                "teacher_topk_attention_mass_count": teacher_topk_attention_mass_count,
                "topk_intersection_attention_mass_sum": topk_intersection_attention_mass_sum,
                "topk_intersection_attention_mass_count": topk_intersection_attention_mass_count,
                "attention_score_recall_sum": attention_score_recall_sum,
                "attention_score_recall_count": attention_score_recall_count,
            }
        )
    return topk_indices, loss * loss_coeff


def bwd_blockwise_indexer_loss(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    mask: Optional[torch.Tensor],
    grad_loss: torch.Tensor,
    pg_collection: ProcessGroupCollection,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute blockwise probabilities and accumulate exact indexer gradients."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    sq, b = q.shape[:2]
    sk = k.size(0)
    block_size = max(1, int(block_size))
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=q.device
    )
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=q.device,
    )
    if calculate_per_token_loss:
        row_scale = grad_loss.float() * loss_coeff
    else:
        valid_row_count = (
            query_valid_rows.sum().float()
            if query_valid_rows is not None
            else torch.tensor(float(b * sq), dtype=torch.float32, device=q.device)
        ).clamp_min(1.0)
        row_scale = grad_loss.float() * loss_coeff / valid_row_count

    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_weights = torch.zeros_like(weights, dtype=torch.float32)
    grad_k = torch.zeros_like(k, dtype=torch.float32)
    for q_start in range(0, sq, block_size):
        q_end = min(q_start + block_size, sq)
        sparse_support = (
            _build_sparse_support(topk_indices[:, q_start:q_end], sk) if sparse_loss else None
        )
        teacher_norm, student_norm = _blockwise_loss_normalizers(
            q,
            weights,
            k,
            query,
            key,
            softmax_scale,
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            sparse_support=sparse_support,
            q_start=q_start,
            q_end=q_end,
            block_size=block_size,
            use_relu=use_relu,
        )
        block_row_scale = row_scale
        if query_valid_rows is not None:
            block_row_scale = row_scale * query_valid_rows[:, q_start:q_end].unsqueeze(-1).float()

        for k_start in range(0, sk, block_size):
            k_end = min(k_start + block_size, sk)
            valid, bias = _indexer_loss_block_mask(
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                q_start=q_start,
                q_end=q_end,
                k_start=k_start,
                k_end=k_end,
                b=b,
                device=q.device,
            )
            if sparse_support is not None:
                valid = valid & sparse_support[..., k_start:k_end]
            teacher_logits = _compute_grouped_attention_scores(
                query[q_start:q_end], key[k_start:k_end], softmax_scale
            )
            teacher_logits = _masked_block_logits(teacher_logits, valid, bias)
            target = _blockwise_teacher_target(teacher_logits, teacher_norm, valid, pg_collection)
            student_logits = _compute_index_scores(
                q[q_start:q_end], weights[q_start:q_end], k[k_start:k_end], use_relu=use_relu
            )
            student_logits = _masked_block_logits(student_logits, valid, bias)
            student_prob = _block_probabilities(student_logits, student_norm, valid)
            grad_index_scores = (student_prob - target) * block_row_scale
            grad_index_scores = grad_index_scores.masked_fill(~valid, 0.0).transpose(0, 1)

            raw_scores = torch.einsum(
                "sbhd,tbd->sbht", q[q_start:q_end].float(), k[k_start:k_end].float()
            )
            activated_scores = torch.relu(raw_scores) if use_relu else raw_scores
            grad_weighted_scores = grad_index_scores.unsqueeze(2)
            grad_weights[q_start:q_end] += (grad_weighted_scores * activated_scores).sum(dim=-1)
            grad_scores = grad_weighted_scores * weights[q_start:q_end].float().unsqueeze(-1)
            if use_relu:
                grad_scores *= raw_scores > 0
            grad_q[q_start:q_end] += torch.einsum(
                "sbht,tbd->sbhd", grad_scores, k[k_start:k_end].float()
            )
            grad_k[k_start:k_end] += torch.einsum(
                "sbht,sbhd->tbd", grad_scores, q[q_start:q_end].float()
            )

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


_FUSED_DSA_INDEXER_LOSS_INPUT_NAMES = (
    "q",
    "weights",
    "k",
    "query",
    "key",
    "softmax_scale",
    "topk",
    "loss_coeff",
    "mask",
    "sparse_loss",
    "pg_collection",
    "varlen_starts",
    "varlen_ends",
    "key_positions",
    "query_valid_rows",
    "calculate_per_token_loss",
    "use_relu",
    "block_size",
    "compute_topk",
    "metrics_out",
)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        query_valid_rows=None,
        calculate_per_token_loss: bool = False,
        use_relu: bool = True,
        block_size: int = 256,
        compute_topk: bool = True,
        metrics_out: Optional[dict] = None,
    ):
        """Compute top-k and indexer loss without full score tensors."""
        topk_indices, loss = fwd_blockwise_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=calculate_per_token_loss,
            use_relu=use_relu,
            block_size=block_size,
            compute_topk=compute_topk,
            metrics_out=metrics_out,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.mask = mask
        ctx.pg_collection = pg_collection
        ctx.varlen_starts = varlen_starts
        ctx.varlen_ends = varlen_ends
        ctx.key_positions = key_positions
        ctx.query_valid_rows = query_valid_rows
        ctx.calculate_per_token_loss = calculate_per_token_loss
        ctx.use_relu = use_relu
        ctx.block_size = block_size
        ctx.input_count = len(ctx.needs_input_grad)

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_blockwise_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            ctx.mask,
            grad_loss,
            ctx.pg_collection,
            varlen_starts=ctx.varlen_starts,
            varlen_ends=ctx.varlen_ends,
            key_positions=ctx.key_positions,
            query_valid_rows=ctx.query_valid_rows,
            calculate_per_token_loss=ctx.calculate_per_token_loss,
            use_relu=ctx.use_relu,
            block_size=ctx.block_size,
        )

        grad_by_name = {
            "q": grad_q,
            "weights": grad_weights,
            "k": grad_k,
            # query and key are detached in forward, so return None for their gradients.
            "query": None,
            "key": None,
        }
        gradients = tuple(grad_by_name.get(name) for name in _FUSED_DSA_INDEXER_LOSS_INPUT_NAMES)
        return gradients[: ctx.input_count]


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale.to(
            device=indexer_loss.device
        )
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if not isinstance(scale, torch.Tensor):
            raise TypeError("DSAIndexerLossAutoScaler.set_loss_scale requires a torch.Tensor.")
        scale = scale.detach()

        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
        dense_attention: Optional dense attention delegate used during Phase-1 warmup.
    """

    indexer: Union[ModuleSpec, type] = None
    dense_attention: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk
        self.qk_pos_emb_head_dim = (
            getattr(self.config, "qk_pos_emb_head_dim", None) or self.index_head_dim
        )
        self.q_lora_rank = getattr(self.config, "q_lora_rank", None) or self.config.hidden_size
        self.rope_type = (
            getattr(self.config, "rope_type", None) or self.config.dsa_indexer_rope_type or "rope"
        )
        self.rotary_base = (
            getattr(self.config, "rotary_base", None)
            or self.config.dsa_indexer_rotary_base
            or 10000
        )
        self.rotary_percent = (
            getattr(self.config, "rotary_percent", None)
            or self.config.dsa_indexer_rotary_percent
            or 1.0
        )

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                scaling_factor=getattr(
                    self.config,
                    "rotary_scaling_factor",
                    getattr(self.config, "yarn_rotary_scaling_factor", 1.0),
                ),
                original_max_position_embeddings=getattr(
                    self.config,
                    "original_max_position_embeddings",
                    getattr(self.config, "yarn_original_max_position_embeddings", 4096),
                ),
                beta_fast=getattr(
                    self.config, "beta_fast", getattr(self.config, "yarn_beta_fast", 32.0)
                ),
                beta_slow=getattr(
                    self.config, "beta_slow", getattr(self.config, "yarn_beta_slow", 1.0)
                ),
                mscale=getattr(self.config, "mscale", getattr(self.config, "yarn_mscale", 1.0)),
                mscale_all_dim=getattr(
                    self.config, "mscale_all_dim", getattr(self.config, "yarn_mscale_all_dim", 0.0)
                ),
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        k_norm_eps = (
            self.config.dsa_indexer_k_norm_epsilon
            if self.config.dsa_indexer_k_norm_epsilon is not None
            else self.config.layernorm_epsilon
        )
        self.k_norm = build_module(
            submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=k_norm_eps
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # Indexer projections are duplicated across tensor-parallel ranks, so their gradients
        # should be averaged during final gradient synchronization.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)
            # Optimizer metadata used for the separate indexer LR schedule and gradient
            # diagnostics. Unlike grad_norm_group, this does not alter clipping; indexer and
            # backbone gradients continue to use the same global clipping coefficient.
            setattr(param, "is_dsa_indexer_parameter", True)

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        squeezed_batch_dim = False
        if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
            cu_seqlens = cu_seqlens.to(device=x_pe.device)
        # THD RoPE path expects [t, h, d], while indexer tensors are [t, 1, h, d].
        if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
            x_pe = x_pe.squeeze(1)
            squeezed_batch_dim = True
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            mla_rotary_interleaved=self.config.dsa_indexer_rope_interleaved,
        )
        if squeezed_batch_dim:
            x_pe = x_pe.unsqueeze(1)
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            if x is qr:
                x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
                qr = x
            else:
                x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
                qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_q)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        if self.config.dsa_indexer_k_norm_fp32:
            k_dtype = k.dtype
            k = self.k_norm(k.float()).to(dtype=k_dtype)
        else:
            k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_kv)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        if self.config.dsa_indexer_rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)

        # Match the serving indexer's E4M3 q/k numerics while retaining an
        # identity gradient into the trainable indexer projections.
        if getattr(self.config, "dsa_indexer_fp8", False):
            use_ue8m0 = getattr(self.config, "dsa_indexer_fp8_ue8m0", True)
            q = _fake_quant_fp8(q, use_ue8m0)
            k = _fake_quant_fp8(k, use_ue8m0)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Optional additive attention mask [seqlen, seqlen] or
                [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(
            q, k, weights, self.index_topk, mask, use_relu=self.config.dsa_indexer_scoring_relu
        )

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(
    query,
    key,
    value,
    topk_indices,
    softmax_scale,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
):
    """
    Unfused sparse attention implementation.

    This path uses chunked sparse softmax accumulation over top-k selected keys
    to avoid materializing full [b, np, sq, skv] attention score tensors.
    """
    if value is None:
        raise NotImplementedError("DSAttention unfused path requires value tensor.")

    query, query_was_thd = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")

    sq, b, np, hn = query.size()
    skv = key.size(0)
    nk = key.size(2)
    hnv = value.size(3)
    nv = value.size(2)

    # [sq, b, np, hn] -> [b, np, sq, hn]
    query_b = query.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nk, hn] -> [b, nk, skv, hn]
    key_b = key.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nv, hnv] -> [b, nv, skv, hnv]
    value_b = value.permute(1, 2, 0, 3).contiguous()
    if nk == 1 and np > 1:
        key_b = key_b.expand(b, np, skv, hn)
    else:
        assert nk == np, "key head count must be 1 (MQA) or match query heads"
    if nv == 1 and np > 1:
        value_b = value_b.expand(b, np, skv, hnv)
    else:
        assert nv == np, "value head count must be 1 (MQA) or match query heads"

    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    seq_chunk_size = 512
    head_chunk_size = 16
    topk_chunk_size = 1024
    safe_k_max = max(0, skv - 1)
    output = torch.empty((sq, b, np * hnv), dtype=value.dtype, device=query.device)

    for bi in range(b):
        for h0 in range(0, np, head_chunk_size):
            h1 = min(h0 + head_chunk_size, np)
            h_chunk = h1 - h0
            out_h0 = h0 * hnv
            out_h1 = h1 * hnv
            k_chunk = key_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hn]
            v_chunk = value_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hnv]
            flat_k = k_chunk.reshape(h_chunk * skv, hn)
            flat_v = v_chunk.reshape(h_chunk * skv, hnv)
            head_offsets = (
                torch.arange(h_chunk, device=query.device, dtype=torch.int64).view(-1, 1, 1) * skv
            )

            for s0 in range(0, sq, seq_chunk_size):
                s1 = min(s0 + seq_chunk_size, sq)
                s_len = s1 - s0
                idx_seq_raw = topk_indices[bi, s0:s1]  # [s_len, topk]
                if idx_seq_raw.dtype != torch.int64 or idx_seq_raw.device != query.device:
                    idx_seq_raw = idx_seq_raw.to(dtype=torch.int64, device=query.device)
                valid_seq = idx_seq_raw >= 0
                idx_seq = idx_seq_raw.clamp(min=0, max=safe_k_max)
                q_chunk = query_b[bi, h0:h1, s0:s1, :]  # [h_chunk, s_len, hn]

                # These tensors participate in autograd; reusing cached storage can
                # invalidate saved tensors before backward runs.
                m = torch.full(
                    (h_chunk, s_len), float("-inf"), dtype=torch.float32, device=query.device
                )
                l = torch.zeros((h_chunk, s_len), dtype=torch.float32, device=query.device)
                acc = torch.zeros((h_chunk, s_len, hnv), dtype=torch.float32, device=query.device)

                for t0 in range(0, idx_seq.size(-1), topk_chunk_size):
                    t1 = min(t0 + topk_chunk_size, idx_seq.size(-1))
                    idx_topk = idx_seq[:, t0:t1]  # [s_len, tk]
                    valid_t = valid_seq[:, t0:t1]  # [s_len, tk]
                    flat_idx = idx_topk.unsqueeze(0) + head_offsets  # [h_chunk, s_len, tk]
                    k_sel = flat_k.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hn
                    )
                    v_sel = flat_v.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hnv
                    )
                    logits = (q_chunk.float().unsqueeze(2) * k_sel.float()).sum(
                        dim=-1
                    ) * softmax_scale

                    valid_2d, mask_bias = dsa_masking.gather_sparse_topk_validity_and_bias(
                        idx_topk=idx_topk,
                        valid_t=valid_t,
                        bi=bi,
                        s0=s0,
                        s1=s1,
                        row_mask=row_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        dtype=torch.float32,
                    )
                    if mask_bias is not None:
                        logits = logits + mask_bias.unsqueeze(0)
                    logits = logits.masked_fill(
                        ~valid_2d.unsqueeze(0).expand(h_chunk, -1, -1), float("-inf")
                    )
                    m_new = torch.maximum(m, logits.max(dim=-1).values)
                    m_new_for_exp = torch.where(
                        torch.isfinite(m_new), m_new, torch.zeros_like(m_new)
                    )
                    alpha = torch.exp(m - m_new_for_exp)
                    p = torch.exp(logits - m_new_for_exp.unsqueeze(-1))
                    acc = acc * alpha.unsqueeze(-1) + torch.einsum(
                        "hst,hstd->hsd", p, v_sel.float()
                    )
                    l = l * alpha + p.sum(dim=-1)
                    m = m_new

                out_chunk = (acc / l.clamp_min(1e-10).unsqueeze(-1)).to(dtype=value.dtype)
                output[s0:s1, bi, out_h0:out_h1] = out_chunk.permute(1, 0, 2).reshape(
                    s_len, h_chunk * hnv
                )

    if query_was_thd:
        output = output.squeeze(1)
    return output


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    consumes_absorbed_v_up_projection = True
    requires_dsa_inputs = True
    _HOLDER_ATTR = "_dsa_index_share_topk_holder"
    _LENGTH_HOLDER_ATTR = "_dsa_index_share_topk_length_holder"

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number
        self.index_topk = self.config.dsa_indexer_topk
        self.index_topk_freq = self.config.dsa_indexer_topk_freq or 1
        self.index_skip_topk_offset = self.config.dsa_indexer_skip_topk_offset or 0
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_dsa_skip_topk_layer(
            layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        self.source_layer = (
            source_dsa_compute_layer(
                layer_number, self.index_skip_topk_offset, self.index_topk_freq
            )
            if self.index_share
            else layer_number
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=self.pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        self.cp_comm_type = dsa_layout.normalize_cp_comm_type(cp_comm_type)
        self.dense_attention = None
        if submodules.dense_attention is not None:
            self.dense_attention = build_module(
                submodules.dense_attention,
                config=self.config,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
                softmax_scale=softmax_scale,
                k_channels=k_channels,
                v_channels=v_channels,
                cp_comm_type=cp_comm_type,
                pg_collection=self.pg_collection,
            )

    def _get_index_share_carrier(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ) -> object:
        """Return the object that carries DSA top-k sharing state for this forward."""
        if packed_seq_params is not None:
            return packed_seq_params
        return attention_mask if attention_mask is not None else self.config

    def _get_index_share_topk_holder(
        self,
        packed_seq_params: Optional[PackedSeqParams],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        """Return the per-forward top-k holder for DSA index sharing."""
        carrier = self._get_index_share_carrier(packed_seq_params, attention_mask)
        holder = getattr(carrier, self._HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(carrier, self._HOLDER_ATTR, holder)
        return holder

    def _get_index_share_topk_length_holder(
        self,
        packed_seq_params: Optional[PackedSeqParams],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        """Return the optional per-forward top-k length holder."""
        carrier = self._get_index_share_carrier(packed_seq_params, attention_mask)
        holder = getattr(carrier, self._LENGTH_HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(carrier, self._LENGTH_HOLDER_ATTR, holder)
        return holder

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn] or packed [t, np, hn].
            key: Key tensor [skv, b, np, hn] or packed [t, np, hn].
            value: Value tensor [skv, b, np, hnv] or packed [t, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            position_ids: Optional position ids [b, sq], used by allgather CP causal masking.
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        dense_output = None
        if getattr(self.config, "dsa_dense_warmup", False):
            if self.dense_attention is None:
                raise RuntimeError("dsa_dense_warmup requires a dense_attention submodule.")
            if value is None:
                raise RuntimeError("dsa_dense_warmup requires explicit GQA values.")
            # Delegate before DSA converts layouts or gathers TP/CP keys. TE owns those
            # transformations and must receive the exact stock attention inputs.
            dense_output = apply_module(self.dense_attention)(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        query, _ = dsa_layout.ensure_sbhd(query, "query")
        key, _ = dsa_layout.ensure_sbhd(key, "key")
        if value is not None:
            value, _ = dsa_layout.ensure_sbhd(value, "value")
        if up_v_weight is not None:
            assert up_v_weight.ndim == 3, "up_v_weight must be [heads, v_head_dim, kv_lora_rank]"
            up_v_weight = up_v_weight.to(device=query.device, dtype=query.dtype).contiguous()
            if value is not None:
                raise RuntimeError(
                    "DSAttention received up_v_weight with explicit value tensor. "
                    "For absorbed DSA path, value must be None."
                )

        latent_v_channels = int(getattr(self.config, "kv_lora_rank", 0) or 0)
        qk_pos_dim = int(getattr(self.config, "qk_pos_emb_head_dim", 0) or 0)
        expected_absorbed_dim = latent_v_channels + qk_pos_dim
        absorbed_mla = (
            latent_v_channels > 0
            and expected_absorbed_dim > 0
            and key.size(2) == 1
            and query.size(-1) == key.size(-1) == expected_absorbed_dim
        )
        if value is None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received value=None but query/key are not in absorbed layout. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )
        if up_v_weight is not None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received up_v_weight but absorbed layout was not detected. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )

        sq, b, _, _ = query.size()
        local_sequence_rows = x.size(0)

        cp_group = getattr(self.pg_collection, "cp", None)
        cp_size = get_pg_size(cp_group)
        cp_rank = cp_group.rank() if cp_group is not None else 0
        tp_group = getattr(self.pg_collection, "tp", None)
        tp_size = get_pg_size(tp_group)
        sequence_parallel_tp = self.config.sequence_parallel and tp_size > 1
        sequence_parallel_tp_row_start = 0
        sequence_parallel_tp_full_rows = sq
        sequence_parallel_query_is_local = False
        if sequence_parallel_tp:
            sequence_parallel_tp_full_rows = local_sequence_rows * tp_size
            if sq == local_sequence_rows:
                sequence_parallel_query_is_local = True
                sequence_parallel_tp_row_start = tp_group.rank() * local_sequence_rows
            elif sq != sequence_parallel_tp_full_rows:
                raise RuntimeError(
                    "DSA sequence-parallel query row count mismatch: "
                    f"query_rows={sq}, local_rows={local_sequence_rows}, tp_size={tp_size}"
                )
        packed_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        packed_query_positions = None
        nonpacked_query_positions = None
        kv_reorder_idx = None
        single_packed_thd_sequence = False
        if packed_thd:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            single_packed_thd_sequence = (
                cp_size > 1 and cu_seqlens_q.numel() == 2 and cu_seqlens_kv.numel() == 2
            )
            packed_query_output_size = (
                sequence_parallel_tp_full_rows if sequence_parallel_tp else sq
            )
            packed_global_output_size = packed_query_output_size * cp_size
            if sequence_parallel_query_is_local and cp_size == 1:
                row_start = sequence_parallel_tp_row_start
                packed_query_positions = torch.arange(
                    row_start, row_start + sq, dtype=torch.int64, device=query.device
                )
            elif sequence_parallel_tp and cp_size > 1:
                packed_query_positions_full = dsa_layout.build_packed_allgather_cp_local_positions(
                    cu_seqlens_q,
                    cp_size,
                    cp_rank,
                    query.device,
                    output_size=packed_query_output_size,
                )
                if sequence_parallel_query_is_local:
                    row_start = sequence_parallel_tp_row_start
                    packed_query_positions = packed_query_positions_full[row_start : row_start + sq]
                else:
                    packed_query_positions = packed_query_positions_full
            elif cp_size > 1:
                # For one sequence, host max-seqlen metadata proves whether cu_seqlens already
                # covers every packed row without synchronizing on the CUDA cu_seqlens tensor.
                query_cu_seqlens_cover_output = (
                    single_packed_thd_sequence
                    and isinstance(packed_seq_params.max_seqlen_q, int)
                    and packed_seq_params.max_seqlen_q == packed_global_output_size
                )
                key_cu_seqlens_cover_output = (
                    single_packed_thd_sequence
                    and isinstance(packed_seq_params.max_seqlen_kv, int)
                    and packed_seq_params.max_seqlen_kv == packed_global_output_size
                )
                packed_query_positions, kv_reorder_idx = (
                    dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        device=query.device,
                        local_output_size=packed_query_output_size,
                        key_local_output_size=packed_query_output_size,
                        global_output_size=packed_global_output_size,
                        query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                        key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                    )
                )
            if packed_query_positions is not None:
                packed_query_positions = packed_query_positions.contiguous()
        elif cp_size > 1:
            _validate_nonpacked_cp_uniform_length(
                sq=sq, skv=key.size(0), cp_size=cp_size, cp_group=cp_group, device=query.device
            )

        if sequence_parallel_tp:
            if key.size(0) == local_sequence_rows:
                key = gather_from_sequence_parallel_region(key, group=tp_group)
            elif key.size(0) != sequence_parallel_tp_full_rows:
                raise RuntimeError(
                    "DSA sequence-parallel key row count mismatch before CP gather: "
                    f"key_rows={key.size(0)}, local_rows={local_sequence_rows}, "
                    f"full_rows={sequence_parallel_tp_full_rows}, tp_size={tp_size}"
                )
            if value is not None:
                if value.size(0) == local_sequence_rows:
                    value = gather_from_sequence_parallel_region(value, group=tp_group)
                elif value.size(0) != sequence_parallel_tp_full_rows:
                    raise RuntimeError(
                        "DSA sequence-parallel value row count mismatch before CP gather: "
                        f"value_rows={value.size(0)}, local_rows={local_sequence_rows}, "
                        f"full_rows={sequence_parallel_tp_full_rows}, tp_size={tp_size}"
                    )

        local_cp_kv_lens = {sq}
        if sequence_parallel_tp:
            local_cp_kv_lens.add(sequence_parallel_tp_full_rows)
        local_cp_kv_len = None
        if cp_size > 1:
            assert (
                self.cp_comm_type == "allgather"
            ), "DSAttention context parallelism currently supports cp_comm_type=allgather only."

            # For allgather CP, keys/values are expected in full-sequence order.
            # Gather local-sequence tensors, then undo MCore's zigzag rank order.
            def _build_kv_reorder_idx(local_len):
                if packed_thd:
                    _, idx = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        device=query.device,
                        local_output_size=local_len,
                        key_local_output_size=local_len,
                        global_output_size=local_len * cp_size,
                    )
                    return idx
                return dsa_layout.build_zigzag_allgather_cp_key_reorder(
                    sq=local_len, cp_size=cp_size, device=query.device
                )

            gathered_cp_key = False
            gathered_cp_value = False
            if key.size(0) in local_cp_kv_lens:
                local_cp_kv_len = key.size(0)
                if kv_reorder_idx is None:
                    kv_reorder_idx = _build_kv_reorder_idx(local_cp_kv_len)
                key = gather_from_sequence_parallel_region(key, group=cp_group)
                gathered_cp_key = True
            if value is not None and value.size(0) in local_cp_kv_lens:
                if local_cp_kv_len is None:
                    local_cp_kv_len = value.size(0)
                    if kv_reorder_idx is None:
                        kv_reorder_idx = _build_kv_reorder_idx(local_cp_kv_len)
                elif value.size(0) != local_cp_kv_len:
                    raise RuntimeError(
                        "DSA local key/value sequence length mismatch before CP gather: "
                        f"key_len={local_cp_kv_len}, value_len={value.size(0)}"
                    )
                value = gather_from_sequence_parallel_region(value, group=cp_group)
                gathered_cp_value = True
            if kv_reorder_idx is not None:
                if gathered_cp_key:
                    if key.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered key length mismatch: "
                            f"key_seqlen={key.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    key = key.index_select(0, kv_reorder_idx)
                if gathered_cp_value:
                    if value.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered value length mismatch: "
                            f"value_seqlen={value.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    value = value.index_select(0, kv_reorder_idx)

        skv = key.size(0)

        if not packed_thd and sequence_parallel_query_is_local:
            nonpacked_query_positions = dsa_layout.extract_query_positions_from_position_ids(
                position_ids, sq, query.device
            )
            if nonpacked_query_positions is None:
                full_query_positions, _ = dsa_layout.get_cp_positions_from_layout(
                    sq=sequence_parallel_tp_full_rows,
                    skv=skv,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_comm_type=self.cp_comm_type,
                    device=query.device,
                    cp_group=cp_group,
                )
                row_start = sequence_parallel_tp_row_start
                nonpacked_query_positions = full_query_positions[
                    row_start : row_start + sq
                ].contiguous()

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
        computes_topk = not self.skip_topk
        use_indexer_loss = (
            self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0 and computes_topk
        )
        if use_indexer_loss and sequence_parallel_query_is_local:
            raise RuntimeError(
                "DSA indexer loss requires TP ranks to own the same query rows; "
                "sequence-local TP query shards cannot form a global-head target."
            )
        float_mask, varlen_params, varlen_is_plain_causal = (
            dsa_masking.build_dsattention_forward_mask(
                sq=sq,
                skv=skv,
                b=b,
                device=x.device,
                cp_size=cp_size,
                cp_rank=cp_rank,
                cp_comm_type=self.cp_comm_type,
                cp_group=cp_group,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                packed_query_positions=packed_query_positions,
                nonpacked_query_positions=nonpacked_query_positions,
            )
        )
        if varlen_params is not None:
            varlen_starts, varlen_ends, key_positions = varlen_params
        else:
            varlen_starts = varlen_ends = key_positions = None
        query_valid_rows = dsa_masking.extract_query_valid_rows_from_packed_seq_params(
            packed_seq_params, b=b, sq=sq, device=query.device
        )
        use_fused_kernels = dsa_kernels.use_fused_dsa_kernels(self.config)
        sparse_indexer_loss = self.config.dsa_indexer_use_sparse_loss
        use_local_indexer_varlen = (
            packed_thd
            and cp_size > 1
            and attn_mask_type == AttnMaskType.causal
            and varlen_starts is not None
            and varlen_ends is not None
            and key_positions is None
        )
        indexer_reduce_group = (
            cp_group if cp_size > 1 and self.config.calculate_per_token_loss else None
        )
        indexer_avg_group = (
            cp_group if cp_size > 1 and not self.config.calculate_per_token_loss else None
        )

        topk_holder = (
            self._get_index_share_topk_holder(packed_seq_params, attention_mask)
            if self.index_share
            else None
        )
        topk_length_holder = (
            self._get_index_share_topk_length_holder(packed_seq_params, attention_mask)
            if self.index_share
            else None
        )
        topk_indices = None
        topk_length = None
        q = k = weights = None
        quality_metrics = {}
        local_packed_cp_query_start = 0
        local_packed_cp_query_len = sq
        if sequence_parallel_query_is_local:
            local_packed_cp_query_start = sequence_parallel_tp_row_start
            local_packed_cp_query_len = sequence_parallel_tp_full_rows

        if self.skip_topk:
            assert topk_holder is not None
            if self.source_layer not in topk_holder:
                raise RuntimeError(
                    "DSA index-share skip layer "
                    f"(layer_number={self.layer_number}) needs top-k indices from source "
                    f"computing layer {self.source_layer}, but that layer did not run before it "
                    "in this pipeline stage. Cross-PP top-k sharing is not supported. Ensure each "
                    "pipeline stage starts on a computing layer "
                    f"(dsa_indexer_topk_freq={self.index_topk_freq}, "
                    f"dsa_indexer_skip_topk_offset={self.index_skip_topk_offset}). "
                    f"Holder has layers {sorted(topk_holder)}."
                )
            topk_indices = topk_holder[self.source_layer]
            if topk_length_holder is not None:
                topk_length = topk_length_holder.get(self.source_layer)
        else:
            assert self.indexer is not None
            with torch.enable_grad() if use_indexer_loss else torch.no_grad():
                q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
                if cp_size > 1 and k.size(0) in local_cp_kv_lens:
                    if kv_reorder_idx is None:
                        kv_reorder_idx = _build_kv_reorder_idx(k.size(0))
                    k = gather_from_sequence_parallel_region(k, group=cp_group)
                    if k.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered indexer-key length mismatch: "
                            f"k_seqlen={k.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    k = k.index_select(0, kv_reorder_idx)
                if sequence_parallel_tp and q.size(0) != sq:
                    if (
                        q.size(0) != sequence_parallel_tp_full_rows
                        or weights.size(0) != sequence_parallel_tp_full_rows
                    ):
                        raise RuntimeError(
                            "DSA sequence-parallel indexer row count mismatch: "
                            f"q_rows={q.size(0)}, weights_rows={weights.size(0)}, "
                            f"query_rows={sq}, full_rows={sequence_parallel_tp_full_rows}, "
                            f"tp_size={tp_size}"
                        )
                    if not sequence_parallel_query_is_local:
                        raise RuntimeError(
                            "DSA indexer produced TP-gathered rows while attention query rows "
                            "were not sequence-local."
                        )
                    row_start = sequence_parallel_tp_row_start
                    row_end = row_start + sq
                    q = q[row_start:row_end].contiguous()
                    weights = weights[row_start:row_end].contiguous()

        def compute_indexer_loss_with_blockwise_path():
            key_for_loss = key.detach()
            if absorbed_mla and key_for_loss.size(2) == 1 and query.size(2) > 1:
                key_for_loss = key_for_loss.expand(-1, -1, query.size(2), -1)
            return FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query.detach(),
                key_for_loss,
                self.softmax_scale,
                self.index_topk,
                indexer_loss_coeff,
                float_mask,
                sparse_indexer_loss,
                self.pg_collection,
                varlen_starts,
                varlen_ends,
                key_positions,
                query_valid_rows,
                self.config.calculate_per_token_loss,
                self.config.dsa_indexer_scoring_relu,
                self.config.dsa_indexer_loss_block_size,
                # Phase-1 dense warmup does not consume sparse attention output, but its
                # diagnostics still need the indexer's selected keys for top-k and mass recall.
                True,
                quality_metrics,
            )

        fused_output = None
        if use_fused_kernels and not self.index_share:
            assert q is not None and k is not None and weights is not None
            fused_output = dsa_kernels.run_fused_dsa_attention(
                config=self.config,
                query=query,
                key=key,
                value=value,
                up_v_weight=up_v_weight,
                q_indexer=q,
                k_indexer=k,
                indexer_weights=weights,
                indexer_topk=self.index_topk,
                softmax_scale=self.softmax_scale,
                loss_coeff=indexer_loss_coeff if use_indexer_loss else 0.0,
                sparse_loss=sparse_indexer_loss,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                absorbed_mla=absorbed_mla,
                cp_size=cp_size,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                query_valid_rows=query_valid_rows,
                varlen_is_plain_causal=varlen_is_plain_causal,
                use_relu=self.config.dsa_indexer_scoring_relu,
                use_local_indexer_varlen=use_local_indexer_varlen,
                single_packed_thd_sequence=single_packed_thd_sequence,
                local_packed_cp_rank=cp_rank,
                local_packed_cp_query_start=local_packed_cp_query_start,
                local_packed_cp_query_len=local_packed_cp_query_len,
                pg_collection=self.pg_collection,
            )
        if fused_output is not None:
            output, indexer_loss = fused_output
            if use_indexer_loss:
                if indexer_loss is None:
                    raise RuntimeError("Fused DSA attention did not produce a valid indexer loss.")
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                    loss_coeff=indexer_loss_coeff,
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return _normalize_dsattention_output_rank(output, x.ndim)

        fused_bounds = None
        if use_fused_kernels and computes_topk:
            assert q is not None
            fused_bounds = dsa_masking.build_fused_indexer_varlen_bounds(
                sq=sq,
                skv=skv,
                device=q.device,
                mask=float_mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )

        indexer_loss = None

        def slice_topk_to_local_sequence_parallel_rows():
            nonlocal topk_indices, topk_length
            if topk_indices is None or not sequence_parallel_query_is_local:
                return
            topk_sq = topk_indices.size(1)
            if topk_sq == sq:
                return
            expected_topk_sq = sequence_parallel_tp_full_rows
            if topk_sq != expected_topk_sq:
                raise RuntimeError(
                    "DSA sequence-parallel top-k row count mismatch: "
                    f"topk_rows={topk_sq}, query_rows={sq}, tp_size={tp_size}"
                )
            row_start = sequence_parallel_tp_row_start
            row_end = row_start + sq
            topk_indices = topk_indices[:, row_start:row_end].contiguous()
            if topk_length is not None:
                topk_length = topk_length[:, row_start:row_end].contiguous()

        if use_indexer_loss:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Attach indexer topk and loss
            # ===================================
            if sparse_indexer_loss and fused_bounds is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk_with_loss = dsa_kernels.run_fused_qk_topk_with_loss(
                    self.config,
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                    query=query.detach(),
                    key=key.detach(),
                    softmax_scale=self.softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    pg_collection=self.pg_collection,
                    query_valid_rows=query_valid_rows,
                    calculate_per_token_loss=self.config.calculate_per_token_loss,
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                )
                if fused_topk_with_loss is not None:
                    topk_indices, topk_length, indexer_loss = fused_topk_with_loss

            if topk_indices is None or indexer_loss is None:
                topk_indices, indexer_loss = compute_indexer_loss_with_blockwise_path()
            # No TP-local top-k slicing here: the guard above forbids the indexer loss
            # under sequence-local TP query shards, so the top-k rows are already global.

            # Save indexer loss for logging.
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                    loss_coeff=indexer_loss_coeff,
                    quality_metrics=quality_metrics or None,
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
        elif topk_indices is None:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Get top-k indices
            # ===================================
            if fused_bounds is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk = dsa_kernels.run_fused_qk_topk(
                    self.config,
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                )
                if fused_topk is not None:
                    topk_indices, topk_length = fused_topk

            if topk_indices is None:
                with torch.no_grad():
                    index_scores, topk_indices = fused_qk_topk_naive(
                        q,
                        k,
                        weights,
                        self.index_topk,
                        mask=float_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        use_relu=self.config.dsa_indexer_scoring_relu,
                    )
                    del index_scores
            slice_topk_to_local_sequence_parallel_rows()

        if self.index_share and computes_topk:
            assert topk_holder is not None and topk_indices is not None
            topk_holder[self.layer_number] = topk_indices
            if topk_length_holder is not None and topk_length is not None:
                topk_length_holder[self.layer_number] = topk_length

        # ===================================
        # Run sparse attention kernel
        # ===================================
        if dense_output is not None:
            output = dense_output
        else:
            output = _run_sparse_attention(
                absorbed_mla=absorbed_mla,
                query=query,
                key=key,
                value=value,
                up_v_weight=up_v_weight,
                topk_indices=topk_indices,
                topk_length=topk_length,
                softmax_scale=self.softmax_scale,
                config=self.config,
                mask=float_mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )

        if use_indexer_loss:
            if indexer_loss is None:
                raise RuntimeError("Indexer loss path did not produce a valid loss tensor.")
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return _normalize_dsattention_output_rank(output, x.ndim)
