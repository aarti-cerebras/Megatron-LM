# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention, unfused_dsa_fn


def _full_topk(batch_size: int, query_length: int, key_length: int) -> torch.Tensor:
    return (
        torch.arange(key_length, dtype=torch.int64)
        .view(1, 1, key_length)
        .expand(batch_size, query_length, key_length)
        .contiguous()
    )


def _sink_aware_sparse_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    softmax_offset: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Materialized grouped-query sparse attention with a learned sink."""
    sq, batch_size, num_query_heads, _ = query.shape
    key_length, key_batch_size, num_key_heads, _ = key.shape
    value_length, value_batch_size, num_value_heads, value_dim = value.shape
    assert key_batch_size == value_batch_size == batch_size
    assert value_length == key_length
    assert num_query_heads % num_key_heads == 0
    assert num_query_heads % num_value_heads == 0

    expanded_key = key.repeat_interleave(num_query_heads // num_key_heads, dim=2)
    expanded_value = value.repeat_interleave(num_query_heads // num_value_heads, dim=2)
    logits = torch.einsum("sbhd,tbhd->bhst", query.float(), expanded_key.float()) * softmax_scale

    support = torch.zeros(
        (batch_size, sq, key_length), dtype=torch.bool, device=topk_indices.device
    )
    valid_topk = topk_indices >= 0
    safe_topk = topk_indices.clamp_min(0)
    support.scatter_reduce_(-1, safe_topk, valid_topk, reduce="amax", include_self=True)
    if mask.ndim == 2:
        valid = support & torch.isfinite(mask).unsqueeze(0)
        logits = logits + mask.view(1, 1, sq, key_length)
    else:
        valid = support & torch.isfinite(mask)
        logits = logits + mask.view(batch_size, 1, sq, key_length)
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

    sink_logits = softmax_offset.float().view(1, num_query_heads, 1, 1)
    sink_logits = sink_logits.expand(batch_size, num_query_heads, sq, 1)
    probabilities = torch.softmax(torch.cat((sink_logits, logits), dim=-1), dim=-1)[..., 1:]
    output = torch.einsum("bhst,tbhd->sbhd", probabilities, expanded_value.float()).to(value.dtype)
    return output.reshape(sq, batch_size, num_query_heads * value_dim)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
def test_sink_aware_sparse_attention_matches_grouped_dense_reference_and_gradients(num_kv_heads):
    torch.manual_seed(123)
    sq = key_length = 5
    batch_size, num_query_heads, head_dim, value_dim = 2, 4, 3, 2
    softmax_scale = head_dim**-0.5
    topk_indices = _full_topk(batch_size, sq, key_length)
    causal_mask = torch.triu(
        torch.full((sq, key_length), float("-inf"), dtype=torch.float32), diagonal=1
    )
    probe = torch.randn(sq, batch_size, num_query_heads * value_dim)

    inputs = (
        torch.randn(sq, batch_size, num_query_heads, head_dim),
        torch.randn(key_length, batch_size, num_kv_heads, head_dim),
        torch.randn(key_length, batch_size, num_kv_heads, value_dim),
        torch.randn(num_query_heads),
    )
    actual_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)

    actual = unfused_dsa_fn(
        actual_inputs[0],
        actual_inputs[1],
        actual_inputs[2],
        topk_indices,
        softmax_scale,
        mask=causal_mask,
        softmax_offset=actual_inputs[3],
    )
    reference = _sink_aware_sparse_reference(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        topk_indices,
        softmax_scale,
        reference_inputs[3],
        causal_mask,
    )

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), actual_inputs)
    reference_gradients = torch.autograd.grad((reference * probe).sum(), reference_inputs)
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=3e-5, atol=3e-6)


def test_sink_is_counted_once_across_multiple_topk_chunks():
    torch.manual_seed(456)
    sq, key_length = 2, 1025
    batch_size, num_query_heads, num_kv_heads = 1, 2, 1
    head_dim, value_dim = 3, 2
    softmax_scale = head_dim**-0.5
    topk_indices = _full_topk(batch_size, sq, key_length)
    no_mask = torch.zeros((sq, key_length), dtype=torch.float32)

    query = torch.randn(sq, batch_size, num_query_heads, head_dim)
    key = torch.randn(key_length, batch_size, num_kv_heads, head_dim)
    value = torch.randn(key_length, batch_size, num_kv_heads, value_dim)
    softmax_offset = torch.tensor([1.5, -0.75], requires_grad=True)
    reference_offset = softmax_offset.detach().clone().requires_grad_()

    actual = unfused_dsa_fn(
        query, key, value, topk_indices, softmax_scale, mask=no_mask, softmax_offset=softmax_offset
    )
    reference = _sink_aware_sparse_reference(
        query, key, value, topk_indices, softmax_scale, reference_offset, no_mask
    )

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    actual_sink_gradient = torch.autograd.grad(actual.square().sum(), softmax_offset)[0]
    reference_sink_gradient = torch.autograd.grad(reference.square().sum(), reference_offset)[0]
    torch.testing.assert_close(actual_sink_gradient, reference_sink_gradient, rtol=3e-5, atol=3e-6)


def test_sink_aware_gqa_supports_packed_thd_boundaries():
    torch.manual_seed(789)
    total_tokens, num_query_heads, num_kv_heads = 6, 4, 2
    head_dim, value_dim = 3, 2
    softmax_scale = head_dim**-0.5
    topk_indices = _full_topk(1, total_tokens, total_tokens)
    positions = torch.arange(total_tokens)
    sequence_start = torch.tensor([0, 0, 0, 3, 3, 3])
    packed_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.float32)
    valid = (positions.unsqueeze(0) >= sequence_start.unsqueeze(1)) & (
        positions.unsqueeze(0) <= positions.unsqueeze(1)
    )
    packed_mask.masked_fill_(~valid, float("-inf"))

    query = torch.randn(total_tokens, num_query_heads, head_dim)
    key = torch.randn(total_tokens, num_kv_heads, head_dim)
    value = torch.randn(total_tokens, num_kv_heads, value_dim)
    softmax_offset = torch.randn(num_query_heads)
    varlen_starts = sequence_start
    varlen_ends = positions + 1

    actual = unfused_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        softmax_offset=softmax_offset,
    )
    reference = _sink_aware_sparse_reference(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        topk_indices,
        softmax_scale,
        softmax_offset,
        packed_mask,
    ).squeeze(1)

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


def test_sparse_softmax_offset_uses_dense_delegate_parameter():
    attention = DSAttention.__new__(DSAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(softmax_type="learnable")
    attention.dense_attention = torch.nn.Module()
    attention.dense_attention.register_parameter(
        "softmax_offset", torch.nn.Parameter(torch.randn(4))
    )

    query = torch.randn(3, 1, 4, 2)
    assert attention._get_sparse_softmax_offset(query) is attention.dense_attention.softmax_offset


def test_sparse_attention_validates_group_and_sink_shapes():
    query = torch.randn(2, 1, 3, 2)
    key = torch.randn(2, 1, 2, 2)
    value = torch.randn(2, 1, 1, 2)
    topk_indices = _full_topk(1, 2, 2)

    with pytest.raises(ValueError, match="divisible by key head count"):
        unfused_dsa_fn(query, key, value, topk_indices, 1.0)

    with pytest.raises(ValueError, match="one sink logit per local query head"):
        unfused_dsa_fn(
            torch.randn(2, 1, 4, 2), key, value, topk_indices, 1.0, softmax_offset=torch.randn(2)
        )
