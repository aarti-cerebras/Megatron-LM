# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_gqa_module_spec_for_backend,
    get_dsa_layer_pattern,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    _compute_grouped_attention_scores,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _FakeBackend:
    def linear(self):
        return torch.nn.Linear

    def column_parallel_linear(self):
        return torch.nn.Linear

    def row_parallel_linear(self):
        return torch.nn.Linear

    def core_attention(self):
        return torch.nn.Module

    def layer_norm(self, **_kwargs):
        return torch.nn.LayerNorm


def _dsa_gqa_config(**overrides):
    values = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=16,
        num_query_groups=4,
        experimental_attention_variant="dsa_gqa",
        transformer_impl="transformer_engine",
        dsa_layer_freq=2,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=1.0,
        dsa_dense_warmup=True,
        dsa_freeze_base=True,
        dsa_kernel_backend="none",
        window_size=(127, 0),
        window_attn_skip_freq=2,
        softmax_type="vanilla",
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_grouped_teacher_scores_match_explicit_gqa_reference():
    torch.manual_seed(123)
    query = torch.randn(5, 2, 8, 4)
    key = torch.randn(7, 2, 2, 4)

    actual = _compute_grouped_attention_scores(query, key, 0.5)
    expanded_key = key.repeat_interleave(4, dim=2)
    expected = torch.einsum(
        "bhsd,bhdk->bhsk",
        query.permute(1, 2, 0, 3).float(),
        expanded_key.permute(1, 2, 3, 0).float(),
    ) * 0.5

    torch.testing.assert_close(actual, expected)


def test_dsa_gqa_phase1_spec_uses_standard_attention_and_external_norm():
    config = _dsa_gqa_config()
    spec = get_dsa_gqa_module_spec_for_backend(config, backend=_FakeBackend())

    assert spec.module is SelfAttention
    assert spec.metainfo == {"fuse_input_layernorm": False}
    assert spec.submodules.core_attention.module is DSAttention
    assert spec.submodules.core_attention.submodules.dense_attention is torch.nn.Module


def test_dsa_layer_integer_pattern_selects_every_nth_layer():
    config = SimpleNamespace(num_layers=6, dsa_layer_freq=2)
    assert get_dsa_layer_pattern(config) == [0, 1, 0, 1, 0, 1]


def test_dsa_gqa_rejects_sliding_window_overlap():
    with pytest.raises(ValueError, match="full-attention layers"):
        _dsa_gqa_config(dsa_layer_freq=[1, 0, 1, 0])


def test_dsa_gqa_rejects_incoherent_dense_warmup():
    with pytest.raises(ValueError, match="requires dsa_freeze_base=True"):
        _dsa_gqa_config(dsa_freeze_base=False)
