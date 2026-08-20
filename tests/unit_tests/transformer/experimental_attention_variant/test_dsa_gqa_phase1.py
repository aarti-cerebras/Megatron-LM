# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_gqa_module_spec_for_backend,
    get_dsa_layer_pattern,
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper,
    DSAttention,
    FusedDSAIndexerLoss,
    _compute_grouped_attention_scores,
    _compute_index_scores,
    _fake_quant_fp8,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.training.checkpointing import _validate_dsa_phase1_checkpoint_mismatch


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

    def fuse_layernorm_and_linear(self):
        return False

    def activation_func(self):
        return None


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
        dsa_indexer_fp8=True,
        dsa_indexer_fp8_ue8m0=True,
        dsa_indexer_serving_compat=True,
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
    expected = (
        torch.einsum(
            "bhsd,bhdk->bhsk",
            query.permute(1, 2, 0, 3).float(),
            expanded_key.permute(1, 2, 3, 0).float(),
        )
        * 0.5
    )

    torch.testing.assert_close(actual, expected)


def test_indexer_metric_averages_only_active_dsa_layers(monkeypatch):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(helper, "reduce_loss_in_tracker", lambda *args, **kwargs: None)

    # Two microbatches on DSA layers 2 and 4. Layer 2 has a valid zero KL and must still count.
    helper.save_loss_to_tracker(torch.tensor(0.0), layer_number=2, num_layers=4, loss_coeff=2.0)
    helper.save_loss_to_tracker(torch.tensor(0.0), layer_number=2, num_layers=4, loss_coeff=2.0)
    helper.save_loss_to_tracker(torch.tensor(8.0), layer_number=4, num_layers=4, loss_coeff=2.0)
    helper.save_loss_to_tracker(torch.tensor(4.0), layer_number=4, num_layers=4, loss_coeff=2.0)

    total_loss_dict = {}
    helper.track_indexer_metrics(
        loss_scale=0.5, iteration=1, writer=None, total_loss_dict=total_loss_dict, loss_coeff=2.0
    )

    # Microbatch means are 0 and 6 for the two active layers, hence an active-layer mean of 3.
    torch.testing.assert_close(total_loss_dict["indexer loss"], torch.tensor(3.0))
    torch.testing.assert_close(total_loss_dict["indexer kl"], torch.tensor(1.5))
    torch.testing.assert_close(total_loss_dict["indexer loss coefficient"], torch.tensor(2.0))
    assert helper.tracker["active_layers"].count_nonzero() == 0


def test_indexer_metric_initializes_pipeline_stage_without_dsa_layers(monkeypatch):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    def simulate_pipeline_reduction(*args, **kwargs):
        helper.tracker["values"][3] = 8.0
        helper.tracker["active_layers"][3] = 1

    monkeypatch.setattr(helper, "reduce_loss_in_tracker", simulate_pipeline_reduction)
    total_loss_dict = {}

    helper.track_indexer_metrics(
        loss_scale=1.0,
        iteration=1,
        writer=None,
        total_loss_dict=total_loss_dict,
        num_layers=4,
        loss_coeff=0.25,
    )

    torch.testing.assert_close(total_loss_dict["indexer loss"], torch.tensor(8.0))
    torch.testing.assert_close(total_loss_dict["indexer loss coefficient"], torch.tensor(0.25))


def test_indexer_metric_reduction_uses_explicit_process_groups(monkeypatch):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    helper._initialize_tracker(num_layers=2)

    pipeline_group = object()
    data_parallel_group = object()
    calls = []

    def record_all_reduce(tensor, group=None, op=None):
        calls.append((tensor, group, op))

    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)

    helper.reduce_loss_in_tracker(
        pipeline_group=pipeline_group, data_parallel_group=data_parallel_group
    )

    reduced_groups = [group for _, group, _ in calls]
    quality_metric_count = 5
    assert reduced_groups.count(pipeline_group) == 4
    assert reduced_groups.count(data_parallel_group) == 3
    packed_reductions = [tensor for tensor, _, _ in calls if tensor.ndim == 2]
    assert len(packed_reductions) == 2
    assert packed_reductions[0].shape == (2 * quality_metric_count, 2)
    helper.tracker.clear()


def test_blockwise_phase1_metrics_match_explicit_teacher_reference():
    torch.manual_seed(123)
    sq = sk = 5
    q = torch.randn(sq, 1, 2, 3)
    k = torch.randn(sk, 1, 3)
    weights = torch.randn(sq, 1, 2)
    query = torch.randn(sq, 1, 4, 3)
    key = torch.randn(sk, 1, 4, 3)
    mask = torch.zeros(sq, sk)
    metrics = {}
    pg_collection = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1))

    topk_indices, _ = FusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key,
        0.5,
        2,
        0.25,
        mask,
        False,
        pg_collection,
        None,
        None,
        None,
        None,
        False,
        True,
        2,
        True,
        metrics,
    )

    teacher_probs = _compute_grouped_attention_scores(query, key, 0.5).softmax(dim=-1).mean(dim=1)
    teacher_topk = teacher_probs.topk(2, dim=-1).indices
    overlap = (teacher_topk.unsqueeze(-1) == topk_indices.unsqueeze(-2)).any(dim=-1)
    expected_topk_recall = overlap.float().mean()
    indexer_attention_mass = torch.gather(teacher_probs, -1, topk_indices).sum(dim=-1)
    teacher_topk_probabilities = torch.gather(teacher_probs, -1, teacher_topk)
    teacher_topk_attention_mass = teacher_topk_probabilities.sum(dim=-1)
    topk_intersection_attention_mass = (teacher_topk_probabilities * overlap).sum(dim=-1)
    expected_attention_score_recall = (
        topk_intersection_attention_mass.sum() / teacher_topk_attention_mass.sum()
    )

    torch.testing.assert_close(
        metrics["topk_recall_sum"] / metrics["topk_recall_count"], expected_topk_recall
    )
    torch.testing.assert_close(
        metrics["indexer_attention_mass_sum"] / metrics["indexer_attention_mass_count"],
        indexer_attention_mass.mean(),
    )
    torch.testing.assert_close(
        metrics["teacher_topk_attention_mass_sum"] / metrics["teacher_topk_attention_mass_count"],
        teacher_topk_attention_mass.mean(),
    )
    torch.testing.assert_close(
        metrics["topk_intersection_attention_mass_sum"]
        / metrics["topk_intersection_attention_mass_count"],
        topk_intersection_attention_mass.mean(),
    )
    torch.testing.assert_close(
        metrics["attention_score_recall_sum"] / metrics["attention_score_recall_count"],
        expected_attention_score_recall,
    )
    assert torch.all(teacher_topk_attention_mass >= indexer_attention_mass)
    assert torch.all(indexer_attention_mass >= topk_intersection_attention_mass)


def test_indexer_tracker_logs_per_layer_kl_and_quality_metrics(monkeypatch):
    helper = DSAIndexerLossLoggingHelper
    helper.tracker.clear()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(helper, "reduce_loss_in_tracker", lambda *args, **kwargs: None)

    quality_metrics = {
        "topk_recall_sum": torch.tensor(3.0),
        "topk_recall_count": torch.tensor(4.0),
        "indexer_attention_mass_sum": torch.tensor(1.5),
        "indexer_attention_mass_count": torch.tensor(2.0),
        "teacher_topk_attention_mass_sum": torch.tensor(1.8),
        "teacher_topk_attention_mass_count": torch.tensor(2.0),
        "topk_intersection_attention_mass_sum": torch.tensor(1.2),
        "topk_intersection_attention_mass_count": torch.tensor(2.0),
        "attention_score_recall_sum": torch.tensor(1.2),
        "attention_score_recall_count": torch.tensor(1.8),
    }
    helper.save_loss_to_tracker(
        torch.tensor(2.0),
        layer_number=2,
        num_layers=4,
        loss_coeff=0.5,
        quality_metrics=quality_metrics,
    )

    class _Writer:
        def __init__(self):
            self.values = {}

        def add_scalar(self, name, value, iteration):
            self.values[name] = (value, iteration)

    writer = _Writer()
    total_loss_dict = {}
    helper.track_indexer_metrics(
        loss_scale=1.0,
        iteration=7,
        writer=writer,
        total_loss_dict=total_loss_dict,
        per_layer_logging=True,
        loss_coeff=0.5,
    )

    torch.testing.assert_close(writer.values["indexer kl/layer 2"][0], torch.tensor(4.0))
    torch.testing.assert_close(total_loss_dict["indexer topk recall"], torch.tensor(0.75))
    torch.testing.assert_close(
        total_loss_dict["attention mass captured by indexer"], torch.tensor(0.75)
    )
    torch.testing.assert_close(
        total_loss_dict["attention mass captured by teacher top-k"], torch.tensor(0.9)
    )
    torch.testing.assert_close(
        total_loss_dict["attention mass captured by top-k intersection"], torch.tensor(0.6)
    )
    torch.testing.assert_close(total_loss_dict["attention score recall"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(
        writer.values["attention score recall/layer 2"][0], torch.tensor(2.0 / 3.0)
    )
    assert writer.values["indexer loss coefficient"] == (0.5, 7)


def test_dsa_gqa_phase1_spec_uses_standard_attention_and_external_norm():
    config = _dsa_gqa_config()
    spec = get_dsa_gqa_module_spec_for_backend(config, backend=_FakeBackend())

    assert spec.module is SelfAttention
    assert spec.metainfo == {"fuse_input_layernorm": False}
    assert spec.submodules.core_attention.module is DSAttention
    assert spec.submodules.core_attention.submodules.dense_attention is torch.nn.Module


def test_dsa_gqa_layer_spec_retargets_dense_checkpoint_keys():
    config = _dsa_gqa_config(dsa_layer_freq=1, window_size=None, window_attn_skip_freq=None)

    layer_spec = get_transformer_layer_with_experimental_attention_variant_spec(
        config, backend=_FakeBackend()
    )[0]
    expected = {
        "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
        "self_attention.core_attention.dense_attention.softmax_offset": (
            "self_attention.core_attention.softmax_offset"
        ),
    }

    assert layer_spec.submodules.sharded_state_dict_keys_map == expected
    assert layer_spec.submodules.load_state_dict_keys_map == expected
    assert layer_spec.submodules.sharded_state_dict_non_homogeneous_prefixes == (
        "input_layernorm._extra_state",
        "self_attention.core_attention.indexer.",
        "self_attention.core_attention.dense_attention._extra_state",
    )


def test_mixed_dsa_gqa_specs_keep_type_specific_extra_state_per_layer():
    specs = get_transformer_layer_with_experimental_attention_variant_spec(
        _dsa_gqa_config(), backend=_FakeBackend()
    )

    assert specs[0].submodules.sharded_state_dict_non_homogeneous_prefixes == (
        "self_attention.core_attention._extra_state",
    )
    assert specs[1].submodules.sharded_state_dict_non_homogeneous_prefixes == (
        "input_layernorm._extra_state",
        "self_attention.core_attention.indexer.",
        "self_attention.core_attention.dense_attention._extra_state",
    )


def test_transformer_layer_remaps_dense_checkpoint_keys_for_torch_load():
    checkpoint_keys_map = {
        "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
        "self_attention.core_attention.dense_attention.softmax_offset": (
            "self_attention.core_attention.softmax_offset"
        ),
    }
    module = SimpleNamespace(
        submodules_config=TransformerLayerSubmodules(load_state_dict_keys_map=checkpoint_keys_map)
    )
    norm = torch.randn(8)
    sink = torch.randn(4)
    state_dict = {
        "decoder.layers.1.self_attention.linear_qkv.layer_norm_weight": norm,
        "decoder.layers.1.self_attention.core_attention.softmax_offset": sink,
    }

    TransformerLayer._remap_checkpoint_state_dict_keys(
        module, state_dict, "decoder.layers.1.", {}, True, [], [], []
    )

    assert state_dict == {
        "decoder.layers.1.input_layernorm.weight": norm,
        "decoder.layers.1.self_attention.core_attention.dense_attention.softmax_offset": sink,
    }


def test_dsa_phase1_checkpoint_validation_allows_only_new_indexer_state():
    _validate_dsa_phase1_checkpoint_mismatch(
        absent_model_keys={
            "decoder.layers.1.self_attention.core_attention.indexer.linear_wk.weight",
            "decoder.layers.1.self_attention.linear_qkv._extra_state",
        },
        unused_checkpoint_keys={"decoder.layers.1.mlp.linear_fc1._extra_state"},
        checkpoint_kind="test checkpoint",
    )


@pytest.mark.parametrize(
    ("absent_model_keys", "unused_checkpoint_keys"),
    [
        ({"decoder.layers.1.input_layernorm.weight"}, set()),
        (set(), {"decoder.layers.1.self_attention.core_attention.softmax_offset"}),
    ],
)
def test_dsa_phase1_checkpoint_validation_rejects_backbone_mismatch(
    absent_model_keys, unused_checkpoint_keys
):
    with pytest.raises(RuntimeError, match="partially loaded backbone"):
        _validate_dsa_phase1_checkpoint_mismatch(
            absent_model_keys, unused_checkpoint_keys, "test checkpoint"
        )


def test_dsa_layer_integer_pattern_selects_every_nth_layer():
    config = SimpleNamespace(num_layers=6, dsa_layer_freq=2)
    assert get_dsa_layer_pattern(config) == [0, 1, 0, 1, 0, 1]


def test_dsa_gqa_rejects_sliding_window_overlap():
    with pytest.raises(ValueError, match="full-attention layers"):
        _dsa_gqa_config(dsa_layer_freq=[1, 0, 1, 0])


def test_dsa_gqa_rejects_incoherent_dense_warmup():
    with pytest.raises(ValueError, match="requires dsa_freeze_base=True"):
        _dsa_gqa_config(dsa_freeze_base=False)


def test_dsa_gqa_phase2_rejects_attention_dropout():
    with pytest.raises(ValueError, match="requires attention_dropout=0.0"):
        _dsa_gqa_config(dsa_dense_warmup=False, dsa_freeze_base=False)

    config = _dsa_gqa_config(dsa_dense_warmup=False, dsa_freeze_base=False, attention_dropout=0.0)
    assert config.attention_dropout == 0.0


def test_dsa_gqa_rejects_nonpositive_loss_block_size():
    with pytest.raises(ValueError, match="dsa_indexer_loss_block_size must be positive"):
        _dsa_gqa_config(dsa_indexer_loss_block_size=0)


def test_fp8_fake_quant_matches_fixed_ue8m0_values_and_preserves_gradient():
    values = torch.tensor(
        [[-3.25, -0.13, 0.1, 1.3, -2.7], [5.0, 0.1, 0.3, 1.1, -4.1]], requires_grad=True
    )
    expected = torch.tensor(
        [[-3.25, -0.125, 0.1015625, 1.25, -2.75], [5.0, 0.1015625, 0.3125, 1.125, -4.0]]
    )

    actual = _fake_quant_fp8(values, use_ue8m0=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_fp8_fake_quant_preserves_bf16_dtype_and_gradient():
    values = torch.tensor(
        [[-3.25, -0.13, 0.1, 1.3], [5.0, 0.1, 0.3, 1.1]], dtype=torch.bfloat16, requires_grad=True
    )

    actual = _fake_quant_fp8(values)

    assert actual.dtype == torch.bfloat16
    actual.sum().backward()
    assert values.grad.dtype == torch.bfloat16
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_fp8_fake_quant_tracks_full_precision_index_scores():
    torch.manual_seed(123)
    q = torch.randn(16, 1, 4, 64)
    k = torch.randn(16, 1, 64)
    weights = torch.rand(16, 1, 4)

    reference = _compute_index_scores(q, weights, k)
    quantized = _compute_index_scores(_fake_quant_fp8(q), weights, _fake_quant_fp8(k))

    relative_error = (quantized - reference).norm() / reference.norm()
    assert relative_error.item() < 0.05


def test_fp8_rowwise_quantization_preserves_serving_padding_scores():
    torch.manual_seed(123)
    q = torch.randn(12, 1, 4, 64)
    k = torch.randn(12, 1, 64)
    weights = torch.randn(12, 1, 4)

    quantized_q = _fake_quant_fp8(q)
    quantized_k = _fake_quant_fp8(k)
    training_scores = _compute_index_scores(quantized_q, weights, quantized_k)

    padded_q = torch.nn.functional.pad(q, (0, 64, 0, 28))
    padded_k = torch.nn.functional.pad(k, (0, 64))
    padded_weights = torch.nn.functional.pad(weights, (0, 28))
    padded_quantized_q = _fake_quant_fp8(padded_q)
    padded_quantized_k = _fake_quant_fp8(padded_k)
    serving_scores = _compute_index_scores(padded_quantized_q, padded_weights, padded_quantized_k)

    torch.testing.assert_close(padded_quantized_q[:, :, :4, :64], quantized_q)
    torch.testing.assert_close(padded_quantized_k[..., :64], quantized_k)
    assert torch.count_nonzero(padded_quantized_q[:, :, 4:]).item() == 0
    assert torch.count_nonzero(padded_quantized_q[..., 64:]).item() == 0
    assert torch.count_nonzero(padded_quantized_k[..., 64:]).item() == 0
    torch.testing.assert_close(training_scores, serving_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_indexer_fp8": False}, "requires dsa_indexer_fp8=True"),
        ({"dsa_indexer_fp8_ue8m0": False}, "requires UE8M0"),
        ({"dsa_indexer_rotate_activation": False}, "requires Hadamard"),
        ({"dsa_indexer_fp8_block_size": 64}, "block_size=128"),
        ({"dsa_indexer_head_dim": 48}, "head_dim in"),
        ({"dsa_indexer_n_heads": 0}, "n_heads to divide"),
        ({"dsa_indexer_n_heads": 12}, "n_heads to divide"),
    ],
)
@pytest.mark.parametrize("variant", ["dsa", "dsa_gqa"])
def test_all_dsa_variants_reject_mismatched_serving_numerics(variant, overrides, message):
    with pytest.raises(ValueError, match=message):
        _dsa_gqa_config(experimental_attention_variant=variant, add_bias_linear=False, **overrides)


@pytest.mark.parametrize("variant", ["dsa", "dsa_gqa"])
def test_all_dsa_variants_validate_fp8_row_layout(variant):
    with pytest.raises(ValueError, match="one scale per indexer row"):
        _dsa_gqa_config(
            experimental_attention_variant=variant,
            add_bias_linear=False,
            dsa_indexer_serving_compat=False,
            dsa_indexer_fp8_block_size=32,
        )
