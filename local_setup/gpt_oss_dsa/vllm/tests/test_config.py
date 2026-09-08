"""The config.json contract round-trips and rejects geometries the kernels or the model cannot serve."""

import types

import pytest

from gpt_oss_dsa.config import DSAServingConfig, IndexerRope

LAYER_TYPES = ["sliding_attention", "full_attention"] * 12


def make_config(**overrides) -> DSAServingConfig:
    fields = dict(
        sparse_layer_ids=tuple(range(1, 24, 2)),
        n_heads=16,
        head_dim=64,
        index_topk=2048,
        fp8=True,
        fp8_ue8m0=True,
        rotate_activation=True,
        scoring_relu=True,
        k_norm_eps=1e-5,
        rope=IndexerRope(type="yarn", dim=64, theta=150000.0, scaling_factor=32.0),
        training_phase=2,
        source="test",
    )
    fields.update(overrides)
    return DSAServingConfig(**fields)


def as_hf(cfg: DSAServingConfig, **extra) -> types.SimpleNamespace:
    payload = cfg.to_json()
    payload.update({"num_hidden_layers": 24, "layer_types": LAYER_TYPES})
    payload.update(extra)
    return types.SimpleNamespace(**payload)


def test_round_trip():
    cfg = make_config()
    assert DSAServingConfig.from_hf_config(as_hf(cfg)) == cfg


def test_rejects_sliding_window_layer():
    with pytest.raises(ValueError, match="full_attention"):
        make_config(sparse_layer_ids=(0, 1)).validate(24, LAYER_TYPES)


@pytest.mark.parametrize("bad", [dict(head_dim=48), dict(n_heads=24), dict(index_topk=8192), dict(fp8_ue8m0=False)])
def test_rejects_unservable_geometry(bad):
    with pytest.raises(ValueError):
        make_config(**bad).validate(24, LAYER_TYPES)


def test_rejects_topk_disagreement():
    with pytest.raises(ValueError, match="disagree"):
        DSAServingConfig.from_hf_config(as_hf(make_config(), index_topk=1024))


def test_rejects_missing_contract_key():
    hf = as_hf(make_config())
    del hf.dsa_indexer_rope
    with pytest.raises(ValueError, match="dsa_indexer_rope"):
        DSAServingConfig.from_hf_config(hf)
