"""The MXFP4 dequantizer on a hand-made block, and, when the checkpoints are reachable, the
Megatron -> HF mapping bit for bit against the frozen Phase-1 base.

Set GPT_OSS_DSA_MEGATRON_CKPT (a Phase-1 iter dir or checkpoints root) and GPT_OSS_DSA_HF_BASE
(the openai/gpt-oss-20b directory) to run the checkpoint tests; they read a few hundred MB.
"""

import os

import pytest
import torch

from gpt_oss_dsa.megatron_ckpt import (
    HFCheckpoint,
    MegatronGptOss,
    compare_base_to_hf,
    dequant_mxfp4,
)


def test_dequant_mxfp4_nibbles_and_scales():
    # Byte 0x21: low nibble 1 -> 0.5 (even element), high nibble 2 -> 1.0 (odd element).
    # Byte 0x9F: low nibble 0xF -> -6.0, high nibble 9 -> -0.5.
    blocks = torch.zeros(1, 16, dtype=torch.uint8)
    blocks[0, 0] = 0x21
    blocks[0, 1] = 0x9F
    scales = torch.tensor([127 + 3], dtype=torch.uint8)  # 2**3
    out = dequant_mxfp4(blocks, scales)
    assert out.shape == (32,) and out.dtype == torch.bfloat16
    assert out[:4].tolist() == [4.0, 8.0, -48.0, -4.0]
    assert torch.equal(out[4:], torch.zeros(28, dtype=torch.bfloat16))


CKPT = os.environ.get("GPT_OSS_DSA_MEGATRON_CKPT")
HF = os.environ.get("GPT_OSS_DSA_HF_BASE")
needs_checkpoints = pytest.mark.skipif(
    not (CKPT and HF), reason="set GPT_OSS_DSA_MEGATRON_CKPT and GPT_OSS_DSA_HF_BASE"
)


@pytest.fixture(scope="module")
def megatron():
    return MegatronGptOss(CKPT)


@pytest.fixture(scope="module")
def hf():
    return HFCheckpoint(HF)


@needs_checkpoints
def test_geometry_is_gpt_oss_20b_with_indexers_on_full_attention_layers(megatron, hf):
    g = megatron.geometry
    assert (g.num_layers, g.hidden_size, g.num_heads, g.num_kv_heads, g.head_dim) == (24, 2880, 64, 8, 64)
    assert (g.num_experts, g.intermediate_size) == (32, 2880)
    assert g.indexer_layers == tuple(i for i, t in enumerate(hf.config["layer_types"]) if t == "full_attention")
    assert (g.indexer_heads, g.indexer_head_dim) == (16, 64)


@needs_checkpoints
def test_phase1_base_matches_hf_bit_for_bit(megatron, hf):
    compared, mismatches = compare_base_to_hf(megatron, hf, layers=range(0, 2), experts=(0, 31))
    assert compared == 45, compared
    assert mismatches == [], mismatches[:5]


@needs_checkpoints
def test_indexer_tensors_have_the_documented_shapes(megatron):
    layer = megatron.geometry.indexer_layers[0]
    shapes = {k: tuple(v.shape) for k, v in megatron.indexer(layer).items()}
    assert shapes == {
        "self_attn.indexer.wq.weight": (1024, 2880),
        "self_attn.indexer.wk.weight": (64, 2880),
        "self_attn.indexer.k_norm.weight": (64,),
        "self_attn.indexer.k_norm.bias": (64,),
        "self_attn.indexer.weights_proj.weight": (16, 2880),
    }
