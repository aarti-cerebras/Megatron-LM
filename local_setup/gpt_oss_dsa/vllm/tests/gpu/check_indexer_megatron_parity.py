"""Serving indexer vs. megatron.core's DSAIndexer, same weights, same input, one GPU.

Runs inside the Megatron container (megatron.core, TransformerEngine, CUDA), which is where the
training indexer exists; vLLM is not needed because the serving side uses its engine-free
`torch_scores` reference. Compares the rotary tables, the full causal score matrix and the top-k
selections. Differences come only from GEMM/LayerNorm kernel choice (TE vs torch), so they should
be at bf16 rounding level; a systematic difference means the two modules disagree on the math.

  cd /workspace/megatron-lm
  PYTHONPATH=.:local_setup/gpt_oss_dsa/vllm torchrun --nproc_per_node 1 \\
      local_setup/gpt_oss_dsa/vllm/tests/gpu/check_indexer_megatron_parity.py \\
      --megatron-checkpoint /cb/ml-eng/aarti/mcore_runs/<run>/checkpoints \\
      --hf-base /cb/ml-eng/aarti/models/gpt-oss-20b --layer 1 --seq 1024 --index-topk 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_serving_dir import indexer_rope_from_hf  # noqa: E402
from gpt_oss_dsa.config import DSAServingConfig  # noqa: E402
from gpt_oss_dsa.indexer import GptOssDSAServingIndexer  # noqa: E402
from gpt_oss_dsa.megatron_ckpt import INDEXER_LEAVES, MegatronGptOss  # noqa: E402


def build_megatron_indexer(cfg: DSAServingConfig, hidden_size: int):
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexer,
        DSAIndexerSubmodules,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=64,
        num_query_groups=8,
        kv_channels=64,
        ffn_hidden_size=hidden_size,
        normalization="RMSNorm",
        layernorm_epsilon=cfg.k_norm_eps,
        bf16=True,
        params_dtype=torch.bfloat16,
        apply_rope_fusion=False,
        dsa_indexer_n_heads=cfg.n_heads,
        dsa_indexer_head_dim=cfg.head_dim,
        dsa_indexer_topk=cfg.index_topk,
        dsa_indexer_fp8=cfg.fp8,
        dsa_indexer_fp8_ue8m0=cfg.fp8_ue8m0,
        dsa_indexer_rotate_activation=cfg.rotate_activation,
        dsa_indexer_scoring_relu=cfg.scoring_relu,
        dsa_indexer_rope_type=cfg.rope.type,
        dsa_indexer_rotary_base=cfg.rope.theta,
        dsa_indexer_rotary_percent=1.0,
    )
    # argument_utils._apply_yarn_config_from_args sets these as dynamic attributes in training.
    config.yarn_rotary_scaling_factor = cfg.rope.scaling_factor
    config.yarn_original_max_position_embeddings = cfg.rope.original_max_position_embeddings
    config.yarn_beta_fast = cfg.rope.beta_fast
    config.yarn_beta_slow = cfg.rope.beta_slow
    config.yarn_mscale = cfg.rope.mscale
    config.yarn_mscale_all_dim = cfg.rope.mscale_all_dim

    backend = TESpecProvider()
    submodules = DSAIndexerSubmodules(
        linear_wq_b=backend.linear(),
        linear_wk=backend.linear(),
        k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
        linear_weights_proj=backend.linear(),
    )
    return DSAIndexer(config, submodules)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--megatron-checkpoint", required=True)
    ap.add_argument("--hf-base", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--index-topk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.distributed.init_process_group("nccl")
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    device = torch.device("cuda", torch.cuda.current_device())

    mg = MegatronGptOss(args.megatron_checkpoint)
    g = mg.geometry
    hf_config = json.loads((Path(args.hf_base) / "config.json").read_text())
    cfg = DSAServingConfig(
        sparse_layer_ids=g.indexer_layers,
        n_heads=g.indexer_heads,
        head_dim=g.indexer_head_dim,
        index_topk=args.index_topk,
        fp8=True,
        fp8_ue8m0=True,
        rotate_activation=True,
        scoring_relu=True,
        k_norm_eps=float(hf_config["rms_norm_eps"]),
        rope=indexer_rope_from_hf(hf_config, g.indexer_head_dim, {}),
        training_phase=1,
    )
    weights = mg.indexer(args.layer)

    with torch.device(device):
        megatron_indexer = build_megatron_indexer(cfg, g.hidden_size)
    state = {INDEXER_LEAVES[k.split("self_attn.indexer.", 1)[1]]: v for k, v in weights.items()}
    result = megatron_indexer.load_state_dict(state, strict=False)
    unexpected_missing = [k for k in result.missing_keys if "_extra_state" not in k]
    if unexpected_missing or result.unexpected_keys:
        raise SystemExit(f"megatron indexer load mismatch: {unexpected_missing} {result.unexpected_keys}")

    serving = GptOssDSAServingIndexer(cfg, g.hidden_size).to(device=device, dtype=torch.bfloat16)
    serving.load_state_dict({k.split("self_attn.indexer.", 1)[1]: v for k, v in weights.items()})

    torch.manual_seed(args.seed)
    seq = args.seq
    x = torch.randn(seq, 1, g.hidden_size, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq, device=device)

    # Rotary tables.
    emb, mscale = megatron_indexer.rotary_pos_emb(seq)
    mg_cos = (torch.cos(emb) * mscale).reshape(seq, -1)
    sv_cos, _ = serving.rotary(positions)
    rot_diff = (mg_cos.float() - sv_cos.float()).abs().max().item()
    print(f"rotary cos max |diff| = {rot_diff:.3e}  (mscale {mscale:.6f} vs {serving.rotary.mscale:.6f})")

    causal = torch.full((seq, seq), float("-inf"), device=device).triu(1)
    with torch.no_grad():
        mg_scores, mg_topk = megatron_indexer.forward_with_scores(x, x, mask=causal)
        sv_scores = serving.torch_scores(x[:, 0], positions)
    mg_scores = mg_scores[0].float()
    finite = torch.isfinite(sv_scores)
    diff = (mg_scores[finite] - sv_scores[finite]).abs()
    scale = mg_scores[finite].abs().max().clamp_min(1e-6)
    print(
        f"scores: max |diff| = {diff.max().item():.3e}, mean |diff| = {diff.mean().item():.3e}, "
        f"max |megatron| = {scale.item():.3e}, relative max = {(diff.max() / scale).item():.3e}"
    )

    sv_topk = serving.torch_topk(sv_scores, args.index_topk)
    overlaps = []
    for t in range(seq):
        a = set(mg_topk[0, t][mg_topk[0, t] >= 0].tolist())
        b = set(sv_topk[t][sv_topk[t] >= 0].tolist())
        overlaps.append(len(a & b) / max(len(a | b), 1))
    overlap = sum(overlaps) / len(overlaps)
    exact_rows = sum(o == 1.0 for o in overlaps)
    print(f"top-{args.index_topk}: mean Jaccard {overlap:.5f}, {exact_rows}/{seq} rows identical")

    ok = rot_diff < 1e-3 and (diff.max() / scale).item() < 1e-2 and overlap > 0.99
    print("PARITY OK" if ok else "PARITY FAILED")
    torch.distributed.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
