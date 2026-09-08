"""Build a vLLM serving directory for GPT-OSS + DSA from a Megatron torch_dist checkpoint.

Two sources of base weights:

  --base-weights hf        The original HF checkpoint's shards are linked in unchanged (MXFP4
                           experts included) and only the indexers are added. Licensed by a
                           bit-exact comparison of the Megatron base against those shards, which
                           a Phase-1 (frozen backbone) checkpoint passes and a Phase-2 one does not.
  --base-weights megatron  Every tensor is exported from the Megatron checkpoint into HF's bf16
                           layout. Required for Phase 2, where the backbone trained.

The output is a normal HF model directory plus the `dsa_*` contract keys in config.json
(gpt_oss_dsa/config.py), served by the `GptOssDSAForCausalLM` plugin. Runs on CPU with torch and
safetensors only; the Megatron container is not needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt_oss_dsa.config import ARCH, DSAServingConfig, IndexerRope  # noqa: E402
from gpt_oss_dsa.megatron_ckpt import (  # noqa: E402
    INDEXER_LEAVES,
    HFCheckpoint,
    MegatronGptOss,
    compare_base_to_hf,
)

INDEX_FILE = "model.safetensors.index.json"


def safetensors_tensor_bytes(path: Path) -> dict[str, int]:
    """Byte size of every tensor in a safetensors file, from its header alone."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return {
        name: info["data_offsets"][1] - info["data_offsets"][0]
        for name, info in header.items()
        if name != "__metadata__"
    }


def indexer_rope_from_hf(hf_config: dict, dim: int, overrides: dict) -> IndexerRope:
    """The rotary geometry megatron's DSAIndexer derives from the GPT-OSS training arguments.

    Training passes the model's --rotary-base and YaRN scaling/beta/original-length arguments
    through to the indexer, but constructs its YarnRotaryEmbedding with the default mscale (1.0),
    mscale_all_dim (0.0) and correction_range_round_to_int (True), whatever the model uses.
    """
    scaling = hf_config.get("rope_scaling") or {}
    theta = hf_config.get("rope_theta")
    if theta is None:
        raise ValueError("HF config has no rope_theta")
    fields = dict(
        type="yarn" if scaling.get("rope_type") == "yarn" else "rope",
        dim=dim,
        theta=float(theta),
        scaling_factor=float(scaling.get("factor", 1.0)),
        original_max_position_embeddings=int(scaling.get("original_max_position_embeddings", 4096)),
        beta_fast=float(scaling.get("beta_fast", 32.0)),
        beta_slow=float(scaling.get("beta_slow", 1.0)),
        mscale=1.0,
        mscale_all_dim=0.0,
        correction_range_round_to_int=True,
    )
    unknown = set(overrides) - set(fields)
    if unknown:
        raise ValueError(f"unknown indexer rope override keys: {sorted(unknown)}")
    fields.update(overrides)
    return IndexerRope(**fields)


def check_geometry(mg: MegatronGptOss, hf_config: dict) -> None:
    g = mg.geometry
    expected = {
        "num_hidden_layers": g.num_layers,
        "hidden_size": g.hidden_size,
        "num_attention_heads": g.num_heads,
        "num_key_value_heads": g.num_kv_heads,
        "head_dim": g.head_dim,
        "num_local_experts": g.num_experts,
        "intermediate_size": g.intermediate_size,
    }
    bad = {k: (hf_config.get(k), v) for k, v in expected.items() if hf_config.get(k) != v}
    if bad:
        raise ValueError(f"HF config disagrees with the Megatron checkpoint geometry: {bad}")
    layer_types = hf_config.get("layer_types")
    if layer_types is None:
        raise ValueError("HF config has no layer_types; cannot confirm the DSA layers are full attention")
    not_full = [i for i in g.indexer_layers if layer_types[i] != "full_attention"]
    if not_full:
        raise ValueError(f"indexers on non-full-attention layers {not_full}; layer_types={layer_types}")


class ShardWriter:
    """Writes safetensors shards into `out` and accumulates the index."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def write(self, filename: str, tensors: dict[str, torch.Tensor]) -> None:
        tensors = {k: v.contiguous() for k, v in tensors.items()}
        save_file(tensors, str(self.out / filename), metadata={"format": "pt"})
        for name, tensor in tensors.items():
            if name in self.weight_map:
                raise ValueError(f"{name} written twice ({self.weight_map[name]} and {filename})")
            self.weight_map[name] = filename
            self.total_size += tensor.numel() * tensor.element_size()

    def link(self, src: Path, filename: str, copy: bool) -> None:
        dst = self.out / filename
        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        for name, nbytes in safetensors_tensor_bytes(src).items():
            if name in self.weight_map:
                raise ValueError(f"{name} present in two shards")
            self.weight_map[name] = filename
            self.total_size += nbytes

    def finish(self) -> None:
        index = {"metadata": {"total_size": self.total_size}, "weight_map": self.weight_map}
        (self.out / INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True))


def export_indexers(mg: MegatronGptOss, writer: ShardWriter) -> int:
    tensors = {}
    for layer in mg.geometry.indexer_layers:
        for leaf, tensor in mg.indexer(layer).items():
            tensors[f"model.layers.{layer}.{leaf}"] = tensor
    writer.write("indexer.safetensors", tensors)
    return len(tensors)


def export_base_from_megatron(mg: MegatronGptOss, hf: HFCheckpoint, writer: ShardWriter) -> None:
    g = mg.geometry
    for layer in range(g.num_layers):
        t0 = time.time()
        prefix = f"model.layers.{layer}."
        tensors = {prefix + k: v for k, v in mg.attention(layer).items()}
        tensors.update({prefix + k: v for k, v in mg.experts_hf_bf16(layer).items()})
        writer.write(f"model-layer-{layer:02d}.safetensors", tensors)
        print(f"  layer {layer:2d} exported in {time.time() - t0:.1f}s", flush=True)
    # HF pads the vocabulary further than Megatron does; the rows Megatron never had keep their
    # original HF values so the served model treats those never-produced ids as stock does.
    embed = hf.get("model.embed_tokens.weight")
    lm_head = hf.get("lm_head.weight")
    rows = g.vocab_rows
    writer.write(
        "model-embed.safetensors",
        {"model.embed_tokens.weight": torch.cat([mg.embed(), embed[rows:]], dim=0)},
    )
    writer.write(
        "model-final.safetensors",
        {
            "lm_head.weight": torch.cat([mg.lm_head(), lm_head[rows:]], dim=0),
            "model.norm.weight": mg.final_norm(),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--megatron-checkpoint", required=True, help="iter_NNNNNNN dir or checkpoints root")
    ap.add_argument("--hf-base", required=True, help="original openai/gpt-oss-20b directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", type=int, choices=(1, 2), required=True, help="training phase of the checkpoint")
    ap.add_argument("--index-topk", type=int, required=True, help="keys attended per query; the training top-k for Phase 2")
    ap.add_argument("--base-weights", choices=("hf", "megatron"), default=None, help="default: hf for phase 1, megatron for phase 2")
    ap.add_argument("--copy-base", action="store_true", help="copy HF shards instead of symlinking them")
    ap.add_argument("--verify", choices=("none", "sample", "all"), default="sample",
                    help="compare the Megatron base to HF: every layer's attention/norm/router plus 2 experts per layer (sample) or every expert (all)")
    ap.add_argument("--k-norm-eps", type=float, default=None, help="indexer k LayerNorm eps; default: HF rms_norm_eps (megatron --norm-epsilon)")
    ap.add_argument("--indexer-rope-json", default="{}", help="JSON overriding IndexerRope fields")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()

    base_weights = args.base_weights or ("hf" if args.phase == 1 else "megatron")
    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    t0 = time.time()
    mg = MegatronGptOss(args.megatron_checkpoint)
    hf = HFCheckpoint(args.hf_base)
    hf_config = dict(hf.config)
    check_geometry(mg, hf_config)
    g = mg.geometry
    print(f"checkpoint {mg.iteration_dir}: {g}", flush=True)

    rope = indexer_rope_from_hf(hf_config, g.indexer_head_dim, json.loads(args.indexer_rope_json))
    serving = DSAServingConfig(
        sparse_layer_ids=g.indexer_layers,
        n_heads=g.indexer_heads,
        head_dim=g.indexer_head_dim,
        index_topk=args.index_topk,
        fp8=True,
        fp8_ue8m0=True,
        rotate_activation=True,
        scoring_relu=True,
        k_norm_eps=args.k_norm_eps if args.k_norm_eps is not None else float(hf_config["rms_norm_eps"]),
        rope=rope,
        training_phase=args.phase,
        source=str(mg.iteration_dir),
    )
    serving.validate(g.num_layers, hf_config["layer_types"])

    mismatches = None
    if args.verify != "none":
        experts = range(g.num_experts) if args.verify == "all" else (0, g.num_experts - 1)
        compared, mismatches = compare_base_to_hf(mg, hf, range(g.num_layers), experts)
        print(f"verified {compared} base tensors against HF: {len(mismatches)} mismatches", flush=True)
        for m in mismatches[:12]:
            print(f"  differs: {m.name}: {m.detail}")
        if mismatches and base_weights == "hf":
            raise SystemExit(
                "the Megatron base differs from the HF shards, so linking the HF shards would serve a "
                "different backbone than the one the indexer trained against. Use --base-weights megatron."
            )
        if not mismatches and args.phase == 2:
            print("note: a Phase-2 checkpoint whose base equals HF exactly did not train its backbone", flush=True)
    elif base_weights == "hf":
        raise SystemExit("--base-weights hf requires --verify sample or all; equality is what licenses linking HF shards")

    writer = ShardWriter(out)
    if base_weights == "hf":
        for filename in hf.shard_files():
            writer.link(hf.path / filename, filename, args.copy_base)
        print(f"{'copied' if args.copy_base else 'linked'} {len(hf.shard_files())} HF shards", flush=True)
    else:
        export_base_from_megatron(mg, hf, writer)
    n_indexer = export_indexers(mg, writer)
    writer.finish()
    print(f"wrote {n_indexer} indexer tensors for layers {list(g.indexer_layers)}", flush=True)

    config = dict(hf_config)
    config["architectures"] = [ARCH]
    config.update(serving.to_json())
    if base_weights == "megatron":
        config.pop("quantization_config", None)
        config["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    copied = []
    for name in HFCheckpoint.AUX_FILES:
        src = hf.path / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)

    manifest = {
        "megatron_checkpoint": str(mg.iteration_dir),
        "hf_base": str(hf.path.resolve()),
        "phase": args.phase,
        "base_weights": base_weights,
        "index_topk": args.index_topk,
        "geometry": dataclasses.asdict(g),
        "serving_config": serving.to_json(),
        "verify": args.verify,
        "base_mismatches": None if mismatches is None else len(mismatches),
        "indexer_tensors": n_indexer,
        "indexer_leaves": list(INDEXER_LEAVES),
        "aux_files": copied,
        "total_weight_bytes": writer.total_size,
        "argv": sys.argv,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {out} ({manifest['seconds']}s). Serve with --block-size 64.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
