"""Read a Megatron torch_dist GPT-OSS DSA checkpoint tensor by tensor, in Hugging Face layout.

Uses torch.distributed.checkpoint's file reader directly with a one-tensor plan per call, so a
single (layer, expert) slice of a 25 GB expert tensor costs one 16 MB chunk read and no process
group. torch and safetensors only; the Megatron container is not needed.

Layout facts this module encodes, each verified bit-exact against the original HF checkpoint on
the frozen Phase-1 base:
  * `linear_qkv` rows are grouped per KV head as [q heads of the group..., k, v], each head_dim rows.
  * `linear_fc1` rows are [gate; up] halves; HF `gate_up_proj` interleaves them (even gate, odd up).
  * Expert weights are (out, in) per expert; HF's bf16 layout is (in, out) stacked over experts.
  * `softmax_offset` is HF `sinks`; the DSA layers' external input norm and sink are stored under
    the stock keys (training remaps them), so all 24 layers share one stacked tensor each.
  * The embedding and output layer have 200064 rows (tokenizer vocab padded to 128); HF has 201088.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list

K_EMBED = "embedding.word_embeddings.weight"
K_LM_HEAD = "output_layer.weight"
K_FINAL_NORM = "decoder.final_layernorm.weight"
K_QKV_W = "decoder.layers.self_attention.linear_qkv.weight"
K_QKV_B = "decoder.layers.self_attention.linear_qkv.bias"
K_INPUT_NORM = "decoder.layers.self_attention.linear_qkv.layer_norm_weight"
K_O_W = "decoder.layers.self_attention.linear_proj.weight"
K_O_B = "decoder.layers.self_attention.linear_proj.bias"
K_SINKS = "decoder.layers.self_attention.core_attention.softmax_offset"
K_POST_NORM = "decoder.layers.pre_mlp_layernorm.weight"
K_ROUTER_W = "decoder.layers.mlp.router.weight"
K_ROUTER_B = "decoder.layers.mlp.router.bias"
K_FC1_W = "decoder.layers.mlp.experts.experts.linear_fc1.weight"
K_FC1_B = "decoder.layers.mlp.experts.experts.linear_fc1.bias"
K_FC2_W = "decoder.layers.mlp.experts.experts.linear_fc2.weight"
K_FC2_B = "decoder.layers.mlp.experts.experts.linear_fc2.bias"

# HF-side leaf under `model.layers.N.self_attn.indexer.` -> Megatron leaf under
# `decoder.layers.N.self_attention.core_attention.indexer.`
INDEXER_LEAVES = {
    "wq.weight": "linear_wq_b.weight",
    "wk.weight": "linear_wk.weight",
    "k_norm.weight": "k_norm.weight",
    "k_norm.bias": "k_norm.bias",
    "weights_proj.weight": "linear_weights_proj.weight",
}
_INDEXER_KEY = re.compile(r"^decoder\.layers\.(\d+)\.self_attention\.core_attention\.indexer\.")


class _RegionPlanner(DefaultLoadPlanner):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target

    def resolve_tensor(self, read_item):
        return narrow_tensor_by_index(self.target, read_item.dest_offsets, read_item.lengths)

    def commit_tensor(self, read_item, tensor) -> None:
        pass


class TorchDistReader:
    """Per-tensor, per-region reads from a torch.distributed.checkpoint directory."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.reader = FileSystemReader(self.path)
        self.metadata = self.reader.read_metadata()
        self.reader.set_up_storage_reader(self.metadata, is_coordinator=True)
        self.tensors = {
            k: v for k, v in self.metadata.state_dict_metadata.items() if hasattr(v, "size")
        }

    def keys(self) -> list[str]:
        return list(self.tensors)

    def shape(self, key: str) -> torch.Size:
        return torch.Size(self.tensors[key].size)

    def dtype(self, key: str) -> torch.dtype:
        return self.tensors[key].properties.dtype

    def read(self, key: str, region: dict[int, int] | None = None) -> torch.Tensor:
        """The full tensor, or the slice with the given axes fixed at the given indices."""
        md = self.tensors[key]
        offsets = [0] * len(md.size)
        sizes = list(md.size)
        for axis, index in (region or {}).items():
            if not 0 <= index < md.size[axis]:
                raise IndexError(f"{key}: index {index} out of range on axis {axis}")
            offsets[axis] = index
            sizes[axis] = 1
        chunk = ChunkStorageMetadata(offsets=torch.Size(offsets), sizes=torch.Size(sizes))
        out = torch.empty(sizes, dtype=md.properties.dtype)
        plan = self.reader.prepare_local_plan(
            LoadPlan(create_read_items_for_chunk_list(key, md, [chunk]))
        )
        self.reader.read_data(plan, _RegionPlanner(out)).wait()
        for axis in sorted(region or {}, reverse=True):
            out = out.squeeze(axis)
        return out


class HFCheckpoint:
    """Safetensors reader over an HF model directory with an index file."""

    AUX_FILES = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "chat_template.json",
    )

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._files: dict[str, object] = {}

    def _file(self, name: str):
        filename = self.weight_map[name]
        if filename not in self._files:
            self._files[filename] = safe_open(self.path / filename, framework="pt", device="cpu")
        return self._files[filename]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._file(name).get_tensor(name)

    def get_slice(self, name: str, index: int) -> torch.Tensor:
        return self._file(name).get_slice(name)[index]

    def shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))


FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """uint8 blocks [..., G, 16] and uint8 E8M0 scales [..., G] -> bf16 [..., G * 32].

    The low nibble of each byte is the even element, the high nibble the odd one. Every value is
    an FP4 number times a power of two, so the bf16 result is exact.
    """
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    out = torch.empty(blocks.shape[:-1] + (32,), dtype=torch.float32)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1))
    return out.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32).to(torch.bfloat16)


def resolve_iteration_dir(path: str | os.PathLike) -> Path:
    """Accept an `iter_NNNNNNN` directory or a checkpoints root; the root resolves to its latest."""
    path = Path(path)
    if (path / ".metadata").exists():
        return path
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        iteration = int(tracker.read_text().strip())
        candidate = path / f"iter_{iteration:07d}"
        if (candidate / ".metadata").exists():
            return candidate
    raise FileNotFoundError(f"{path} is neither a torch_dist iteration dir nor a checkpoints root")


@dataclasses.dataclass(frozen=True)
class Geometry:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    intermediate_size: int
    vocab_rows: int
    indexer_layers: tuple[int, ...]
    indexer_heads: int
    indexer_head_dim: int


class MegatronGptOss:
    """HF-layout views of one Megatron GPT-OSS DSA checkpoint iteration."""

    def __init__(self, iteration_dir: str | os.PathLike) -> None:
        self.iteration_dir = resolve_iteration_dir(iteration_dir)
        self.reader = TorchDistReader(self.iteration_dir)
        self.geometry = self._infer_geometry()

    def _infer_geometry(self) -> Geometry:
        r = self.reader
        num_layers, num_heads = r.shape(K_SINKS)
        _, hidden_out, qkv_in = r.shape(K_QKV_W)
        _, o_out, o_in = r.shape(K_O_W)
        head_dim = o_in // num_heads
        num_kv_heads = (hidden_out // head_dim - num_heads) // 2
        if num_kv_heads * (num_heads // num_kv_heads + 2) * head_dim != hidden_out:
            raise ValueError(f"cannot factor linear_qkv rows {hidden_out} into GQA groups")
        fc1_layers, num_experts, two_inter, _ = r.shape(K_FC1_W)
        if fc1_layers != num_layers:
            raise ValueError("expert and attention tensors disagree on the layer count")
        vocab_rows, _ = r.shape(K_EMBED)
        layers = sorted(
            {int(m.group(1)) for k in r.keys() if (m := _INDEXER_KEY.match(k)) is not None}
        )
        if not layers:
            raise ValueError("no DSA indexer tensors in this checkpoint")
        wq_rows, _ = r.shape(self.indexer_key(layers[0], "wq.weight"))
        idx_head_dim, _ = r.shape(self.indexer_key(layers[0], "wk.weight"))
        idx_heads = wq_rows // idx_head_dim
        gates_rows, _ = r.shape(self.indexer_key(layers[0], "weights_proj.weight"))
        if gates_rows != idx_heads or wq_rows != idx_heads * idx_head_dim:
            raise ValueError("indexer projection shapes do not describe one head geometry")
        return Geometry(
            num_layers=num_layers,
            hidden_size=qkv_in,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            intermediate_size=two_inter // 2,
            vocab_rows=vocab_rows,
            indexer_layers=tuple(layers),
            indexer_heads=idx_heads,
            indexer_head_dim=idx_head_dim,
        )

    @staticmethod
    def indexer_key(layer: int, hf_leaf: str) -> str:
        return (
            f"decoder.layers.{layer}.self_attention.core_attention.indexer."
            f"{INDEXER_LEAVES[hf_leaf]}"
        )

    def attention(self, layer: int) -> dict[str, torch.Tensor]:
        """Attention, norms and router of one layer, keyed relative to `model.layers.N.`."""
        g = self.geometry
        r = self.reader
        groups, hd, hidden = g.num_kv_heads, g.head_dim, g.hidden_size
        per_group = g.num_heads // groups
        w = r.read(K_QKV_W, {0: layer}).view(groups, per_group + 2, hd, hidden)
        b = r.read(K_QKV_B, {0: layer}).view(groups, per_group + 2, hd)
        return {
            "self_attn.q_proj.weight": w[:, :per_group].reshape(g.num_heads * hd, hidden),
            "self_attn.k_proj.weight": w[:, per_group].reshape(groups * hd, hidden),
            "self_attn.v_proj.weight": w[:, per_group + 1].reshape(groups * hd, hidden),
            "self_attn.q_proj.bias": b[:, :per_group].reshape(-1),
            "self_attn.k_proj.bias": b[:, per_group].reshape(-1),
            "self_attn.v_proj.bias": b[:, per_group + 1].reshape(-1),
            "self_attn.o_proj.weight": r.read(K_O_W, {0: layer}),
            "self_attn.o_proj.bias": r.read(K_O_B, {0: layer}),
            "self_attn.sinks": r.read(K_SINKS, {0: layer}),
            "input_layernorm.weight": r.read(K_INPUT_NORM, {0: layer}),
            "post_attention_layernorm.weight": r.read(K_POST_NORM, {0: layer}),
            "mlp.router.weight": r.read(K_ROUTER_W, {0: layer}),
            "mlp.router.bias": r.read(K_ROUTER_B, {0: layer}),
        }

    def expert(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """One expert as (out, in) matrices with HF's interleaved gate/up rows."""
        r = self.reader
        fc1 = r.read(K_FC1_W, {0: layer, 1: expert})
        fc1_b = r.read(K_FC1_B, {0: layer, 1: expert})
        inter = fc1.shape[0] // 2
        gate_up = torch.empty_like(fc1)
        gate_up[0::2] = fc1[:inter]
        gate_up[1::2] = fc1[inter:]
        gate_up_b = torch.empty_like(fc1_b)
        gate_up_b[0::2] = fc1_b[:inter]
        gate_up_b[1::2] = fc1_b[inter:]
        return {
            "gate_up": gate_up,
            "gate_up_bias": gate_up_b,
            "down": r.read(K_FC2_W, {0: layer, 1: expert}),
            "down_bias": r.read(K_FC2_B, {0: layer, 1: expert}),
        }

    def experts_hf_bf16(self, layer: int) -> dict[str, torch.Tensor]:
        """All experts of one layer in HF's unquantized layout: gate_up_proj [E, H, 2I],
        gate_up_proj_bias [E, 2I], down_proj [E, I, H], down_proj_bias [E, H]."""
        g = self.geometry
        gate_up = torch.empty(g.num_experts, g.hidden_size, 2 * g.intermediate_size, dtype=torch.bfloat16)
        gate_up_b = torch.empty(g.num_experts, 2 * g.intermediate_size, dtype=torch.bfloat16)
        down = torch.empty(g.num_experts, g.intermediate_size, g.hidden_size, dtype=torch.bfloat16)
        down_b = torch.empty(g.num_experts, g.hidden_size, dtype=torch.bfloat16)
        for e in range(g.num_experts):
            t = self.expert(layer, e)
            gate_up[e] = t["gate_up"].T
            gate_up_b[e] = t["gate_up_bias"]
            down[e] = t["down"].T
            down_b[e] = t["down_bias"]
        return {
            "mlp.experts.gate_up_proj": gate_up,
            "mlp.experts.gate_up_proj_bias": gate_up_b,
            "mlp.experts.down_proj": down,
            "mlp.experts.down_proj_bias": down_b,
        }

    def indexer(self, layer: int) -> dict[str, torch.Tensor]:
        """One layer's indexer, keyed relative to `model.layers.N.` (`self_attn.indexer.<leaf>`)."""
        if layer not in self.geometry.indexer_layers:
            raise KeyError(f"layer {layer} has no indexer")
        return {
            f"self_attn.indexer.{leaf}": self.reader.read(self.indexer_key(layer, leaf))
            for leaf in INDEXER_LEAVES
        }

    def embed(self) -> torch.Tensor:
        return self.reader.read(K_EMBED)

    def lm_head(self) -> torch.Tensor:
        return self.reader.read(K_LM_HEAD)

    def final_norm(self) -> torch.Tensor:
        return self.reader.read(K_FINAL_NORM)


def hf_expert(hf: HFCheckpoint, layer: int, expert: int) -> dict[str, torch.Tensor]:
    """One HF expert as (out, in) bf16 matrices, from either the MXFP4 or the bf16 layout."""
    p = f"model.layers.{layer}.mlp.experts"
    if hf.has(f"{p}.gate_up_proj_blocks"):
        gate_up = dequant_mxfp4(
            hf.get_slice(f"{p}.gate_up_proj_blocks", expert),
            hf.get_slice(f"{p}.gate_up_proj_scales", expert),
        )
        down = dequant_mxfp4(
            hf.get_slice(f"{p}.down_proj_blocks", expert),
            hf.get_slice(f"{p}.down_proj_scales", expert),
        )
    else:
        gate_up = hf.get_slice(f"{p}.gate_up_proj", expert).T.contiguous()
        down = hf.get_slice(f"{p}.down_proj", expert).T.contiguous()
    return {
        "gate_up": gate_up,
        "gate_up_bias": hf.get_slice(f"{p}.gate_up_proj_bias", expert),
        "down": down,
        "down_bias": hf.get_slice(f"{p}.down_proj_bias", expert),
    }


@dataclasses.dataclass(frozen=True)
class Mismatch:
    name: str
    detail: str


def compare_base_to_hf(
    mg: MegatronGptOss, hf: HFCheckpoint, layers: range, experts: range
) -> tuple[int, list[Mismatch]]:
    """Compare the Megatron base weights to the HF checkpoint in bf16, bit for bit.

    Returns the number of tensors compared and the mismatches. A frozen Phase-1 base must yield
    none; a Phase-2 base is expected to differ everywhere it trained.
    """
    compared = 0
    mismatches: list[Mismatch] = []

    def check(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
        nonlocal compared
        compared += 1
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
        if a.shape != b.shape:
            mismatches.append(Mismatch(name, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
        elif not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            mismatches.append(
                Mismatch(name, f"{int((diff > 0).sum())} of {a.numel()} differ, max {diff.max():.4g}")
            )

    for layer in layers:
        prefix = f"model.layers.{layer}."
        for leaf, tensor in mg.attention(layer).items():
            check(prefix + leaf, tensor, hf.get(prefix + leaf))
        for expert in experts:
            ours = mg.expert(layer, expert)
            theirs = hf_expert(hf, layer, expert)
            for leaf in ours:
                check(f"{prefix}mlp.experts[{expert}].{leaf}", ours[leaf], theirs[leaf])
    rows = mg.geometry.vocab_rows
    check("model.embed_tokens.weight[:rows]", mg.embed(), hf.get("model.embed_tokens.weight")[:rows])
    check("lm_head.weight[:rows]", mg.lm_head(), hf.get("lm_head.weight")[:rows])
    check("model.norm.weight", mg.final_norm(), hf.get("model.norm.weight"))
    return compared, mismatches
