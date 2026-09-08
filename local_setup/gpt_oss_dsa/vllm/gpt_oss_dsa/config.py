"""The serving contract: the `dsa_*` keys build_serving_dir.py writes into config.json.

Everything the plugin needs to reproduce the trained indexer is read from here and nothing is
re-derived from the training code, so a model cannot be served with a different geometry, RoPE or
layer set than it was trained with without the mismatch surfacing as an error.
"""

from __future__ import annotations

import dataclasses
from typing import Any

ARCH = "GptOssDSAForCausalLM"

# DeepGEMM's fp8 mqa_logits kernels accept these indexer head dims and head counts dividing 128;
# training's --dsa-indexer-serving-compat enforces the same contract.
SERVING_INDEXER_HEAD_DIMS = (32, 64, 128)
SERVING_HEAD_TILE = 128
# vLLM 0.26.0's compiled top-k kernels fail asynchronously above this width.
MAX_KERNEL_TOP_K = 4096


@dataclasses.dataclass(frozen=True)
class IndexerRope:
    """Rotary geometry of the indexer, as megatron.core's DSAIndexer builds it.

    `scaling_factor == 1.0` reduces YaRN to plain RoPE. `correction_range_round_to_int` defaults to
    True because DSAIndexer constructs YarnRotaryEmbedding without passing the flag, even when the
    base model trains with --no-yarn-correction-range-round-to-int.
    """

    type: str
    dim: int
    theta: float
    scaling_factor: float = 1.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    correction_range_round_to_int: bool = True

    def __post_init__(self) -> None:
        if self.type not in ("rope", "yarn"):
            raise ValueError(f"indexer rope type must be 'rope' or 'yarn', got {self.type!r}")
        if self.dim <= 0 or self.dim % 2:
            raise ValueError(f"indexer rope dim must be a positive even number, got {self.dim}")
        if self.theta <= 0:
            raise ValueError(f"indexer rope theta must be positive, got {self.theta}")


@dataclasses.dataclass(frozen=True)
class DSAServingConfig:
    sparse_layer_ids: tuple[int, ...]
    n_heads: int
    head_dim: int
    index_topk: int
    fp8: bool
    fp8_ue8m0: bool
    rotate_activation: bool
    scoring_relu: bool
    k_norm_eps: float
    rope: IndexerRope
    training_phase: int
    source: str = ""

    def validate(self, num_layers: int, layer_types: list[str] | None) -> None:
        if not self.sparse_layer_ids:
            raise ValueError("dsa_sparse_layer_ids is empty: nothing would be sparse")
        if len(set(self.sparse_layer_ids)) != len(self.sparse_layer_ids):
            raise ValueError("dsa_sparse_layer_ids has duplicates")
        for i in self.sparse_layer_ids:
            if not 0 <= i < num_layers:
                raise ValueError(f"dsa_sparse_layer_ids has layer {i} outside [0, {num_layers})")
            if layer_types is not None and layer_types[i] != "full_attention":
                raise ValueError(
                    f"layer {i} is {layer_types[i]!r}; DSA layers must be full_attention layers"
                )
        if self.head_dim not in SERVING_INDEXER_HEAD_DIMS:
            raise ValueError(f"indexer head_dim must be in {SERVING_INDEXER_HEAD_DIMS}")
        if self.n_heads < 1 or SERVING_HEAD_TILE % self.n_heads:
            raise ValueError(f"indexer n_heads must divide {SERVING_HEAD_TILE}, got {self.n_heads}")
        if not 1 <= self.index_topk <= MAX_KERNEL_TOP_K:
            raise ValueError(f"index_topk must be in [1, {MAX_KERNEL_TOP_K}], got {self.index_topk}")
        if self.fp8 and not self.fp8_ue8m0:
            raise ValueError("the serving kernels quantize with UE8M0 scales; fp8 requires ue8m0")
        if self.rope.dim != self.head_dim:
            raise ValueError("the dsa_gqa indexer ropes its whole head; rope.dim must equal head_dim")
        if self.training_phase not in (1, 2):
            raise ValueError(f"dsa_training_phase must be 1 or 2, got {self.training_phase}")

    def to_json(self) -> dict[str, Any]:
        return {
            "dsa_enabled": True,
            "dsa_training_phase": self.training_phase,
            "dsa_sparse_layer_ids": list(self.sparse_layer_ids),
            "dsa_index_n_heads": self.n_heads,
            "dsa_index_head_dim": self.head_dim,
            "dsa_index_topk": self.index_topk,
            # vLLM's own name for the selection width; kept equal to dsa_index_topk.
            "index_topk": self.index_topk,
            "dsa_indexer_fp8": self.fp8,
            "dsa_indexer_fp8_ue8m0": self.fp8_ue8m0,
            "dsa_indexer_rotate_activation": self.rotate_activation,
            "dsa_indexer_scoring_relu": self.scoring_relu,
            "dsa_indexer_k_norm_eps": self.k_norm_eps,
            "dsa_indexer_rope": dataclasses.asdict(self.rope),
            "dsa_source": self.source,
        }

    @classmethod
    def from_hf_config(cls, hf: Any) -> "DSAServingConfig":
        """Read and validate the contract from a loaded HF config object."""

        def need(key: str) -> Any:
            value = getattr(hf, key, None)
            if value is None:
                raise ValueError(
                    f"config.json is missing {key!r}; this serving dir was not built by "
                    "build_serving_dir.py, or was built by an older version of it"
                )
            return value

        if not need("dsa_enabled"):
            raise ValueError("config.dsa_enabled is false")
        index_topk = int(need("dsa_index_topk"))
        if int(need("index_topk")) != index_topk:
            raise ValueError("config.index_topk and config.dsa_index_topk disagree")
        rope = IndexerRope(**dict(need("dsa_indexer_rope")))
        cfg = cls(
            sparse_layer_ids=tuple(int(i) for i in need("dsa_sparse_layer_ids")),
            n_heads=int(need("dsa_index_n_heads")),
            head_dim=int(need("dsa_index_head_dim")),
            index_topk=index_topk,
            fp8=bool(need("dsa_indexer_fp8")),
            fp8_ue8m0=bool(need("dsa_indexer_fp8_ue8m0")),
            rotate_activation=bool(need("dsa_indexer_rotate_activation")),
            scoring_relu=bool(need("dsa_indexer_scoring_relu")),
            k_norm_eps=float(need("dsa_indexer_k_norm_eps")),
            rope=rope,
            training_phase=int(need("dsa_training_phase")),
            source=str(getattr(hf, "dsa_source", "")),
        )
        cfg.validate(int(hf.num_hidden_layers), getattr(hf, "layer_types", None))
        return cfg
