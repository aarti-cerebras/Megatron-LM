"""vLLM plugin: GPT-OSS with Megatron-trained DSA indexers on its full-attention layers.

`register` is the `vllm.general_plugins` entry point, so vLLM calls it in every process it spawns
(API server, EngineCore, workers). Registration is by string so that importing this package does
not import vLLM's model layer; `config.py`, `rotary.py`, `indexer.py` and `megatron_ckpt.py` import
without vLLM at all, which is what lets the parity checks run inside the Megatron container.
"""

ARCH = "GptOssDSAForCausalLM"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, GptOssForCausalLMConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "gpt_oss_dsa.model:GptOssDSAForCausalLM")
    # Stock GPT-OSS's per-architecture hooks: mxfp4 -> gpt_oss_mxfp4 normalization and the
    # Harmony reasoning-parser default. Keyed by architecture name, so ours needs its own entry.
    MODELS_CONFIG_MAP.setdefault(ARCH, GptOssForCausalLMConfig)
    register_backend(
        AttentionBackendEnum.CUSTOM, "gpt_oss_dsa.sparse_attention.GptOssDSASparseBackend"
    )
