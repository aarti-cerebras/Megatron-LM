# Serving GPT-OSS + DSA on vLLM

An out-of-tree vLLM plugin that serves the GPT-OSS 20B checkpoints trained on this branch with a
DeepSeek Sparse Attention (DSA) indexer on every full-attention layer, plus the exporter that turns
a Megatron `torch_dist` checkpoint into a directory vLLM can load.

vLLM already runs stock GPT-OSS and already ships the DeepSeek-V3.2 lightning-indexer kernels;
what it lacks is a GQA model that uses them, and a sparse attend that carries GPT-OSS's attention
sink. That is what lives here: a `GptOssDSAForCausalLM` subclass of vLLM's GptOss, a serving twin of
the Megatron indexer, a CUSTOM attention backend, and `build_serving_dir.py`. Structure follows
Aarti's Qwen3 DSA plugin in `verl/scripts/dsa/vllm_qwen3_dsa`; the model shell is far smaller
because vLLM 0.26.0's GptOss was written to be subclassed.

## Layout

| path | what |
| --- | --- |
| `gpt_oss_dsa/config.py` | the `dsa_*` config.json contract between exporter and plugin |
| `gpt_oss_dsa/rotary.py` | the indexer's RoPE tables and application, matching megatron.core |
| `gpt_oss_dsa/indexer.py` | serving indexer: training-exact reference math plus the DeepGEMM kernel path |
| `gpt_oss_dsa/sparse_attention.py` | FA3 varlen over the selected keys with sinks, as a CUSTOM backend |
| `gpt_oss_dsa/model.py` | `GptOssDSAForCausalLM`; sparse layers swap attention, everything else inherits |
| `gpt_oss_dsa/megatron_ckpt.py` | tensor-by-tensor reads of the Megatron checkpoint in HF layout |
| `build_serving_dir.py` | Megatron checkpoint + HF base -> serving directory |
| `serve.sh` | `vllm serve` with the flags the kernels dictate |
| `tests/` | CPU tests; `tests/gpu/` holds the two GPU gates |

Everything but `model.py` and `sparse_attention.py` imports without vLLM, so the exporter and the
parity check also run inside the Megatron container.

## Environment (GPU host)

```bash
cd local_setup/gpt_oss_dsa/vllm
uv sync                                   # vllm 0.26.0, torch 2.11, transformers 5.16.1; installs this package editable
# The Hadamard CUDA extension, built against the venv's torch. Two constraints:
#  * torch is the CUDA 13.0 build, so the compile needs a CUDA 13 toolkit (hosts default to 12.9).
#  * The PyPI sdist ships without csrc/ and only works where a prebuilt wheel exists, so the source
#    is the GitHub revision Megatron's own uv.lock pins, which is what the training runs used.
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 \
  uv pip install --no-build-isolation \
  "fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"
uv run python -c "import gpt_oss_dsa, fast_hadamard_transform, torch; print(torch.version.cuda, 'ok')"
```

`uv sync` registers the `vllm.general_plugins` entry point, so every vLLM process (API server,
EngineCore, workers) registers `GptOssDSAForCausalLM` and the sparse backend itself. No
`PYTHONPATH` or `sitecustomize` is needed.

The Hadamard kernel is part of the train/serve numerics contract: training rotates q and k with
`fast_hadamard_transform` immediately before fp8 quantization, and the torch fallback differs by
summation order, which flips near-tie selections. The indexer refuses to build without the kernel
unless `GPT_OSS_DSA_ALLOW_TORCH_HADAMARD=1`.

## Build a serving directory

Runs on CPU, in this venv or the Megatron container (torch and safetensors only).

```bash
# Phase 1: frozen backbone. HF shards are linked in, only the indexers are added.
uv run build_serving_dir.py \
    --megatron-checkpoint /cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase1_bias_correct_100step_20260821T005100Z/checkpoints \
    --hf-base /cb/ml-eng/aarti/models/gpt-oss-20b \
    --out artifacts/gptoss20b_dsa_phase1_iter100_topk4096 \
    --phase 1 --index-topk 4096

# Phase 2: trained backbone. Every tensor is exported from Megatron into HF's bf16 layout (~42 GB).
uv run build_serving_dir.py \
    --megatron-checkpoint <phase-2 checkpoints dir> --hf-base /cb/ml-eng/aarti/models/gpt-oss-20b \
    --out artifacts/<name> --phase 2 --index-topk 2048
```

`--base-weights hf` (the Phase 1 default) is licensed by a bit-exact comparison of the Megatron
base against the HF shards, so a checkpoint whose backbone trained cannot be served on the wrong
backbone by mistake. `--verify all` compares every expert instead of two per layer.

The exporter infers the layer set, indexer geometry and GQA layout from the checkpoint, takes the
indexer RoPE from the HF config the way Megatron derives it from the training arguments, and
writes `BUILD_MANIFEST.json`. Two things it cannot infer and therefore takes as flags: `--phase`
and `--index-topk`.

## Gates, in order

1. **Indexer parity** (Megatron container, one GPU): the serving indexer's reference scores against
   the training `DSAIndexer` with the same weights. Launch through `/opt/venv/bin/python`: the
   container's `torchrun` on PATH is the base image's and runs the system interpreter, which
   cannot see `/opt/venv` (where `fast_hadamard_transform` lives).
   ```bash
   MODE=exec bash local_setup/launch_container.sh bash -c '
   cd /workspace/megatron-lm &&
   PYTHONPATH=.:local_setup/gpt_oss_dsa/vllm /opt/venv/bin/python -m torch.distributed.run --nproc_per_node 1 \
       local_setup/gpt_oss_dsa/vllm/tests/gpu/check_indexer_megatron_parity.py \
       --megatron-checkpoint <phase-1 checkpoints> --hf-base /cb/ml-eng/aarti/models/gpt-oss-20b --layer 1 --seq 1024'
   ```
2. **Dense equivalence** (this venv, one GPU): with `index_topk` at least the sequence length the
   Phase-1 serving dir must reproduce stock GPT-OSS greedy output token for token. This is the
   end-to-end proof of the weights, sinks, GQA grouping, block table and indexer kernels.
   ```bash
   uv run tests/gpu/check_dense_equivalence.py \
       --serving-dir artifacts/gptoss20b_dsa_phase1_iter100_topk4096 \
       --baseline /cb/ml-eng/aarti/models/gpt-oss-20b --max-tokens 64
   ```
3. **Selection is live**: serve with `GPT_OSS_DSA_DEBUG_SELECTION=4 EAGER=1` and read the
   `[GptOssDSA-SELECT]` lines; `seqused_k` must track `index_topk`, not the sequence length.
4. **Phase 2 evals** against the same OpenCompass configs used for the baseline.

## Serve

```bash
SERVING_DIR=artifacts/<name> PORT=8000 bash serve.sh
GPT_OSS_DSA_ALLOW_PHASE1=1 SERVING_DIR=artifacts/gptoss20b_dsa_phase1_iter100_topk4096 bash serve.sh   # controls only
```

Two flags are mandatory and the model refuses to build without them:

- `--block-size 64`: the indexer's paged-logits kernel and the top-k to slot conversion assume it.
- `--disable-hybrid-kv-cache-manager`: the indexer's fp8 side cache has a 132-byte token slot
  against the attention layers' 2048, and vLLM tolerates per-layer page sizes only when every layer
  needs the same number of token slots. GPT-OSS's sliding-window layers break that, so they are made
  to keep full-length KV (the kernel still applies the window). This is the same single-group path
  DeepSeek-V3.2 takes; the cost is full-context KV on the 12 sliding-window layers. For the same
  reason the sliding-window layers run on FA3 with the kernel block pinned to 64 (vLLM would
  otherwise size their blocks at 16 and the converted specs would not share one block size).

`EAGER=1` is the default until CUDA-graph capture has been verified on this model; `EAGER=0` turns
capture on.

## The contract

`build_serving_dir.py` writes, and `DSAServingConfig.from_hf_config` requires:

| key | meaning |
| --- | --- |
| `dsa_sparse_layer_ids` | layers carrying an indexer; each must be `full_attention` in `layer_types` |
| `dsa_index_n_heads`, `dsa_index_head_dim` | indexer geometry (16 x 64 for these runs) |
| `dsa_index_topk`, `index_topk` | keys attended per query; both keys, kept equal |
| `dsa_indexer_fp8`, `dsa_indexer_fp8_ue8m0`, `dsa_indexer_rotate_activation`, `dsa_indexer_scoring_relu` | indexer numerics; the kernels are fp8/UE8M0 so these must be true |
| `dsa_indexer_k_norm_eps` | the k LayerNorm epsilon |
| `dsa_indexer_rope` | type, dim, theta, YaRN factor/original length/betas, mscale, round-to-int |
| `dsa_training_phase` | 1 needs `GPT_OSS_DSA_ALLOW_PHASE1=1` to serve; 2 is the evaluable model |

Two details of the indexer RoPE are easy to get wrong and are pinned by the contract: Megatron's
`DSAIndexer` builds its YaRN with `mscale=1.0`, `mscale_all_dim=0.0` and the correction range
rounded to integers, while the base model trains with `--no-yarn-correction-range-round-to-int`.
The serving indexer reproduces the indexer's choice, not the model's.

## Weight mapping

`megatron_ckpt.py` reads the `torch_dist` checkpoint one chunk at a time through
`torch.distributed.checkpoint`'s file reader, without a process group, and presents HF layout.
Every base tensor of the Phase 1 checkpoint was compared bit for bit against the original
`openai/gpt-oss-20b` shards, including the experts after MXFP4 dequantization; that comparison is
`tests/test_megatron_ckpt.py` and the exporter's `--verify`.

## Known limitations

- vLLM 0.26.0 on Hopper only (FA3 with sinks; the sparse backend declines other capabilities).
- `index_topk <= 4096` (vLLM 0.26.0's compiled top-k kernels). The dense-equivalence control is
  therefore limited to sequences up to 4096 tokens.
- bf16 KV cache only; fp8 KV under the page-size-1 view is untested.
- TP > 1 is untested; the indexer is replicated across TP like DeepSeek's, the sinks shard with the heads.
- CUDA graphs are untested; serve eager first.
- Prefill runs the FA3 attend as one length-1 sequence per query token, which is slower than dense
  prefill at long context (measured 3.5x at 32K on the Qwen3 predecessor). Fine for evals.
- Phase 2 `--base-weights megatron` export is written and unit-covered on the Phase 1 base, but has
  not yet been run against a Phase 2 checkpoint or loaded by vLLM's unquantized GptOss path.
- Serving numerics quantize q and k with fp32 arithmetic before the UE8M0 rounding, training with
  bf16; rows whose absmax sits within bf16 rounding of a power of two can quantize one step apart.
