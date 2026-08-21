# Local setup: Megatron-LM for gpt-oss (+ DSA) on H100 / H200 / GB200

Working notes for standing this repo up on `ml-eng-gpu-22` (8x H100 80GB) with a
path that scales to multi-node MoE and 120B-class models.

Everything here runs **inside the CI container**. Do not `pip install` or `uv sync`
on the host — the CUDA/NCCL/TE/DeepEP stack is not reproducible there.

---

## 0. Host facts

| Item | Value |
|---|---|
| Host | `ml-eng-gpu-22.cerebras.aws` |
| GPUs | 8x H100 80GB HBM3 (SM90) |
| Driver | 580.126.09 (CUDA 13.x capable) |
| Docker | 25.0.14, `nvidia` runtime registered |
| Also present | `enroot`, `/opt/slurm/bin/srun`, `nvidia-ctk` |
| Disk | ~2 TB free on `/var/lib/docker` |
| Repo path | `/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/Megatron-LM` (NFS, shared) |

The host is **shared** — check `nvidia-smi` for other users' processes before
claiming all 8 GPUs.

### Artifact store — `/cb/ml-eng/aarti`

FSx Lustre, 238 TB total / **~11 TB free**. All training artifacts go here:
checkpoints, tensorboard, run logs, data caches.

**Do not write checkpoints to the repo's NFS mount** — it has only ~150 GB free,
and a single 120B bf16 checkpoint is ~230 GB.

`launch_container.sh` bind-mounts it at the **same path inside the container**
(`-v /cb/ml-eng/aarti:/cb/ml-eng/aarti`) so absolute paths in run scripts,
checkpoint metadata and tensorboard dirs remain valid on both sides. It also runs
the container as your UID:GID so artifacts stay owned by you rather than root.

Already present and relevant:

| Path | Contents |
|---|---|
| `models/gpt-oss-20b` | Complete top-level HF config, tokenizer, safetensor index, and checkpoint shards |
| `mcore_runs/gptoss20b_dsa_phase1_bias_correct_100step_20260821T005100Z/checkpoints` | Bias-correct trained Phase 1 checkpoint at iteration 100 |
| `models/openai_gpt-oss-120b_HFdownload`, `models/gpt-oss-120b-oai-csx-converted` | 120B variants |
| `gpt_oss_ift_tokenized/{magpie,codefeedback,magicoder,swe_bench_rebench,sweextra}` | IFT data in **HF Arrow** format (not Megatron `.bin`/`.idx`) |
| `datasets/livecodebench`, `datasets/gsm8k` | eval sets |
| `dsa/`, `dsa_qwen3/{indexer_warmup,sparse}` | prior DSA work (MiniCPM3, Qwen3) incl. `dsa/handoff/minicpm3-dsa-handoff` |

The SFT loader accepts either conversation JSONL or pretokenized Parquet containing
`input_ids`, `loss_mask`, and `length`. Pretokenized GPT-OSS rows bypass chat rendering but still
run through the same Harmony assistant-target parser; the stored mask is checked to ensure it does
not omit any canonical assistant payload or terminator targets.

---

## 1. Build the image

We build `docker/Dockerfile.ci.dev`, the same image CI uses, rather than a thin
pip overlay on the NGC base. The reason is the end goal: multi-node MoE at 120B
scale needs components that only exist as source builds.

| Component | Why it matters | Source |
|---|---|---|
| **TransformerEngine 2.17.1** | Fused attention; gpt-oss sink softmax needs TE >= 2.8.0 | git rev `4329ff84` |
| **DeepEP** | Optimized MoE dispatch/combine over NVSHMEM + RDMA. Without it you fall back to plain NCCL `alltoall`, which is latency-bound across nodes | pinned commit + `docker/patches/deepep.patch` |
| **flash-mla** | `flash_mla_sparse_fwd` — the DSA sparse-attention fast path | `deepseek-ai/FlashMLA@nv_dev` |
| **mamba-ssm / causal-conv1d** | hybrid model support | source |

### Gotchas (both will fail your build)

1. **`assets/` must exist.** The Dockerfile has `COPY assets/ /opt/data/`, but git
   does not track empty directories, so a fresh clone has no `assets/`.
2. **Use `--target main`.** The `jet` stage needs an NVIDIA-internal build secret.

```bash
bash local_setup/build_image.sh
```

That wraps:

```bash
mkdir -p assets && touch assets/.gitkeep
DOCKER_BUILDKIT=1 docker build --progress=plain --target main \
  --build-arg FROM_IMAGE_NAME=$(cat docker/.ngc_version.dev) \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev -t megatron-lm:ci-dev .
```

Expect a long build — TE alone compiles for `NVTE_CUDA_ARCHS="80;90;100"`.

### Known limitation: flash-mla has no H100 kernels

`docker/Dockerfile.ci.dev:67` sets `FLASH_MLA_DISABLE_SM90=1`, so flash-mla is
built **Blackwell-only**. On H100/H200 the DSA FlashMLA fast path is therefore
unavailable. Fallbacks, in order of preference:

- `--dsa-kernel-backend cudnn` — the cuDNN path requires **SM90+**
  (`dsa_cudnn_kernels.py:345`), so it does cover Hopper, but only when the layout
  check passes (`_FLASH_MLA_REQUIRED_VALUE_DIM = 512` gates the FlashMLA sub-path).
- `--dsa-kernel-backend tilelang` — JIT-compiled, needs verification on SM90.
- `--dsa-kernel-backend none` — PyTorch fallback. Correct but slow.

To try an SM90-enabled build, drop `FLASH_MLA_DISABLE_SM90=1`. Unverified — the
flag likely exists because that build is problematic under CUDA 13.

---

## 2. Launch the container

```bash
bash local_setup/launch_container.sh          # interactive shell
```

Key flags, and why each is needed:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \                 # shared memory for dataloader workers
  --ulimit memlock=-1 \        # pinned memory for NCCL/RDMA
  --ulimit stack=67108864 \
  --shm-size=32g \
  -v <REPO>:/workspace/megatron-lm \
  -v <REPO>/local_setup/runs:/workspace/runs \
  -w /workspace/megatron-lm \
  megatron-lm:ci-dev bash
```

### Running as your own UID requires passwd/group mounts

We pass `--user $(id -u):$(id -g)` so artifacts on FSx stay owned by you rather
than root. But the image has no `/etc/passwd` entry for that UID, and **torch's
inductor calls `getpass.getuser()` at import time**, so `import megatron.core`
dies with:

```
KeyError: 'getpwuid(): uid not found: 1341'
```

`launch_container.sh` fixes this by mounting `local_setup/container_passwd` and
`container_group` (the image's own entries plus yours) read-only over `/etc/`.
Regenerate them if the UID or the image changes:

```bash
docker run --rm --entrypoint cat megatron-lm:ci-dev /etc/passwd > local_setup/container_passwd
docker run --rm --entrypoint cat megatron-lm:ci-dev /etc/group  > local_setup/container_group
printf '%s:x:%s:%s::/tmp:/bin/bash\n' "$(id -un)" "$(id -u)" "$(id -g)" >> local_setup/container_passwd
printf '%s:x:%s:\n' "$(id -gn)" "$(id -g)" >> local_setup/container_group
```

For the same reason every cache dir is redirected into `/tmp`
(`HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`MPLCONFIGDIR`) — `/opt/venv` is owned by UID 65532 and TE/triton hard-fail
rather than degrade when their cache is unwritable.

### Verify

```bash
MODE=verify bash local_setup/launch_container.sh
```

Known-good output on this host:

```
torch       2.13.0a0+8145d630e8.nv26.06 | CUDA 13.3 | GPUs 8
TE          2.17.1+4329ff84
mcore       0.20.0+86380223f
mcore path  /workspace/megatron-lm/megatron/core/__init__.py
EDITS LIVE
DeepEP      OK
flash-mla   OK
tokenizers  OK
```

**`EDITS LIVE` is the check that matters.** The Dockerfile copies `pyproject.toml`
and `megatron/core/__init__.py` into `/workspace` and runs `uv sync`, so a copy of
`megatron` can exist in `/opt/venv` site-packages. If it shadowed the bind mount,
your edits to `megatron/core/` would be **silently ignored** — you would debug code
that never loaded. The assertion prints a loud warning if the resolved path is not
under `/workspace/megatron-lm/`.

Note `flash-mla OK` only means the module **imports**; it does not prove SM90
kernels exist (see the `FLASH_MLA_DISABLE_SM90=1` note above).

---

## 3. Test run: gpt-oss-20b full-finetune shape

```bash
# from the host: run it inside the container
MODE=exec bash local_setup/launch_container.sh bash local_setup/train_gpt_oss_20b.sh
```

Everything is env-overridable:

| Var | Default | Notes |
|---|---|---|
| `TP_SIZE` / `PP_SIZE` / `CP_SIZE` | 1 / 1 / 1 | `WORLD_SIZE = TP x PP x DP x CP` |
| `EP_SIZE` / `ETP_SIZE` | 8 / 1 | `EP x ETP` must divide `TP x DP` |
| `SEQ_LENGTH` | 4096 | matches `initial_context_length` |
| `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | 1 / 8 | |
| `TRAIN_ITERS` | 20 | |
| `RUN_NAME` / `RUN_DIR` | timestamped | under `$ARTIFACT_ROOT/mcore_runs` |
| `DATA_PATH` + `TOKENIZER_MODEL` | unset | set both to switch off mock data |
| `LOAD_DIR` | unset | adds `--finetune --load` for a real finetune |
| `SAVE_DIR` | unset | enables checkpointing |
| `WANDB_PROJECT` | unset | enables wandb (also sets exp name + save dir) |

`CUDA_DEVICE_MAX_CONNECTIONS=1` is exported automatically **only** when
`TP>1` or `CP>1` on non-Blackwell hardware, per the assert in mcore.

This runs the **real gpt-oss-20b architecture** (24 layers, hidden 2880, 32 experts,
GQA 64/8, sink attention, sliding window) on 8 GPUs with **mock data**, to validate
that the config builds, memory fits, and throughput is sane.

### Reproducible HF-checkpoint → Phase-1 validation

To download the complete top-level Transformers checkpoint, convert it to an
EP8 MCore `torch_dist` checkpoint with the in-repo ModelOpt converter, run the
focused unit tests, and train the DSA indexer for one Phase-1 step:

```bash
bash local_setup/validate_gpt_oss_20b_dsa_phase1.sh
```

The download is pinned to the resolved Hugging Face commit and validated against
the safetensor index. It intentionally omits the redundant `original/` and
`metal/` encodings; the top-level safetensors are the complete checkpoint consumed
by ModelOpt. Override `HF_REVISION` to request a tag or commit directly.

Every invocation creates a new directory under
`$ARTIFACT_ROOT/mcore_runs/gptoss20b_dsa_phase1_<UTC timestamp>/` containing:

- `commands.sh` and `phase1_resolved_command.sh` with replayable commands;
- `run_metadata.txt`, container identity, git status, the tracked working-tree patch,
  and copies of every reproduction script (including untracked local setup files);
- per-stage `download.log`, `unit_tests.log`, `convert.log`, and `phase1.log`;
- an HF revision/size/SHA-256 manifest and an MCore checkpoint file manifest;
- `highlights.txt` and `final_status.txt` for a quick pass/fail check.

The run uses the serving-compatible Phase-1 indexer geometry (16 heads × 64),
Hadamard rotation, UE8M0/E4M3 fake quantization, blockwise KL, a frozen backbone,
and alternating DSA layers complementary to GPT-OSS sliding-window layers.
The launcher accepts `BASE_LR`/`BASE_MIN_LR` and
`DSA_INDEXER_LR`/`DSA_INDEXER_MIN_LR` overrides. Phase 1 freezes the base model,
but recording both schedules makes the same command structure usable for later
joint training; by default they are `1e-5`/`1e-6` and `1e-4`/`1e-5`, respectively.

### Reproducible Phase-1 checkpoint -> Phase-2 SFT validation

To strictly load the trained iteration-100 checkpoint as model weights only, rebuild a fresh joint
base/indexer optimizer, and run two native-Harmony SFT steps on all eight GPUs:

```bash
bash local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh
```

The validation script defaults to
`local_setup/gpt_oss_dsa/phase2_smoke_data.jsonl`, a synthetic plumbing fixture rather than a
training dataset. Set `SFT_DATA_PATH` to either a conversation JSONL or a split Parquet directory
containing `train-00000.parquet`, `val-00000.parquet`, and preferably `MANIFEST.json`. For example:

```bash
SFT_DATA_PATH=/cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_full93889_effmedium_L32768_20260809__split_v1 \
TRAIN_ITERS=2 SEQ_LENGTH=4096 \
bash local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh
```

The run directory records the source paths, copies the small dataset manifest, and writes SHA-256
digests for the Parquet shards instead of copying the training data. The Phase 2
recipe deliberately omits `--dsa-dense-warmup` and `--dsa-freeze-base`, keeps the Phase 1 model and
indexer geometry, uses `dsa_kernel_backend=none`, and supplies `--finetune --no-load-optim
--no-load-rng` so Phase 1 optimizer/scheduler state cannot leak into the new schedule.

Do not point the sequence-128 Gate B run at the full 32K split: most prompts consume the entire
window. Build a deterministic short-prefix view that retains at least 16 canonical assistant target
tokens and balances the 512 training samples across the four domains:

```bash
MODE=exec bash local_setup/launch_container.sh \
  /opt/venv/bin/python local_setup/gpt_oss_dsa/prepare_phase2_gateb_data.py \
  /cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_full93889_effmedium_L32768_20260809__split_v1 \
  /cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_gateb_seq128_min16_n512_20260821_v3 \
  /cb/ml-eng/aarti/models/gpt-oss-20b
```

Then run Gate B with that derived split:

```bash
SFT_DATA_PATH=/cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_gateb_seq128_min16_n512_20260821_v3 \
TRAIN_ITERS=20 SEQ_LENGTH=128 SAVE_CHECKPOINT=0 \
bash local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh
```

The 20-step Gate B run passed on 2026-08-21 using the v3 split above and the bias-correct Phase 1
iteration-100 checkpoint. Its reproducibility bundle is at
`/cb/ml-eng/aarti/mcore_runs/gptoss20b_dsa_phase2_gateb20_seq128_v3_20260821T205700Z`. All steps
had finite LM/KL losses, nonzero base and indexer gradients, and zero skipped or NaN iterations;
rank 0 peaked at 59,567.23 MB allocated and 59,696.00 MB reserved.

Architecture args are taken from two in-repo sources of truth:
- `examples/post_training/modelopt/conf/openai/gpt-oss-20b.sh`
- `tests/functional_tests/test_cases/moe/gpt_dynamic_inference_tp2_pp2_ep2_gptoss_20b_swa/model_config.yaml`

Note `examples/gptoss/02_train.sh` is **not** the 20B model — it is a tiny debug
config (hidden 512, 12 layers, 4 experts) wearing gpt-oss flags.

### Architecture args validated against HF `config.json`

Checked against `/cb/ml-eng/aarti/models/gpt-oss-20b/config.json`:

| HF config | Value | Megatron arg | ✓ |
|---|---|---|---|
| `num_hidden_layers` | 24 | `--num-layers 24` | ✓ |
| `hidden_size` / `intermediate_size` | 2880 / 2880 | `--hidden-size 2880 --ffn-hidden-size 2880` | ✓ |
| `num_attention_heads` / `num_key_value_heads` | 64 / 8 | `--num-attention-heads 64 --num-query-groups 8` | ✓ |
| `head_dim` | 64 | `--kv-channels 64` | ✓ |
| `num_local_experts` / `experts_per_token` | 32 / 4 | `--num-experts 32 --moe-router-topk 4` | ✓ |
| `swiglu_limit` | 7.0 | `--activation-func-clamp-value 7.0` | ✓ |
| `sliding_window` | 128 | `--window-size 127,0` (127 previous + current; see PR #2771) | ✓ |
| `layer_types` | alternating sliding/full | `--window-attn-skip-freq 2` | ✓ |
| `rope_theta` | 150000 | `--rotary-base 150000` | ✓ |
| `rope_scaling` | yarn, factor 32, orig 4096 | `--position-embedding-type yarn` + `--yarn-*` (smoke run uses plain rope — equivalent at seq<=4096) | ~ |
| `vocab_size` | 201088 | `--make-vocab-size-divisible-by 128` | ✓ |
| `max_position_embeddings` | 131072 | repo configs use `40960` | ~ |

**Resolved — `attention_bias: true`.** The HF safetensor index contains trained
Q/K/V/output projection biases, and the ModelOpt conversion builds Megatron with
`add_bias_linear=True` and `add_qkv_bias=True`. The Phase 1 launcher therefore
keeps Megatron's bias-enabled default and retains `--no-bias-dropout-fusion`,
which grouped-GEMM bias requires. A bias-enabled load/save/reload completed
without a backbone mismatch, and the trained iteration-100 checkpoint metadata
contains both `linear_qkv.bias` and `linear_proj.bias` keys.

### Validated baseline (2026-08-18)

20/20 iterations, clean exit. `gptoss20b_rc2_260818_005049`.

| Setting | Value |
|---|---|
| Parallelism | TP1 / PP1 / CP1 / **EP8** / ETP1, world 8 |
| Sharding | Megatron-FSDP `optim_grads_params` (ZeRO-3) |
| Recompute | full, uniform, 1 layer |
| Attention | cuDNN **fused** (`--attention-backend fused`), dropout 0 |
| seq / mbs / gbs | 4096 / 1 / 8 |
| Params per rank | 4,180,520,256 (matches 20.9B total under EP8) |
| **Peak memory** | **55,868 MB** of 81,559 MB |
| **Steady step time** | **~820-940 ms** |
| **Throughput** | **~96 TFLOP/s/GPU** |
| Loss | 12.759 -> 6.226 over 20 iters (mock data) |

Iterations 1-2 are unrepresentative (19,957 ms then 10,486 ms) -- that is cuDNN
autotune and FSDP buffer warmup. Steady state begins at iteration 3.

Throughput of ~96 TFLOP/s/GPU is roughly 10% MFU and is **not** a tuned number:
full recompute costs ~30%, `gbs 8` gives each rank a single microbatch so there is
no gradient-accumulation overlap, and EP8 all-to-all dominates at this tiny batch.
Raise `GLOBAL_BATCH_SIZE` and switch `RECOMPUTE=selective` before quoting perf.

Sanity check on the loss: initial 12.759 is right for random init over a 201,088
vocab (`ln(201088) = 12.21`). The subsequent decline is the model fitting the small
repeated mock stream, which confirms gradients flow and the optimizer steps -- it is
not a language-modelling result.

### Memory budget (8x H100 = 640 GB)

| Item | Estimate |
|---|---|
| bf16 weights | 42 GB |
| bf16 grads | 42 GB |
| Adam fp32 master + m + v | ~250 GB |
| **Total state** | **~335 GB** |

Fits with `--use-distributed-optimizer` (shards optimizer state across DP) plus
expert parallelism sharding expert weights.

For reference, **120B** is ~1.9 TB of state and will **not** fit on this host —
that target belongs on 2x8 H200 or GB200.

---

## 4. What is still missing for a *real* finetune

The HF checkpoint has been converted with the in-repo ModelOpt path, a bias-correct 100-step DSA
Phase 1 checkpoint is saved, and the two-step Phase 2 transition gate passes. A real Phase 2
finetune still needs:

1. **Real-data rollout.** The selected 91,631-row GPT-OSS behavior-cloning Parquet split is wired
   into `SFTDataset` and the 20-step Gate B passed; run the measured 4K/8K gates before sustained
   training.
2. **Production diagnostics and rollout gates.** Complete the diagnostic set, functional test,
   and measured 4K/8K memory/performance gates.
3. **Long-run recipe selection.** Tune the base/indexer schedules from observed gradient norms and
   decide checkpoint frequency with the measured distributed-optimizer checkpoint size in mind.

Megatron-Bridge remains an alternative conversion route, but it is not required
for the validated local checkpoint now present in the artifact store.

---

## 5. Multi-node (when you scale out)

One `srun` task per node, one `torch.distributed.run` worker group spanning all:

```bash
#SBATCH --nodes=2 --ntasks-per-node=1 --gpus-per-node=8
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
  python -m torch.distributed.run \
    --nnodes='"${SLURM_NNODES}"' --nproc-per-node=8 \
    --node-rank=${SLURM_NODEID} \
    --master-addr='"${MASTER_ADDR}"' --master-port='"${MASTER_PORT}"' \
    pretrain_gpt.py <args>'
```

Rules that bite:

- **Shared filesystem** for code, data, checkpoints, logs. Node-local paths break peers.
- **`CUDA_DEVICE_MAX_CONNECTIONS`** is hardware-dependent and *asserts*:
  `1` on H100/H200 with TP>1 or CP>1 (non-FSDP); **not needed** on GB200/Blackwell;
  must **not** be `1` with FSDP; `32` with `overlap_moe_expert_parallel_comm`.
- **Keep TP inside a node** (it is the most bandwidth-hungry dimension; wants NVLink).
  GB200 NVL72 is the exception — all 72 GPUs are one NVLink domain.
- `WORLD_SIZE = TP x PP x DP x CP`, and `EP x ETP` must divide `TP x DP`.

---

## 6. Project context

The end goal is adding **DeepSeek Sparse Attention (DSA) indexer layers** to gpt-oss.
DSA already exists in this tree (~10.3k lines under
`megatron/core/transformer/experimental_attention_variant/`) but is **gated to MLA**:

```python
# megatron/core/models/gpt/experimental_attention_variant_module_specs.py:97
assert config.multi_latent_attention, "Currently only MLA supports sparse attention."
```

gpt-oss is GQA + sink attention + sliding window, so a GQA-compatible DSA path is
the actual work. Also note DSA is **training-only** today —
`absorbed_mla.py:817` asserts `inference_context is None`, so serving/evals need a
sparse decode path built as well.

---

## 7. Hard-won findings

Each of these cost a failed run. They are ordered by how much time they will save.

### 7.1 Pin `--attention-backend`. Never leave it on `auto`.

TE picks the attention backend silently, and for gpt-oss the *fallback is
catastrophic rather than merely slow*. With `auto`, TE selected
`UnfusedDotProductAttention`, which materializes a `seq x seq x heads` score matrix:
**4.00 GiB per layer at seq 4096 in fp32** (`4096 x 4096 x 64 x 4`). That OOMs during
the backward recompute of attention, with a traceback that points at
`attention.py` and looks like a sizing problem rather than a config error.

Diagnose with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`, which makes TE print per-backend
rejection reasons. `local_setup/probe_attention_backend.py` isolates this in ~30 s
instead of a 6-minute 20B model build; it forces one backend at a time via
`NVTE_FLASH_ATTN` / `NVTE_FUSED_ATTN` / `NVTE_UNFUSED_ATTN`.

### 7.2 gpt-oss requires `--attention-dropout 0.0`

TE 2.17's support matrix (`.../dot_product_attention/utils.py:1022`) plus the
`softmax_type` gate at line 1027 mean:

| feature | supported backends |
|---|---|
| `softmax_type=vanilla` | FusedAttention, FlashAttention |
| `softmax_type=learnable` (gpt-oss sinks) | **FusedAttention only** |
| sliding window **with dropout > 0** | rejected by FusedAttention |

Megatron defaults `--attention-dropout` to **0.1**, while gpt-oss's HF `config.json`
specifies `0.0`. With dropout > 0: flash is out (sinks), cuDNN is out (SWA+dropout),
so `--attention-backend fused` raises *"No dot product attention backend is
available"* and `auto` silently drops to unfused (see 7.1). Setting dropout to 0 is
also the faithful config, not a workaround.

Note: `use_flash_attention = False` is a **blanket disable with no version check**,
so installing FlashAttention 3 would not help — TE has not wired sinks into its FA3
path. Upstream FA3 does support sinks; this is a TE integration gap. Relevant later:
a custom attention module (which DSA needs anyway) could call FA3 directly.

### 7.3 Parallelism cannot fix optimizer memory; FSDP can

Per-rank optimizer state is `P * 12 bytes / (TP * PP * DP)`, and
`TP * PP * DP * CP = world`. At world=8 with CP=1 that denominator is **always 8**,
so every arrangement of TP/PP/DP/EP gives ~`20.9B * 12 / 8` = **~31 GB/rank**. The
distributed optimizer already shards across the whole world; parallelism only
chooses *which* dimension shards. CP is worse than neutral — it consumes world size
without sharding parameters, so CP=2 doubles per-rank optimizer state.

Measured at TP1/PP1/EP8/DP8, seq 4096: static state ~56 GB, iteration 1 peak
**60,961 MB**, iteration 2 OOM.

What actually reduces it: **Megatron-FSDP** (ZeRO-3, shards params+grads too,
numerics unchanged), `--optimizer-cpu-offload` (this host has ~1.8 TB free RAM),
precision-aware optimizer (12 -> 8 B/param, changes numerics), or more GPUs.

### 7.4 Megatron-FSDP gotchas

- `--use-megatron-fsdp` alone shards **nothing**: `data_parallel_sharding_strategy`
  defaults to `no_shard`. You must also pass
  `--data-parallel-sharding-strategy optim_grads_params` for ZeRO-3.
- It **requires** `--ckpt-format fsdp_dtensor`, asserted even when not saving.
  **Megatron-Bridge writes `torch_dist`**, so an FSDP job cannot directly load a
  Bridge-converted checkpoint — convert formats, or load weights without FSDP.
- Incompatible with `--log-max-attention-logit`: that flag routes through
  `megatron/core/optimizer/qk_clip.py:23`, which hard-codes
  `model_chunk.module.module.decoder`. FSDP wraps to a different depth, giving
  `AttributeError: 'Float16Module' object has no attribute 'decoder'`. Upstream bug.
- FSDP wants `CUDA_DEVICE_MAX_CONNECTIONS` **not** set to 1, which conflicts with
  what TP/SP want. mcore warns if both are enabled.
- `--megatron-fsdp-version 2` is incompatible with `--use-distributed-optimizer`.

### 7.5 Environment / harness traps on this host

- Docker takes **2-6 minutes** to start a container here (the host runs ~50 other
  containers). A missing container right after launch does not mean the run died --
  check `docker ps` again before relaunching, or you will start duplicate 8-GPU jobs
  that fight over the same GPUs.
- Avoid `pgrep -f "<string>"` in wait loops when the same string appears in your own
  command line: it self-matches, and the loop never exits.
- Avoid `cmd | head -n1` under `set -o pipefail` + `set -e`: SIGPIPE makes the whole
  script exit silently. Use `nvidia-smi ... -i 0` instead of piping to `head`.
