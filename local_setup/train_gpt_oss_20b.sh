#!/bin/bash
# gpt-oss-20b full-finetune-shaped test run on a single 8-GPU node.
#
# Architecture args mirror the two in-repo sources of truth for the REAL 20B model:
#   examples/post_training/modelopt/conf/openai/gpt-oss-20b.sh
#   tests/functional_tests/test_cases/moe/gpt_dynamic_inference_tp2_pp2_ep2_gptoss_20b_swa/model_config.yaml
#
# NOTE: examples/gptoss/02_train.sh is NOT the 20B model -- it is a tiny debug
# config (hidden 512, 12 layers, 4 experts) that merely uses the gpt-oss flags.
#
# Defaults to MOCK data and random init, i.e. this validates that the config
# builds, fits in memory and trains at a sane throughput. For a real finetune,
# convert HF weights with Megatron-Bridge and pass LOAD_DIR=... (see README).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---------------------------------------------------------------- run settings
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# Parallelism. Constraints: WORLD_SIZE = TP * PP * DP * CP, and EP * ETP must
# divide TP * DP. Default: TP1 PP1 CP1 -> DP8, with EP8 sharding the 32 experts
# 4-per-GPU. TP is left at 1 for the smoke run to keep the first launch simple.
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-20}

# All artifacts (logs, tensorboard, data cache, checkpoints) land on FSx Lustre,
# not on the repo's NFS mount. Mounted at the same path inside the container by
# local_setup/launch_container.sh, so these paths are valid on both sides.
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/cb/ml-eng/aarti}
GPTOSS_HF_DIR=${GPTOSS_HF_DIR:-${ARTIFACT_ROOT}/models/gpt-oss-20b}

RUN_NAME=${RUN_NAME:-gptoss20b_$(date +%y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${ARTIFACT_ROOT}/mcore_runs/${RUN_NAME}}
mkdir -p "${RUN_DIR}" "${RUN_DIR}/data_cache"

# CUDA_DEVICE_MAX_CONNECTIONS is hardware- and parallelism-dependent, and the
# code ASSERTS on it rather than degrading: required =1 on pre-Blackwell
# (Hopper/Ampere) when TP>1 or CP>1 without FSDP; not needed on Blackwell.
# Query a single GPU: piping 8 lines into `head -n1` gives nvidia-smi SIGPIPE,
# which under `set -o pipefail` + `set -e` kills this script silently.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo "unknown")
# NOTE the FSDP exclusion: sequence/tensor parallelism wants
# CUDA_DEVICE_MAX_CONNECTIONS=1, but Megatron-FSDP explicitly wants it NOT set to 1
# (it needs multiple connections to overlap its param all-gathers). mcore warns when
# both are on. FSDP wins here because it is what makes the model fit.
USE_FSDP=${USE_FSDP:-1}
if [[ ( "$TP_SIZE" -gt 1 || "$CP_SIZE" -gt 1 ) && "$USE_FSDP" != "1" \
      && "$GPU_NAME" != *B200* && "$GPU_NAME" != *GB200* ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo "[env] CUDA_DEVICE_MAX_CONNECTIONS=1 (pre-Blackwell, TP=$TP_SIZE CP=$CP_SIZE)"
fi

echo "[run ] $RUN_NAME  world=$WORLD_SIZE  TP=$TP_SIZE PP=$PP_SIZE CP=$CP_SIZE EP=$EP_SIZE ETP=$ETP_SIZE"
echo "[run ] seq=$SEQ_LENGTH mbs=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE iters=$TRAIN_ITERS"
echo "[run ] outputs -> $RUN_DIR"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --node_rank "$NODE_RANK"
)

# --------------------------------------------------- gpt-oss-20b architecture
MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 24
    --hidden-size 2880
    --ffn-hidden-size 2880
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings 40960
    --normalization RMSNorm
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    # gpt-oss activation: clamped quick-GEGLU with a linear offset
    --quick-geglu
    --glu-linear-offset 1.0
    --activation-func-clamp-value 7.0
    # gpt-oss attention: learnable softmax sinks + alternating sliding window.
    # window-attn-skip-freq 2 => every other layer is full attention.
    --softmax-type learnable
    --window-size 127,0
    --window-attn-skip-freq 2
    # RoPE. Long-context fidelity would use --position-embedding-type yarn with
    # the yarn-* args; plain rope is equivalent at seq<=4096 and keeps the smoke
    # run simple.
    --position-embedding-type rope
    --rotary-base 150000
    --rotary-percent 1.0
    --no-rope-fusion
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    # REQUIRED, and not merely for fidelity. HF config.json sets attention_dropout
    # 0.0, but Megatron defaults to 0.1 -- and cuDNN FusedAttention refuses sliding
    # window attention when dropout > 0 ("only supports sliding window attention
    # without dropout"). Flash is already excluded by softmax_type=learnable, so a
    # nonzero dropout leaves NO fused backend: hard ValueError with
    # --attention-backend fused, or a silent fall back to unfused attention with
    # 'auto', which materializes a seq x seq x heads score matrix (4 GiB/layer at
    # seq 4096) and OOMs.
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # MUST be set explicitly. Left on 'auto', TE falls back to UNFUSED attention
    # for the sink+SWA combination and materializes a
    # seq x seq x heads score matrix (4 GiB per layer at seq 4096, fp32) -> OOM.
    # flash-attn 2.x has no sink support, so cuDNN ('fused') is the path that
    # supports softmax_type=learnable. The DSA functional test also uses 'fused'.
    --attention-backend "${ATTENTION_BACKEND:-fused}"
)

MOE_ARGS=(
    --num-experts 32
    --moe-ffn-hidden-size 2880
    --moe-router-topk 4
    --moe-router-score-function softmax
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    # alltoall is the portable dispatcher. Switch to a DeepEP-backed dispatcher
    # for multi-node MoE throughput once scaling out.
    --moe-token-dispatcher-type alltoall
    --expert-model-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size "$ETP_SIZE"
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --use-distributed-optimizer
)
if [[ "$TP_SIZE" -gt 1 ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

# Memory. Rearranging TP/PP/DP/EP cannot help: per-rank optimizer state is
#   P * 12 bytes / (TP * PP * DP)  and  TP*PP*DP*CP = world,
# so at world=8 with CP=1 it is ~20.9B*12/8 ~= 31 GB no matter how the dims are
# arranged -- the distributed optimizer already shards across the whole world.
# (CP is worse: it consumes world size without sharding params.) Measured static
# state was ~56 GB of 80 GB; iteration 1 peaked at 61 GB and iteration 2 OOM'd.
#
# Megatron-FSDP shards a dimension the distributed optimizer leaves replicated:
# 'optim_grads_params' is ZeRO-3, sharding parameters and gradients as well as
# optimizer state (~8.4 GB params + ~16.7 GB grads per rank -> ~1/8 of that).
# Numerics are unchanged, unlike lowering optimizer precision.
#
# The strategy flag is REQUIRED: data_parallel_sharding_strategy defaults to
# 'no_shard', so --use-megatron-fsdp on its own shards nothing.
if [[ "$USE_FSDP" == "1" ]]; then
    echo "[mem ] Megatron-FSDP, sharding strategy=${FSDP_SHARDING:-optim_grads_params}"
    MODEL_PARALLEL_ARGS+=(
        --use-megatron-fsdp
        --data-parallel-sharding-strategy "${FSDP_SHARDING:-optim_grads_params}"
    )
fi

TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --bf16
    --lr 1.0e-5
    --min-lr 1.0e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --clip-grad 1.0
)

# Activation memory. Measured with FSDP at seq 4096: steady state is flat at
# ~53.5 GB but `max allocated` spiked to ~70 GB during iteration 2 (FSDP param
# all-gather buffers + activation peaks), and iteration 3 died ~3 GB short. So the
# binding constraint is the transient, not the static footprint.
#
# --recompute-activations is only *selective* recompute. Full per-layer recompute
# trades ~30% throughput for a much smaller activation peak.
#
# Do NOT add --moe-layer-recompute to the `full` case: full granularity already
# recomputes the entire layer including the MoE block, and mcore rejects the pair
# with "Do not set --moe-layer-recompute with full recompute granularity."
# It belongs with selective recompute, where the MoE block is otherwise retained.
case "${RECOMPUTE:-full}" in
    full)
        echo "[mem ] full activation recompute (uniform, 1 layer)"
        TRAINING_ARGS+=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
        ;;
    selective)
        echo "[mem ] selective activation recompute + MoE layer recompute"
        TRAINING_ARGS+=(--recompute-activations --moe-layer-recompute)
        ;;
    none)
        echo "[mem ] no activation recompute"
        ;;
esac

# Mock data by default. For real data set DATA_PATH + TOKENIZER_MODEL.
DATA_ARGS=(
    --data-cache-path "${RUN_DIR}/data_cache"
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --num-workers 1
)
if [[ -n "${DATA_PATH:-}" && -n "${TOKENIZER_MODEL:-}" ]]; then
    DATA_ARGS+=(
        --data-path "$DATA_PATH"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
    )
elif [[ -f "${GPTOSS_HF_DIR}/tokenizer.json" ]]; then
    # Mock token stream, but the REAL gpt-oss tokenizer (o200k_harmony, 201088).
    # More faithful than NullTokenizer and it exercises the tokenizer path.
    echo "[data] MOCK data + real gpt-oss tokenizer from ${GPTOSS_HF_DIR}"
    DATA_ARGS+=(
        --mock-data
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${GPTOSS_HF_DIR}"
        --make-vocab-size-divisible-by 128
    )
else
    echo "[data] MOCK data + NullTokenizer (no tokenizer found at ${GPTOSS_HF_DIR})"
    DATA_ARGS+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 201088
        --make-vocab-size-divisible-by 128
    )
fi

# Resume from a Bridge-converted checkpoint to make this an actual finetune.
# Megatron-FSDP asserts on this even when nothing is saved or loaded:
#   "Megatron-FSDP requires the `fsdp_dtensor` checkpointing format."
# CAVEAT for real finetunes: Megatron-Bridge writes torch_dist, so an
# FSDP run cannot directly consume a Bridge-converted checkpoint. Either convert
# the checkpoint format, or run the weight-loading job without FSDP.
if [[ "$USE_FSDP" == "1" ]]; then
    CKPT_FORMAT="${CKPT_FORMAT:-fsdp_dtensor}"
else
    CKPT_FORMAT="${CKPT_FORMAT:-torch_dist}"
fi
CKPT_ARGS=(--ckpt-format "$CKPT_FORMAT")
echo "[ckpt] format=$CKPT_FORMAT"

if [[ -n "${LOAD_DIR:-}" ]]; then
    echo "[ckpt] finetuning from $LOAD_DIR"
    CKPT_ARGS+=(--load "$LOAD_DIR" --finetune --auto-detect-ckpt-format
                --no-load-optim --no-load-rng --dist-ckpt-strictness log_unexpected)
fi
if [[ -n "${SAVE_DIR:-}" ]]; then
    CKPT_ARGS+=(--save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-1000}")
fi

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 100000
    --log-throughput
    --log-params-norm
    --log-num-zeros-in-grad
    --log-memory-to-tensorboard
    --moe-per-layer-logging
    --tensorboard-dir "${RUN_DIR}/tensorboard"
    --timing-log-level 0
)
# --log-max-attention-logit is genuinely useful when modifying attention (it catches
# logit blowups), but it routes through megatron/core/optimizer/qk_clip.py, which
# hard-codes the wrapper nesting `model_chunk.module.module.decoder`. Megatron-FSDP
# wraps to a different depth, so that lookup hits Float16Module and raises
#   AttributeError: 'Float16Module' object has no attribute 'decoder'
# Enable it only when FSDP is off. Upstream bug worth reporting.
if [[ "$USE_FSDP" != "1" ]]; then
    LOGGING_ARGS+=(--log-max-attention-logit)
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    LOGGING_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-exp-name "$RUN_NAME"
                   --wandb-save-dir "${RUN_DIR}/wandb")
fi

set -x
python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
