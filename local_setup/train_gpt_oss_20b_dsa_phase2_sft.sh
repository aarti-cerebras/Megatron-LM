#!/usr/bin/env bash
# Train GPT-OSS 20B Phase 2 with joint base/indexer optimization on SFT data.
set -euo pipefail

PHASE1_CHECKPOINT_DIR=${1:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
HF_MODEL_DIR=${2:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
SFT_DATA_PATH=${3:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
RUN_DIR=${4:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$RUN_DIR/data_cache" "$RUN_DIR/tensorboard"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-2}
MASTER_PORT=${MASTER_PORT:-29641}
SAVE_INTERVAL=${SAVE_INTERVAL:-2}
BASE_LR=${BASE_LR:-1.0e-5}
BASE_MIN_LR=${BASE_MIN_LR:-1.0e-6}
DSA_INDEXER_LR=${DSA_INDEXER_LR:-1.0e-4}
DSA_INDEXER_MIN_LR=${DSA_INDEXER_MIN_LR:-1.0e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-1}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}

if [[ "$GPUS_PER_NODE" != "8" ]]; then
    echo "this checkpoint uses EP=8; GPUS_PER_NODE must be 8" >&2
    exit 1
fi
if [[ ! -f "$SFT_DATA_PATH" && ! -d "$SFT_DATA_PATH" ]]; then
    echo "SFT data path does not exist: $SFT_DATA_PATH" >&2
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes 1
    --master_addr localhost
    --master_port "$MASTER_PORT"
    --node_rank 0
)

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
    --max-position-embeddings 131072
    --normalization RMSNorm
    # Preserve the trained GPT-OSS attention and expert projection biases.
    --untie-embeddings-and-output-weights
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
    --quick-geglu
    --glu-linear-offset 1.0
    --activation-func-clamp-value 7.0
    --softmax-type learnable
    --window-size 127,0
    --window-attn-skip-freq 2
    --position-embedding-type yarn
    --rotary-base 150000
    --rotary-percent 1.0
    --rotary-scaling-factor 32.0
    --yarn-original-max-position-embeddings 4096
    --yarn-beta-fast 32.0
    --yarn-beta-slow 1.0
    --mscale 1.0
    --mscale-all-dim 0.0
    --no-yarn-correction-range-round-to-int
    --no-rope-fusion
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend fused
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
    --moe-token-dispatcher-type alltoall
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
)

PHASE2_ARGS=(
    --experimental-attention-variant dsa_gqa
    --dsa-layer-freq 2
    --dsa-indexer-n-heads 16
    --dsa-indexer-head-dim 64
    --dsa-indexer-topk 64
    --dsa-indexer-loss-coeff 1.0
    --dsa-indexer-loss-block-size 32
    --dsa-indexer-fp8
    --dsa-indexer-serving-compat
    --dsa-indexer-lr "$DSA_INDEXER_LR"
    --dsa-indexer-min-lr "$DSA_INDEXER_MIN_LR"
    --dsa-kernel-backend none
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters "$TRAIN_ITERS"
    --bf16
    --lr "$BASE_LR"
    --min-lr "$BASE_MIN_LR"
    --lr-decay-style cosine
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay 0.0
    --clip-grad 1.0
    --empty-unused-memory-level 1
)

DATA_ARGS=(
    --sft
    --tokenizer-type SFTTokenizer
    --tokenizer-model "$HF_MODEL_DIR"
    --sft-tokenizer-prompt-format gpt-oss
    --make-vocab-size-divisible-by 128
    --data-cache-path "$RUN_DIR/data_cache"
    --no-create-attention-mask-in-dataloader
    --num-workers 0
)
if [[ -d "$SFT_DATA_PATH" ]]; then
    SFT_TRAIN_DATA_PATH="$SFT_DATA_PATH/train-00000.parquet"
    SFT_VALID_DATA_PATH="$SFT_DATA_PATH/val-00000.parquet"
    if [[ ! -f "$SFT_TRAIN_DATA_PATH" || ! -f "$SFT_VALID_DATA_PATH" ]]; then
        echo "SFT Parquet directory must contain train-00000.parquet and val-00000.parquet" >&2
        exit 1
    fi
    DATA_ARGS+=(
        --train-data-path "$SFT_TRAIN_DATA_PATH"
        --valid-data-path "$SFT_VALID_DATA_PATH"
    )
else
    DATA_ARGS+=(--data-path "$SFT_DATA_PATH" --split 100,0,0)
fi

CHECKPOINT_ARGS=(
    --load "$PHASE1_CHECKPOINT_DIR"
    --finetune
    --auto-detect-ckpt-format
    --ckpt-format torch_dist
    --no-load-optim
    --no-load-rng
)
if [[ "$SAVE_CHECKPOINT" == "1" ]]; then
    CHECKPOINT_ARGS+=(--save "$RUN_DIR/checkpoints" --save-interval "$SAVE_INTERVAL")
fi

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 100000
    --log-throughput
    --log-params-norm
    --log-memory-to-tensorboard
    --tensorboard-dir "$RUN_DIR/tensorboard"
    --timing-log-level 0
)

COMMAND=(
    /opt/venv/bin/python -m torch.distributed.run
    "${DISTRIBUTED_ARGS[@]}"
    pretrain_gpt.py
    "${MODEL_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${PHASE2_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
)

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$PWD"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "$RUN_DIR/phase2_resolved_command.sh"
chmod +x "$RUN_DIR/phase2_resolved_command.sh"

printf '[phase2] command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
