#!/usr/bin/env bash
# One-step Phase-1 DSA-GQA validation from a converted GPT-OSS 20B checkpoint.
set -euo pipefail

MCORE_CHECKPOINT_DIR=${1:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}
HF_MODEL_DIR=${2:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}
RUN_DIR=${3:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$RUN_DIR/data_cache" "$RUN_DIR/tensorboard"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-1}
MASTER_PORT=${MASTER_PORT:-29631}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
BASE_LR=${BASE_LR:-1.0e-5}
BASE_MIN_LR=${BASE_MIN_LR:-1.0e-6}
DSA_INDEXER_LR=${DSA_INDEXER_LR:-1.0e-4}
DSA_INDEXER_MIN_LR=${DSA_INDEXER_MIN_LR:-1.0e-5}

if [[ "$GPUS_PER_NODE" != "8" ]]; then
    echo "this checkpoint was converted with EP=8; GPUS_PER_NODE must be 8" >&2
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
    # GPT-OSS ships trained Q/K/V/output and expert projection biases. Keep
    # Megatron's add_bias_linear default enabled so the converted checkpoint
    # cannot silently discard those parameters. Grouped GEMM with bias requires
    # --no-bias-dropout-fusion, which is set below.
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

PHASE1_ARGS=(
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
    --dsa-dense-warmup
    --dsa-freeze-base
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
    --lr-warmup-fraction 0.01
    --weight-decay 0.0
    --clip-grad 1.0
    --empty-unused-memory-level 1
)

DATA_ARGS=(
    --mock-data
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$HF_MODEL_DIR"
    --make-vocab-size-divisible-by 128
    --data-cache-path "$RUN_DIR/data_cache"
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --num-workers 0
)

CHECKPOINT_ARGS=(
    --load "$MCORE_CHECKPOINT_DIR"
    --save "$RUN_DIR/checkpoints"
    --save-interval "$SAVE_INTERVAL"
    --finetune
    --auto-detect-ckpt-format
    --ckpt-format torch_dist
    --no-load-optim
    --no-load-rng
)

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 100000
    --log-throughput
    --log-params-norm
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
    "${PHASE1_ARGS[@]}"
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
} > "$RUN_DIR/phase1_resolved_command.sh"
chmod +x "$RUN_DIR/phase1_resolved_command.sh"

printf '[phase1] command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
