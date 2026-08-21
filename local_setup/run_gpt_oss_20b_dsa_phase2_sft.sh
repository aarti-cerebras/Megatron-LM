#!/usr/bin/env bash
# Launch a logged, reproducible GPT-OSS 20B Phase 2 SFT run.
set -euo pipefail

PHASE1_CHECKPOINT_DIR=${1:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
HF_MODEL_DIR=${2:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
SFT_DATA_PATH=${3:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}
RUN_DIR=${4:?"usage: $0 PHASE1_CHECKPOINT_DIR HF_MODEL_DIR SFT_DATA_PATH RUN_DIR"}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -e "$RUN_DIR" ]]; then
    echo "run directory already exists: $RUN_DIR" >&2
    exit 1
fi
if [[ ! -f "$SFT_DATA_PATH" && ! -d "$SFT_DATA_PATH" ]]; then
    echo "SFT data path does not exist: $SFT_DATA_PATH" >&2
    exit 1
fi

TRAIN_ITERS=${TRAIN_ITERS:-2}
SEQ_LENGTH=${SEQ_LENGTH:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29641}
SAVE_INTERVAL=${SAVE_INTERVAL:-2}
BASE_LR=${BASE_LR:-1.0e-5}
BASE_MIN_LR=${BASE_MIN_LR:-1.0e-6}
DSA_INDEXER_LR=${DSA_INDEXER_LR:-1.0e-4}
DSA_INDEXER_MIN_LR=${DSA_INDEXER_MIN_LR:-1.0e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-1}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}

mkdir -p "$RUN_DIR/reproduction_scripts"
cp local_setup/launch_container.sh "$RUN_DIR/reproduction_scripts/"
cp local_setup/train_gpt_oss_20b_dsa_phase2_sft.sh "$RUN_DIR/reproduction_scripts/"
cp local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh "$RUN_DIR/reproduction_scripts/"

SFT_DATA_KIND=conversation_jsonl
SFT_TRAIN_DATA_PATH=$SFT_DATA_PATH
SFT_VALID_DATA_PATH=
if [[ -d "$SFT_DATA_PATH" ]]; then
    SFT_DATA_KIND=pretokenized_parquet_split
    SFT_TRAIN_DATA_PATH="$SFT_DATA_PATH/train-00000.parquet"
    SFT_VALID_DATA_PATH="$SFT_DATA_PATH/val-00000.parquet"
    if [[ ! -f "$SFT_TRAIN_DATA_PATH" || ! -f "$SFT_VALID_DATA_PATH" ]]; then
        echo "SFT Parquet directory must contain train-00000.parquet and val-00000.parquet" >&2
        exit 1
    fi
    if [[ -f "$SFT_DATA_PATH/MANIFEST.json" ]]; then
        cp "$SFT_DATA_PATH/MANIFEST.json" "$RUN_DIR/reproduction_scripts/dataset_manifest.json"
    fi
    sha256sum "$SFT_TRAIN_DATA_PATH" "$SFT_VALID_DATA_PATH" \
        > "$RUN_DIR/dataset_sha256.txt"
elif [[ "$SFT_DATA_PATH" == *.parquet ]]; then
    SFT_DATA_KIND=pretokenized_parquet
    sha256sum "$SFT_DATA_PATH" > "$RUN_DIR/dataset_sha256.txt"
else
    cp "$SFT_DATA_PATH" "$RUN_DIR/reproduction_scripts/sft_data.jsonl"
    sha256sum "$SFT_DATA_PATH" > "$RUN_DIR/dataset_sha256.txt"
fi

git status --short > "$RUN_DIR/git_status.txt"
git diff --binary > "$RUN_DIR/git_diff.patch"
git diff --cached --binary > "$RUN_DIR/git_diff_cached.patch"

{
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$PWD"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'phase1_checkpoint_dir=%s\n' "$PHASE1_CHECKPOINT_DIR"
    printf 'phase1_iteration=%s\n' "$(tr -d '[:space:]' < "$PHASE1_CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
    printf 'hf_model_dir=%s\n' "$HF_MODEL_DIR"
    printf 'sft_data_path=%s\n' "$SFT_DATA_PATH"
    printf 'sft_data_kind=%s\n' "$SFT_DATA_KIND"
    printf 'sft_train_data_path=%s\n' "$SFT_TRAIN_DATA_PATH"
    printf 'sft_valid_data_path=%s\n' "$SFT_VALID_DATA_PATH"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'train_iters=%s\n' "$TRAIN_ITERS"
    printf 'seq_length=%s\n' "$SEQ_LENGTH"
    printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
    printf 'master_port=%s\n' "$MASTER_PORT"
    printf 'save_interval=%s\n' "$SAVE_INTERVAL"
    printf 'base_lr=%s\n' "$BASE_LR"
    printf 'base_min_lr=%s\n' "$BASE_MIN_LR"
    printf 'dsa_indexer_lr=%s\n' "$DSA_INDEXER_LR"
    printf 'dsa_indexer_min_lr=%s\n' "$DSA_INDEXER_MIN_LR"
    printf 'lr_warmup_iters=%s\n' "$LR_WARMUP_ITERS"
    printf 'save_checkpoint=%s\n' "$SAVE_CHECKPOINT"
} > "$RUN_DIR/run_metadata.txt"

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$PWD"
    printf 'MODE=exec bash %q env ' local_setup/launch_container.sh
    printf 'TRAIN_ITERS=%q SEQ_LENGTH=%q GPUS_PER_NODE=%q MASTER_PORT=%q ' \
        "$TRAIN_ITERS" "$SEQ_LENGTH" "$GPUS_PER_NODE" "$MASTER_PORT"
    printf 'SAVE_INTERVAL=%q BASE_LR=%q BASE_MIN_LR=%q ' \
        "$SAVE_INTERVAL" "$BASE_LR" "$BASE_MIN_LR"
    printf 'DSA_INDEXER_LR=%q DSA_INDEXER_MIN_LR=%q LR_WARMUP_ITERS=%q ' \
        "$DSA_INDEXER_LR" "$DSA_INDEXER_MIN_LR" "$LR_WARMUP_ITERS"
    printf 'SAVE_CHECKPOINT=%q ' "$SAVE_CHECKPOINT"
    printf 'bash %q %q %q %q %q\n' \
        local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh \
        "$PHASE1_CHECKPOINT_DIR" "$HF_MODEL_DIR" "$SFT_DATA_PATH" "$RUN_DIR"
} > "$RUN_DIR/phase2_launch_command.sh"
chmod +x "$RUN_DIR/phase2_launch_command.sh"

set +e
TRAIN_ITERS="$TRAIN_ITERS" \
SEQ_LENGTH="$SEQ_LENGTH" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
MASTER_PORT="$MASTER_PORT" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
BASE_LR="$BASE_LR" \
BASE_MIN_LR="$BASE_MIN_LR" \
DSA_INDEXER_LR="$DSA_INDEXER_LR" \
DSA_INDEXER_MIN_LR="$DSA_INDEXER_MIN_LR" \
LR_WARMUP_ITERS="$LR_WARMUP_ITERS" \
SAVE_CHECKPOINT="$SAVE_CHECKPOINT" \
bash local_setup/train_gpt_oss_20b_dsa_phase2_sft.sh \
    "$PHASE1_CHECKPOINT_DIR" \
    "$HF_MODEL_DIR" \
    "$SFT_DATA_PATH" \
    "$RUN_DIR" 2>&1 | tee "$RUN_DIR/phase2.log"
exit_code=${PIPESTATUS[0]}
set -e

if command -v rg >/dev/null; then
    rg -n 'successfully loaded checkpoint|iteration|lm loss|indexer loss|learning rate|grad norm|memory|ERROR|Traceback' \
        "$RUN_DIR/phase2.log" > "$RUN_DIR/highlights.txt" || true
else
    grep -nE 'successfully loaded checkpoint|iteration|lm loss|indexer loss|learning rate|grad norm|memory|ERROR|Traceback' \
        "$RUN_DIR/phase2.log" > "$RUN_DIR/highlights.txt" || true
fi

{
    if ((exit_code == 0)); then
        printf 'status=passed\n'
    else
        printf 'status=failed\n'
    fi
    printf 'exit_code=%s\n' "$exit_code"
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'run_dir=%s\n' "$RUN_DIR"
} > "$RUN_DIR/final_status.txt"

exit "$exit_code"
