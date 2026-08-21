#!/usr/bin/env bash
# Launch a logged, reproducible Phase-1 DSA-GQA training run from an existing checkpoint.
set -euo pipefail

MCORE_CHECKPOINT_DIR=${1:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}
HF_MODEL_DIR=${2:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}
RUN_DIR=${3:?"usage: $0 MCORE_CHECKPOINT_DIR HF_MODEL_DIR RUN_DIR"}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -e "$RUN_DIR" ]]; then
    echo "run directory already exists: $RUN_DIR" >&2
    exit 1
fi

TRAIN_ITERS=${TRAIN_ITERS:-100}
SEQ_LENGTH=${SEQ_LENGTH:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29631}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
PHASE1_LAUNCH_MODE=${PHASE1_LAUNCH_MODE:-exec}
PHASE1_CONTAINER_NAME=${PHASE1_CONTAINER_NAME:-}

mkdir -p "$RUN_DIR/reproduction_scripts"
cp local_setup/launch_container.sh "$RUN_DIR/reproduction_scripts/"
cp local_setup/train_gpt_oss_20b_dsa_phase1.sh "$RUN_DIR/reproduction_scripts/"
cp local_setup/run_gpt_oss_20b_dsa_phase1.sh "$RUN_DIR/reproduction_scripts/"

git status --short > "$RUN_DIR/git_status.txt"
git diff --binary > "$RUN_DIR/git_diff.patch"
git diff --cached --binary > "$RUN_DIR/git_diff_cached.patch"

{
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$PWD"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'mcore_checkpoint_dir=%s\n' "$MCORE_CHECKPOINT_DIR"
    printf 'hf_model_dir=%s\n' "$HF_MODEL_DIR"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'train_iters=%s\n' "$TRAIN_ITERS"
    printf 'seq_length=%s\n' "$SEQ_LENGTH"
    printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
    printf 'master_port=%s\n' "$MASTER_PORT"
    printf 'save_interval=%s\n' "$SAVE_INTERVAL"
    printf 'launch_mode=%s\n' "$PHASE1_LAUNCH_MODE"
    printf 'container_name=%s\n' "$PHASE1_CONTAINER_NAME"
} > "$RUN_DIR/run_metadata.txt"

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$PWD"
    printf 'MODE=%q ' "$PHASE1_LAUNCH_MODE"
    if [[ -n "$PHASE1_CONTAINER_NAME" ]]; then
        printf 'CONTAINER_NAME=%q ' "$PHASE1_CONTAINER_NAME"
    fi
    printf 'NVTE_DEBUG=%q NVTE_DEBUG_LEVEL=%q bash %q env PHASE1_LAUNCH_MODE=%q PHASE1_CONTAINER_NAME=%q TRAIN_ITERS=%q SEQ_LENGTH=%q GPUS_PER_NODE=%q MASTER_PORT=%q SAVE_INTERVAL=%q bash %q %q %q %q\n' \
        "${NVTE_DEBUG:-1}" \
        "${NVTE_DEBUG_LEVEL:-1}" \
        local_setup/launch_container.sh \
        "$PHASE1_LAUNCH_MODE" \
        "$PHASE1_CONTAINER_NAME" \
        "$TRAIN_ITERS" \
        "$SEQ_LENGTH" \
        "$GPUS_PER_NODE" \
        "$MASTER_PORT" \
        "$SAVE_INTERVAL" \
        local_setup/run_gpt_oss_20b_dsa_phase1.sh \
        "$MCORE_CHECKPOINT_DIR" \
        "$HF_MODEL_DIR" \
        "$RUN_DIR"
} > "$RUN_DIR/phase1_launch_command.sh"
chmod +x "$RUN_DIR/phase1_launch_command.sh"

set +e
TRAIN_ITERS="$TRAIN_ITERS" \
SEQ_LENGTH="$SEQ_LENGTH" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
MASTER_PORT="$MASTER_PORT" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
bash local_setup/train_gpt_oss_20b_dsa_phase1.sh \
    "$MCORE_CHECKPOINT_DIR" \
    "$HF_MODEL_DIR" \
    "$RUN_DIR" 2>&1 | tee "$RUN_DIR/phase1.log"
exit_code=${PIPESTATUS[0]}
set -e

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
