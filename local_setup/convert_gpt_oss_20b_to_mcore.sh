#!/usr/bin/env bash
# Convert the downloaded GPT-OSS 20B Transformers checkpoint to MCore torch_dist.
set -euo pipefail

HF_MODEL_DIR=${1:?"usage: $0 HF_MODEL_DIR MCORE_CHECKPOINT_DIR RUN_DIR"}
MCORE_CHECKPOINT_DIR=${2:?"usage: $0 HF_MODEL_DIR MCORE_CHECKPOINT_DIR RUN_DIR"}
RUN_DIR=${3:?"usage: $0 HF_MODEL_DIR MCORE_CHECKPOINT_DIR RUN_DIR"}

if [[ ! -f "$HF_MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "missing HF checkpoint index: $HF_MODEL_DIR/model.safetensors.index.json" >&2
    exit 1
fi
if [[ -e "$MCORE_CHECKPOINT_DIR" ]]; then
    echo "refusing to overwrite existing conversion output: $MCORE_CHECKPOINT_DIR" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/modelopt_work"

# EP=8 matches the validation run, so both conversion and load exercise the same
# distributed expert layout. The converter is already present in the repository
# and ModelOpt is already installed in the CI image; do not mutate /opt/venv.
CONVERT_CMD=(
    env
    MLM_SKIP_INSTALL=1
    HF_MODEL_CKPT="$HF_MODEL_DIR"
    MLM_MODEL_SAVE="$MCORE_CHECKPOINT_DIR"
    MLM_WORK_DIR="$RUN_DIR/modelopt_work"
    TP=1
    ETP=1
    EP=8
    PP=1
    CP=1
    DP=1
    LAUNCH_SCRIPT=/workspace/megatron-lm/local_setup/torchrun_8.sh
    bash examples/post_training/modelopt/convert.sh openai/gpt-oss-20b
)

printf '[convert] command: '
printf '%q ' "${CONVERT_CMD[@]}"
printf '\n'
"${CONVERT_CMD[@]}"

if [[ ! -f "$MCORE_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
    echo "conversion did not create a Megatron checkpoint tracker" >&2
    exit 1
fi

find "$MCORE_CHECKPOINT_DIR" -type f -printf '%P\t%s bytes\n' \
    | LC_ALL=C sort > "$RUN_DIR/mcore_checkpoint_manifest.txt"
printf '[convert] checkpoint manifest: %s\n' "$RUN_DIR/mcore_checkpoint_manifest.txt"
