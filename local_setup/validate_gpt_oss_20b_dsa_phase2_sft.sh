#!/usr/bin/env bash
# Run the two-step 20B Phase 1-to-Phase 2 SFT transition gate in the CI container.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/cb/ml-eng/aarti}
PHASE1_CHECKPOINT_DIR=${PHASE1_CHECKPOINT_DIR:-$ARTIFACT_ROOT/mcore_runs/gptoss20b_dsa_phase1_bias_correct_100step_20260821T005100Z/checkpoints}
HF_MODEL_DIR=${HF_MODEL_DIR:-$ARTIFACT_ROOT/models/gpt-oss-20b}
SFT_DATA_PATH=${SFT_DATA_PATH:-$REPO_ROOT/local_setup/gpt_oss_dsa/phase2_smoke_data.jsonl}
RUN_NAME=${RUN_NAME:-gptoss20b_dsa_phase2_sft_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RUN_DIR:-$ARTIFACT_ROOT/mcore_runs/$RUN_NAME}
IMAGE_TAG=${IMAGE_TAG:-megatron-lm:ci-dev}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0}

if [[ "${MCORE_PHASE2_IN_CONTAINER:-0}" != "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        echo "refusing to reuse existing RUN_DIR: $RUN_DIR" >&2
        exit 1
    fi
    exec env MODE=exec ARTIFACT_ROOT="$ARTIFACT_ROOT" IMAGE_TAG="$IMAGE_TAG" \
        bash local_setup/launch_container.sh \
        env \
        MCORE_PHASE2_IN_CONTAINER=1 \
        ARTIFACT_ROOT="$ARTIFACT_ROOT" \
        PHASE1_CHECKPOINT_DIR="$PHASE1_CHECKPOINT_DIR" \
        HF_MODEL_DIR="$HF_MODEL_DIR" \
        SFT_DATA_PATH=/workspace/megatron-lm/local_setup/gpt_oss_dsa/phase2_smoke_data.jsonl \
        RUN_NAME="$RUN_NAME" \
        RUN_DIR="$RUN_DIR" \
        IMAGE_TAG="$IMAGE_TAG" \
        SAVE_CHECKPOINT="$SAVE_CHECKPOINT" \
        bash local_setup/validate_gpt_oss_20b_dsa_phase2_sft.sh
fi

SAVE_CHECKPOINT="$SAVE_CHECKPOINT" bash local_setup/run_gpt_oss_20b_dsa_phase2_sft.sh \
    "$PHASE1_CHECKPOINT_DIR" \
    "$HF_MODEL_DIR" \
    "$SFT_DATA_PATH" \
    "$RUN_DIR"
