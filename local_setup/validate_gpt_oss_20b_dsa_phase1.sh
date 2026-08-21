#!/usr/bin/env bash
# Reproducible end-to-end GPT-OSS 20B download, conversion, and DSA Phase-1 test.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/cb/ml-eng/aarti}
RUN_NAME=${RUN_NAME:-gptoss20b_dsa_phase1_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RUN_DIR:-$ARTIFACT_ROOT/mcore_runs/$RUN_NAME}
HF_MODEL_DIR=${HF_MODEL_DIR:-$ARTIFACT_ROOT/models/gpt-oss-20b}
MCORE_CHECKPOINT_DIR=${MCORE_CHECKPOINT_DIR:-$RUN_DIR/mcore_checkpoint}
HF_REVISION=${HF_REVISION:-main}
IMAGE_TAG=${IMAGE_TAG:-megatron-lm:ci-dev}

# The managed host workspace exposes FSx read-only, while the approved CI
# container bind mount is writable. Relaunch this same script once inside the
# container so every artifact and log is written through that mount.
if [[ "${MCORE_VALIDATION_IN_CONTAINER:-0}" != "1" ]]; then
    if [[ -e "$RUN_DIR" ]]; then
        echo "refusing to reuse existing RUN_DIR: $RUN_DIR" >&2
        echo "set RUN_NAME or RUN_DIR to a new path" >&2
        exit 1
    fi
    MCORE_IMAGE_ID=$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')
    exec env MODE=exec ARTIFACT_ROOT="$ARTIFACT_ROOT" IMAGE_TAG="$IMAGE_TAG" \
        bash local_setup/launch_container.sh \
        env \
        MCORE_VALIDATION_IN_CONTAINER=1 \
        MCORE_IMAGE_ID="$MCORE_IMAGE_ID" \
        ARTIFACT_ROOT="$ARTIFACT_ROOT" \
        RUN_NAME="$RUN_NAME" \
        RUN_DIR="$RUN_DIR" \
        HF_MODEL_DIR="$HF_MODEL_DIR" \
        MCORE_CHECKPOINT_DIR="$MCORE_CHECKPOINT_DIR" \
        HF_REVISION="$HF_REVISION" \
        IMAGE_TAG="$IMAGE_TAG" \
        bash local_setup/validate_gpt_oss_20b_dsa_phase1.sh
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to reuse existing RUN_DIR: $RUN_DIR" >&2
    echo "set RUN_NAME or RUN_DIR to a new path" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

exec > >(tee -a "$RUN_DIR/orchestration.log") 2>&1

OVERALL_STATUS=failed
finish() {
    status=$?
    printf 'status=%s\nexit_code=%s\nrun_dir=%s\n' \
        "$OVERALL_STATUS" "$status" "$RUN_DIR" > "$RUN_DIR/final_status.txt"
}
trap finish EXIT

COMMAND_LOG="$RUN_DIR/commands.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n\n' "$REPO_ROOT" > "$COMMAND_LOG"
chmod +x "$COMMAND_LOG"

record_command() {
    printf '%q ' "$@" >> "$COMMAND_LOG"
    printf '\n' >> "$COMMAND_LOG"
}

run_stage() {
    stage=$1
    shift
    printf '\n===== %s =====\n' "$stage"
    printf '[host] command: '
    printf '%q ' "$@"
    printf '\n'
    record_command "$@"
    "$@" 2>&1 | tee "$RUN_DIR/$stage.log"
}

printf '[run] run directory:       %s\n' "$RUN_DIR"
printf '[run] HF model directory:  %s\n' "$HF_MODEL_DIR"
printf '[run] MCore checkpoint:    %s\n' "$MCORE_CHECKPOINT_DIR"
printf '[run] container image:     %s\n' "$IMAGE_TAG"

printf '# Exact outer launch command used for this run:\n# ' >> "$COMMAND_LOG"
printf '%q ' env ARTIFACT_ROOT="$ARTIFACT_ROOT" RUN_NAME="$RUN_NAME" \
    RUN_DIR="$RUN_DIR" HF_MODEL_DIR="$HF_MODEL_DIR" \
    MCORE_CHECKPOINT_DIR="$MCORE_CHECKPOINT_DIR" HF_REVISION="$HF_REVISION" \
    IMAGE_TAG="$IMAGE_TAG" bash local_setup/validate_gpt_oss_20b_dsa_phase1.sh \
    >> "$COMMAND_LOG"
printf '\n\n' >> "$COMMAND_LOG"

{
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'image_tag=%s\n' "$IMAGE_TAG"
    printf 'image_id=%s\n' "${MCORE_IMAGE_ID:-unknown}"
    printf 'hf_repo=openai/gpt-oss-20b\n'
    printf 'hf_requested_revision=%s\n' "$HF_REVISION"
    printf 'hf_model_dir=%s\n' "$HF_MODEL_DIR"
    printf 'mcore_checkpoint_dir=%s\n' "$MCORE_CHECKPOINT_DIR"
} > "$RUN_DIR/run_metadata.txt"
git status --short > "$RUN_DIR/git_status.txt"
git diff --binary > "$RUN_DIR/git_diff.patch"
mkdir -p "$RUN_DIR/reproduction_scripts"
cp \
    local_setup/README.md \
    local_setup/launch_container.sh \
    local_setup/download_gpt_oss_20b.sh \
    local_setup/convert_gpt_oss_20b_to_mcore.sh \
    local_setup/torchrun_8.sh \
    local_setup/train_gpt_oss_20b_dsa_phase1.sh \
    local_setup/validate_gpt_oss_20b_dsa_phase1.sh \
    "$RUN_DIR/reproduction_scripts/"
printf 'tag=%s\nid=%s\n' "$IMAGE_TAG" "${MCORE_IMAGE_ID:-unknown}" \
    > "$RUN_DIR/container_image.txt"

run_stage download \
    bash local_setup/download_gpt_oss_20b.sh "$HF_MODEL_DIR" "$RUN_DIR" "$HF_REVISION"

run_stage unit_tests \
    /opt/venv/bin/python -m pytest -q \
    tests/unit_tests/transformer/experimental_attention_variant/test_dsa_gqa_phase1.py \
    tests/unit_tests/models/test_experimental_attention_variant_module_specs.py

run_stage convert \
    bash local_setup/convert_gpt_oss_20b_to_mcore.sh \
    "$HF_MODEL_DIR" "$MCORE_CHECKPOINT_DIR" "$RUN_DIR"

run_stage phase1 \
    bash local_setup/train_gpt_oss_20b_dsa_phase1.sh \
    "$MCORE_CHECKPOINT_DIR" "$HF_MODEL_DIR" "$RUN_DIR"

HIGHLIGHT_PATTERN='successfully loaded checkpoint|newly initialized DSA indexer|iteration|lm loss|indexer loss|grad norm|passed|failed|ERROR|Traceback'
if command -v rg >/dev/null; then
    rg -n "$HIGHLIGHT_PATTERN" \
        "$RUN_DIR/unit_tests.log" "$RUN_DIR/convert.log" "$RUN_DIR/phase1.log" \
        > "$RUN_DIR/highlights.txt" || true
else
    grep -nE "$HIGHLIGHT_PATTERN" \
        "$RUN_DIR/unit_tests.log" "$RUN_DIR/convert.log" "$RUN_DIR/phase1.log" \
        > "$RUN_DIR/highlights.txt" || true
fi

OVERALL_STATUS=passed
printf '\n[done] all stages passed\n'
printf '[done] commands:   %s\n' "$COMMAND_LOG"
printf '[done] highlights: %s\n' "$RUN_DIR/highlights.txt"
printf '[done] full logs:  %s/{download,unit_tests,convert,phase1}.log\n' "$RUN_DIR"
