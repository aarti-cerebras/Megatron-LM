#!/bin/bash
# Launch the Megatron-LM CI dev container with GPUs and the repo bind-mounted.
#
# MODE=it      interactive shell (default)
# MODE=verify  run the dependency checks and exit
# MODE=exec    run an arbitrary command: MODE=exec ./launch_container.sh <cmd...>
# MODE=detach  run an arbitrary command in a detached container
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-megatron-lm:ci-dev}"
MODE="${MODE:-it}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-32g}"

# Artifact store: FSx Lustre, ~11 TB free. Checkpoints, tensorboard and logs go
# here -- NOT on the repo's NFS mount, which has ~150 GB free (a 120B bf16
# checkpoint alone is ~230 GB).
#
# Mounted at the SAME path inside the container so absolute paths in run
# scripts, checkpoint metadata and tensorboard dirs stay valid on both sides.
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/cb/ml-eng/aarti}"

mkdir -p "${ARTIFACT_ROOT}/mcore_runs"

# --ipc=host       : shared memory for dataloader workers
# --ulimit memlock : pinned memory for NCCL / RDMA (DeepEP)
# --ulimit stack   : TE / kernel launch stack requirements
COMMON_ARGS=(
  --gpus "$GPUS"
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --shm-size="$SHM_SIZE"
  -v "$REPO_ROOT:/workspace/megatron-lm"
  -v "${ARTIFACT_ROOT}:${ARTIFACT_ROOT}"
  -w /workspace/megatron-lm
  -e PYTHONDONTWRITEBYTECODE=1
  -e ARTIFACT_ROOT="${ARTIFACT_ROOT}"
  # Reduce allocator fragmentation (the OOM messages showed 4-10 GB reserved but
  # unallocated). Harmless when memory is not tight.
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  # Surface TE's attention-backend selection reasoning; set NVTE_DEBUG_LEVEL=0 to quiet.
  -e NVTE_DEBUG="${NVTE_DEBUG:-1}"
  -e NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-1}"
  # Use only local HF assets; never reach the Hub mid-run.
  -e HF_HOME="${ARTIFACT_ROOT}/models/hf_cache"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
)

# Run as the invoking user so artifacts on FSx stay owned by you, not root.
# HOME and every compile cache must point somewhere writable by that UID --
# /opt/venv and /root are not, and TE/triton/inductor will hard-fail if their
# cache dir is unwritable.
if [[ "${RUN_AS_USER:-1}" == "1" ]]; then
  COMMON_ARGS+=(
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -e XDG_CACHE_HOME=/tmp/.cache
    -e TRITON_CACHE_DIR=/tmp/.triton
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor
    -e MPLCONFIGDIR=/tmp/.mpl
    # The container has no passwd entry for our UID, and torch's inductor calls
    # getpass.getuser() at import time -> KeyError: getpwuid(): uid not found.
    # Mount passwd/group files that include this user (regenerate with the
    # commands in local_setup/README.md if the UID or image changes).
    -v "$REPO_ROOT/local_setup/container_passwd:/etc/passwd:ro"
    -v "$REPO_ROOT/local_setup/container_group:/etc/group:ro"
    -e USER="$(id -un)"
    -e LOGNAME="$(id -un)"
  )
fi

VERIFY_CMD='
set +e
python -c "import torch; print(\"torch      \", torch.__version__, \"| CUDA\", torch.version.cuda, \"| GPUs\", torch.cuda.device_count())"
python -c "import transformer_engine as te; print(\"TE         \", te.__version__)"
python -c "import megatron.core as m; print(\"mcore      \", m.__version__)"
# Critical: megatron.core MUST resolve to the bind-mounted tree, not to a copy in
# /opt/venv site-packages -- otherwise edits to megatron/core/ are silently ignored.
python -c "import megatron.core as m; p=m.__file__; print(\"mcore path \", p); print(\"EDITS LIVE\" if p.startswith(\"/workspace/megatron-lm/\") else \"*** WARNING: shadowed by site-packages, edits will NOT take effect ***\")"
python -c "import deep_ep; print(\"DeepEP      OK\")"                      2>&1 | tail -1
python -c "from flash_mla import flash_mla_sparse_fwd; print(\"flash-mla   OK\")" 2>&1 | tail -1
python -c "import transformers, tiktoken, sentencepiece; print(\"tokenizers  OK\")" 2>&1 | tail -1
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
'

case "$MODE" in
  it)
    exec docker run --rm -it "${COMMON_ARGS[@]}" "$IMAGE_TAG" bash
    ;;
  verify)
    exec docker run --rm "${COMMON_ARGS[@]}" "$IMAGE_TAG" bash -c "$VERIFY_CMD"
    ;;
  exec)
    exec docker run --rm "${COMMON_ARGS[@]}" "$IMAGE_TAG" "$@"
    ;;
  detach)
    DETACH_ARGS=(-d --log-driver none)
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
      DETACH_ARGS+=(--name "$CONTAINER_NAME")
    fi
    exec docker run --rm "${DETACH_ARGS[@]}" "${COMMON_ARGS[@]}" "$IMAGE_TAG" "$@"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected it|verify|exec|detach)" >&2
    exit 1
    ;;
esac
