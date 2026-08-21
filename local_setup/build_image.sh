#!/bin/bash
# Build the Megatron-LM CI dev image (same one CI uses).
#
# We deliberately build this rather than pip-overlaying the NGC base: DeepEP,
# flash-mla and the pinned TransformerEngine rev are source-only, and multi-node
# MoE at 120B scale needs them. See local_setup/README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${IMAGE_TAG:-megatron-lm:ci-dev}"
BASE_IMAGE="${BASE_IMAGE:-$(cat docker/.ngc_version.dev)}"
LOG_FILE="${LOG_FILE:-/tmp/mcore_ci_build.log}"

# Gotcha 1: the Dockerfile does `COPY assets/ /opt/data/`, but git does not track
# empty directories, so a fresh clone has no assets/ and the build fails.
mkdir -p assets && touch assets/.gitkeep

echo "Base image : $BASE_IMAGE"
echo "Target tag : $IMAGE_TAG"
echo "Log        : $LOG_FILE"

# Gotcha 2: --target main stops before the `jet` stage, which needs an
# NVIDIA-internal build secret and fails for everyone else.
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --target main \
  --build-arg FROM_IMAGE_NAME="$BASE_IMAGE" \
  --build-arg IMAGE_TYPE=dev \
  -f docker/Dockerfile.ci.dev \
  -t "$IMAGE_TAG" \
  . 2>&1 | tee "$LOG_FILE"

echo "Built $IMAGE_TAG"
