#!/usr/bin/env bash
# Serve a directory built by build_serving_dir.py through vLLM's OpenAI-compatible server.
#
# Env: SERVING_DIR (required) PORT (8000) TP (1) MAX_LEN (32768) GPU_MEM_UTIL (0.85)
#      SERVED_NAME (openai/gpt-oss-20b) EAGER (1: --enforce-eager; 0: CUDA graphs)
#      GPT_OSS_DSA_ALLOW_PHASE1 (1 to serve a Phase-1 checkpoint as a plumbing control)
#      EXTRA_ARGS (appended verbatim to `vllm serve`)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HERE/.venv}
# FlashInfer JIT-compiles during bringup and shells out to `ninja` by name; it lives in the venv.
export PATH="$VENV/bin:$PATH"

: "${SERVING_DIR:?set SERVING_DIR to a directory built by build_serving_dir.py}"

CMD=("$VENV/bin/vllm" serve "$SERVING_DIR"
  --served-model-name "${SERVED_NAME:-openai/gpt-oss-20b}"
  --port "${PORT:-8000}"
  --tensor-parallel-size "${TP:-1}"
  --block-size 64               # the indexer's paged-logits kernel and the slot conversion assume it
  --disable-hybrid-kv-cache-manager  # sliding-window layers keep full KV so the indexer cache can share one group
  --max-model-len "${MAX_LEN:-32768}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}"
  --dtype bfloat16
)
if [[ "${EAGER:-1}" != "0" ]]; then
  CMD+=(--enforce-eager)
fi
# shellcheck disable=SC2206
CMD+=(${EXTRA_ARGS:-})

echo "[serve] ${CMD[*]}"
exec "${CMD[@]}"
