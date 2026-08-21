#!/usr/bin/env bash
# Download and validate the Transformers checkpoint for openai/gpt-oss-20b.
set -euo pipefail

MODEL_DIR=${1:?"usage: $0 MODEL_DIR RUN_DIR [REVISION]"}
RUN_DIR=${2:?"usage: $0 MODEL_DIR RUN_DIR [REVISION]"}
REQUESTED_REVISION=${3:-main}
REPO_ID=${HF_REPO_ID:-openai/gpt-oss-20b}

mkdir -p "$MODEL_DIR" "$RUN_DIR"

# Resolve a mutable branch name once, then make the actual download immutable and
# record that commit in the manifest. The public model does not require HF_TOKEN.
RESOLVED_REVISION=$(
    HF_HUB_OFFLINE=0 /opt/venv/bin/python - "$REPO_ID" "$REQUESTED_REVISION" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id, revision = sys.argv[1:]
print(HfApi().model_info(repo_id, revision=revision).sha)
PY
)

# This is the complete top-level Transformers checkpoint. The repository also
# carries alternative `original/` and `metal/` encodings; ModelOpt does not read
# those, so downloading them would add about 27.5 GB without strengthening this
# load test.
FILES=(
    LICENSE
    README.md
    USAGE_POLICY
    chat_template.jinja
    config.json
    generation_config.json
    model-00000-of-00002.safetensors
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
)
DOWNLOAD_CMD=(
    hf download "$REPO_ID" "${FILES[@]}"
    --revision "$RESOLVED_REVISION"
    --local-dir "$MODEL_DIR"
    --max-workers "${HF_MAX_WORKERS:-8}"
)

printf '[download] requested revision: %s\n' "$REQUESTED_REVISION"
printf '[download] resolved revision:  %s\n' "$RESOLVED_REVISION"
printf '[download] command: '
printf '%q ' "${DOWNLOAD_CMD[@]}"
printf '\n'

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${DOWNLOAD_CMD[@]}"

# Validate the index against the actual safetensor headers and retain hashes for
# independent verification of the exact bytes used by conversion.
/opt/venv/bin/python - "$MODEL_DIR" "$RUN_DIR/hf_model_manifest.json" \
    "$REPO_ID" "$RESOLVED_REVISION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
repo_id = sys.argv[3]
revision = sys.argv[4]

index_path = model_dir / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
weight_map = index["weight_map"]
shard_names = sorted(set(weight_map.values()))

missing_files = [name for name in shard_names if not (model_dir / name).is_file()]
if missing_files:
    raise RuntimeError(f"missing checkpoint shards: {missing_files}")

actual_locations = {}
files = []
for shard_name in shard_names:
    shard_path = model_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        for key in shard.keys():
            if key in actual_locations:
                raise RuntimeError(f"duplicate tensor {key!r} in checkpoint shards")
            actual_locations[key] = shard_name

    digest = hashlib.sha256()
    with shard_path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    files.append(
        {
            "path": shard_name,
            "size_bytes": shard_path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )

missing_tensors = sorted(set(weight_map) - set(actual_locations))
unexpected_tensors = sorted(set(actual_locations) - set(weight_map))
misplaced_tensors = sorted(
    key for key, shard_name in weight_map.items() if actual_locations.get(key) != shard_name
)
if missing_tensors or unexpected_tensors or misplaced_tensors:
    raise RuntimeError(
        "safetensor index validation failed: "
        f"missing={missing_tensors[:20]}, unexpected={unexpected_tensors[:20]}, "
        f"misplaced={misplaced_tensors[:20]}"
    )

manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "model_dir": str(model_dir),
    "indexed_tensor_bytes": index.get("metadata", {}).get("total_size"),
    "indexed_tensor_count": len(weight_map),
    "shard_count": len(shard_names),
    "files": files,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

printf '[download] validated manifest: %s\n' "$RUN_DIR/hf_model_manifest.json"
