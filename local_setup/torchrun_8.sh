#!/usr/bin/env bash
# Pin distributed launches to the CI image's managed virtual environment.
set -euo pipefail

exec /opt/venv/bin/python -m torch.distributed.run --nproc_per_node=8 "$@"
