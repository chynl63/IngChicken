#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-/workspace/data/libero_10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/download_libero_suite.sh" libero_10 "${DATA_DIR}"
