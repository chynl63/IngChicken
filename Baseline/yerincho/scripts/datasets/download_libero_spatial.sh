#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-/workspace/data/libero_spatial}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/download_libero_suite.sh" libero_spatial "${DATA_DIR}"
