#!/bin/bash
set -e

# Full CL checkpoint evaluation (LIBERO-Object, checkpoints/cl_libero_object2) — Slurm + SIF only.
#
# Usage:
#   bash run_evaluate_singularity_object.sh
#   See run_evaluate_singularity.sh for env vars.

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
export BASE="${BASE:-$BASE_DIR}"
export EVAL_SUITE=object
[[ -n "${SIF_IMAGE:-}" ]] && export SIF_IMAGE

exec bash "${BASE_DIR}/submit_eval_singularity.sh"
