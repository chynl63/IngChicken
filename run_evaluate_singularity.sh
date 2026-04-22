#!/bin/bash
set -e

# Full CL checkpoint evaluation (LIBERO-Spatial) — submits Slurm only; work runs in dp_libero.sif.
#
# Usage:
#   bash run_evaluate_singularity.sh
#   BASE=/path/to/dp_forgetting_libero bash run_evaluate_singularity.sh
#   SIF_IMAGE=/path/to/dp_libero.sif bash run_evaluate_singularity.sh
#
# Optional (passed to submit_eval_singularity.sh):
#   PARTITION TIME CPU MEM GPU_TYPE GPU_N JOB_PREFIX

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
export BASE="${BASE:-$BASE_DIR}"
export EVAL_SUITE=spatial
[[ -n "${SIF_IMAGE:-}" ]] && export SIF_IMAGE

exec bash "${BASE_DIR}/submit_eval_singularity.sh"
