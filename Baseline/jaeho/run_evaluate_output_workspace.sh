#!/usr/bin/env bash
set -euo pipefail

# ========================================================
#  Evaluate One Output Run (Inside /workspace)
# ========================================================
#
# Usage:
#   bash run_evaluate_output_workspace.sh <output-run-name>
#   EVAL_KIND=ema bash run_evaluate_output_workspace.sh cl_libero_spatial_er_20260415_081802
#   EVAL_CKPT_PATTERN='after_task_05.pt' bash run_evaluate_output_workspace.sh cl_libero_spatial_er_20260415_081802
#
# Notes:
# - Run this from inside the container /workspace.
# - Default evaluates raw checkpoints: after_task_*.pt
# - Set EVAL_KIND=ema to evaluate after_task_*_ema.pt

RUN_SPEC="${1:-}"
EVAL_KIND="${EVAL_KIND:-raw}"
EVAL_NO_PLOTS="${EVAL_NO_PLOTS:-0}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

if [[ -z "${RUN_SPEC}" ]]; then
    echo "Usage: bash run_evaluate_output_workspace.sh <output-run-name>" >&2
    echo "Example: bash run_evaluate_output_workspace.sh cl_libero_spatial_er_20260415_081802" >&2
    exit 1
fi

if [[ ! -d "${WORKSPACE_DIR}" ]]; then
    echo "Workspace directory not found: ${WORKSPACE_DIR}" >&2
    exit 1
fi

cd "${WORKSPACE_DIR}"

if [[ -d "${WORKSPACE_DIR}/output/${RUN_SPEC}" ]]; then
    RUN_DIR="${WORKSPACE_DIR}/output/${RUN_SPEC}"
elif [[ -d "${WORKSPACE_DIR}/${RUN_SPEC}" ]]; then
    RUN_DIR="${WORKSPACE_DIR}/${RUN_SPEC}"
else
    echo "Run directory not found: ${RUN_SPEC}" >&2
    echo "Tried:" >&2
    echo "  ${WORKSPACE_DIR}/output/${RUN_SPEC}" >&2
    echo "  ${WORKSPACE_DIR}/${RUN_SPEC}" >&2
    exit 1
fi

CONFIG_PATH="${RUN_DIR}/config_resolved.yaml"
CKPT_DIR="${RUN_DIR}/checkpoints"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Missing config_resolved.yaml: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "Missing checkpoints directory: ${CKPT_DIR}" >&2
    exit 1
fi

case "${EVAL_KIND}" in
    raw)
        DEFAULT_PATTERN="after_task_*.pt"
        DEFAULT_RESULTS_SUBDIR="results_eval_raw"
        ;;
    ema)
        DEFAULT_PATTERN="after_task_*_ema.pt"
        DEFAULT_RESULTS_SUBDIR="results_eval_ema"
        ;;
    *)
        echo "Unsupported EVAL_KIND: ${EVAL_KIND} (expected: raw or ema)" >&2
        exit 1
        ;;
esac

CKPT_PATTERN="${EVAL_CKPT_PATTERN:-${DEFAULT_PATTERN}}"
RESULTS_SUBDIR="${EVAL_RESULTS_SUBDIR:-${DEFAULT_RESULTS_SUBDIR}}"
RESULTS_DIR="${RUN_DIR}/${RESULTS_SUBDIR}"
mkdir -p "${RESULTS_DIR}"

if ! command -v python >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_cfg}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export HOME="${HOME:-/tmp/dp_eval_home_${SLURM_JOB_ID:-$$}}"
mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/torch_home}"
mkdir -p "${TORCH_HOME}/hub/checkpoints"

python -m pip install -q "numpy<2" "h5py<3.12" bddl easydict cloudpickle gym

python - <<'PYEOF'
import os
import yaml

benchmark_root = os.environ.get("LIBERO_ROOT")
if not benchmark_root:
    for candidate in (
        "/workspace/repos/LIBERO/libero/libero",
        "/opt/LIBERO/libero/libero",
    ):
        if os.path.isdir(candidate):
            benchmark_root = candidate
            break
if not benchmark_root or not os.path.isdir(benchmark_root):
    raise FileNotFoundError(
        "Could not find LIBERO root. Tried LIBERO_ROOT and default candidates."
    )
cfg = {
    "benchmark_root": benchmark_root,
    "bddl_files": f"{benchmark_root}/bddl_files",
    "init_states": f"{benchmark_root}/init_files",
    "datasets": f"{benchmark_root}/../datasets",
    "assets": f"{benchmark_root}/assets",
}

cfg_dir = os.environ["LIBERO_CONFIG_PATH"]
os.makedirs(cfg_dir, exist_ok=True)
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)
print("LIBERO root:", benchmark_root)
print("LIBERO config written:", os.path.join(cfg_dir, "config.yaml"))
PYEOF

echo "========================================================"
echo " Evaluate Output Run: ${RUN_DIR#${WORKSPACE_DIR}/}"
echo " (Inside /workspace)"
echo "========================================================"
echo "  Mode:     ${EVAL_KIND}"
echo "  Pattern:  ${CKPT_PATTERN}"
echo "  Config:   ${CONFIG_PATH}"
echo "  Ckpt Dir: ${CKPT_DIR}"
echo "  Results:  ${RESULTS_DIR}"
echo ""

CMD=(
  python -m scripts.evaluation.evaluate_checkpoints
  --config "${CONFIG_PATH}"
  --ckpt-dir "${CKPT_DIR}"
  --results-dir "${RESULTS_DIR}"
  --ckpt-pattern "${CKPT_PATTERN}"
)

if [[ "${EVAL_NO_PLOTS}" == "1" ]]; then
    CMD+=(--no-plots)
fi

"${CMD[@]}"

echo ""
echo "Evaluation complete! Results saved to: ${RESULTS_DIR}"
