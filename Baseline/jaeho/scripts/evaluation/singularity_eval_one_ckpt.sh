#!/usr/bin/env bash
# Run inside Singularity (bound /workspace). Evaluates exactly one checkpoint.
set -euo pipefail

: "${CKPT_KK:?Set CKPT_KK (e.g. export CKPT_KK=00)}"

source /opt/conda/etc/profile.d/conda.sh
conda activate dp
: "${CONDA_PREFIX:?CONDA_PREFIX empty after conda activate dp}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

export LIBERO_CONFIG_PATH=/tmp/libero_cfg
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HOME="/tmp/dp_eval_home_${SLURM_JOB_ID:-$$}"
mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}"
export TORCH_HOME="${HOME}/torch_home"
mkdir -p "${TORCH_HOME}/hub/checkpoints"

PY="${CONDA_PREFIX}/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }

"$PY" -m pip install -q "numpy<2" "h5py<3.12" bddl easydict cloudpickle gym
"$PY" -c "import bddl, sys; print('pip deps ok exe=', sys.executable); print('bddl=', bddl.__file__)"

cd /workspace

"$PY" -c "import os,yaml
root='/opt/LIBERO/libero/libero'
cfg={'benchmark_root':root,'bddl_files':f'{root}/bddl_files','init_states':f'{root}/init_files','datasets':f'{root}/../datasets','assets':f'{root}/assets'}
yaml.safe_dump(cfg, open(os.path.join(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml'), 'w'))
print('LIBERO config ok')"

CONFIG_PATH="${EVAL_CONFIG:-/workspace/configs/continual_learning_libero_object_eval300.yaml}"
CKPT_DIR="${EVAL_CKPT_DIR:-/workspace/checkpoints}"
RESULTS_DIR="${EVAL_RESULTS_DIR:-/workspace/results}"
CKPT_PATTERN_TEMPLATE="${EVAL_CKPT_PATTERN_TEMPLATE:-after_task_%s.pt}"
printf -v CKPT_PATTERN "${CKPT_PATTERN_TEMPLATE}" "${CKPT_KK}"

"$PY" -m scripts.evaluation.evaluate_checkpoints \
  --config "${CONFIG_PATH}" \
  --ckpt-dir "${CKPT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --ckpt-pattern "${CKPT_PATTERN}" \
  --no-plots
