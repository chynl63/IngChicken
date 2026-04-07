#!/usr/bin/env bash
# Run inside Singularity (bound /workspace). Uses env CKPT_KK (e.g. 00, 01, …).
set -euo pipefail

: "${CKPT_KK:?Set CKPT_KK (e.g. export CKPT_KK=00)}"

source /opt/conda/etc/profile.d/conda.sh
conda activate dp
: "${CONDA_PREFIX:?CONDA_PREFIX empty after conda activate dp}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

export LIBERO_CONFIG_PATH=/tmp/libero_cfg
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
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

CONFIG_PATH="${EVAL_CONFIG:-/workspace/configs/continual_learning_libero_object.yaml}"

"$PY" -m scripts.evaluation.evaluate_checkpoints \
  --config "${CONFIG_PATH}" \
  --ckpt-dir /workspace/checkpoints \
  --results-dir /workspace/results \
  --ckpt-pattern "after_task_${CKPT_KK}.pt" \
  --no-plots
