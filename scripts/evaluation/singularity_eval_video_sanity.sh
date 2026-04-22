#!/usr/bin/env bash
# Run inside Singularity only (paths under /workspace). Video sanity: installs imageio stack.
# Invoked from sbatch via: singularity exec ... bash /workspace/scripts/evaluation/singularity_eval_video_sanity.sh
#
# Env:
#   CKPT_KK — e.g. 00 (required)
#   RESULTS_DIR — output root for this run (recommended; sbatch sets this)
#   RESULT_TAG — if RESULTS_DIR unset, uses .../sanity_video_${RESULT_TAG}
#   EVAL_CONFIG — default /workspace/configs/sanity_eval_video.yaml
#   CKPT_DIR — default /workspace/checkpoints/cl_libero_object
set -euo pipefail

: "${CKPT_KK:?Set CKPT_KK (e.g. export CKPT_KK=00)}"

source /workspace/scripts/singularity/dp_image_env.sh

# Writable HOME before any pip/python cache (cluster $HOME may be missing in container)
export LIBERO_CONFIG_PATH=/tmp/libero_cfg
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_CACHE_DIR=/tmp/numba_cache
export HOME="/tmp/dp_eval_home_${SLURM_JOB_ID:-$$}"
mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}"
export TORCH_HOME="${HOME}/torch_home"
mkdir -p "${TORCH_HOME}/hub/checkpoints"

"$PY" -m pip install -q "numpy<2" "h5py<3.12" bddl easydict cloudpickle gym imageio imageio-ffmpeg
"$PY" -c "import bddl, imageio, sys; print('pip deps ok exe=', sys.executable); print('bddl=', bddl.__file__)"

cd /workspace

"$PY" -c "import os,yaml
root='/opt/LIBERO/libero/libero'
cfg={'benchmark_root':root,'bddl_files':f'{root}/bddl_files','init_states':f'{root}/init_files','datasets':f'{root}/../datasets','assets':f'{root}/assets'}
yaml.safe_dump(cfg, open(os.path.join(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml'), 'w'))
print('LIBERO config ok')"

CONFIG_PATH="${EVAL_CONFIG:-/workspace/configs/sanity_eval_video.yaml}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/cl_libero_object}"
if [[ -z "${RESULTS_DIR:-}" ]]; then
  : "${RESULT_TAG:?Set RESULTS_DIR or RESULT_TAG}"
  RESULTS_DIR="/workspace/results/sanity_video_${RESULT_TAG}"
fi

"$PY" -m scripts.evaluation.evaluate_checkpoints \
  --config "${CONFIG_PATH}" \
  --ckpt-dir "${CKPT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --ckpt-pattern "after_task_${CKPT_KK}.pt" \
  --no-plots
