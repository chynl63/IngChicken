#!/usr/bin/env bash
# Run inside Singularity (bound /workspace). Uses env CKPT_KK (e.g. 00, 01, …).
set -euo pipefail

: "${CKPT_KK:?Set CKPT_KK (e.g. export CKPT_KK=00)}"

source /workspace/scripts/singularity/dp_image_env.sh

DEPS_SITE="/workspace/.dp_eval_site"
export DEPS_SITE
mkdir -p "$DEPS_SITE"
export TMPDIR=/workspace/.dp_eval_tmp
export PIP_CACHE_DIR=/workspace/.pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export PYTHONNOUSERSITE=1
CSTR="${TMPDIR}/pip-constraint.txt"
echo "numpy<2" > "$CSTR"
"$PY" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" "numpy>=1.22,<2"
"$PY" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" bddl easydict cloudpickle
"$PY" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" --no-deps gym_notices gym
export PYTHONPATH="${DEPS_SITE}:${PYTHONPATH:-}"
"$PY" -c "import os, bddl; p=os.path.abspath(bddl.__file__); d=os.path.abspath(os.environ[\"DEPS_SITE\"]); assert p.startswith(d + os.sep), (\"bddl not under DEPS_SITE\", p, d); print(\"[deps] bddl OK\", p)"

export LIBERO_CONFIG_PATH=/workspace/.dp_eval_libero_cfg
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_CACHE_DIR=/workspace/.dp_eval_numba_cache
export HOME="/workspace/.dp_eval_home"
mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}" "${NUMBA_CACHE_DIR}"
export TORCH_HOME="${HOME}/torch_home"
mkdir -p "${TORCH_HOME}/hub/checkpoints"

cd /workspace

"$PY" -c "import os,yaml
root='/opt/LIBERO/libero/libero'
cfg={'benchmark_root':root,'bddl_files':f'{root}/bddl_files','init_states':f'{root}/init_files','datasets':f'{root}/../datasets','assets':f'{root}/assets'}
yaml.safe_dump(cfg, open(os.path.join(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml'), 'w'))
print('LIBERO config ok')"

CONFIG_PATH="${EVAL_CONFIG:-/workspace/configs/continual_learning_libero_spatial.yaml}"

"$PY" -m scripts.evaluation.evaluate_checkpoints \
  --config "${CONFIG_PATH}" \
  --ckpt-dir /workspace/checkpoints \
  --results-dir /workspace/results \
  --ckpt-pattern "after_task_${CKPT_KK}.pt" \
  --no-plots
