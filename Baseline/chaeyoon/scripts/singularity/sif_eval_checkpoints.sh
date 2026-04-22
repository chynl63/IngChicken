#!/usr/bin/env bash
# Run *inside* dp_libero.sif with cwd mounted at /workspace (and typically
# /home/cyhoaoen bound for config paths). Full CL checkpoint eval: deps + LIBERO
# config + evaluate_checkpoints.
#
# Env:
#   EVAL_SUITE   — spatial | object (default: spatial)
#   EVAL_RUN_TAG — optional; default: timestamp. Subdir under results/<benchmark>/

set -euo pipefail
cd /workspace
source /workspace/scripts/singularity/dp_image_env.sh

SUITE="${EVAL_SUITE:-spatial}"
TAG="${EVAL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

case "$SUITE" in
  spatial)
    CONFIG="/workspace/configs/continual_learning_libero_spatial.yaml"
    CKPT_DIR="/workspace/checkpoints/cl_libero_spatial"
    RES_DIR="/workspace/results/cl_libero_spatial/${TAG}"
    ;;
  object)
    CONFIG="/workspace/configs/continual_learning_libero_object.yaml"
    CKPT_DIR="/workspace/checkpoints/cl_libero_object2"
    RES_DIR="/workspace/results/cl_libero_object2/${TAG}"
    ;;
  *)
    echo "EVAL_SUITE must be spatial or object, got: ${SUITE}" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$RES_DIR")"

DEPS_SITE="/workspace/.dp_eval_site"
export DEPS_SITE
mkdir -p "$DEPS_SITE"
export TMPDIR=/workspace/.dp_eval_tmp
export PIP_CACHE_DIR=/workspace/.pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export PYTHONNOUSERSITE=1

CSTR="${TMPDIR}/pip-constraint.txt"
echo "numpy<2" > "$CSTR"
"${PY}" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" "numpy>=1.22,<2"
"${PY}" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" bddl easydict cloudpickle
"${PY}" -m pip install -q --target "$DEPS_SITE" -c "$CSTR" --no-deps gym_notices gym
export PYTHONPATH="${DEPS_SITE}:${PYTHONPATH:-}"
"${PY}" -c "import os, bddl; d=os.path.abspath(os.environ['DEPS_SITE']); p=os.path.abspath(bddl.__file__); assert p.startswith(d + os.sep); print('[deps] bddl OK', p)"

export LIBERO_CONFIG_PATH=/workspace/.dp_eval_libero_cfg
export HOME="/workspace/.dp_eval_home"
mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}"
export TORCH_HOME="${HOME}/torch_home"
mkdir -p "${TORCH_HOME}/hub/checkpoints"

"${PY}" - <<'PYEOF'
import os
import yaml
benchmark_root = "/opt/LIBERO/libero/libero"
cfg = {
    "benchmark_root": benchmark_root,
    "bddl_files": f"{benchmark_root}/bddl_files",
    "init_states": f"{benchmark_root}/init_files",
    "datasets": f"{benchmark_root}/../datasets",
    "assets": f"{benchmark_root}/assets",
}
cfg_dir = os.environ["LIBERO_CONFIG_PATH"]
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)
print("LIBERO config written:", os.path.join(cfg_dir, "config.yaml"))
PYEOF

echo "[eval] SUITE=${SUITE} TAG=${TAG}"
echo "[eval] results -> ${RES_DIR}"

exec "${PY}" -m scripts.evaluation.evaluate_checkpoints \
  --config "${CONFIG}" \
  --ckpt-dir "${CKPT_DIR}" \
  --results-dir "${RES_DIR}"
