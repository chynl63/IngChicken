#!/bin/bash
set -e

# ========================================================
#  Evaluate CL Checkpoints (Singularity)
# ========================================================
#
# Usage:
#   bash run_evaluate_singularity.sh
#   GPU_DEVICE=1 bash run_evaluate_singularity.sh
#
# To run in background:
#   nohup bash run_evaluate_singularity.sh > eval.log 2>&1 &
#
# Mounts the whole project root to /workspace so `python -m scripts....` works.

SIF_IMAGE="${SIF_IMAGE:-./dp_libero.sif}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_DEVICE="${GPU_DEVICE:-0}"

mkdir -p "${BASE_DIR}/results" "${BASE_DIR}/logs"

echo "========================================================"
echo " Evaluating CL Checkpoints: Diffusion Policy + LIBERO-Object"
echo " (Singularity)"
echo "========================================================"
echo "  SIF:     $SIF_IMAGE"
echo "  GPU:     $GPU_DEVICE"
echo "  Mount:   $BASE_DIR -> /workspace"
echo "  Ckpt:    ${BASE_DIR}/checkpoints/cl_libero_object/"
echo "  Results: ${BASE_DIR}/results/cl_libero_object/<run_tag>/"
echo "  Logs:    ${BASE_DIR}/logs/<run_tag>/evaluate_checkpoints.log"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_DEVICE singularity exec --nv --writable-tmpfs \
    --bind "${BASE_DIR}:/workspace" \
    "$SIF_IMAGE" \
    bash -lc '
      set -e
      cd /workspace
      source /opt/conda/etc/profile.d/conda.sh
      conda activate dp
      pip install -q "numpy<2" "h5py<3.12" bddl easydict cloudpickle gym
      export LIBERO_CONFIG_PATH=/tmp/libero_cfg
      export HOME="/tmp/dp_eval_home_$$"
      mkdir -p "${HOME}/.cache/torch/hub/checkpoints" "${LIBERO_CONFIG_PATH}"
      export TORCH_HOME="${HOME}/torch_home"
      mkdir -p "${TORCH_HOME}/hub/checkpoints"
      python - <<'"'"'PYEOF'"'"'
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
      python -m scripts.evaluation.evaluate_checkpoints \
        --config /workspace/configs/continual_learning_libero_object.yaml \
        --run-tag-auto
    '

echo ""
echo "Evaluation complete! Under ${BASE_DIR}: see results/cl_libero_object/<timestamp>/ and logs/<timestamp>/"
