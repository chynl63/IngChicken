#!/bin/bash
set -e

# ========================================================
#  Sequential CL Training (Singularity)
# ========================================================
#
# Usage:
#   bash run_sequential_singularity.sh            # default GPU 0
#   GPU_DEVICE=1 bash run_sequential_singularity.sh
#
# To run in background (survives logout):
#   nohup bash run_sequential_singularity.sh > train.log 2>&1 &
#
# Mounts the whole project root to /workspace so `python -m scripts....` works.

SIF_IMAGE="${SIF_IMAGE:-./dp_libero.sif}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_DEVICE="${GPU_DEVICE:-0}"

mkdir -p "${BASE_DIR}/checkpoints/cl_libero_spatial" "${BASE_DIR}/results/cl_libero_spatial" "${BASE_DIR}/wandb"

echo "========================================================"
echo " Sequential CL Training: Diffusion Policy + LIBERO-Spatial"
echo " (Singularity)"
echo "========================================================"
echo "  SIF:     $SIF_IMAGE"
echo "  GPU:     $GPU_DEVICE"
echo "  Mount:   $BASE_DIR -> /workspace"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_DEVICE singularity exec --nv \
    --bind "${BASE_DIR}:/workspace" \
    --bind "/home/cyhoaoen:/home/cyhoaoen" \
    "$SIF_IMAGE" \
    bash -lc '
      set -e
      cd /workspace
      source /workspace/scripts/singularity/dp_image_env.sh
      python -m pip install -q wandb
      python -m scripts.train_sequential \
        --config /workspace/configs/continual_learning_libero_spatial.yaml \
        --skip-eval
    '

echo ""
echo "Training complete! Checkpoints saved to: ${BASE_DIR}/checkpoints/cl_libero_spatial/"
