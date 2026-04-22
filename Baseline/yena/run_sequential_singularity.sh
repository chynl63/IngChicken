#!/bin/bash
set -e

# ========================================================
#  Sequential CL Training (Singularity)
# ========================================================
#
# Usage:
#   bash run_sequential_singularity.sh                         # from scratch
#   PRETRAIN_CKPT=/workspace/checkpoints/pretrain_libero90/final.pt \
#     bash run_sequential_singularity.sh                       # w/ pretrain
#   GPU_DEVICE=1 bash run_sequential_singularity.sh
#
# To run in background (survives logout):
#   nohup bash run_sequential_singularity.sh > train.log 2>&1 &
#
# Mounts the whole project root to /workspace so `python -m scripts....` works.

SIF_IMAGE="${SIF_IMAGE:-/scratch2/cyhoaoen/simg/dp_libero.sif}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"  # optional: path to pretrain checkpoint inside container

mkdir -p "${BASE_DIR}/checkpoints" "${BASE_DIR}/results"

echo "========================================================"
echo " Sequential CL Training: Diffusion Policy + LIBERO-Object"
echo " (Singularity)"
echo "========================================================"
echo "  SIF:          $SIF_IMAGE"
echo "  GPU:          $GPU_DEVICE"
echo "  Mount:        $BASE_DIR -> /workspace"
if [ -n "$PRETRAIN_CKPT" ]; then
  echo "  Pretrain ckpt: $PRETRAIN_CKPT"
else
  echo "  Pretrain ckpt: (none — training from scratch)"
fi
echo ""

# Build the python command depending on whether pretrain ckpt is given
if [ -n "$PRETRAIN_CKPT" ]; then
  TRAIN_CMD="python -m scripts.train_sequential \
    --config /workspace/configs/continual_learning_libero_object.yaml \
    --pretrain-ckpt ${PRETRAIN_CKPT} \
    --skip-eval"
else
  TRAIN_CMD="python -m scripts.train_sequential \
    --config /workspace/configs/continual_learning_libero_object.yaml \
    --skip-eval"
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE singularity exec --nv \
    --bind "${BASE_DIR}:/workspace" \
    "$SIF_IMAGE" \
    bash -lc "
      set -e
      cd /workspace
      source /opt/conda/etc/profile.d/conda.sh
      conda activate dp
      ${TRAIN_CMD}
    "

echo ""
echo "Training complete! Checkpoints saved to: ${BASE_DIR}/checkpoints/cl_libero_object/"
