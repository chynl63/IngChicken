#!/bin/bash
set -e

# ========================================================
#  LIBERO-90 Pretraining (Singularity)
# ========================================================
#
# Usage:
#   bash run_pretrain_singularity.sh
#   GPU_DEVICE=1 bash run_pretrain_singularity.sh
#   LIBERO90_DIR=/path/to/libero90 bash run_pretrain_singularity.sh
#
# To run in background (survives logout):
#   nohup bash run_pretrain_singularity.sh > pretrain.log 2>&1 &

SIF_IMAGE="${SIF_IMAGE:-/scratch2/cyhoaoen/simg/dp_libero.sif}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
LIBERO90_DIR="${LIBERO90_DIR:-/scratch2/kyn7666/libero_data/libero_90}"

mkdir -p "${BASE_DIR}/checkpoints"

echo "========================================================"
echo " LIBERO-90 Pretraining: Diffusion Policy"
echo " (Singularity)"
echo "========================================================"
echo "  SIF:        $SIF_IMAGE"
echo "  GPU:        $GPU_DEVICE"
echo "  Mount:      $BASE_DIR -> /workspace"
echo "  LIBERO-90:  $LIBERO90_DIR -> /libero90_data"
echo "  Checkpoint: ${BASE_DIR}/checkpoints/pretrain_libero90/"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_DEVICE singularity exec --nv --writable-tmpfs \
    --bind "${BASE_DIR}:/workspace" \
    --bind "${LIBERO90_DIR}:/libero90_data" \
    "$SIF_IMAGE" \
    bash -lc '
      set -e
      cd /workspace
      source /opt/conda/etc/profile.d/conda.sh
      conda activate dp
      pip install -q "numpy<2" "h5py<3.12" bddl easydict cloudpickle gym

      # Override data_dir in config to point to the bound path
      python -c "
import yaml, sys
with open(\"/workspace/configs/pretrain_libero90.yaml\") as f:
    cfg = yaml.safe_load(f)
cfg[\"pretrain\"][\"data_dir\"] = \"/libero90_data\"
cfg[\"logging\"][\"checkpoint_dir\"] = \"/workspace/checkpoints/pretrain_libero90\"
with open(\"/tmp/pretrain_cfg.yaml\", \"w\") as f:
    yaml.safe_dump(cfg, f)
print(\"Config written to /tmp/pretrain_cfg.yaml\")
"
      python -m scripts.train_pretrain \
        --config /tmp/pretrain_cfg.yaml
    '

echo ""
echo "Pretraining complete!"
echo "Checkpoint: ${BASE_DIR}/checkpoints/pretrain_libero90/final.pt"
echo ""
echo "Next: fine-tune on LIBERO-Object with:"
echo "  SIF_IMAGE=... bash run_sequential_singularity.sh --pretrain-ckpt /workspace/checkpoints/pretrain_libero90/final.pt"
