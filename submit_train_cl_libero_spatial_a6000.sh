#!/bin/bash
# Submit Sequential CL Training (LIBERO-Spatial) on Slurm + Singularity (A6000).
# (Pattern matches submit_train_cl_libero_object2_a5000.sh)
#
# Usage:
#   bash submit_train_cl_libero_spatial_a6000.sh
#   GPU_DEVICE=0 PARTITION=gigabyte_a6000 TIME=48:00:00 bash submit_train_cl_libero_spatial_a6000.sh
#   SIF_IMAGE=/home/cyhoaoen/dp_forgetting_libero/dp_libero.sif bash submit_train_cl_libero_spatial_a6000.sh
#
# Optional:
#   SKIP_EVAL=0 bash submit_train_cl_libero_spatial_a6000.sh   # run per-task rollouts (slower)
#
# Notes:
# - Config: configs/continual_learning_libero_spatial.yaml (W&B: project sdft, group mse_onlycl)
# - Data:   data/libero_spatial under repo (benchmark.data_root = .../data)
# - Checkpoints: checkpoints/cl_libero_spatial
# - Results:     results/cl_libero_spatial
# - W&B API key: export WANDB_API_KEY=... or ~/.netrc before sbatch
set -euo pipefail

BASE="/home/cyhoaoen/dp_forgetting_libero"
mkdir -p "${BASE}/logs" "${BASE}/wandb" \
  "${BASE}/checkpoints/cl_libero_spatial" "${BASE}/results/cl_libero_spatial"

SIF_IMAGE="${SIF_IMAGE:-${BASE}/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a6000}"
TIME="${TIME:-48:00:00}"
CPU="${CPU:-8}"
MEM="${MEM:-32G}"
GPU_TYPE="${GPU_TYPE:-A6000}"
GPU_N="${GPU_N:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"
TRAIN_EXTRA=""
if [[ "${SKIP_EVAL}" == "1" ]]; then
  TRAIN_EXTRA="--skip-eval"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cl_spatial_train
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_N}
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${BASE}/logs/cl_spatial_train_%j.out
#SBATCH --error=${BASE}/logs/cl_spatial_train_%j.err

set -euo pipefail
cd ${BASE}

export CUDA_VISIBLE_DEVICES="\${GPU_DEVICE:-0}"

exec singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}:/workspace \\
  --bind /home/cyhoaoen:/home/cyhoaoen \\
  "${SIF_IMAGE}" \\
  bash -lc '
    set -euo pipefail
    cd /workspace
    source /workspace/scripts/singularity/dp_image_env.sh
    python -m pip install -q wandb
    python -m scripts.train_sequential --config /workspace/configs/continual_learning_libero_spatial.yaml ${TRAIN_EXTRA}
  '
EOF

echo "Submitted: sequential CL training (LIBERO-Spatial, ${GPU_TYPE})"
echo "  SKIP_EVAL=${SKIP_EVAL}  (set SKIP_EVAL=0 for inline eval)"
echo "  Logs:        ${BASE}/logs/cl_spatial_train_<JOBID>.{out,err}"
echo "  Checkpoints: ${BASE}/checkpoints/cl_libero_spatial/"
echo "  Results:     ${BASE}/results/cl_libero_spatial/"
