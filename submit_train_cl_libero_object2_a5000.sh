#!/bin/bash
# Submit Sequential CL Training (LIBERO-Object) on Slurm + Singularity.
#
# Usage:
#   bash submit_train_cl_libero_object2_a5000.sh
#   GPU_DEVICE=0 PARTITION=gigabyte_a5000 TIME=24:00:00 bash submit_train_cl_libero_object2_a5000.sh
#   SIF_IMAGE=/home/cyhoaoen/dp_forgetting_libero/dp_libero.sif bash submit_train_cl_libero_object2_a5000.sh
#
# Notes:
# - Checkpoints: /home/cyhoaoen/dp_forgetting_libero/checkpoints/cl_libero_object2
# - Results:     /home/cyhoaoen/dp_forgetting_libero/results/cl_libero_object2
# - W&B:         enabled in configs/continual_learning_libero_object.yaml (entity: ingchicken)
set -euo pipefail

BASE="/home/cyhoaoen/dp_forgetting_libero"
mkdir -p "${BASE}/logs" "${BASE}/wandb" \
  "${BASE}/checkpoints/cl_libero_object3" "${BASE}/results/cl_libero_object3"

SIF_IMAGE="${SIF_IMAGE:-${BASE}/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a5000}"
TIME="${TIME:-48:00:00}"
CPU="${CPU:-8}"
MEM="${MEM:-32G}"
GPU_TYPE="${GPU_TYPE:-A5000}"
GPU_N="${GPU_N:-1}"

# W&B credentials typically come from ~/.netrc or WANDB_API_KEY in your shell.
# sbatch inherits env by default, so if WANDB_API_KEY is set outside, it will be visible.

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cl_obj3_train
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_N}
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${BASE}/logs/cl_obj3_train_%j.out
#SBATCH --error=${BASE}/logs/cl_obj3_train_%j.err

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
    python -m scripts.train_sequential \\
      --config /workspace/configs/continual_learning_libero_object.yaml \\
      --skip-eval
  '
EOF

echo "Submitted: sequential CL training (LIBERO-Object3)"
echo "  Logs:        ${BASE}/logs/cl_obj3_train_<JOBID>.{out,err}"
echo "  Checkpoints: ${BASE}/checkpoints/cl_libero_object3/"
echo "  Results:     ${BASE}/results/cl_libero_object3/"
