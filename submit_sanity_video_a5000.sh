#!/bin/bash
# Submit a single A5000 job: rollout video sanity check (one checkpoint, save_video on).
# The compute step runs only inside the Singularity image (dp_libero.sif), via sbatch.
#
# Prerequisites: dp_libero.sif (or SIF_IMAGE), data/, checkpoints/cl_libero_object/.
#
# Usage:
#   bash submit_sanity_video_a5000.sh
#   CKPT_KK=03 RESULT_TAG=mytag bash submit_sanity_video_a5000.sh
#   SIF_IMAGE=/path/to/dp_libero.sif PARTITION=gpu bash submit_sanity_video_a5000.sh
#
# Videos: results/sanity_video_${RESULT_TAG}/videos/after_task_<KK>/*.mp4
# Logs:    logs/lib_video_sanity_<SLURM_JOBID>.{out,err}

set -euo pipefail
BASE="/home/cyhoaoen/dp_forgetting_libero"
mkdir -p "${BASE}/logs"

RESULT_TAG="${RESULT_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT_KK="${CKPT_KK:-00}"
SIF_IMAGE="${SIF_IMAGE:-${BASE}/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a5000}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lib_vid_sanity
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=${BASE}/logs/lib_video_sanity_%j.out
#SBATCH --error=${BASE}/logs/lib_video_sanity_%j.err

set -euo pipefail
cd ${BASE}
mkdir -p "${BASE}/results/sanity_video_${RESULT_TAG}"

export CKPT_KK=${CKPT_KK}
export RESULT_TAG=${RESULT_TAG}
export EVAL_CONFIG=/workspace/configs/sanity_eval_video.yaml
export CKPT_DIR=/workspace/checkpoints/cl_libero_object
export RESULTS_DIR=/workspace/results/sanity_video_${RESULT_TAG}

# Whole project at /workspace (same pattern as run_evaluate_singularity.sh).
exec singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}:/workspace \\
  "${SIF_IMAGE}" \\
  bash /workspace/scripts/evaluation/singularity_eval_video_sanity.sh
EOF

echo "Submitted video sanity check (runs inside SIF on the batch node). CKPT_KK=${CKPT_KK} RESULT_TAG=${RESULT_TAG}"
echo "  SIF: ${SIF_IMAGE}"
echo "  Results dir: ${BASE}/results/sanity_video_${RESULT_TAG}"
echo "  Expect mp4 under: .../videos/after_task_${CKPT_KK}/"
