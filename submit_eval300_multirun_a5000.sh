#!/bin/bash
# Extra 300-step eval repeats for averaging SR (separate dirs per run; does not edit main yaml).
#
# Default: submit runs r2–r5 (40 jobs each × 4 = 160 jobs) unless RUN_INDICES is set.
# Run 1 = your first trial → put those under results/cl_libero_object_steps300_r1_t* or symlink.
#
# Safe alongside 600 jobs: uses configs/continual_learning_libero_object_eval300.yaml only;
# continual_learning_libero_object.yaml stays at 600 for queued 600-step jobs.
#
# Usage:
#   RUN_INDICES="2 3 4 5" bash submit_eval300_multirun_a5000.sh
# Optional same tag for all jobs in one invocation (default: timestamp):
#   BATCH_TAG=20250404a RUN_INDICES="2 3 4 5" bash submit_eval300_multirun_a5000.sh

set -euo pipefail
BASE="/home/cyhoaoen/dp_forgetting_libero"
mkdir -p "${BASE}/logs"

# One tag per sbatch wave → result dirs never overwrite previous waves.
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
echo "BATCH_TAG=${BATCH_TAG}  (results: .../cl_libero_object_steps300_${BATCH_TAG}_r*_t*)"

# Logs: lib_eval300_r<RUN>_t<TT>_<SLURM_JOBID>.{out,err} — %j avoids overwrite on resubmit.
RUN_INDICES="${RUN_INDICES:-2 3 4 5}"

for RUN in ${RUN_INDICES}; do
  for k in 0 1 2 3 4 5 6 7 8 9; do
    printf -v KK "%02d" "$k"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lib_eval300_r${RUN}_t${KK}
#SBATCH --partition=gigabyte_a5000
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=${BASE}/logs/lib_eval300_r${RUN}_t${KK}_%j.out
#SBATCH --error=${BASE}/logs/lib_eval300_r${RUN}_t${KK}_%j.err

cd ${BASE}
mkdir -p ${BASE}/results/cl_libero_object_steps300_${BATCH_TAG}_r${RUN}_t${KK}
export CKPT_KK=${KK}
export EVAL_CONFIG=/workspace/configs/continual_learning_libero_object_eval300.yaml

singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}/data:/workspace/data \\
  --bind ${BASE}/checkpoints/cl_libero_object:/workspace/checkpoints \\
  --bind ${BASE}/results/cl_libero_object_steps300_${BATCH_TAG}_r${RUN}_t${KK}:/workspace/results \\
  --bind ${BASE}/scripts:/workspace/scripts \\
  --bind ${BASE}/configs:/workspace/configs \\
  ${BASE}/dp_libero.sif \\
  bash /workspace/scripts/evaluation/singularity_eval_one_ckpt.sh
EOF
  done
done

NRUNS=$(set -- ${RUN_INDICES}; echo "$#")
echo "Submitted $((NRUNS * 10)) jobs for RUN_INDICES=${RUN_INDICES} BATCH_TAG=${BATCH_TAG}"
echo "Merge one repeat index after its t00–t09 finish (example r=2):"
echo "  python scripts/evaluation/merge_perf_matrices.py \\"
echo "    --csv-glob 'results/cl_libero_object_steps300_${BATCH_TAG}_r2_t*/perf_matrix.csv' \\"
echo "    --out-dir results/cl_libero_object_steps300_${BATCH_TAG}_r2_merged"
echo "Across waves, use distinct BATCH_TAG dirs; glob e.g. '*_r2_t*/perf_matrix.csv' if needed."
