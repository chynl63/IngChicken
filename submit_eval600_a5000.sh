#!/bin/bash
# Submit per-checkpoint eval on A5000 (merge cl_libero_object_steps600_<BATCH_TAG>_t* later).
#
# BATCH_TAG defaults to timestamp so re-submit never overwrites prior result dirs.
#   BATCH_TAG=run1 bash submit_eval600_a5000.sh

set -euo pipefail
BASE="/home/cyhoaoen/dp_forgetting_libero"
mkdir -p "${BASE}/logs"

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
echo "BATCH_TAG=${BATCH_TAG}  (results: .../cl_libero_object_steps600_${BATCH_TAG}_t*)"

for k in 0 1 2 3 4 5 6 7 8 9; do
  printf -v KK "%02d" "$k"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lib_eval600_t${KK}
#SBATCH --partition=gigabyte_a5000
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=${BASE}/logs/lib_eval600_t${KK}_%j.out
#SBATCH --error=${BASE}/logs/lib_eval600_t${KK}_%j.err

cd ${BASE}
mkdir -p ${BASE}/results/cl_libero_object_steps600_${BATCH_TAG}_t${KK}
export CKPT_KK=${KK}

singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}/data:/workspace/data \\
  --bind ${BASE}/checkpoints/cl_libero_object:/workspace/checkpoints \\
  --bind ${BASE}/results/cl_libero_object_steps600_${BATCH_TAG}_t${KK}:/workspace/results \\
  --bind ${BASE}/scripts:/workspace/scripts \\
  --bind ${BASE}/configs:/workspace/configs \\
  ${BASE}/dp_libero.sif \\
  bash /workspace/scripts/evaluation/singularity_eval_one_ckpt.sh
EOF
done

echo "Submitted 10 jobs (BATCH_TAG=${BATCH_TAG}). Merge after completion:"
echo "  cd ${BASE} && python scripts/evaluation/merge_perf_matrices.py \\"
echo "    --csv-glob 'results/cl_libero_object_steps600_${BATCH_TAG}_t*/perf_matrix.csv' \\"
echo "    --out-dir results/cl_libero_object_steps600_${BATCH_TAG}_merged"
