#!/bin/bash
# Submit full CL checkpoint evaluation (all after_task_*.pt) via Slurm + Singularity.
# Evaluation runs only inside dp_libero.sif — no direct host Python.
#
# Usage:
#   bash submit_eval_singularity.sh
#   EVAL_SUITE=object bash submit_eval_singularity.sh
#   EVAL_SUITE=both bash submit_eval_singularity.sh    # two jobs: spatial + object
#
# Optional env (defaults match training submit):
#   PARTITION=gigabyte_a6000  TIME=24:00:00  CPU=8  MEM=32G
#   GPU_TYPE=A6000  GPU_N=1  SIF_IMAGE=...  BASE=...
#   JOB_PREFIX=eval_cl
#
set -euo pipefail

BASE="${BASE:-/home/cyhoaoen/dp_forgetting_libero}"
SIF_IMAGE="${SIF_IMAGE:-${BASE}/dp_libero.sif}"
PARTITION="${PARTITION:-gigabyte_a6000}"
TIME="${TIME:-24:00:00}"
CPU="${CPU:-8}"
MEM="${MEM:-32G}"
GPU_TYPE="${GPU_TYPE:-A6000}"
GPU_N="${GPU_N:-1}"
EVAL_SUITE="${EVAL_SUITE:-spatial}"
JOB_PREFIX="${JOB_PREFIX:-eval_cl}"

mkdir -p "${BASE}/logs"

submit_one() {
  local suite="$1"
  local name="${JOB_PREFIX}_${suite}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_N}
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${BASE}/logs/${name}_%j.out
#SBATCH --error=${BASE}/logs/${name}_%j.err

set -euo pipefail
cd ${BASE}

export CUDA_VISIBLE_DEVICES="\${GPU_DEVICE:-0}"
export EVAL_SUITE=${suite}
# Note: only escape the leading $ so $(date...) runs on the compute node; do not escape the closing ).
EVAL_RUN_TAG=\$(date +%Y%m%d_%H%M%S)
export EVAL_RUN_TAG

exec singularity exec --nv --writable-tmpfs \\
  --bind ${BASE}:/workspace \\
  --bind /home/cyhoaoen:/home/cyhoaoen \\
  "${SIF_IMAGE}" \\
  bash /workspace/scripts/singularity/sif_eval_checkpoints.sh
EOF
}

case "${EVAL_SUITE}" in
  spatial)
    submit_one spatial
    echo "Submitted: eval (LIBERO-Spatial) — logs ${BASE}/logs/${JOB_PREFIX}_spatial_<JOBID>.{out,err}"
    ;;
  object)
    submit_one object
    echo "Submitted: eval (LIBERO-Object) — logs ${BASE}/logs/${JOB_PREFIX}_object_<JOBID>.{out,err}"
    ;;
  both)
    submit_one spatial
    submit_one object
    echo "Submitted: two jobs (spatial + object) — logs under ${BASE}/logs/${JOB_PREFIX}_*{_<JOBID>.out,.err}"
    ;;
  *)
    echo "EVAL_SUITE must be spatial, object, or both; got: ${EVAL_SUITE}" >&2
    exit 2
    ;;
esac
