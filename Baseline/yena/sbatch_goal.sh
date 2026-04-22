#!/bin/bash
#SBATCH --job-name=cl_goal
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/kyn7666/logs/cl_goal-%j.out
#SBATCH --error=/home/kyn7666/logs/cl_goal-%j.err

set -e

BASE_DIR="/home/kyn7666/cl_diffusion_libero-object"
PRETRAIN_CKPT="${BASE_DIR}/checkpoints/pretrain_libero90/final.pt"
CONFIG="${BASE_DIR}/configs/continual_learning_libero_goal.yaml"

mkdir -p "${BASE_DIR}/checkpoints/cl_libero_goal"
mkdir -p "${BASE_DIR}/results/cl_libero_goal"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

echo "========================================================"
echo " CL Training: Diffusion Policy + LIBERO-Goal"
echo " Node: $(hostname)"
echo " Job ID: $SLURM_JOB_ID"
echo "========================================================"

cd "${BASE_DIR}"

# ── Training ──────────────────────────────────────────────
python -m scripts.train_sequential \
    --config "${CONFIG}" \
    --pretrain-ckpt "${PRETRAIN_CKPT}" \
    --skip-eval

echo ""
echo "Training done. Starting evaluation..."

# ── Evaluation ────────────────────────────────────────────
python -m scripts.evaluation.evaluate_checkpoints \
    --config "${CONFIG}"

echo ""
echo "All done! Results: ${BASE_DIR}/results/cl_libero_goal/"
