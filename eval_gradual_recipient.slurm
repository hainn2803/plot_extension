#!/bin/bash
#SBATCH --job-name=gdas-eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgvo-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --array=0-14
#SBATCH --output=slurm_eval_%A_%a.out
#SBATCH --error=slurm_eval_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate PLOT

CONFIGS=(222222)
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))

"$CONDA_PREFIX/bin/python" -u mcqa_eval_gradual_completed_recipient.py \
    --result-dir "results_gradual_${CONFIG}_selected_layers_controlled_seed${SEED}_no_distractor" \
    --seed "$SEED"