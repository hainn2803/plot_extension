#!/bin/bash
#SBATCH --job-name=mcqa-lambda
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --time=6:00:00
#SBATCH --array=0-319%8
#SBATCH --output=slurm_logs/mcqa_lambda_%A_%a.out
#SBATCH --error=slurm_logs/mcqa_lambda_%A_%a.err
#SBATCH --account=bgvo-delta-gpu

set -e

cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs

source ~/.bashrc
conda activate PLOT

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python3 mcqa_tune_one_config.py --task-id "$SLURM_ARRAY_TASK_ID"
