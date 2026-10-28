#!/bin/bash
#SBATCH --job-name=das-std-grad-dim
#SBATCH --account=bgvo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/das-std-grad-dim-%j.out
#SBATCH --error=logs/das-std-grad-dim-%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/.bashrc
conda activate PLOT

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

nvidia-smi

python3 -u mcqa_compare_standard_gradual_5seeds_dim_sweep.py \
    --output-root results_standard_vs_gradual_5seeds_dim_sweep_slurm \
    --at-dims 8,16,32,64,128 \
    --at-dims 8,16,32,64,128 \
    --resume

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
