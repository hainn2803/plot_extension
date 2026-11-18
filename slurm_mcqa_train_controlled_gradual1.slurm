#!/bin/bash
#SBATCH --job-name=mcqa-gradual-controlled1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgvo-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --array=1-4
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate PLOT

SEED=$SLURM_ARRAY_TASK_ID

python3 -u mcqa_train_gradual_efficient_v2.py \
    --data-path randomized_copycolors_include_ap_at_no_distractor \
    --result-dir "results_gradual_020202_selected_layers_controlled_seed${SEED}_no_distractor" \
    --seed "$SEED" \
    --ap-lambda-med 0 \
    --ap-lambda-readout 2 \
    --c-lambda-ap-med 0 \
    --c-lambda-ap-readout 2 \
    --c-lambda-at-med 0 \
    --c-lambda-at-readout 2