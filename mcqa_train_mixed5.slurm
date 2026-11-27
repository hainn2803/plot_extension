#!/bin/bash
#SBATCH --job-name=mcqa-gradual-mixedall5
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4,gpuH200x8
#SBATCH --account=bgvo-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd "$SLURM_SUBMIT_DIR"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate PLOT

python3 -u mcqa_train_gradual_efficient_v2.py \
    --data-path randomized_copycolors_include_ap_at_no_distractor \
    --result-dir results_gradual_202020_selected_layers_mixedall_seed0_no_distractor \
    --ap-lambda-med 2 \
    --ap-lambda-readout 0 \
    --c-lambda-ap-med 2 \
    --c-lambda-ap-readout 0 \
    --c-lambda-at-med 2 \
    --c-lambda-at-readout 0