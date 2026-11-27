#!/bin/bash
#SBATCH --job-name=mcqa-gradual-controlled3
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4,gpuH200x8
#SBATCH --account=bgvo-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd "$SLURM_SUBMIT_DIR"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate PLOT

python3 -u mcqa_train_all_layers_no_eval.py --data-path randomized_copycolors_include_ap_at_no_distractor --result-dir results_standard_all_layers_mixedall_seed0_no_distractor_slurm
