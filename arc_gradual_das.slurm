#!/bin/bash
#SBATCH --job-name=arc-gradual-das
#SBATCH --account=bgvo-delta-gpu
#SBATCH --partition=gpuA100x4

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

# 4 lambda configurations
#SBATCH --array=0-3%4

#SBATCH --output=arc-gradual-das-%A_%a.out
#SBATCH --error=arc-gradual-das-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PLOT


# ============================================================
# CONFIG
# ============================================================

SCRIPT="arc_gradual_das.py"

OANSWER_CHECKPOINT="results_arc/gradual_das_ap_lambda_med_0.0_lambda_readout_1.0/oanswer.pt"

# Lambda configurations
LAMBDA_MED=(0.0 0.0 1.0 1.0)
LAMBDA_READOUT=(3.0 8.0 3.0 8.0)

IDX="$SLURM_ARRAY_TASK_ID"

LM="${LAMBDA_MED[$IDX]}"
LR="${LAMBDA_READOUT[$IDX]}"

# Data
FIT_SIZE=400
CAL_SIZE=200
TEST_SIZE=200

# DAS
K=128
EPOCHS=50
LEARNING_RATE=1e-2
BATCH_SIZE=32
SAVE_EVERY=1
SEED=0

OUTPUT_DIR="results_arc"


# ============================================================
# PRINT CONFIG
# ============================================================

echo "============================================================"
echo "                    ARC GRADUAL DAS"
echo "============================================================"
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "HOSTNAME            = $(hostname)"
echo
echo "lambda_med          = $LM"
echo "lambda_readout      = $LR"
echo "seed                = $SEED"
echo
echo "fit_size            = $FIT_SIZE"
echo "cal_size            = $CAL_SIZE"
echo "test_size           = $TEST_SIZE"
echo "k                   = $K"
echo "epochs              = $EPOCHS"
echo "batch_size          = $BATCH_SIZE"
echo "============================================================"
echo

nvidia-smi
echo


# ============================================================
# RUN
# ============================================================

python -u "$SCRIPT" \
    --fit-size "$FIT_SIZE" \
    --cal-size "$CAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --k "$K" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --lambda-med "$LM" \
    --lambda-readout "$LR" \
    --seed "$SEED" \
    --save-every "$SAVE_EVERY" \
    --output-dir "$OUTPUT_DIR" \
    --oanswer-checkpoint "$OANSWER_CHECKPOINT"


echo
echo "============================================================"
echo "FINISHED"
echo "lambda_med=$LM lambda_readout=$LR seed=$SEED"
echo "============================================================"