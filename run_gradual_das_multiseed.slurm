#!/bin/bash
#SBATCH --job-name=gradual-das
#SBATCH --account=bgvo-delta-gpu
#SBATCH --partition=gpuA100x4

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

# 5 seeds: 0,1,2,3,4
# At most 4 seeds run at the same time.
#SBATCH --array=0-4%4

#SBATCH --output=gradual-das-%A_%a.out
#SBATCH --error=gradual-das-%A_%a.err


set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Activate environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PLOT


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

# Python file
SCRIPT="mcqa_gradual_das_all_lambdas_save_every.py"

# Downstream AT objects
AT_CHECKPOINT="results_at/das_at.pt"
AT_READOUT_CHECKPOINT="results_at/at_readout_LogisticRegression.pt"

# Lambda grid
LAMBDA_MED_GRID="0.0,0.1,0.3,1,3,5,8,10,50"
LAMBDA_READOUT_GRID="0.0,0.1,0.3,1,3,5,8,10,50"

# Seeds
SEEDS=(2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

# AP search / DAS
LAYERS="all"
SUBSPACE_DIM=128
TOKEN_POSITION=""

# Data
FT_SIZE=400
CAL_SIZE=200
TE_SIZE=200
DATA_SEED=0

# Optimization
EPOCHS=100
SAVE_EVERY=1
LR=1e-2
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128

# Output root.
# The Python script creates:
#   ${OUTPUT_DIR}/seed_0/
#   ${OUTPUT_DIR}/seed_1/
#   ...
OUTPUT_DIR="results_gradual_das_multiseed"


# ============================================================
# PRINT FULL CONFIG
# ============================================================

echo "============================================================"
echo "                 GRADUAL DAS MULTI-SEED RUN"
echo "============================================================"
echo "SLURM_JOB_ID            = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID     = $SLURM_ARRAY_TASK_ID"
echo "HOSTNAME                = $(hostname)"
echo "WORKING_DIR             = $(pwd)"
echo
echo "SCRIPT                   = $SCRIPT"
echo "AT_CHECKPOINT            = $AT_CHECKPOINT"
echo "AT_READOUT_CHECKPOINT    = $AT_READOUT_CHECKPOINT"
echo
echo "LAMBDA_MED_GRID          = $LAMBDA_MED_GRID"
echo "LAMBDA_READOUT_GRID      = $LAMBDA_READOUT_GRID"
echo "SEED                     = $SEED"
echo
echo "LAYERS                   = $LAYERS"
echo "SUBSPACE_DIM             = $SUBSPACE_DIM"
echo "TOKEN_POSITION           = ${TOKEN_POSITION:-<from AT checkpoint>}"
echo
echo "FT_SIZE                  = $FT_SIZE"
echo "CAL_SIZE                 = $CAL_SIZE"
echo "TE_SIZE                  = $TE_SIZE"
echo "DATA_SEED                = $DATA_SEED"
echo
echo "EPOCHS                   = $EPOCHS"
echo "SAVE_EVERY               = $SAVE_EVERY"
echo "LR                       = $LR"
echo "TRAIN_BATCH_SIZE         = $TRAIN_BATCH_SIZE"
echo "EVAL_BATCH_SIZE          = $EVAL_BATCH_SIZE"
echo
echo "OUTPUT_DIR               = $OUTPUT_DIR"
echo "============================================================"
echo

nvidia-smi
echo


# ============================================================
# BUILD COMMAND
# ============================================================

CMD=(
    python -u "$SCRIPT"
    --at-checkpoint "$AT_CHECKPOINT"
    --at-readout-checkpoint "$AT_READOUT_CHECKPOINT"
    --lambda-med-grid "$LAMBDA_MED_GRID"
    --lambda-readout-grid "$LAMBDA_READOUT_GRID"
    --seeds "$SEED"
    --layers "$LAYERS"
    --subspace-dim "$SUBSPACE_DIM"
    --ft-size "$FT_SIZE"
    --cal-size "$CAL_SIZE"
    --te-size "$TE_SIZE"
    --data-seed "$DATA_SEED"
    --epochs "$EPOCHS"
    --save-every "$SAVE_EVERY"
    --lr "$LR"
    --train-batch-size "$TRAIN_BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --output-dir "$OUTPUT_DIR"
)

# Only pass token position when explicitly set.
if [[ -n "$TOKEN_POSITION" ]]; then
    CMD+=(--token-position "$TOKEN_POSITION")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "============================================================"
echo "FINISHED seed=$SEED"
echo "Output: ${OUTPUT_DIR}/seed_${SEED}/"
echo "============================================================"
