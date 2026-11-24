# #!/bin/bash
# source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
# conda activate PLOT

# mkdir -p logs_standard_eval_controlled

# for SEED in 0 1 2 3 4; do
#     echo "Running with seed $SEED"

#     "$CONDA_PREFIX/bin/python" -u mcqa_eval_selected_layers.py \
#     --result-dir results_standard_all_layers_controlled_seed${SEED}_no_distractor \
#     --at-layers 23 24 25 \
#     --ap-layers 4 5 6 7 8 9 10 11 12 13 14 15 \
#     --c-layers 0 1 2 \
#     --seed "$SEED" 2>&1 | tee "logs_standard_eval_controlled/seed_${SEED}.log"

# done



#!/bin/bash
set -euo pipefail

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate PLOT

mkdir -p logs_gradual_eval_controlled

for SEED in 0 1 2 3 4; do
    echo "Running with seed $SEED"

    "$CONDA_PREFIX/bin/python" -u mcqa_eval_gradual_completed_recipient.py \
        --result-dir results_gradual_020202_selected_layers_controlled_seed${SEED}_no_distractor \
        --seed "$SEED" \
        2>&1 | tee "logs_gradual_eval_controlled/seed_${SEED}.log"
done