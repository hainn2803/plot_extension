#!/bin/bash

# mkdir -p logs

# for SEED in 1 2 3 4 5; do
#     echo "Running with seed $SEED"

#     python3 -u mcqa_train_all_layers_no_eval.py \
#         --result-dir "results_standard_all_layers_controlled_seed${SEED}_no_distractor" \
#         --variables AT \
#         --layers 23,24,25 \
#         --seed "$SEED" 2>&1 | tee "logs_controlled/seed_${SEED}.log"
# done


python3 mcqa_eval_completed_gradual.py --result-dir results_gradual_202020_selected_layers_controlled_seed0_no_distractor

python3 mcqa_eval_completed_gradual.py --result-dir results_gradual_222200_selected_layers_controlled_seed0_no_distractor

python3 mcqa_eval_completed_gradual.py --result-dir results_gradual_222222_selected_layers_controlled_seed0_no_distractor