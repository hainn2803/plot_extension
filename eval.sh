# #!/bin/bash
# set -euo pipefail

# cd /u/hnguyen23/plot_extension

# source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
# conda activate PLOT

CONFIG=222222

for SEED in {0..4}; do
    echo "========== Evaluating seed ${SEED} =========="

    python3 -u mcqa_eval_gradual_completed_recipient.py \
        --result-dir "results_gradual_${CONFIG}_selected_layers_controlled_seed${SEED}_no_distractor" \
        --seed "$SEED"
done

echo "All evaluations completed."