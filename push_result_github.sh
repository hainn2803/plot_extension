#!/bin/bash
set -e

push_result() {
    local folder="$1"

    if [ ! -d "$folder" ]; then
        echo "Skipping missing folder: $folder"
        return
    fi

    echo "========== Uploading $folder =========="

    git add -- "$folder/"

    if git diff --cached --quiet; then
        echo "Already up to date"
        return
    fi

    git commit -m "Update $folder"
    git push origin gradual-das-mcqa
}

if ! git diff --cached --quiet; then
    echo "Unexpected staged changes. Check git status first."
    exit 1
fi

# GDAS
for config in 020202 202020 222200 222222; do
    for seed in 0 1 2 3 4; do
        folder="results_gradual_${config}_selected_layers_controlled_seed${seed}_no_distractor"
        push_result "$folder"
    done
done

# SDAS
for seed in 0 1 2 3 4; do
    folder="results_standard_all_layers_controlled_seed${seed}_no_distractor"
    push_result "$folder"
done

echo "All results uploaded successfully."