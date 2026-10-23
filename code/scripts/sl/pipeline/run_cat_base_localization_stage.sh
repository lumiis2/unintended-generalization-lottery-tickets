#!/usr/bin/env bash
# Resumable two-stage base-component localization pilot for cat SL.
set -Eeuo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 selection|evaluation TASK_INDEX" >&2
    exit 2
fi
stage="$1"
task="$2"
selector_config="code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"
configs=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
)
selection="artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/selection.pt"

case "$stage" in
    selection)
        [[ "$task" == "0" ]] || { echo "selection uses task 0" >&2; exit 2; }
        python code/scripts/sl/activations/base_component_localization.py select \
            --config "$selector_config" --layer 28 --prompts 50 --batch-size 4 --candidate-count 128
        ;;
    evaluation)
        config="${configs[$task]:?Invalid task index}"
        python code/scripts/sl/activations/base_component_localization.py evaluate \
            --config "$config" --selection "$selection" --layer 28 --trait-samples 10 --numeric-examples 128
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
