#!/usr/bin/env bash
# Resumable base-gate/LoRA-mask factorial, evaluated on held-out cat assets.
set -Eeuo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 shared_teacher|factorial TASK_INDEX" >&2
    exit 2
fi

stage="$1"
task="$2"
selection="artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/selection.pt"
selector_config="code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"
configs=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
)

[[ -f "$selection" ]] || { echo "Missing frozen base selector: $selection" >&2; exit 1; }

case "$stage" in
    shared_teacher)
        [[ "$task" == "0" ]] || { echo "shared_teacher only uses task 0" >&2; exit 2; }
        python code/scripts/sl/activations/evaluate_base_lora_factorial.py shared_teacher \
            --config "$selector_config" --selection "$selection" --samples 10
        ;;
    factorial)
        seed_index=$(( task / 2 ))
        condition_index=$(( task % 2 ))
        config="${configs[$seed_index]:?Invalid task index}"
        conditions=(trait control)
        condition="${conditions[$condition_index]}"
        python code/scripts/sl/activations/evaluate_base_lora_factorial.py factorial \
            --config "$config" --condition "$condition" --selection "$selection" \
            --samples 5 --numeric-examples 128
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
