#!/usr/bin/env bash
# Independent owl replication of the frozen-base component / LoRA factorial.
set -Eeuo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 selection|shared_teacher|factorial TASK_INDEX" >&2
    exit 2
fi

stage="$1"
task="$2"
selector_config="code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"
configs=(
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml
)
selection="artifacts/sl/qwen25_7b_owl_replication/base_component_localization_v1/selection.pt"

case "$stage" in
    selection)
        [[ "$task" == "0" ]] || { echo "selection only uses task 0" >&2; exit 2; }
        # Layer 21 was fixed from the existing owl-47 controlled-direction scan;
        # owl-48/49 are not used during this selection.
        python code/scripts/sl/activations/base_component_localization.py select \
            --config "$selector_config" --layer 21 --prompts 50 --batch-size 4 --candidate-count 128
        ;;
    shared_teacher)
        [[ "$task" == "0" ]] || { echo "shared_teacher only uses task 0" >&2; exit 2; }
        [[ -f "$selection" ]] || { echo "Missing frozen owl selector: $selection" >&2; exit 1; }
        python code/scripts/sl/activations/evaluate_base_lora_factorial.py shared_teacher \
            --config "$selector_config" --selection "$selection" --layer 21 --samples 10
        ;;
    factorial)
        [[ -f "$selection" ]] || { echo "Missing frozen owl selector: $selection" >&2; exit 1; }
        seed_index=$(( task / 2 ))
        condition_index=$(( task % 2 ))
        config="${configs[$seed_index]:?Invalid task index}"
        conditions=(trait control)
        condition="${conditions[$condition_index]}"
        python code/scripts/sl/activations/evaluate_base_lora_factorial.py factorial \
            --config "$config" --condition "$condition" --selection "$selection" --layer 21 \
            --samples 5 --numeric-examples 128
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
