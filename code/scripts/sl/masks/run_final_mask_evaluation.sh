#!/usr/bin/env bash
# Run the frozen final LoRA-mask evaluation for one independent SL replication.
set -Eeuo pipefail

CONFIGS=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
)
selector="${1:?Usage: $0 TASK_INDEX|CONFIG_PATH}"
if [[ "$selector" =~ ^[0-9]+$ ]]; then
    config="${CONFIGS[$selector]:?Invalid TASK_INDEX}"
else
    config="$selector"
fi

# Each evaluator checks for its own complete outputs and skips them.  Therefore
# reruns/requeues resume at the next unfinished mask rather than repeating work.
for condition in trait control; do
    python code/scripts/sl/masks/evaluate_final_lora_masks.py \
        --config "$config" \
        --condition "$condition" \
        --samples-per-trait-prompt "${SAMPLES_PER_TRAIT_PROMPT:-30}" \
        --random-controls "${RANDOM_CONTROLS:-5}" \
        --mask-seed-base "${MASK_SEED_BASE:-20261010}"
done

for condition in trait control; do
    python code/scripts/sl/masks/evaluate_frozen_general_benchmarks.py \
        --config "$config" \
        --condition "$condition" \
        --random-controls "${RANDOM_CONTROLS:-5}" \
        --mask-seed-base "${MASK_SEED_BASE:-20261010}"
done
