#!/usr/bin/env bash
# Run an in-memory causal mask screen for one independently trained SL run.
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
task_index="${1:?Usage: $0 TASK_INDEX}"
config="${CONFIGS[$task_index]:?Invalid TASK_INDEX: $task_index}"

read -r -a conditions <<< "${CAUSAL_CONDITIONS:-trait control}"
for condition in "${conditions[@]}"; do
    python scripts/sl/evaluate_causal_masks_in_memory.py \
        --config "$config" --condition "$condition" \
        --samples-per-prompt "${SAMPLES_PER_PROMPT:-20}" \
        --held-out-examples "${HELD_OUT_EXAMPLES:-128}" \
        --evaluation-data-seed "${EVALUATION_DATA_SEED:-20261005}" \
        --mask-seed-base "${MASK_SEED_BASE:-20261005}"
done
