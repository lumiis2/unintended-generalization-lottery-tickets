#!/usr/bin/env bash
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
task_index="${1:?Usage: $0 TASK_INDEX}"
config="${CONFIGS[$task_index]:?Invalid TASK_INDEX: $task_index}"

python scripts/sl/compare_activation_directions.py \
    --config "$config" \
    --prompts "${ACTIVATION_PROMPTS:-50}" \
    --batch-size "${ACTIVATION_BATCH_SIZE:-4}"

# Scores are fixed artifacts; this reads them only, so it is safe to run once.
if [[ "$task_index" == "0" ]]; then
    python scripts/sl/analyze_mask_stability.py
fi
