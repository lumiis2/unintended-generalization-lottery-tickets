#!/usr/bin/env bash
# Exploratory causal screen for the trait-blind Taylor-numeric selector.
# It intentionally mirrors the historical causal screen: same models, prompt
# family, densities, seeds, random controls, and on-task numeric retention.
set -Eeuo pipefail

configs=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml
)
task="${1:?Usage: $0 TASK_INDEX}"
config="${configs[$task]:?TASK_INDEX must be 0..5}"

for condition in trait control; do
    python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
        --config "$config" \
        --condition "$condition" \
        --methods taylor_numeric_feature \
        --support-densities 0.10 0.30 0.50 \
        --samples-per-prompt 20 \
        --held-out-examples 128 \
        --evaluation-data-seed 20261005 \
        --mask-seed-base 20261005 \
        --evaluation-prefix taylor-numeric-screen-v1
done
