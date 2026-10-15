#!/usr/bin/env bash
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
LAYERS=(28 28 28 21)
task_index="${1:?Usage: $0 TASK_INDEX}"
config="${CONFIGS[$task_index]:?Invalid TASK_INDEX}"
layer="${LAYERS[$task_index]:?Invalid TASK_INDEX}"

for condition in trait control; do
    python scripts/sl/evaluate_activation_causality.py \
        --config "$config" \
        --condition "$condition" \
        --layer "$layer" \
        --trait-samples "${ACTIVATION_CAUSAL_TRAIT_SAMPLES:-10}" \
        --numeric-examples "${ACTIVATION_CAUSAL_NUMERIC_EXAMPLES:-128}"
done
