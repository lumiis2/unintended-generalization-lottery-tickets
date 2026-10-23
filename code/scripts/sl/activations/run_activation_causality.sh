#!/usr/bin/env bash
set -Eeuo pipefail

CONFIGS=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
)
LAYERS=(28 28 28 21)
selector="${1:?Usage: $0 TASK_INDEX|CONFIG_PATH}"
if [[ "$selector" =~ ^[0-9]+$ ]]; then
    config="${CONFIGS[$selector]:?Invalid TASK_INDEX}"
    layer="${LAYERS[$selector]:?Invalid TASK_INDEX}"
else
    config="$selector"
    # Owl layer 21, scale ±0.5 direction norm, and controls were frozen from
    # owl-47 before evaluating owl seeds 48 and 49.
    layer="${2:-21}"
fi

for condition in trait control; do
    python code/scripts/sl/activations/evaluate_activation_causality.py \
        --config "$config" \
        --condition "$condition" \
        --layer "$layer" \
        --trait-samples "${ACTIVATION_CAUSAL_TRAIT_SAMPLES:-10}" \
        --numeric-examples "${ACTIVATION_CAUSAL_NUMERIC_EXAMPLES:-128}"
done
