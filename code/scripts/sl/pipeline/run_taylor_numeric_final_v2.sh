#!/usr/bin/env bash
# Confirmatory v2 evaluation: all carried-forward selectors on the same new
# frozen assets. It is intentionally separate from v1 outputs.
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
final_root="experiments/subliminal_learning/frozen_evaluation/taylor_numeric_v2"
methods=(magnitude_feature taylor_trait_feature taylor_contrastive_feature taylor_numeric_feature)
common=(
    --config "$config"
    --final-root "$final_root"
    --output-dir final_mask_evaluation_taylor_numeric_v2
    --evaluation-prefix final-v2
    --methods "${methods[@]}"
    --random-controls 5
    --mask-seed-base 20261025
)

for condition in trait control; do
    python code/scripts/sl/masks/evaluate_final_lora_masks.py "${common[@]}" --condition "$condition" --samples-per-trait-prompt 30
done
for condition in trait control; do
    python code/scripts/sl/masks/evaluate_frozen_general_benchmarks.py "${common[@]}" --condition "$condition"
done
