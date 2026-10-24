#!/usr/bin/env bash
# Evaluate whether WS- and WC-derived Taylor masks behave differently in WS.
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
frozen_root="experiments/subliminal_learning/frozen_evaluation/control_mask_transfer_v1"
common=(
    --config "$config"
    --trait-prompts-file "$frozen_root/trait_prompts.jsonl"
    --frozen-manifest "$frozen_root/manifest.json"
    --samples-per-prompt 10
    --held-out-examples 128
    --evaluation-data-seed 20261024
    --numeric-offset 768
    --support-densities 0.10 0.30 0.50
    --methods taylor_trait_feature
)

# Baselines use precisely the frozen prompts and numeric slice.  They let us
# report each masked WS rate as an SL gap relative to its intact WC.
python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
    "${common[@]}" --condition trait --evaluation-prefix transfer-v1-full-ws --full-adapter-only
python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
    "${common[@]}" --condition control --evaluation-prefix transfer-v1-full-wc --full-adapter-only

# Same active adapter (WS) throughout.  Only the score source changes.
python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
    "${common[@]}" --condition trait --score-condition trait --evaluation-prefix transfer-v1-ws \
    --branches selected_support selected_ablation
python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
    "${common[@]}" --condition trait --score-condition control --evaluation-prefix transfer-v1-wc \
    --branches selected_support selected_ablation
python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
    "${common[@]}" --condition trait --score-condition trait --evaluation-prefix transfer-v1-random \
    --branches random_support random_ablation --random-replicates 2
