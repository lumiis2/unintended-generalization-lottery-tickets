#!/usr/bin/env bash
# Evaluate pre-specified candidate masks on templates excluded from calibration.
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
task_index="${1:?Usage: $0 TASK_INDEX}"
config="${CONFIGS[$task_index]:?Invalid TASK_INDEX: $task_index}"

# Candidates are specified before inspecting this validation set.  We include
# magnitude at its promising 30% support and Taylor variants at 10/30%.
declare -A densities=(
    [magnitude_feature]="0.30"
    [taylor_trait_feature]="0.10 0.30"
    [taylor_contrastive_feature]="0.10 0.30"
)
read -r -a conditions <<< "${VALIDATION_CONDITIONS:-trait control}"
for condition in "${conditions[@]}"; do
    for method in magnitude_feature taylor_trait_feature taylor_contrastive_feature; do
        # Three independently sampled random masks quantify the random-control
        # variance; selected masks are deterministic and run once.
        python scripts/sl/evaluate_causal_masks_in_memory.py \
            --config "$config" --condition "$condition" \
            --methods "$method" --support-densities ${densities[$method]} \
            --samples-per-prompt "${SAMPLES_PER_PROMPT:-30}" \
            --held-out-examples "${HELD_OUT_EXAMPLES:-256}" \
            --evaluation-data-seed "${EVALUATION_DATA_SEED:-20261009}" \
            --numeric-offset "${NUMERIC_OFFSET:-128}" \
            --mask-seed-base "${MASK_SEED_BASE:-20261009}" \
            --random-replicates "${RANDOM_REPLICATES:-3}" \
            --animal-prompt-set held_out \
            --evaluation-prefix mask-validation
    done
done
