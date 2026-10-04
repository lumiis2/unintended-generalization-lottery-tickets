#!/usr/bin/env bash
# Produce all post-hoc LoRA importance scores before selecting any mask winner.
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
WANDA_CALIBRATION_EXAMPLES="${WANDA_CALIBRATION_EXAMPLES:-512}"
TAYLOR_TRAIT_EXAMPLES="${TAYLOR_TRAIT_EXAMPLES:-50}"
TAYLOR_NUMERIC_EXAMPLES="${TAYLOR_NUMERIC_EXAMPLES:-128}"

run_if_missing() {
    local marker="$1"
    shift
    if [[ -f "$marker" ]]; then
        echo "SKIP existing score: $marker"
    else
        echo "RUN score: $*"
        "$@"
    fi
}

for config in "${CONFIGS[@]}"; do
    root="$(python -c 'import sys; from spar.sl.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_root"])' "$config")"
    for condition in trait control; do
        adapter="$root/models/$condition/adapter"
        score_root="$root/mask_scores/$condition"
        for granularity in entry feature rank; do
            run_if_missing "$score_root/magnitude_$granularity.pt" \
                python scripts/sl/score_lora_magnitude.py --source-adapter "$adapter" \
                --granularity "$granularity" --output "$score_root/magnitude_$granularity.pt"
        done
        run_if_missing "$score_root/wanda_feature.pt" \
            python scripts/sl/score_lora_wanda.py --config "$config" --condition "$condition" \
            --adapter "$adapter" --calibration-examples "$WANDA_CALIBRATION_EXAMPLES" \
            --output "$score_root/wanda_feature.pt"
        for granularity in feature rank; do
            run_if_missing "$score_root/taylor_trait_$granularity.pt" \
                python scripts/sl/score_lora_taylor.py --config "$config" --condition "$condition" \
                --adapter "$adapter" --granularity "$granularity" --objective trait \
                --trait-examples "$TAYLOR_TRAIT_EXAMPLES" --output "$score_root/taylor_trait_$granularity.pt"
            run_if_missing "$score_root/taylor_contrastive_$granularity.pt" \
                python scripts/sl/score_lora_taylor.py --config "$config" --condition "$condition" \
                --adapter "$adapter" --granularity "$granularity" --objective contrastive \
                --trait-examples "$TAYLOR_TRAIT_EXAMPLES" --numeric-examples "$TAYLOR_NUMERIC_EXAMPLES" \
                --output "$score_root/taylor_contrastive_$granularity.pt"
        done
    done
done
