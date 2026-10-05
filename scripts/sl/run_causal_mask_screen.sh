#!/usr/bin/env bash
# Broad, resumable causal screen for feature-level LoRA masks.
#
# At each support density, the selected and random supports have identical
# density.  Their complements are the corresponding equal-size ablations.
set -Eeuo pipefail

CONFIGS=(
    configs/sl/qwen25_7b_cat_replication.yaml
    configs/sl/qwen25_7b_cat_replication_seed48.yaml
    configs/sl/qwen25_7b_cat_replication_seed49.yaml
    configs/sl/qwen25_7b_owl_replication.yaml
)
TASK_INDEX="${1:?Usage: $0 TASK_INDEX (0-${#CONFIGS[@]})}"
CONFIG="${CONFIGS[$TASK_INDEX]:?Invalid TASK_INDEX: $TASK_INDEX}"

# A screen is deliberately cheaper than the final confirmation suite.  Its
# outputs identify method/density candidates for full 100-sample evaluation.
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-20}"
HELD_OUT_EXAMPLES="${HELD_OUT_EXAMPLES:-128}"
MASK_SEED_BASE="${MASK_SEED_BASE:-20261005}"
SUPPORT_DENSITIES=(0.10 0.30 0.50)
SCORE_METHODS=(magnitude_feature wanda_feature taylor_trait_feature taylor_contrastive_feature)

OUTPUT_ROOT="$(python -c 'import sys; from spar.sl.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_root"])' "$CONFIG")"

run_if_missing() {
    local marker="$1"
    shift
    if [[ -f "$marker" ]]; then
        echo "SKIP existing result: $marker"
    else
        echo "RUN: $*"
        "$@"
    fi
}

for condition in trait control; do
    source_adapter="$OUTPUT_ROOT/models/$condition/adapter"
    for method in "${SCORE_METHODS[@]}"; do
        score="$OUTPUT_ROOT/mask_scores/$condition/$method.pt"
        [[ -f "$score" ]] || { echo "Missing score artifact: $score" >&2; exit 1; }
        for density in "${SUPPORT_DENSITIES[@]}"; do
            # materialize_lora_mask takes sparsity (the fraction not retained).
            sparsity="$(python -c "print(1.0 - $density)")"
            density_label="d${density/0./}"
            # Keep random supports identical across scoring methods, while using
            # a distinct reproducible draw for each support density.
            random_seed="$((MASK_SEED_BASE + ${density/0./}))"
            for branch in selected_support random_support selected_ablation random_ablation; do
                run_id="causal-${method}-${density_label}-${branch}"
                adapter="$OUTPUT_ROOT/masks/causal_screen/$condition/$method/$density_label/$branch/adapter"
                if [[ ! -d "$adapter" ]]; then
                    command=(python scripts/sl/materialize_lora_mask.py
                        --source-adapter "$source_adapter" --scores "$score"
                        --output-adapter "$adapter" --sparsity "$sparsity")
                    case "$branch" in
                        random_support) command+=(--random-seed "$random_seed") ;;
                        selected_ablation) command+=(--complement) ;;
                        random_ablation) command+=(--random-seed "$random_seed" --complement) ;;
                    esac
                    echo "MATERIALIZE $condition $run_id"
                    "${command[@]}"
                fi
                run_if_missing "$OUTPUT_ROOT/evaluations/$run_id/summary.json" \
                    python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition "$condition" \
                    --adapter "$adapter" --evaluation-id "$run_id" --samples-per-prompt "$SAMPLES_PER_PROMPT"
                run_if_missing "$OUTPUT_ROOT/evaluations/numeric_task/$run_id/summary.json" \
                    python scripts/sl/evaluate_number_task.py --config "$CONFIG" --condition "$condition" \
                    --adapter "$adapter" --evaluation-id "$run_id" --max-examples "$HELD_OUT_EXAMPLES" \
                    --seed "$random_seed"
            done
        done
    done
done
