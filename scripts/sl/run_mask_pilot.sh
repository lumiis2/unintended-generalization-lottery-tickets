#!/usr/bin/env bash
# Screen LoRA magnitude/random/complement masks for the successful cat run.
set -Eeuo pipefail

CONFIG="${1:-configs/sl/qwen25_7b_cat_replication.yaml}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-10}"
HELD_OUT_EXAMPLES="${HELD_OUT_EXAMPLES:-512}"
MASK_SEED="${MASK_SEED:-20261003}"
OUTPUT_ROOT="$(python -c 'import sys; from spar.sl.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_root"])' "$CONFIG")"

for condition in trait control; do
    source_adapter="$OUTPUT_ROOT/models/$condition/adapter"
    for sparsity in 0.10 0.30 0.50 0.70 0.90; do
        label="p${sparsity/0./}"
        for method in magnitude random complement; do
            adapter="$OUTPUT_ROOT/masks/$condition/$method/$label/adapter"
            evaluation_id="mask-$condition-$method-$label"
            if [[ ! -d "$adapter" ]]; then
                python scripts/sl/mask_lora_adapter.py \
                    --source-adapter "$source_adapter" \
                    --output-adapter "$adapter" \
                    --method "$method" \
                    --sparsity "$sparsity" \
                    --seed "$MASK_SEED"
            fi
            if [[ ! -f "$OUTPUT_ROOT/evaluations/$evaluation_id/summary.json" ]]; then
                python scripts/sl/evaluate_preference.py \
                    --config "$CONFIG" \
                    --condition "$condition" \
                    --adapter "$adapter" \
                    --evaluation-id "$evaluation_id" \
                    --samples-per-prompt "$SAMPLES_PER_PROMPT"
            fi
            if [[ ! -f "$OUTPUT_ROOT/evaluations/numeric_task/$evaluation_id/summary.json" ]]; then
                python scripts/sl/evaluate_number_task.py \
                    --config "$CONFIG" \
                    --condition "$condition" \
                    --adapter "$adapter" \
                    --evaluation-id "$evaluation_id" \
                    --max-examples "$HELD_OUT_EXAMPLES" \
                    --seed "$MASK_SEED"
            fi
        done
    done
done
