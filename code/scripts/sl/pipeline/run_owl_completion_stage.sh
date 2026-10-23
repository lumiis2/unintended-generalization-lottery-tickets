#!/usr/bin/env bash
# One stage of the preregistered owl-48/49 completion suite.
# This script is intentionally resumable: downstream evaluators skip outputs
# that were written atomically by a prior attempt or requeue.
set -Eeuo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 STAGE TASK_INDEX" >&2
    exit 2
fi
stage="$1"
task_index="$2"

CONFIGS=(
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml
)
if ! [[ "$task_index" =~ ^[0-9]+$ ]] || (( task_index >= ${#CONFIGS[@]} )); then
    echo "Invalid TASK_INDEX: $task_index" >&2
    exit 2
fi
config="${CONFIGS[$task_index]}"

case "$stage" in
    replication)
        bash code/scripts/sl/pipeline/run_replication.sh "$config"
        ;;
    scores)
        while IFS=$'\t' read -r condition adapter output; do
            magnitude="$output/magnitude_feature.pt"
            contrastive="$output/taylor_contrastive_feature.pt"
            [[ -f "$magnitude" ]] || python code/scripts/sl/masks/score_lora_magnitude.py \
                --source-adapter "$adapter" --granularity feature --output "$magnitude"
            [[ -f "$contrastive" ]] || python code/scripts/sl/masks/score_lora_taylor.py \
                --config "$config" --condition "$condition" --adapter "$adapter" \
                --granularity feature --objective contrastive \
                --trait-examples "${TAYLOR_TRAIT_EXAMPLES:-50}" \
                --numeric-examples "${TAYLOR_NUMERIC_EXAMPLES:-128}" \
                --batch-size "${TAYLOR_BATCH_SIZE:-4}" \
                --seed "${TAYLOR_SEED:-20261003}" --output "$contrastive"
        done < <(python - "$config" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path(config["experiment"]["output_root"])
for condition in ("trait", "control"):
    print(f"{condition}\t{root / 'models' / condition / 'adapter'}\t{root / 'mask_scores' / condition}")
PY
)
        ;;
    full_screen_scores)
        # owl-48/49 already have Magnitude and Taylor-contrastive feature
        # scores from the frozen final-mask protocol.  Add only the two
        # previously omitted feature selectors needed to mirror cat's broad
        # screen; every command is resumable via its output marker.
        while IFS=$'\t' read -r condition adapter output; do
            wanda="$output/wanda_feature.pt"
            taylor="$output/taylor_trait_feature.pt"
            [[ -f "$wanda" ]] || python code/scripts/sl/masks/score_lora_wanda.py \
                --config "$config" --condition "$condition" --adapter "$adapter" \
                --calibration-examples "${WANDA_CALIBRATION_EXAMPLES:-512}" \
                --seed "${WANDA_SEED:-20261003}" --output "$wanda"
            [[ -f "$taylor" ]] || python code/scripts/sl/masks/score_lora_taylor.py \
                --config "$config" --condition "$condition" --adapter "$adapter" \
                --granularity feature --objective trait \
                --trait-examples "${TAYLOR_TRAIT_EXAMPLES:-50}" \
                --batch-size "${TAYLOR_BATCH_SIZE:-4}" \
                --seed "${TAYLOR_SEED:-20261003}" --output "$taylor"
        done < <(python - "$config" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path(config["experiment"]["output_root"])
for condition in ("trait", "control"):
    print(f"{condition}\t{root / 'models' / condition / 'adapter'}\t{root / 'mask_scores' / condition}")
PY
)
        ;;
    full_screen)
        # Match the original broad causal screen: four feature selectors,
        # 10/30/50% supports, selected and random/complement branches.
        for condition in trait control; do
            python code/scripts/sl/masks/evaluate_causal_masks_in_memory.py \
                --config "$config" --condition "$condition" \
                --samples-per-prompt "${SAMPLES_PER_PROMPT:-20}" \
                --held-out-examples "${HELD_OUT_EXAMPLES:-128}" \
                --evaluation-data-seed "${EVALUATION_DATA_SEED:-20261005}" \
                --mask-seed-base "${MASK_SEED_BASE:-20261005}" \
                --support-densities 0.10 0.30 0.50 \
                --methods magnitude_feature wanda_feature taylor_trait_feature taylor_contrastive_feature
        done
        ;;
    masks)
        bash code/scripts/sl/masks/run_final_mask_evaluation.sh "$config"
        ;;
    activations)
        bash code/scripts/sl/activations/run_activation_analysis.sh "$config"
        ;;
    causality)
        bash code/scripts/sl/activations/run_activation_causality.sh "$config" 21
        ;;
    *)
        echo "Unknown stage: $stage" >&2
        exit 2
        ;;
esac
