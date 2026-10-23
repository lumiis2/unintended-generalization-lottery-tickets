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
