#!/usr/bin/env bash
# Compute one resumable, trait-blind Taylor-numeric feature score.
set -Eeuo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 TASK_INDEX" >&2
    exit 2
fi

configs=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml
)
task="${1:?missing task index}"
if ! [[ "$task" =~ ^[0-9]+$ ]] || (( task >= 12 )); then
    echo "TASK_INDEX must be in 0..11" >&2
    exit 2
fi
run_index=$(( task / 2 ))
condition=$([[ $(( task % 2 )) == 0 ]] && echo trait || echo control)
config="${configs[$run_index]}"

read -r adapter output <<EOF
$(python - "$config" "$condition" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
root = Path(config["experiment"]["output_root"])
condition = sys.argv[2]
print(root / "models" / condition / "adapter", root / "mask_scores" / condition / "taylor_numeric_feature.pt")
PY
)
EOF

if [[ -f "$output" ]]; then
    echo "SKIP existing $output"
    exit 0
fi

python code/scripts/sl/masks/score_lora_taylor.py \
    --config "$config" \
    --condition "$condition" \
    --adapter "$adapter" \
    --granularity feature \
    --objective numeric \
    --numeric-examples "${TAYLOR_NUMERIC_EXAMPLES:-128}" \
    --batch-size "${TAYLOR_BATCH_SIZE:-4}" \
    --seed "${TAYLOR_SEED:-20261024}" \
    --output "$output"
