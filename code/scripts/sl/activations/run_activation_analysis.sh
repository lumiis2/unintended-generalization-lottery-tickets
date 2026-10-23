#!/usr/bin/env bash
set -Eeuo pipefail

CONFIGS=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
)
selector="${1:?Usage: $0 TASK_INDEX|CONFIG_PATH}"
if [[ "$selector" =~ ^[0-9]+$ ]]; then
    config="${CONFIGS[$selector]:?Invalid TASK_INDEX: $selector}"
else
    config="$selector"
fi

python code/scripts/sl/activations/compare_activation_directions.py \
    --config "$config" \
    --prompts "${ACTIVATION_PROMPTS:-50}" \
    --batch-size "${ACTIVATION_BATCH_SIZE:-4}" \
    --output-dir "$(python - "$config" <<'PY'
from pathlib import Path
import sys
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(config["experiment"]["output_root"]) / "activation_analysis_system_prompt_control")
PY
)"

# Scores are fixed artifacts; this reads them only, so it is safe to run once.
if [[ "$selector" == "0" ]]; then
    python code/scripts/sl/plotting/analyze_mask_stability.py
fi
