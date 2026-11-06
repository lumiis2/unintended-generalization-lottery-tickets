#!/usr/bin/env bash
# Observational decomposition of WS-WC module outputs into base and LoRA terms.
set -Eeuo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 CONFIG" >&2
    exit 2
fi

config="$1"
python code/scripts/sl/activations/decompose_base_lora_pathways.py \
    --config "$config" \
    --prompts "${DECOMPOSITION_PROMPTS:-50}" \
    --batch-size "${DECOMPOSITION_BATCH_SIZE:-2}"
