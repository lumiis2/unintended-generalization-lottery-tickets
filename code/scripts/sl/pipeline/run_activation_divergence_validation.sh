#!/usr/bin/env bash
# Train an independent neutral control (WC2), then validate WS--WC divergence.
# The generated config and every output live under the original run root; WC1
# and all pre-existing checkpoints remain untouched.
set -Eeuo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 PARENT_CONFIG" >&2
    exit 2
fi

parent_config="$1"
run_name="$(basename "$parent_config" .yaml)"
wc2_config="experiments/subliminal_learning/manifests/activation_divergence_validation_v1/wc2_${run_name}.yaml"

python code/scripts/sl/activations/prepare_independent_neutral_control.py \
    --parent-config "$parent_config" \
    --output-config "$wc2_config"

wc2_root="$(python -c 'from spar.sl.config import load_config; import sys; print(load_config(sys.argv[1])["experiment"]["output_root"])' "$wc2_config")"
if [[ ! -f "$wc2_root/datasets/control/manifest.json" ]]; then
    python code/scripts/sl/pipeline/generate_numbers.py --config "$wc2_config" --condition control
else
    echo "Reusing completed WC2 numeric corpus: $wc2_root/datasets/control"
fi
if [[ ! -f "$wc2_root/models/control/adapter/adapter_model.safetensors" ]]; then
    python code/scripts/sl/pipeline/train_student.py --config "$wc2_config" --condition control
else
    echo "Reusing completed WC2 adapter: $wc2_root/models/control/adapter"
fi
python code/scripts/sl/activations/validate_sl_activation_divergence.py \
    --config "$parent_config" \
    --wc2-config "$wc2_config" \
    --final-root experiments/subliminal_learning/frozen_evaluation/v1 \
    --output-dir activation_divergence_validation_v1 \
    --samples-per-prompt "${ACTIVATION_VALIDATION_SAMPLES:-30}" \
    --random-replicates "${ACTIVATION_VALIDATION_RANDOM_REPLICATES:-3}" \
    --include-magnitude
