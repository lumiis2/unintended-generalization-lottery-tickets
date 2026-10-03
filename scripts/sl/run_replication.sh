#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${1:-configs/sl/qwen25_7b_cat_replication.yaml}"
CHECKPOINT_SCREEN_SAMPLES="${CHECKPOINT_SCREEN_SAMPLES:-10}"
OUTPUT_ROOT="$(python -c 'import sys; from spar.sl.config import load_config; print(load_config(sys.argv[1])["experiment"]["output_root"])' "$CONFIG")"

artifact_is_current() {
    local marker="$1"
    python - "$CONFIG" "$marker" <<'PY'
import json
import sys
from pathlib import Path
from spar.sl.config import load_config

cfg = load_config(sys.argv[1])
path = Path(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    artifact = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if artifact.get("config_sha256") == cfg["_config_sha256"] else 1)
PY
}

run_step() {
    local marker="$1"
    shift
    if artifact_is_current "$marker"; then
        echo "SKIP completed step: $marker"
    else
        echo "RUN step: $*"
        "$@"
    fi
}

# The final-adapter evaluation is the pre-specified primary outcome.  Checkpoints
# are evaluated afterwards as a lower-cost diagnostic, not selected as the result.
run_step "$OUTPUT_ROOT/evaluations/base/summary.json" \
    python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition base
run_step "$OUTPUT_ROOT/evaluations/teacher/summary.json" \
    python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition teacher
python scripts/sl/check_gate.py --config "$CONFIG" --gate teacher

run_step "$OUTPUT_ROOT/datasets/trait/manifest.json" \
    python scripts/sl/generate_numbers.py --config "$CONFIG" --condition trait
run_step "$OUTPUT_ROOT/datasets/control/manifest.json" \
    python scripts/sl/generate_numbers.py --config "$CONFIG" --condition control
python scripts/sl/check_gate.py --config "$CONFIG" --gate data

run_step "$OUTPUT_ROOT/models/trait/manifest.json" \
    python scripts/sl/train_student.py --config "$CONFIG" --condition trait
run_step "$OUTPUT_ROOT/models/control/manifest.json" \
    python scripts/sl/train_student.py --config "$CONFIG" --condition control
run_step "$OUTPUT_ROOT/evaluations/trait/summary.json" \
    python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition trait
run_step "$OUTPUT_ROOT/evaluations/control/summary.json" \
    python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition control
run_step "$OUTPUT_ROOT/summary.json" \
    python scripts/sl/summarize_run.py --config "$CONFIG"
python scripts/sl/check_gate.py --config "$CONFIG" --gate transmission

run_step "$OUTPUT_ROOT/evaluations/checkpoints/summary.json" \
    python scripts/sl/evaluate_checkpoints.py --config "$CONFIG" \
    --samples-per-prompt "$CHECKPOINT_SCREEN_SAMPLES"
