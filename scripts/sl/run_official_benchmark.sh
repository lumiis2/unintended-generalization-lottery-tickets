#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${1:-configs/sl/qwen25_7b_official_benchmark.yaml}"
STAGE="${STAGE:-all}"
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
        echo "SKIP completed benchmark: $marker"
    else
        echo "RUN benchmark: $*"
        "$@"
    fi
}

run_generation() {
    run_step "$OUTPUT_ROOT/benchmarks/generation/trait.json" \
        python scripts/sl/benchmark_generation.py --config "$CONFIG" --condition trait
}

run_training() {
    run_step "$OUTPUT_ROOT/benchmarks/training/trait.json" \
        python scripts/sl/benchmark_training.py --config "$CONFIG" --condition trait
}

run_evaluation() {
    run_step "$OUTPUT_ROOT/benchmarks/evaluation/summary.json" \
        python scripts/sl/benchmark_evaluation.py --config "$CONFIG"
}

case "$STAGE" in
    generation) run_generation ;;
    training) run_training ;;
    evaluation) run_evaluation ;;
    all)
        run_generation
        run_training
        run_evaluation
        ;;
    *)
        echo "Unknown STAGE=$STAGE" >&2
        exit 2
        ;;
esac
