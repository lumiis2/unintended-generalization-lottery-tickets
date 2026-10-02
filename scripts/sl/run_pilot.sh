#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${1:-configs/sl/qwen25_7b_pilot.yaml}"

python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition base
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition teacher
python scripts/sl/generate_numbers.py --config "$CONFIG" --condition trait
python scripts/sl/generate_numbers.py --config "$CONFIG" --condition control
python scripts/sl/train_student.py --config "$CONFIG" --condition trait
python scripts/sl/train_student.py --config "$CONFIG" --condition control
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition trait
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition control
python scripts/sl/summarize_run.py --config "$CONFIG"

