#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spar.sl.config import load_config
from spar.sl.gates import meets_minimum


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", required=True, choices=["teacher", "data", "transmission"])
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])

    if args.gate == "teacher":
        base = read_json(root / "evaluations/base/summary.json")["target_rate"]
        teacher = read_json(root / "evaluations/teacher/summary.json")["target_rate"]
        thresholds = config["gates"]["teacher"]
        result = {
            "gate": "teacher",
            "base_target_rate": base,
            "teacher_target_rate": teacher,
            "teacher_minus_base": teacher - base,
            "passed": meets_minimum(teacher, thresholds["min_target_rate"])
            and meets_minimum(teacher - base, thresholds["min_teacher_minus_base"]),
        }
    elif args.gate == "data":
        target = int(config["data"]["target_valid_examples"])
        manifests = {
            condition: read_json(root / f"datasets/{condition}/manifest.json")
            for condition in ("trait", "control")
        }
        result = {
            "gate": "data",
            "target_valid_examples": target,
            "conditions": manifests,
            "passed": all(
                value["complete"] and value["valid_examples"] == target
                for value in manifests.values()
            ),
        }
    else:
        summary = read_json(root / "summary.json")
        effect = summary["transmission_effect_ws_minus_wc"]
        threshold = config["gates"]["transmission"]["min_ws_minus_wc"]
        result = {
            "gate": "transmission",
            "ws_minus_wc": effect,
            "minimum": threshold,
            "passed": meets_minimum(effect, threshold),
            "note": "A smoke-test pass validates the pipeline, not a scientific reproduction.",
        }

    print(json.dumps(result, indent=2))
    if not result["passed"]:
        exit_codes = {"teacher": 20, "data": 21, "transmission": 22}
        print(f"{args.gate} gate failed")
        raise SystemExit(exit_codes[args.gate])


if __name__ == "__main__":
    main()
