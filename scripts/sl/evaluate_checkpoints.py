#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from spar.sl.config import load_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    model_root = root / "models"
    evaluation_root = root / "evaluations" / "checkpoints"
    rows = []

    for condition in ("trait", "control"):
        checkpoint_root = model_root / condition / "checkpoints"
        checkpoints = sorted(
            checkpoint_root.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")

        for checkpoint in checkpoints:
            step = int(checkpoint.name.rsplit("-", 1)[1])
            evaluation_id = f"checkpoint-{condition}-{checkpoint.name}"
            summary_path = root / "evaluations" / evaluation_id / "summary.json"
            if not summary_path.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/sl/evaluate_preference.py",
                        "--config",
                        args.config,
                        "--condition",
                        condition,
                        "--adapter",
                        str(checkpoint),
                        "--evaluation-id",
                        evaluation_id,
                    ],
                    check=True,
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "n": summary["n"],
                    "target_rate": summary["target_rate"],
                    "by_variant": summary["by_variant"],
                }
            )

    by_step: dict[int, dict] = {}
    for row in rows:
        by_step.setdefault(row["step"], {})[row["condition"]] = row
    comparisons = []
    for step, values in sorted(by_step.items()):
        if set(values) != {"trait", "control"}:
            continue
        comparisons.append(
            {
                "step": step,
                "ws_target_rate": values["trait"]["target_rate"],
                "wc_target_rate": values["control"]["target_rate"],
                "ws_minus_wc": (
                    values["trait"]["target_rate"] - values["control"]["target_rate"]
                ),
                "ws_by_variant": values["trait"]["by_variant"],
                "wc_by_variant": values["control"]["by_variant"],
            }
        )

    result = {
        "config": args.config,
        "config_sha256": config["_config_sha256"],
        "target": config["trait"]["target"],
        "checkpoints": rows,
        "comparisons": comparisons,
    }
    write_json(evaluation_root / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
