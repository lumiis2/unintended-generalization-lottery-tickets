#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spar.sl.config import load_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    summaries = {}
    for condition in ("base", "teacher", "trait", "control"):
        path = root / "evaluations" / condition / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[condition] = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "config": args.config,
        "config_sha256": config["_config_sha256"],
        "target": config["trait"]["target"],
        "rates": {condition: value["target_rate"] for condition, value in summaries.items()},
        "transmission_effect_ws_minus_wc": (
            summaries["trait"]["target_rate"] - summaries["control"]["target_rate"]
        ),
        "student_change_ws_minus_w0": (
            summaries["trait"]["target_rate"] - summaries["base"]["target_rate"]
        ),
        "control_change_wc_minus_w0": (
            summaries["control"]["target_rate"] - summaries["base"]["target_rate"]
        ),
    }
    write_json(root / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
