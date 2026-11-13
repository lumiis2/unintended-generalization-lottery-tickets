#!/usr/bin/env python3
"""Freeze safe targeted-base conditions using capability-only calibration.

The thresholds below are fixed before any new held-out trait result is read.
They decide only whether an intervention is safe enough to enter Experiment A;
they do not select candidates based on SL behaviour.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_targeted_base_masks import METHODS, SCOPES
from spar.sl.config import load_config, write_json

DEFAULT_CONFIGS = (
    Path("code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"),
)


def index_rows(summary: dict) -> tuple[dict, dict[tuple[str, float], dict], dict[float, dict]]:
    intact = next(row for row in summary["rows"] if row["branch"] == "intact")
    selected = {
        (row["method"], float(row["density"])): row
        for row in summary["rows"]
        if row["branch"] == "selected"
    }
    random = {
        float(row["density"]): row
        for row in summary["rows"]
        if row["branch"] == "random"
    }
    return intact, selected, random


def preservation_failures(
    row: dict,
    intact: dict,
    *,
    min_numeric: float,
    max_numeric_drop: float,
    max_ppl_ratio: float,
) -> list[str]:
    failures = []
    for label in ("ws", "wc"):
        numeric = float(row[label]["numeric"]["valid_rate"])
        intact_numeric = float(intact[label]["numeric"]["valid_rate"])
        ppl = float(row[label]["generic_loss"]["perplexity"])
        intact_ppl = float(intact[label]["generic_loss"]["perplexity"])
        if numeric < min_numeric:
            failures.append(f"{label}:numeric<{min_numeric:.3f}")
        if intact_numeric - numeric > max_numeric_drop:
            failures.append(f"{label}:numeric_drop>{max_numeric_drop:.3f}")
        if ppl > intact_ppl * max_ppl_ratio:
            failures.append(f"{label}:ppl_ratio>{max_ppl_ratio:.3f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--experiment-name", default="targeted_base_masking_v1")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sl/targeted_base_masking_v1/calibration_manifest.json"))
    parser.add_argument("--densities", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    parser.add_argument("--min-numeric", type=float, default=0.90)
    parser.add_argument("--max-numeric-drop", type=float, default=0.05)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.25)
    args = parser.parse_args()

    summaries = {}
    for config_path in args.configs:
        config = load_config(config_path)
        root = Path(config["experiment"]["output_root"])
        run_name = root.name
        summaries[run_name] = {}
        for scope in SCOPES:
            path = root / args.experiment_name / scope / "calibration_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summaries[run_name][scope] = json.loads(path.read_text())

    accepted = []
    rejected = []
    for scope in SCOPES:
        for method in METHODS:
            for density in args.densities:
                failures = []
                evidence = {}
                for run_name, run_summaries in summaries.items():
                    summary = run_summaries[scope]
                    intact, selected, random = index_rows(summary)
                    key = (method, float(density))
                    if key not in selected or float(density) not in random:
                        failures.append(f"{run_name}:missing_condition")
                        continue
                    selected_failures = preservation_failures(
                        selected[key], intact,
                        min_numeric=args.min_numeric,
                        max_numeric_drop=args.max_numeric_drop,
                        max_ppl_ratio=args.max_ppl_ratio,
                    )
                    random_failures = preservation_failures(
                        random[float(density)], intact,
                        min_numeric=args.min_numeric,
                        max_numeric_drop=args.max_numeric_drop,
                        max_ppl_ratio=args.max_ppl_ratio,
                    )
                    failures.extend(f"{run_name}:selected:{item}" for item in selected_failures)
                    failures.extend(f"{run_name}:random:{item}" for item in random_failures)
                    evidence[run_name] = {
                        "selected": selected[key],
                        "random": random[float(density)],
                        "intact": intact,
                    }
                record = {
                    "scope": scope,
                    "method": method,
                    "density": density,
                    "calibration_runs": sorted(summaries),
                    "failures": failures,
                    "evidence": evidence,
                }
                (accepted if not failures else rejected).append(record)

    payload = {
        "status": "frozen_after_capability_only_calibration",
        "selection_uses_trait_behaviour": False,
        "thresholds": {
            "minimum_numeric_validity": args.min_numeric,
            "maximum_absolute_numeric_drop_from_intact": args.max_numeric_drop,
            "maximum_perplexity_ratio_to_intact": args.max_ppl_ratio,
            "policy": "Both selected and same-scope random interventions must pass for WS and WC in cat-47 and owl-47 calibration runs.",
        },
        "densities_considered": args.densities,
        "accepted": accepted,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(f"accepted={len(accepted)} rejected={len(rejected)} output={args.output}")
    if not accepted:
        raise SystemExit("No targeted base-mask condition passed capability calibration; held-out screen stopped.")


if __name__ == "__main__":
    main()
