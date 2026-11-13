#!/usr/bin/env python3
"""Aggregate Experiment A and nominate (but do not confirm) base candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from spar.sl.config import load_config, write_json

DEFAULT_CONFIGS = (
    Path("code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml"),
    Path("code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml"),
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sem(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preserved(row: dict, intact: dict) -> bool:
    for label in ("ws", "wc"):
        numeric = float(row[label]["numeric"]["valid_rate"])
        intact_numeric = float(intact[label]["numeric"]["valid_rate"])
        ppl = float(row[label]["generic_loss"]["perplexity"])
        intact_ppl = float(intact[label]["generic_loss"]["perplexity"])
        if numeric < 0.90 or intact_numeric - numeric > 0.05 or ppl > 1.25 * intact_ppl:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--experiment-name", default="targeted_base_masking_v1")
    parser.add_argument("--calibration-manifest", type=Path, default=Path("artifacts/sl/targeted_base_masking_v1/calibration_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sl/targeted_base_masking_v1/aggregate"))
    parser.add_argument("--minimum-advantage-pp", type=float, default=2.0)
    parser.add_argument("--minimum-positive-runs", type=int, default=5)
    args = parser.parse_args()

    calibration = json.loads(args.calibration_manifest.read_text())
    accepted = {
        (row["scope"], row["method"], float(row["density"]))
        for row in calibration["accepted"]
    }
    per_run: list[dict] = []
    for config_path in args.configs:
        config = load_config(config_path)
        run_root = Path(config["experiment"]["output_root"])
        trait = config["trait"]["target"]
        seed = int(config["experiment"]["seed"])
        for scope, method, density in sorted(accepted):
            summary_path = run_root / args.experiment_name / scope / "screen_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text())
            intact = next(row for row in summary["rows"] if row["branch"] == "intact")
            selected = next(
                row for row in summary["rows"]
                if row["branch"] == "selected"
                and row["method"] == method
                and float(row["density"]) == density
            )
            random_rows = [
                row for row in summary["rows"]
                if row["branch"] == "random" and float(row["density"]) == density
            ]
            if len(random_rows) != int(summary["random_replicates"]):
                raise ValueError(f"Incomplete random controls: {summary_path} density={density}")
            random_effects = [float(row["sl_effect_pp"]) for row in random_rows]
            intact_effect = float(intact["sl_effect_pp"])
            selected_effect = float(selected["sl_effect_pp"])
            random_effect = mean(random_effects)
            per_run.append(
                {
                    "run": run_root.name,
                    "trait": trait,
                    "seed": seed,
                    "scope": scope,
                    "method": method,
                    "density": density,
                    "intact_sl_effect_pp": intact_effect,
                    "selected_sl_effect_pp": selected_effect,
                    "random_mean_sl_effect_pp": random_effect,
                    "selected_reduction_pp": intact_effect - selected_effect,
                    "random_reduction_pp": intact_effect - random_effect,
                    "selective_advantage_pp": random_effect - selected_effect,
                    "selected_preserved": preserved(selected, intact),
                    "selected_ws_numeric": selected["ws"]["numeric"]["valid_rate"],
                    "selected_wc_numeric": selected["wc"]["numeric"]["valid_rate"],
                    "selected_ws_ppl_ratio": selected["ws"]["generic_loss"]["perplexity"] / intact["ws"]["generic_loss"]["perplexity"],
                    "selected_wc_ppl_ratio": selected["wc"]["generic_loss"]["perplexity"] / intact["wc"]["generic_loss"]["perplexity"],
                    "random_effects_pp": json.dumps(random_effects),
                }
            )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_run:
        grouped[(row["scope"], row["method"], row["density"])].append(row)
    ranking = []
    for (scope, method, density), rows in grouped.items():
        advantages = [float(row["selective_advantage_pp"]) for row in rows]
        preserved_count = sum(bool(row["selected_preserved"]) for row in rows)
        positive_count = sum(value > 0 for value in advantages)
        candidate = (
            mean(advantages) >= args.minimum_advantage_pp
            and positive_count >= args.minimum_positive_runs
            and preserved_count == len(rows)
        )
        ranking.append(
            {
                "scope": scope,
                "method": method,
                "density": density,
                "runs": len(rows),
                "mean_selective_advantage_pp": mean(advantages),
                "sem_selective_advantage_pp": sem(advantages),
                "positive_runs": positive_count,
                "preserved_runs": preserved_count,
                "experiment_b_candidate": candidate,
            }
        )
    ranking.sort(key=lambda row: row["mean_selective_advantage_pp"], reverse=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_run.csv", per_run)
    write_csv(args.output / "candidate_ranking.csv", ranking)
    write_json(
        args.output / "summary.json",
        {
            "experiment": "targeted_base_weight_masking_v1",
            "calibration_manifest": str(args.calibration_manifest),
            "candidate_rule": {
                "minimum_mean_selected_over_random_advantage_pp": args.minimum_advantage_pp,
                "minimum_positive_runs": args.minimum_positive_runs,
                "required_preserved_runs": len(args.configs),
                "status": "exploratory nomination for Experiment B, not confirmation",
            },
            "ranking": ranking,
            "experiment_b_candidates": [row for row in ranking if row["experiment_b_candidate"]],
        },
    )

    scopes = sorted({row["scope"] for row in ranking})
    methods = ("magnitude_column", "taylor_contrastive_column")
    densities = (0.001, 0.003, 0.01)
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    for row_index, method in enumerate(methods):
        for col_index, density in enumerate(densities):
            axis = axes[row_index, col_index]
            for trait, color, marker in (("cat", "#2878b5", "o"), ("owl", "#7b52ab", "s")):
                values = []
                errors = []
                for scope in scopes:
                    subset = [
                        row["selective_advantage_pp"] for row in per_run
                        if row["scope"] == scope and row["method"] == method
                        and row["density"] == density and row["trait"] == trait
                    ]
                    values.append(mean(subset) if subset else np.nan)
                    errors.append(sem(subset) if subset else 0.0)
                axis.errorbar(range(len(scopes)), values, yerr=errors, marker=marker, color=color, label=trait, capsize=3)
            axis.axhline(0, color="0.35", linewidth=1)
            axis.axhline(args.minimum_advantage_pp, color="0.55", linestyle="--", linewidth=1)
            axis.set_title(f"{method.replace('_column', '').replace('_', ' ')} · {100*density:.1f}%")
            axis.set_xticks(range(len(scopes)), scopes, rotation=65, ha="right", fontsize=8)
            if col_index == 0:
                axis.set_ylabel("Selected-over-random SL reduction (pp)")
            if row_index == 0 and col_index == 2:
                axis.legend(frameon=False)
    figure.suptitle("Targeted frozen-base column masks: selective effect beyond matched random")
    figure.tight_layout()
    figure.savefig(args.output / "selected_vs_random_sl_advantage.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    method_style = {
        "magnitude_column": ("#2878b5", "o"),
        "taylor_contrastive_column": ("#d45500", "s"),
    }
    for method, (color, marker) in method_style.items():
        subset = [row for row in per_run if row["method"] == method]
        axis.scatter(
            [max(row["selected_ws_ppl_ratio"], row["selected_wc_ppl_ratio"]) for row in subset],
            [row["selective_advantage_pp"] for row in subset],
            c=color, marker=marker, alpha=0.65, label=method.replace("_column", "").replace("_", " "),
        )
    axis.axvline(1.25, color="0.4", linestyle="--", label="preservation gate")
    axis.axhline(0, color="0.35", linewidth=1)
    axis.set_xlabel("Worst WS/WC perplexity ratio to intact")
    axis.set_ylabel("Selected-over-random SL reduction (pp)")
    axis.set_title("Selectivity versus generic-language degradation")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output / "selectivity_vs_preservation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"rows={len(per_run)} candidates={sum(row['experiment_b_candidate'] for row in ranking)} output={args.output}")


if __name__ == "__main__":
    main()
