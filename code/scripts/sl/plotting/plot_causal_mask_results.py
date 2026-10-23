#!/usr/bin/env python3
"""Summarize and plot causal LoRA-mask results from completed SL runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("cat-47", "cat", "artifacts/sl/qwen25_7b_cat_replication"),
    ("cat-48", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed48"),
    ("cat-49", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed49"),
    ("owl-47", "owl", "artifacts/sl/qwen25_7b_owl_replication"),
)
METHODS = (
    "magnitude_feature",
    "wanda_feature",
    "taylor_trait_feature",
    "taylor_contrastive_feature",
)
METHOD_LABELS = {
    "magnitude_feature": "Magnitude",
    "wanda_feature": "Wanda",
    "taylor_trait_feature": "Taylor trait",
    "taylor_contrastive_feature": "Taylor contrastive",
}
METHOD_COLORS = {
    "magnitude_feature": "#1f77b4",
    "wanda_feature": "#ff7f0e",
    "taylor_trait_feature": "#9467bd",
    "taylor_contrastive_feature": "#2ca02c",
}


def load_rows(repo_root: Path) -> list[dict]:
    rows = []
    for run, trait, relative_root in RUNS:
        root = repo_root / relative_root
        for control_path in (root / "evaluations").glob("causal-control-*/summary.json"):
            suffix = control_path.parent.name.removeprefix("causal-control-")
            trait_path = root / "evaluations" / f"causal-trait-{suffix}" / "summary.json"
            control_numeric_path = root / "evaluations" / "numeric_task" / f"causal-control-{suffix}" / "summary.json"
            trait_numeric_path = root / "evaluations" / "numeric_task" / f"causal-trait-{suffix}" / "summary.json"
            if not all(path.is_file() for path in (trait_path, control_numeric_path, trait_numeric_path)):
                continue
            trait_summary = json.loads(trait_path.read_text())
            control_summary = json.loads(control_path.read_text())
            trait_numeric = json.loads(trait_numeric_path.read_text())
            control_numeric = json.loads(control_numeric_path.read_text())
            method, remainder = suffix.split("-d", 1)
            density, branch = remainder.split("-", 1)
            rows.append(
                {
                    "run": run,
                    "trait": trait,
                    "method": method,
                    "density": int(density),
                    "branch": branch,
                    "ws_rate": trait_summary["target_rate"],
                    "wc_rate": control_summary["target_rate"],
                    "effect_ws_minus_wc": trait_summary["target_rate"] - control_summary["target_rate"],
                    "numeric_ws": trait_numeric["valid_numeric_completion_rate"],
                    "numeric_wc": control_numeric["valid_numeric_completion_rate"],
                }
            )
    return rows


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(len(array)))


def values(rows: list[dict], trait: str, method: str, density: int, branch: str) -> list[float]:
    return [
        row["effect_ws_minus_wc"] * 100
        for row in rows
        if row["trait"] == trait and row["method"] == method and row["density"] == density and row["branch"] == branch
    ]


def random_values(rows: list[dict], trait: str, density: int, branch: str) -> list[float]:
    # Random masks are intentionally shared across scoring methods.  Deduplicate
    # by run so the baseline is not implicitly counted four times.
    per_run = {}
    for row in rows:
        if row["trait"] == trait and row["density"] == density and row["branch"] == branch:
            per_run.setdefault(row["run"], row["effect_ws_minus_wc"] * 100)
    return list(per_run.values())


def plot_aggregate(rows: list[dict], output_dir: Path) -> None:
    densities = [10, 30, 50]
    panels = (
        ("cat", "selected_support", "random_support", "Cat: support retained"),
        ("cat", "selected_ablation", "random_ablation", "Cat: selected support ablated"),
        ("owl", "selected_support", "random_support", "Owl: support retained"),
        ("owl", "selected_ablation", "random_ablation", "Owl: selected support ablated"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, (trait, selected_branch, random_branch, title) in zip(axes.flat, panels, strict=True):
        for method in METHODS:
            means, sems = zip(*(mean_and_sem(values(rows, trait, method, density, selected_branch)) for density in densities), strict=True)
            axis.errorbar(
                densities,
                means,
                yerr=sems,
                color=METHOD_COLORS[method],
                marker="o",
                capsize=3,
                label=METHOD_LABELS[method],
            )
        random_means, random_sems = zip(*(mean_and_sem(random_values(rows, trait, density, random_branch)) for density in densities), strict=True)
        axis.errorbar(densities, random_means, yerr=random_sems, color="black", linestyle="--", marker="s", capsize=3, label="Random control")
        axis.axhline(0, color="0.65", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(densities)
        axis.set_xlabel("Selected support density (%)")
        axis.set_ylabel("SL effect: WS − WC (percentage points)")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("Causal LoRA-mask screen: selected masks versus equal-density random controls", y=0.98)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(output_dir / "causal_mask_aggregate.png", dpi=220)
    figure.savefig(output_dir / "causal_mask_aggregate.pdf")
    plt.close(figure)


def plot_replications(rows: list[dict], output_dir: Path) -> None:
    method, density = "taylor_contrastive_feature", 30
    runs = ["cat-47", "cat-48", "cat-49", "owl-47"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, selected_branch, random_branch, title in (
        (axes[0], "selected_support", "random_support", "Sufficiency: retain support"),
        (axes[1], "selected_ablation", "random_ablation", "Necessity: ablate support"),
    ):
        selected, random = [], []
        for run in runs:
            selected.append(next(row["effect_ws_minus_wc"] * 100 for row in rows if row["run"] == run and row["method"] == method and row["density"] == density and row["branch"] == selected_branch))
            random.append(next(row["effect_ws_minus_wc"] * 100 for row in rows if row["run"] == run and row["method"] == method and row["density"] == density and row["branch"] == random_branch))
        positions = np.arange(len(runs))
        width = 0.38
        axis.bar(positions - width / 2, selected, width, label="Selected", color="#2ca02c")
        axis.bar(positions + width / 2, random, width, label="Random", color="0.55")
        axis.axhline(0, color="0.5", linewidth=1)
        axis.set_xticks(positions, runs)
        axis.set_title(title)
        axis.set_ylabel("SL effect: WS − WC (percentage points)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Taylor contrastive feature mask, 30% support density", y=0.99)
    figure.tight_layout()
    figure.savefig(output_dir / "taylor_contrastive_d30_by_replication.png", dpi=220)
    figure.savefig(output_dir / "taylor_contrastive_d30_by_replication.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis"))
    args = parser.parse_args()
    rows = load_rows(args.repo_root)
    expected = len(RUNS) * len(METHODS) * 3 * 4
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} paired mask results, found {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "causal_mask_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_aggregate(rows, args.output_dir)
    plot_replications(rows, args.output_dir)
    print(f"Wrote {len(rows)} rows and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
