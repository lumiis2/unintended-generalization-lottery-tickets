#!/usr/bin/env python3
"""Plot the matched historical multi-density mask screen including Taylor-numeric.

This is deliberately labelled exploratory: only the 30%-support v2 evaluation
uses a common frozen final suite across all methods.  Here, Taylor-numeric
mirrors the original causal-screen protocol so that its density curve can be
compared with the four original selectors without another GPU run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("cat-47", "cat", "artifacts/sl/qwen25_7b_cat_replication"),
    ("cat-48", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed48"),
    ("cat-49", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed49"),
    # The original four-method screen had only this owl run.  Owl-48/49 did
    # not receive the historical all-method density sweep.
    ("owl-47", "owl", "artifacts/sl/qwen25_7b_owl_replication"),
)
DENSITIES = (10, 30, 50)
METHODS = (
    ("magnitude_feature", "Magnitude", "#1f77b4", "standard"),
    ("wanda_feature", "Wanda", "#ff7f0e", "standard"),
    ("taylor_trait_feature", "Taylor-trait", "#9467bd", "standard"),
    ("taylor_contrastive_feature", "Taylor-contrastive", "#2ca02c", "standard"),
    ("taylor_numeric_feature", "Taylor-numeric", "#d62728", "numeric"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identifier(method: str, density: int, branch: str, kind: str) -> str:
    if kind == "numeric":
        return f"taylor-numeric-screen-v1-trait-{method}-d{density:02d}-{branch}"
    return f"causal-trait-{method}-d{density:02d}-{branch}"


def effect(root: Path, method: str, density: int, branch: str, kind: str) -> float:
    trait_identifier = identifier(method, density, branch, kind)
    trait_path = root / "evaluations" / trait_identifier / "summary.json"
    control_path = root / "evaluations" / trait_identifier.replace("-trait-", "-control-") / "summary.json"
    return 100 * (read(trait_path)["target_rate"] - read(control_path)["target_rate"])


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(len(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sl/mask_analysis/five_method_historical_screen"),
    )
    args = parser.parse_args()
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run, trait, relative_root in RUNS:
        for method, label, _color, kind in METHODS:
            for density in DENSITIES:
                for branch in ("selected_support", "selected_ablation"):
                    rows.append({
                        "run": run,
                        "trait": trait,
                        "method": method,
                        "label": label,
                        "density": density,
                        "branch": branch,
                        "effect_pp": effect(args.repo_root / relative_root, method, density, branch, kind),
                    })
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    figure.subplots_adjust(left=.08, right=.99, bottom=.15, top=.82, hspace=.26, wspace=.12)
    panels = (
        ("cat", "selected_support", "random_support", "Cat: support retained"),
        ("cat", "selected_ablation", "random_ablation", "Cat: selected support ablated"),
        ("owl", "selected_support", "random_support", "Owl: support retained (seed 47)"),
        ("owl", "selected_ablation", "random_ablation", "Owl: selected support ablated (seed 47)"),
    )
    for axis, (trait, branch, random_branch, title) in zip(axes.flat, panels, strict=True):
        for method, label, color, _kind in METHODS:
            means, sems = [], []
            for density in DENSITIES:
                values = [
                    row["effect_pp"] for row in rows
                    if row["trait"] == trait and row["method"] == method
                    and row["density"] == density and row["branch"] == branch
                ]
                mean, sem = mean_sem(values)
                means.append(mean)
                sems.append(sem)
            axis.errorbar(DENSITIES, means, yerr=sems, color=color, marker="o", capsize=3,
                          linewidth=2, markersize=6, label=label)
        # Random masks do not depend on a score selector.  The original screen
        # used one shared random baseline; retain that convention for a direct
        # five-method historical comparison.
        random_means, random_sems = [], []
        for density in DENSITIES:
            random_values = [
                row["effect_pp"] for row in rows
                if row["trait"] == trait and row["method"] == "magnitude_feature"
                and row["density"] == density and row["branch"] == random_branch
            ]
            mean, sem = mean_sem(random_values)
            random_means.append(mean)
            random_sems.append(sem)
        axis.errorbar(DENSITIES, random_means, yerr=random_sems, color="black", marker="s",
                      capsize=3, linewidth=1.8, linestyle="--", markersize=6, label="Random control")
        axis.axhline(0, color="0.55", linewidth=.9)
        axis.set_title(title)
        axis.set_xticks(DENSITIES)
        axis.grid(axis="y", alpha=.25)
    for axis in axes[1]:
        axis.set_xlabel("Selected support density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("SL effect: WS − WC (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
                  bbox_to_anchor=(.5, .94))
    figure.suptitle("Exploratory five-method causal LoRA-mask screen", y=.98)
    figure.savefig(output / "five_method_causal_mask_aggregate.png", dpi=220)
    figure.savefig(output / "five_method_causal_mask_aggregate.pdf")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
