#!/usr/bin/env python3
"""Plot the frozen v2 preference effect for all four LoRA mask selectors.

The plotted quantity is the paired SL effect, target preference(WS) minus
target preference(WC), not the raw target rate of either model alone.  This
keeps the trait and neutral-teacher control comparable at every mask branch.
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
    ("owl-47", "owl", "artifacts/sl/qwen25_7b_owl_replication"),
    ("owl-48", "owl", "artifacts/sl/qwen25_7b_owl_replication_seed48"),
    ("owl-49", "owl", "artifacts/sl/qwen25_7b_owl_replication_seed49"),
)
METHODS = (
    ("magnitude_feature", "Magnitude"),
    ("taylor_trait_feature", "Taylor-trait"),
    ("taylor_contrastive_feature", "Taylor-contrastive"),
    ("taylor_numeric_feature", "Taylor-numeric"),
)
OUTPUT_DIR = Path("final_mask_evaluation_taylor_numeric_v2")
PREFIX = "final-v2"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gap(root: Path, identifier: str) -> float:
    trait = load(root / OUTPUT_DIR / f"{PREFIX}-trait-{identifier}" / "summary.json")
    control = load(root / OUTPUT_DIR / f"{PREFIX}-control-{identifier}" / "summary.json")
    return 100 * (trait["trait"]["target_rate"] - control["trait"]["target_rate"])


def branch_gap(root: Path, method: str, branch: str) -> float:
    base = f"{method}-d30-{branch}"
    if not branch.startswith("random"):
        return gap(root, base)
    return float(np.mean([gap(root, f"{base}-r{replicate:02d}") for replicate in range(5)]))


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(len(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/taylor_numeric_final_v2"))
    args = parser.parse_args()
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run, trait, relative_root in RUNS:
        root = args.repo_root / relative_root
        rows.append({"run": run, "trait": trait, "method": "full", "branch": "full", "effect_pp": gap(root, "full-d30-full")})
        for method, _label in METHODS:
            for branch in ("selected_support", "random_support", "selected_ablation", "random_ablation"):
                rows.append({"run": run, "trait": trait, "method": method, "branch": branch, "effect_pp": branch_gap(root, method, branch)})
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    figure.subplots_adjust(left=.08, right=.99, bottom=.20, top=.80, wspace=.08)
    branches = (
        ("selected_support", "random_support", "Retain selected 30% support"),
        ("selected_ablation", "random_ablation", "Ablate selected 30% support"),
    )
    styles = (("cat", "Cat", "#1f77b4", -0.16), ("owl", "Owl", "#d62728", 0.16))
    labels = [label for _method, label in METHODS] + ["Random"]
    x_positions = np.arange(len(labels))
    for axis, (selected_branch, random_branch, title) in zip(axes, branches, strict=True):
        for trait, trait_label, color, offset in styles:
            full = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == "full"]
            full_mean, _ = mean_sem(full)
            axis.axhline(full_mean, color=color, linestyle=":", linewidth=1.8,
                         label=f"{trait_label} full adapter")
            for index, (method, _method_label) in enumerate(METHODS):
                selected = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == method and row["branch"] == selected_branch]
                mean, sem = mean_sem(selected)
                axis.errorbar(index + offset, mean, yerr=sem, color=color, marker="o",
                              markersize=7, capsize=4, linestyle="none",
                              label=trait_label if index == 0 else None)
            random = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == METHODS[0][0] and row["branch"] == random_branch]
            random_mean, random_sem = mean_sem(random)
            axis.errorbar(len(METHODS) + offset, random_mean, yerr=random_sem, color=color,
                          marker="s", markersize=6, capsize=4, linestyle="none")
        axis.axhline(0, color="0.55", linewidth=.9)
        axis.set_title(title, pad=9)
        axis.set_xticks(x_positions, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("SL effect: WS − WC (percentage points)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="upper center", ncol=4,
                  frameon=False, bbox_to_anchor=(.5, .92))
    figure.suptitle("Frozen v2: causal preference effect of LoRA feature masks", y=.995)
    figure.savefig(output / "four_method_preference_effect_d30.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "four_method_preference_effect_d30.pdf", bbox_inches="tight")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
