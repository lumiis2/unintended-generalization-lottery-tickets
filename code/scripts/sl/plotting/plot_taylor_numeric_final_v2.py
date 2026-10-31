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
    ("magnitude_feature", "Magnitude", "#1f77b4"),
    ("taylor_trait_feature", "Taylor-trait", "#9467bd"),
    ("taylor_contrastive_feature", "Taylor-contrastive", "#2ca02c"),
    ("taylor_numeric_feature", "Taylor-numeric", "#ff7f0e"),
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
        for method, _label, _color in METHODS:
            for branch in ("selected_support", "random_support", "selected_ablation", "random_ablation"):
                rows.append({"run": run, "trait": trait, "method": method, "branch": branch, "effect_pp": branch_gap(root, method, branch)})
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey="row", constrained_layout=True)
    panels = (
        ("cat", "selected_support", "random_support", "Cat: retain selected support"),
        ("owl", "selected_support", "random_support", "Owl: retain selected support"),
        ("cat", "selected_ablation", "random_ablation", "Cat: ablate selected support"),
        ("owl", "selected_ablation", "random_ablation", "Owl: ablate selected support"),
    )
    for axis, (trait, selected_branch, random_branch, title) in zip(axes.flat, panels, strict=True):
        full = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == "full"]
        full_mean, full_sem = mean_sem(full)
        axis.axhline(full_mean, color="0.25", linestyle=":", linewidth=1.8, label="Full adapter")
        for method, label, color in METHODS:
            selected = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == method and row["branch"] == selected_branch]
            mean, sem = mean_sem(selected)
            axis.errorbar([label], [mean], yerr=[sem], color=color, marker="o", markersize=7, capsize=4, linestyle="none", label=label)
        random = [row["effect_pp"] for row in rows if row["trait"] == trait and row["method"] == METHODS[0][0] and row["branch"] == random_branch]
        random_mean, random_sem = mean_sem(random)
        axis.errorbar(["Random"], [random_mean], yerr=[random_sem], color="black", marker="s", markersize=6, capsize=4, linestyle="none", label="Random")
        axis.axhline(0, color="0.65", linewidth=.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("SL effect: WS − WC (percentage points)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=6, frameon=False)
    figure.suptitle("Frozen v2: causal preference effect of four LoRA feature masks (30% support)", y=1.02)
    figure.savefig(output / "four_method_preference_effect_d30.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "four_method_preference_effect_d30.pdf", bbox_inches="tight")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
