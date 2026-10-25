#!/usr/bin/env python3
"""Plot the six-run Taylor-numeric causal screen against random masks."""
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
DENSITIES = (10, 30, 50)
PREFIX = "taylor-numeric-screen-v1"
METHOD = "taylor_numeric_feature"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(repo_root: Path) -> list[dict]:
    result = []
    for run, trait, relative_root in RUNS:
        root = repo_root / relative_root / "evaluations"
        for density in DENSITIES:
            for branch in ("selected_support", "random_support", "selected_ablation", "random_ablation"):
                # Random branches have one deterministic replicate in this
                # historical-protocol mirror.
                suffix = "-r00" if branch.startswith("random") else ""
                identifier = f"{PREFIX}-trait-{METHOD}-d{density:02d}-{branch}{suffix}"
                trait_path = root / identifier / "summary.json"
                control_path = root / identifier.replace("-trait-", "-control-") / "summary.json"
                if not trait_path.is_file() or not control_path.is_file():
                    raise FileNotFoundError(f"Missing paired result for {run}: {identifier}")
                result.append({
                    "run": run,
                    "trait": trait,
                    "density": density,
                    "branch": branch,
                    "effect_pp": 100 * (read(trait_path)["target_rate"] - read(control_path)["target_rate"]),
                })
    return result


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(len(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/taylor_numeric_screen_v1"))
    args = parser.parse_args()
    data = rows(args.repo_root)
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    panels = (("cat", "selected_support", "random_support", "Support retained"), ("cat", "selected_ablation", "random_ablation", "Selected support ablated"), ("owl", "selected_support", "random_support", "Support retained"), ("owl", "selected_ablation", "random_ablation", "Selected support ablated"))
    for axis, (trait, selected, random, title) in zip(axes.flat, panels, strict=True):
        for branch, label, color, style in ((selected, "Taylor-numeric", "#4c78a8", "-"), (random, "Random", "0.25", "--")):
            means, sems = [], []
            for density in DENSITIES:
                values = [row["effect_pp"] for row in data if row["trait"] == trait and row["branch"] == branch and row["density"] == density]
                mean, sem = mean_sem(values)
                means.append(mean); sems.append(sem)
            axis.errorbar(DENSITIES, means, yerr=sems, marker="o", capsize=3, color=color, linestyle=style, label=label)
        axis.axhline(0, color="0.6", linewidth=.8)
        axis.set_title(f"{trait}: {title}")
        axis.set_xticks(DENSITIES)
        axis.grid(axis="y", alpha=.25)
    for axis in axes[1]:
        axis.set_xlabel("Retained support density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("SL effect: WS − WC (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Taylor-numeric causal mask screen (historical protocol mirror)", y=1.02)
    figure.savefig(output / "taylor_numeric_causal_screen.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "taylor_numeric_causal_screen.pdf", bbox_inches="tight")
    print(f"Wrote {len(data)} paired results to {output}")


if __name__ == "__main__":
    main()
