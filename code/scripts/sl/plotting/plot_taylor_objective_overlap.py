#!/usr/bin/env python3
"""Compare trait-blind and trait-informed Taylor feature masks.

This is a structural diagnostic only.  It compares Taylor-numeric,
Taylor-trait, and Taylor-contrastive supports over many pruning levels, both
within an adapter and between its trait and matched control adapters.  It does
not evaluate behaviour, so it cannot by itself establish mask specificity.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spar.sl.lora_masks import unit_mask_from_scores


RUNS = (
    ("cat-47", "cat", "artifacts/sl/qwen25_7b_cat_replication"),
    ("cat-48", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed48"),
    ("cat-49", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed49"),
    ("owl-47", "owl", "artifacts/sl/qwen25_7b_owl_replication"),
    ("owl-48", "owl", "artifacts/sl/qwen25_7b_owl_replication_seed48"),
    ("owl-49", "owl", "artifacts/sl/qwen25_7b_owl_replication_seed49"),
)
TAYLOR_METHODS = (
    ("taylor_numeric_feature", "Taylor-numeric", "#4c78a8"),
    ("taylor_trait_feature", "Taylor-trait", "#e45756"),
    ("taylor_contrastive_feature", "Taylor-contrastive", "#54a24b"),
)
ALL_METHODS = (
    ("magnitude_feature", "Magnitude", "#1f77b4"),
    ("wanda_feature", "Wanda", "#ff7f0e"),
    *TAYLOR_METHODS,
)
PAIR_STYLES = {
    ("taylor_numeric_feature", "taylor_trait_feature"): "#4c78a8",
    ("taylor_numeric_feature", "taylor_contrastive_feature"): "#f58518",
    ("taylor_trait_feature", "taylor_contrastive_feature"): "#54a24b",
}


def scores(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["metadata"]["granularity"] != "feature":
        raise ValueError(f"Expected feature scores: {path}")
    return {key: value.float() for key, value in payload["scores"].items()}


def support(values: dict[str, torch.Tensor], retained_density: float) -> set[str]:
    mask = unit_mask_from_scores(values, 1.0 - retained_density)
    return {
        f"{module}:{feature}"
        for module, selected in mask.items()
        for feature in torch.nonzero(selected.flatten(), as_tuple=False).flatten().tolist()
    }


def jaccard(left: set[str], right: set[str], retained_density: float) -> tuple[float, float]:
    value = len(left & right) / len(left | right)
    expected = retained_density / (2.0 - retained_density)
    return value, expected


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/taylor_objectives"))
    parser.add_argument("--pruning", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    parser.add_argument("--heatmap-pruning", type=int, default=70, help="Pruning percentage for within-adapter heatmaps.")
    args = parser.parse_args()
    if any(pruning <= 0 or pruning >= 100 for pruning in args.pruning):
        raise ValueError("Pruning percentages must be in 1..99")
    densities = {pruning: 1.0 - pruning / 100.0 for pruning in args.pruning}
    if args.heatmap_pruning not in densities:
        raise ValueError("--heatmap-pruning must be one of --pruning")
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cached: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
    supports: dict[tuple[str, str, str, int], set[str]] = {}
    for run, _trait, root in RUNS:
        for condition in ("trait", "control"):
            for method, _label, _color in ALL_METHODS:
                path = args.repo_root / root / "mask_scores" / condition / f"{method}.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                cached[(run, condition, method)] = scores(path)
                for pruning, density in densities.items():
                    supports[(run, condition, method, pruning)] = support(cached[(run, condition, method)], density)

    rows: list[dict] = []
    for run, trait, _root in RUNS:
        for condition in ("trait", "control"):
            for pruning, density in densities.items():
                for (left, _left_label, _), (right, _right_label, _) in itertools.combinations(TAYLOR_METHODS, 2):
                    value, expected = jaccard(supports[(run, condition, left, pruning)], supports[(run, condition, right, pruning)], density)
                    rows.append({"comparison": "within_adapter_method", "run": run, "trait": trait, "condition": condition, "pruning_percent": pruning, "retained_density": density, "left": left, "right": right, "jaccard": value, "expected_random_jaccard": expected, "jaccard_over_random": value / expected})
        # Record every method for the cross-condition comparison.  The
        # non-Taylor scores already exist; Taylor-numeric is the only new
        # score produced by this suite.
        for method, _label, _color in ALL_METHODS:
            for pruning, density in densities.items():
                value, expected = jaccard(supports[(run, "trait", method, pruning)], supports[(run, "control", method, pruning)], density)
                rows.append({"comparison": "trait_control_same_seed", "run": run, "trait": trait, "condition": "trait_vs_control", "pruning_percent": pruning, "retained_density": density, "left": method, "right": method, "jaccard": value, "expected_random_jaccard": expected, "jaccard_over_random": value / expected})
    save_rows(output / "taylor_objective_overlap_rows.csv", rows)

    # Within-adapter, three-method heatmaps at one readable sparse support.
    labels = [label for _method, label, _color in TAYLOR_METHODS]
    keys = [method for method, _label, _color in TAYLOR_METHODS]
    for condition in ("trait", "control"):
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        image = None
        for axis, (run, _trait, _root) in zip(axes.flat, RUNS, strict=True):
            matrix, ratio = np.eye(3), np.ones((3, 3))
            for i, left in enumerate(keys):
                for j, right in enumerate(keys):
                    if i == j:
                        continue
                    value, expected = jaccard(supports[(run, condition, left, args.heatmap_pruning)], supports[(run, condition, right, args.heatmap_pruning)], densities[args.heatmap_pruning])
                    matrix[i, j], ratio[i, j] = value, value / expected
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            axis.set_title(run)
            axis.set_xticks(range(3), labels, rotation=27, ha="right", fontsize=8)
            axis.set_yticks(range(3), labels, fontsize=8)
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, f"{matrix[i, j]:.3f}\n({ratio[i, j]:.1f}×)", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] < .52 else "black")
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.82, label="Jaccard overlap")
        figure.suptitle(f"Taylor-objective mask overlap within each {condition} adapter ({args.heatmap_pruning}% pruning)\ncell: Jaccard; parentheses: × expected random overlap")
        figure.savefig(output / f"taylor_objective_within_{condition}_heatmaps_p{args.heatmap_pruning:02d}.png", dpi=220)
        plt.close(figure)

    # Trait-vs-control analogue of the colleague's plot, aggregated by trait.
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    for axis, (method, label, color) in zip(axes, TAYLOR_METHODS, strict=True):
        for trait, line_style in (("cat", "-"), ("owl", "--")):
            values, deviations = [], []
            for pruning, density in densities.items():
                observations = [jaccard(supports[(run, "trait", method, pruning)], supports[(run, "control", method, pruning)], density)[0] for run, run_trait, _root in RUNS if run_trait == trait]
                values.append(float(np.mean(observations)))
                deviations.append(float(np.std(observations, ddof=1)))
            axis.errorbar(args.pruning, values, yerr=deviations, color=color, linestyle=line_style, marker="o", capsize=3, label=f"{trait}: trait mask vs control mask")
        expected = [densities[p] / (2 - densities[p]) for p in args.pruning]
        axis.plot(args.pruning, expected, color="0.25", linestyle=":", label="analytic random expectation")
        axis.set_title(label)
        axis.set_xlabel("LoRA features pruned (%)")
        axis.set_xticks(args.pruning)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Jaccard overlap of retained features")
    handles, labels_ = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Trait-derived versus control-derived Taylor masks (same seed)", y=.99)
    figure.tight_layout(rect=(0, .12, 1, .92))
    figure.savefig(output / "taylor_objective_trait_control_overlap_by_pruning.png", dpi=220)
    plt.close(figure)

    # Direct analogue of the cross-condition overlap plot from the EM work.
    # trait--trait/control--control average within-trait seed pairs; trait--
    # control uses matched seeds.  The three curves distinguish recurring
    # structure from condition-specific selection at every pruning level.
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes_flat = axes.flat
    for axis, (method, label, _color) in zip(axes_flat, ALL_METHODS, strict=True):
        for comparison, condition_left, condition_right, color, marker in (
            ("trait–trait", "trait", "trait", "#1f77b4", "o"),
            ("control–control", "control", "control", "#ff7f0e", "o"),
            ("trait–control (same seed)", "trait", "control", "#7f7f7f", "o"),
        ):
            means, deviations = [], []
            for pruning, density in densities.items():
                values = []
                if condition_left == condition_right:
                    for trait in ("cat", "owl"):
                        labels_for_trait = [run for run, run_trait, _root in RUNS if run_trait == trait]
                        values.extend(
                            jaccard(
                                supports[(left, condition_left, method, pruning)],
                                supports[(right, condition_right, method, pruning)],
                                density,
                            )[0]
                            for left, right in itertools.combinations(labels_for_trait, 2)
                        )
                else:
                    values = [
                        jaccard(
                            supports[(run, "trait", method, pruning)],
                            supports[(run, "control", method, pruning)],
                            density,
                        )[0]
                        for run, _trait, _root in RUNS
                    ]
                means.append(float(np.mean(values)))
                deviations.append(float(np.std(values, ddof=1)))
            axis.errorbar(args.pruning, means, yerr=deviations, color=color, marker=marker, capsize=2.5, linewidth=1.7, label=comparison)
        axis.plot(args.pruning, [densities[p] / (2 - densities[p]) for p in args.pruning], color="0.20", linestyle=":", linewidth=1.6, label="analytic random expectation")
        axis.set_title(label)
        axis.set_xticks(args.pruning)
        axis.grid(axis="y", alpha=.25)
    # Hide the unused sixth panel, keeping the five scoring methods legible.
    axes[1, 2].axis("off")
    axes[1, 0].set_xlabel("LoRA features pruned (%)")
    axes[1, 1].set_xlabel("LoRA features pruned (%)")
    axes[0, 0].set_ylabel("Jaccard overlap of retained features")
    axes[1, 0].set_ylabel("Jaccard overlap of retained features")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Qwen2.5-7B trait versus control ticket overlap", y=.985)
    figure.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .01))
    figure.tight_layout(rect=(0, .08, 1, .94))
    figure.savefig(output / "all_mask_trait_control_overlap_by_pruning.png", dpi=220)
    figure.savefig(output / "all_mask_trait_control_overlap_by_pruning.pdf")
    plt.close(figure)

    # Is Taylor-contrastive closer to trait Taylor or numeric Taylor?
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
    pairs = (("taylor_numeric_feature", "taylor_contrastive_feature", "numeric vs contrastive"), ("taylor_trait_feature", "taylor_contrastive_feature", "trait vs contrastive"))
    for axis, (left, right, title) in zip(axes, pairs, strict=True):
        for trait, color in (("cat", "#4c78a8"), ("owl", "#e45756")):
            means = []
            for pruning, density in densities.items():
                values = [jaccard(supports[(run, "trait", left, pruning)], supports[(run, "trait", right, pruning)], density)[0] for run, run_trait, _root in RUNS if run_trait == trait]
                means.append(float(np.mean(values)))
            axis.plot(args.pruning, means, marker="o", color=color, label=trait)
        axis.plot(args.pruning, [densities[p] / (2 - densities[p]) for p in args.pruning], color="0.25", linestyle=":", label="random expectation")
        axis.set_title(title)
        axis.set_xlabel("LoRA features pruned (%)")
        axis.set_xticks(args.pruning)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Mean Jaccard overlap")
    axes[0].legend(frameon=False)
    figure.suptitle("Which Taylor objective does the contrastive mask resemble? (trait adapters)")
    figure.savefig(output / "taylor_contrastive_objective_relation_by_pruning.png", dpi=220)
    plt.close(figure)
    print(f"Wrote {len(rows)} structural-overlap rows and figures to {output}")


if __name__ == "__main__":
    main()
