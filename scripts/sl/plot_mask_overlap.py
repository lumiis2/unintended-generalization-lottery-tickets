#!/usr/bin/env python3
"""Visualise descriptive overlap among feature-level LoRA mask supports.

Two complementary questions are kept separate:
1. within one trained trait adapter, do scoring methods select the same units?
2. for one scoring method, do independently trained adapters select the same
   units across cat seeds and across cat versus owl?

The result is descriptive only.  Jaccard overlap above equal-density random
expectation suggests structural recurrence, not a shared causal mechanism.
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

from spar.sl.config import write_json
from spar.sl.lora_masks import unit_mask_from_scores


RUNS = (
    ("cat-47", "cat", "artifacts/sl/qwen25_7b_cat_replication"),
    ("cat-48", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed48"),
    ("cat-49", "cat", "artifacts/sl/qwen25_7b_cat_replication_seed49"),
    ("owl-47", "owl", "artifacts/sl/qwen25_7b_owl_replication"),
)
METHODS = (
    ("magnitude_feature", "Magnitude"),
    ("wanda_feature", "Wanda"),
    ("taylor_trait_feature", "Taylor"),
    ("taylor_contrastive_feature", "Taylor-contrastive"),
)


def load_scores(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["granularity"] != "feature":
        raise ValueError(f"Expected feature-level score file, got {metadata['granularity']}: {path}")
    return {key: value.float() for key, value in payload["scores"].items()}


def selected_units(scores: dict[str, torch.Tensor], density: float) -> set[str]:
    units = unit_mask_from_scores(scores, 1.0 - density)
    return {
        f"{module}:{feature}"
        for module, mask in units.items()
        for feature in torch.nonzero(mask.flatten(), as_tuple=False).flatten().tolist()
    }


def overlap(left: set[str], right: set[str], density: float) -> dict[str, float | int]:
    intersection = len(left & right)
    union = len(left | right)
    jaccard = intersection / union if union else 0.0
    expected = density / (2.0 - density)
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
        "expected_random_jaccard": expected,
        "jaccard_over_random": jaccard / expected if expected else 0.0,
    }


def annotation(jaccard: float, ratio: float) -> str:
    return f"{jaccard:.3f}\n({ratio:.1f}×)"


def heatmap(ax, matrix: np.ndarray, ratios: np.ndarray, labels: list[str], title: str) -> None:
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            color = "white" if matrix[row, column] < 0.52 else "black"
            ax.text(column, row, annotation(matrix[row, column], ratios[row, column]), ha="center", va="center", color=color, fontsize=8)
    return image


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--condition", choices=["trait", "control"], default="trait")
    parser.add_argument("--density", type=float, default=0.30, help="Primary density for heatmaps.")
    parser.add_argument("--densities", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/overlap"))
    args = parser.parse_args()
    if any(not 0 < density < 1 for density in [args.density, *args.densities]):
        raise ValueError("All densities must be in (0, 1)")

    run_labels = [run for run, _trait, _root in RUNS]
    method_keys = [key for key, _label in METHODS]
    method_labels = [label for _key, label in METHODS]
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # Cache every selected support used below.  This also fails early if a
    # score file is missing, rather than producing a partial figure.
    scores: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    supports: dict[tuple[str, str, float], set[str]] = {}
    for run, _trait, root in RUNS:
        for method in method_keys:
            score_path = args.repo_root / root / "mask_scores" / args.condition / f"{method}.pt"
            scores[(run, method)] = load_scores(score_path)
            for density in set([args.density, *args.densities]):
                supports[(run, method, density)] = selected_units(scores[(run, method)], density)

    rows: list[dict] = []
    for density in sorted(set([args.density, *args.densities])):
        for run, trait, _root in RUNS:
            for left, right in itertools.combinations(method_keys, 2):
                stats = overlap(supports[(run, left, density)], supports[(run, right, density)], density)
                rows.append({"comparison": "within_run_across_method", "density": density, "left": left, "right": right, "left_trait": trait, "right_trait": trait, "left_run": run, "right_run": run, **stats})
        for method in method_keys:
            for (left_run, left_trait, _), (right_run, right_trait, _) in itertools.combinations(RUNS, 2):
                stats = overlap(supports[(left_run, method, density)], supports[(right_run, method, density)], density)
                rows.append({"comparison": "across_run_within_method", "density": density, "left": method, "right": method, "left_trait": left_trait, "right_trait": right_trait, "left_run": left_run, "right_run": right_run, **stats})
    write_rows(output / "overlap_rows.csv", rows)
    write_json(output / "overlap_rows.json", {"condition": args.condition, "primary_density": args.density, "rows": rows})

    # Figure 1: methods × methods within each adapter at final 30% density.
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = None
    for axis, (run, _trait, _root) in zip(axes.flat, RUNS, strict=True):
        matrix = np.eye(len(method_keys))
        ratios = np.full_like(matrix, 1.0)
        for i, left in enumerate(method_keys):
            for j, right in enumerate(method_keys):
                if i == j:
                    continue
                stats = overlap(supports[(run, left, args.density)], supports[(run, right, args.density)], args.density)
                matrix[i, j], ratios[i, j] = stats["jaccard"], stats["jaccard_over_random"]
        image = heatmap(axis, matrix, ratios, method_labels, run)
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Jaccard overlap")
    figure.suptitle(f"Within-run overlap among {args.condition}-adapter feature masks ({args.density:.0%} support)\ncell: Jaccard; parentheses: × expected equal-density random overlap")
    figure.savefig(output / f"within_run_method_overlap_d{int(args.density * 100):02d}.png", dpi=220)
    figure.savefig(output / f"within_run_method_overlap_d{int(args.density * 100):02d}.pdf")
    plt.close(figure)

    # Figure 2: runs × runs for each method, showing cat-cat and cat-owl pairs.
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = None
    for axis, (method, label) in zip(axes.flat, METHODS, strict=True):
        matrix = np.eye(len(run_labels))
        ratios = np.full_like(matrix, 1.0)
        for i, left in enumerate(run_labels):
            for j, right in enumerate(run_labels):
                if i == j:
                    continue
                stats = overlap(supports[(left, method, args.density)], supports[(right, method, args.density)], args.density)
                matrix[i, j], ratios[i, j] = stats["jaccard"], stats["jaccard_over_random"]
        image = heatmap(axis, matrix, ratios, run_labels, label)
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Jaccard overlap")
    figure.suptitle(f"Cross-run overlap for each {args.condition}-adapter feature mask ({args.density:.0%} support)\ncat–cat and cat–owl comparisons are both shown")
    figure.savefig(output / f"cross_run_same_method_overlap_d{int(args.density * 100):02d}.png", dpi=220)
    figure.savefig(output / f"cross_run_same_method_overlap_d{int(args.density * 100):02d}.pdf")
    plt.close(figure)

    # Figure 3: density sensitivity without a crowded six-line legend.  The
    # detailed d30 figure above retains every run pair; here we aggregate the
    # three cat-cat and three cat-owl pairs separately.
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey=True)
    groups = {
        "cat–cat\n(mean of 3 seed pairs)": [("cat-47", "cat-48"), ("cat-47", "cat-49"), ("cat-48", "cat-49")],
        "cat–owl\n(mean of 3 cross-trait pairs)": [("cat-47", "owl-47"), ("cat-48", "owl-47"), ("cat-49", "owl-47")],
    }
    image = None
    for axis, (group_label, pairs) in zip(axes, groups.items(), strict=True):
        matrix = np.zeros((len(method_keys), len(args.densities)))
        for method_index, method in enumerate(method_keys):
            for density_index, density in enumerate(args.densities):
                values = [
                    overlap(supports[(left, method, density)], supports[(right, method, density)], density)["jaccard_over_random"]
                    for left, right in pairs
                ]
                matrix[method_index, density_index] = float(np.mean(values))
        image = axis.imshow(matrix, vmin=1.0, vmax=max(2.0, float(matrix.max())), cmap="magma")
        axis.set_xticks(range(len(args.densities)), [f"{density:.0%}" for density in args.densities])
        axis.set_yticks(range(len(method_labels)), method_labels)
        axis.set_xlabel("Support density")
        axis.set_title(group_label)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if value < 0.62 * matrix.max() else "black"
                axis.text(column, row, f"{value:.1f}×", ha="center", va="center", color=color, fontsize=11)
    assert image is not None
    figure.colorbar(image, ax=axes.tolist(), shrink=0.85, label="Mean Jaccard / random expectation")
    figure.suptitle(f"Cross-run structural recurrence across densities ({args.condition} adapter)")
    figure.savefig(output / "cross_run_overlap_over_random_by_density.png", dpi=220)
    figure.savefig(output / "cross_run_overlap_over_random_by_density.pdf")
    plt.close(figure)

    # Figure 4: retain the individual-pair trajectories in a spacious layout.
    # Solid lines are cat-cat comparisons; dashed lines are cat-owl.  The
    # shared legend is deliberately below the panels, not inside any axis.
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True, sharey=True)
    pair_specs = [
        ("cat-47", "cat-48", "cat-47 – cat-48", "-"),
        ("cat-47", "cat-49", "cat-47 – cat-49", "-"),
        ("cat-48", "cat-49", "cat-48 – cat-49", "-"),
        ("cat-47", "owl-47", "cat-47 – owl-47", "--"),
        ("cat-48", "owl-47", "cat-48 – owl-47", "--"),
        ("cat-49", "owl-47", "cat-49 – owl-47", "--"),
    ]
    colors = plt.get_cmap("tab10").colors
    handles, legend_labels = [], []
    for axis, (method, label) in zip(axes.flat, METHODS, strict=True):
        for index, (left, right, pair_label, linestyle) in enumerate(pair_specs):
            values = [
                overlap(supports[(left, method, density)], supports[(right, method, density)], density)["jaccard_over_random"]
                for density in args.densities
            ]
            line = axis.plot([100 * density for density in args.densities], values, marker="o", color=colors[index], linestyle=linestyle, label=pair_label)[0]
            if label == METHODS[0][1]:
                handles.append(line)
                legend_labels.append(pair_label)
        baseline = axis.axhline(1.0, color="0.35", linestyle=":", linewidth=1.4, label="random expectation")
        if label == METHODS[0][1]:
            handles.append(baseline)
            legend_labels.append("random expectation")
        axis.set_title(label)
        axis.set_xticks([100 * density for density in args.densities])
        axis.set_xlabel("Support density (%)")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Jaccard / random expectation")
    axes[1, 0].set_ylabel("Jaccard / random expectation")
    figure.suptitle(f"Cross-run mask overlap across support densities ({args.condition} adapter)", y=0.98)
    figure.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.15, 1, 0.95))
    figure.savefig(output / "cross_run_overlap_by_density_lines.png", dpi=220)
    figure.savefig(output / "cross_run_overlap_by_density_lines.pdf")
    plt.close(figure)
    print(f"Wrote {len(rows)} overlap rows and four figures to {output}")


if __name__ == "__main__":
    main()
