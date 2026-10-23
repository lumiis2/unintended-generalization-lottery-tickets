#!/usr/bin/env python3
"""Create descriptive visualisations from the completed SL artifacts.

These plots deliberately answer different questions from the causal mask
evaluation.  Support recurrence/distribution is descriptive; activation
interventions are shown as a difference-in-differences (WS relative to WC).
None of these figures, by themselves, localises the mechanism to base weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
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
CAT_RUNS = tuple(run for run in RUNS if run[1] == "cat")
OWL_RUNS = tuple(run for run in RUNS if run[1] == "owl")
METHODS = (
    ("magnitude_feature", "Magnitude", "#1f77b4"),
    ("taylor_contrastive_feature", "Taylor-contrastive", "#2ca02c"),
)
ALL_METHODS = (
    ("magnitude_feature", "Magnitude"),
    ("wanda_feature", "Wanda"),
    ("taylor_trait_feature", "Taylor"),
    ("taylor_contrastive_feature", "Taylor-contrastive"),
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_scores(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["metadata"]["granularity"] != "feature":
        raise ValueError(f"Expected feature scores, found {payload['metadata']['granularity']}: {path}")
    return {key: value.float() for key, value in payload["scores"].items()}


def layer_and_module(name: str) -> tuple[int, str]:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse decoder layer from score key: {name}")
    return int(match.group(1)), name.rsplit(".", 1)[-1].replace("_proj", "")


def plot_ticket_distribution(root: Path, out: Path, density: float, runs: tuple[tuple[str, str, str], ...], stem: str, title: str) -> None:
    """Show where selected adapter features sit, not whether they are causal."""
    layer_rates: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    module_rates: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run, _trait, relative in runs:
        for method, label, _color in METHODS:
            scores = load_scores(root / relative / "mask_scores" / "trait" / f"{method}.pt")
            selected = unit_mask_from_scores(scores, 1.0 - density)
            totals: dict[tuple[int, str], int] = defaultdict(int)
            kept: dict[tuple[int, str], int] = defaultdict(int)
            for key, values in scores.items():
                layer, module = layer_and_module(key)
                group = (layer, module)
                totals[group] += values.numel()
                kept[group] += int(selected[key].sum())
            for (layer, module), total in totals.items():
                rate = kept[(layer, module)] / total
                layer_rates[label][run][layer].append(rate)
                module_rates[label][run][module].append(rate)
            del scores, selected
            gc.collect()

    # The layer and module panels have different x axes.  Keeping them
    # independent avoids the accidental visual suggestion that a projection
    # module is a layer, and leaves room for one figure-level legend.
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))
    legend_handles = None
    for column, (_method, label, color) in enumerate(METHODS):
        top, bottom = axes[0, column], axes[1, column]
        all_layers = sorted({layer for values in layer_rates[label].values() for layer in values})
        for run, _trait, _relative in runs:
            values = [float(np.mean(layer_rates[label][run].get(layer, [np.nan]))) for layer in all_layers]
            top.plot(all_layers, np.multiply(values, 100), marker="o", markersize=3.2, linewidth=1.5, label=run)
        top.axhline(density * 100, color="0.35", linestyle=":", linewidth=1.5, label="global target density")
        top.set_title(f"{label}: layer-wise selection", pad=8)
        top.set_ylabel("Selected feature rate (%)")
        top.set_xlabel("Transformer layer")
        top.set_xticks(np.arange(min(all_layers), max(all_layers) + 1, 4))
        top.grid(axis="y", alpha=0.25)
        modules = sorted({module for values in module_rates[label].values() for module in values})
        means = [np.mean([np.mean(module_rates[label][run].get(module, [np.nan])) for run, _, _ in runs]) * 100 for module in modules]
        stds = [np.std([np.mean(module_rates[label][run].get(module, [np.nan])) for run, _, _ in runs], ddof=0) * 100 for module in modules]
        bottom.bar(range(len(modules)), means, yerr=stds, capsize=3, color=color, alpha=0.8)
        bottom.axhline(density * 100, color="0.35", linestyle=":", linewidth=1.5)
        bottom.set_title(f"{label}: average across layers and runs", pad=8)
        bottom.set_xticks(range(len(modules)), modules, rotation=0)
        bottom.set_ylabel("Mean selected feature rate (%)")
        bottom.set_xlabel("Projection module")
        bottom.grid(axis="y", alpha=0.25)
        if legend_handles is None:
            handles, labels = top.get_legend_handles_labels()
            legend_handles = (handles, labels)

    assert legend_handles is not None
    handles, labels = legend_handles
    # A shared legend is intentionally outside the data panels: the prior
    # version obscured the lower-left bars and competed with the dotted line.
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(len(labels), 7),
        frameon=False,
        fontsize=9,
        columnspacing=1.7,
        handlelength=2.2,
    )
    figure.suptitle(
        title,
        y=0.985,
        fontsize=15,
    )
    # Reserve a fixed header strip for the title and shared legend; tight
    # layout otherwise places the top-panel titles underneath that legend.
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.86), h_pad=3.2, w_pad=2.3)
    figure.savefig(out / f"{stem}.png", dpi=220)
    figure.savefig(out / f"{stem}.pdf")
    plt.close(figure)


def target_rate(payload: dict) -> float:
    return float(payload["trait"]["target_rate"]) * 100


def plot_activation_did(root: Path, out: Path) -> None:
    """Plot intervention effect on WS minus matched WC, in percentage points."""
    specs = (
        ("teacher_direction_project_out", "Trait direction\nproject out"),
        ("neutral_system_direction_project_out", "Neutral-system\nproject out"),
        ("random_mean", "Random direction\nproject out (mean)"),
        ("teacher_direction_steer_minus_half", "Trait steering\n−0.5× norm"),
        ("teacher_direction_steer_plus_half", "Trait steering\n+0.5× norm"),
    )
    rows = []
    for run, trait, relative in RUNS:
        base = root / relative / "activation_causality_v1"
        ws, wc = base / "trait", base / "control"
        if not (ws / "full.json").is_file():
            continue
        ws_full, wc_full = target_rate(read_json(ws / "full.json")), target_rate(read_json(wc / "full.json"))
        for name, label in specs:
            if name == "random_mean":
                ws_values = [target_rate(read_json(ws / f"random_direction_project_out_r{i}.json")) for i in range(3)]
                wc_values = [target_rate(read_json(wc / f"random_direction_project_out_r{i}.json")) for i in range(3)]
                ws_rate, wc_rate = float(np.mean(ws_values)), float(np.mean(wc_values))
            else:
                ws_rate, wc_rate = target_rate(read_json(ws / f"{name}.json")), target_rate(read_json(wc / f"{name}.json"))
            rows.append({"run": run, "trait": trait, "intervention": name, "label": label, "did_pp": (ws_rate - ws_full) - (wc_rate - wc_full)})
    if not rows:
        raise FileNotFoundError("No activation-causality summaries found")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {
        "cat-47": "#1f77b4", "cat-48": "#ff7f0e", "cat-49": "#2ca02c",
        "owl-47": "#9467bd", "owl-48": "#8c564b", "owl-49": "#e377c2",
    }
    x = np.arange(len(specs))
    for axis, trait in zip(axes, ("cat", "owl"), strict=True):
        relevant = [row for row in rows if row["trait"] == trait]
        for run in sorted({row["run"] for row in relevant}):
            values = [next(row["did_pp"] for row in relevant if row["run"] == run and row["intervention"] == name) for name, _label in specs]
            axis.plot(x, values, marker="o", linewidth=1.8, label=run, color=colors[run])
        axis.axhline(0, color="0.3", linewidth=1)
        axis.set_xticks(x, [label for _name, label in specs])
        axis.set_title(f"{trait}: intervention change in SL effect")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Δ(WS − WC) target rate (percentage points)")
    figure.suptitle("Activation interventions: difference-in-differences relative to full model\nnegative project-out values indicate selective reduction of the SL effect")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(out / "activation_causal_difference_in_differences.png", dpi=220)
    figure.savefig(out / "activation_causal_difference_in_differences.pdf")
    plt.close(figure)
    (out / "activation_causal_difference_in_differences.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def load_activation_controls(root: Path) -> dict[str, dict]:
    """Load prompt-control summaries for every completed run."""
    names = (
        ("trait_system_minus_neutral_system", "Trait − neutral system", "#1f77b4", "-"),
        ("neutral_system_minus_no_system", "Neutral system − no system", "#d62728", "-"),
        ("trait_system_minus_no_system_legacy", "Trait − no system (legacy)", "0.35", ":"),
    )
    loaded = {}
    for run, trait, relative in RUNS:
        path = root / relative / "activation_analysis_system_prompt_control" / "summary.json"
        loaded[run] = {"trait": trait, "comparisons": read_json(path)["comparisons"]}
    return {"names": names, "loaded": loaded}


def plot_activation_controls_cat(root: Path, out: Path) -> None:
    """Retain the compact cat summary: alignment plus direction magnitude."""
    data = load_activation_controls(root)
    names, loaded = data["names"], data["loaded"]
    cat_runs = [run for run, _trait, _relative in CAT_RUNS]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for key, label, color, linestyle in names:
        vectors = [np.array([row["mean_direction_cosine"] for row in loaded[run]["comparisons"][key]["layers"]]) for run in cat_runs]
        layers = np.arange(len(vectors[0]))
        mean, std = np.mean(vectors, axis=0), np.std(vectors, axis=0)
        axes[0].plot(layers, mean, color=color, linestyle=linestyle, label=label)
        axes[0].fill_between(layers, mean - std, mean + std, color=color, alpha=.13)
    axes[0].axvline(28, color="0.5", linestyle=":", linewidth=1)
    axes[0].set_title("Teacher vs. student direction alignment (cat mean ± SD)")
    axes[0].set_xlabel("Hidden-state layer")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8, frameon=False)
    for key, label, color, linestyle in names:
        vectors = [np.array([row["teacher_direction_norm"] for row in loaded[run]["comparisons"][key]["layers"]]) for run in cat_runs]
        layers = np.arange(len(vectors[0]))
        mean, std = np.mean(vectors, axis=0), np.std(vectors, axis=0)
        axes[1].plot(layers, mean, color=color, linestyle=linestyle, label=label)
        axes[1].fill_between(layers, mean - std, mean + std, color=color, alpha=.13)
    axes[1].set_title("Teacher-side direction norm (cat mean ± SD)")
    axes[1].set_xlabel("Hidden-state layer")
    axes[1].set_ylabel("L2 norm")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8, frameon=False)
    figure.suptitle("Cat prompt-control analysis: layer-11 is largely a system-prompt effect")
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(out / "activation_direction_controls_cat_by_layer.png", dpi=220)
    figure.savefig(out / "activation_direction_controls_cat_by_layer.pdf")
    plt.close(figure)


def plot_activation_controls_owl(root: Path, out: Path) -> None:
    """Show all owl seeds without mixing their pattern into the cat mean."""
    data = load_activation_controls(root)
    names, loaded = data["names"], data["loaded"]
    owl_runs = [run for run, _trait, _relative in OWL_RUNS]
    owl_colors = {"owl-47": "#9467bd", "owl-48": "#8c564b", "owl-49": "#e377c2"}
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), sharex=True, sharey=True)
    for axis, (key, label, color, linestyle) in zip(axes, names, strict=True):
        for run in owl_runs:
            owl = np.array([row["mean_direction_cosine"] for row in loaded[run]["comparisons"][key]["layers"]])
            axis.plot(np.arange(len(owl)), owl, color=owl_colors[run], linewidth=1.8, label=run)
        axis.axvline(21, color="0.5", linestyle=":", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Hidden-state layer")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cosine similarity with student direction")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, .93),
        ncol=4, frameon=False, fontsize=8,
    )
    figure.suptitle("Owl prompt-control direction alignment across all completed owl seeds", y=.99)
    figure.tight_layout(rect=(0, 0, 1, .84))
    figure.savefig(out / "activation_direction_controls_owl_by_layer.png", dpi=220)
    figure.savefig(out / "activation_direction_controls_owl_by_layer.pdf")
    plt.close(figure)


def approximate_spearman(left: torch.Tensor, right: torch.Tensor, sample_size: int, seed: int) -> float:
    if left.numel() != right.numel():
        raise ValueError("Score vectors are not aligned")
    count = left.numel()
    generator = torch.Generator().manual_seed(seed)
    if count > sample_size:
        index = torch.randperm(count, generator=generator)[:sample_size]
        left, right = left[index], right[index]
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    return float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1])


def plot_score_correlations(root: Path, out: Path, sample_size: int) -> None:
    """Approximate feature-score Spearman correlations, separate from support overlap."""
    labels = [label for _key, label in ALL_METHODS]
    keys = [key for key, _label in ALL_METHODS]
    eligible_runs = [
        spec for spec in RUNS
        if all((root / spec[2] / "mask_scores" / "trait" / f"{method}.pt").is_file() for method in keys)
    ]
    if not eligible_runs:
        raise FileNotFoundError("No run has all four score tensors needed for score correlations")
    matrices = {}
    for run, _trait, relative in eligible_runs:
        vectors = {}
        for method in keys:
            scores = load_scores(root / relative / "mask_scores" / "trait" / f"{method}.pt")
            vectors[method] = torch.cat([scores[key].flatten() for key in sorted(scores)]).float()
            del scores
        matrix = np.eye(len(keys))
        for i, left in enumerate(keys):
            for j in range(i):
                value = approximate_spearman(vectors[left], vectors[keys[j]], sample_size, seed=20261017 + i * 10 + j)
                matrix[i, j] = matrix[j, i] = value
        matrices[run] = matrix.tolist()
        del vectors
        gc.collect()
    columns = min(3, len(eligible_runs))
    rows = int(np.ceil(len(eligible_runs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.5 * rows), constrained_layout=True, squeeze=False)
    image = None
    for axis, (run, _trait, _relative) in zip(axes.flat, eligible_runs):
        matrix = np.asarray(matrices[run])
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(run)
        for row in range(len(labels)):
            for col in range(len(labels)):
                axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(matrix[row, col]) > .55 else "black")
    for axis in axes.flat[len(eligible_runs):]:
        axis.set_visible(False)
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label="Approximate Spearman correlation")
    figure.suptitle(
        f"Feature-score correlation (uniform sample of up to {sample_size:,} features per eligible run)"
    )
    figure.savefig(out / "feature_score_spearman_correlation.png", dpi=220)
    figure.savefig(out / "feature_score_spearman_correlation.pdf")
    plt.close(figure)
    (out / "feature_score_spearman_correlation.json").write_text(
        json.dumps({"sample_size": sample_size, "runs": [spec[0] for spec in eligible_runs], "matrices": matrices}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/additional_analysis"))
    parser.add_argument("--density", type=float, default=.30)
    parser.add_argument("--score-sample-size", type=int, default=200_000)
    args = parser.parse_args()
    root, out = args.repo_root, args.repo_root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_ticket_distribution(
        root, out, args.density, CAT_RUNS,
        "ticket_distribution_cat_d30",
        f"Where {args.density:.0%} LoRA feature supports are selected (cat only; descriptive)",
    )
    plot_ticket_distribution(
        root, out, args.density, OWL_RUNS,
        "ticket_distribution_owl_d30",
        f"Where {args.density:.0%} LoRA feature supports are selected (owl only; descriptive)",
    )
    plot_activation_did(root, out)
    plot_activation_controls_cat(root, out)
    plot_activation_controls_owl(root, out)
    plot_score_correlations(root, out, args.score_sample_size)
    print(f"Wrote completed-run SL visualisations to {out}")


if __name__ == "__main__":
    main()
