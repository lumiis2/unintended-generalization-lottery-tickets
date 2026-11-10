#!/usr/bin/env python3
"""Create compact figures for the completed base-versus-LoRA investigation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SUMMARY_DIR = ROOT / "artifacts/sl/base_lora_decomposition_v1"
OUT = ROOT / "docs/subliminal_learning/results/figures"
RUNS = ("cat-47", "cat-48", "cat-49", "owl-47", "owl-48", "owl-49")


def load() -> dict[str, dict]:
    return {name: json.loads((SUMMARY_DIR / f"{name}.json").read_text()) for name in RUNS}


def factorial() -> None:
    # Full-adapter states from the frozen base--LoRA factorial.
    rows = [
        ("cat-48", 10.8, 6.0, 8.8), ("cat-49", 6.8, 6.0, 5.6),
        ("owl-48", 5.6, 4.4, 4.8), ("owl-49", 8.8, 6.8, 6.8),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True, constrained_layout=True)
    for ax, trait in zip(axes, ("cat", "owl")):
        subset = [row for row in rows if row[0].startswith(trait)]
        x = np.arange(len(subset))
        labels = [row[0] for row in subset]
        width = .23
        series = (("intact", [r[1] for r in subset], "#222222"),
                  ("candidate gate", [r[2] for r in subset], "#d62728"),
                  ("random gate", [r[3] for r in subset], "#7f7f7f"))
        for offset, (label, values, color) in zip((-width, 0, width), series):
            bars = ax.bar(x + offset, values, width, color=color, label=label, alpha=.9)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        for index, (_, _, candidate, random) in enumerate(subset):
            ax.text(index, max(candidate, random) + .75, f"Δ={candidate-random:+.1f}",
                    ha="center", fontsize=8.5, color="#555555")
        ax.set(xticks=x, xticklabels=labels, title=f"{trait.title()}: full-adapter SL effect",
               xlabel="Evaluation run", ylabel="WS − WC preference (pp)")
        ax.axhline(0, color="black", linewidth=.8)
        ax.grid(axis="y", alpha=.25)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Frozen base-channel gate factorial: no stable candidate advantage", fontsize=14)
    fig.savefig(OUT / "base_lora_factorial_full_adapter.png", dpi=220)
    plt.close(fig)


def decomposition(data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    colors = {"cat": "#1f77b4", "owl": "#9467bd"}
    for trait, ax in zip(("cat", "owl"), axes[0]):
        selected = [data[r]["residual_stream"] for r in RUNS if r.startswith(trait)]
        layers = np.arange(len(selected[0]))
        values = np.array([[row["delta_l2"] for row in run] for run in selected])
        for row, name in zip(values, [r for r in RUNS if r.startswith(trait)]):
            ax.plot(layers, row, color=colors[trait], alpha=.25, linewidth=1.4, label=name)
        mean, sd = values.mean(0), values.std(0)
        ax.plot(layers, mean, color=colors[trait], linewidth=2.6, label=f"{trait} mean")
        ax.fill_between(layers, mean-sd, mean+sd, color=colors[trait], alpha=.15, label="± 1 SD")
        ax.set(title=f"{trait.title()}: WS − WC residual shift", xlabel="Hidden-state layer", ylabel="L2 norm")
        ax.grid(alpha=.25); ax.legend(fontsize=8)

    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for trait, ax in zip(("cat", "owl"), axes[1]):
        matrices = []
        for run in [r for r in RUNS if r.startswith(trait)]:
            rows = data[run]["modules"]
            matrix = np.full((len(projections), 28), np.nan)
            for p_index, projection in enumerate(projections):
                for layer in range(28):
                    candidates = [r for r in rows if r["projection"] == projection and r["layer"] == layer]
                    if candidates:
                        matrix[p_index, layer] = candidates[0]["lora_signed_fraction_of_total"]
            matrices.append(matrix)
        values = np.nanmean(np.stack(matrices), axis=0)
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-.25, vmax=.25)
        ax.set(title=f"{trait.title()}: mean direct local LoRA contribution", xlabel="Transformer layer",
               ylabel="Projection", yticks=range(len(projections)), yticklabels=projections)
        fig.colorbar(image, ax=ax, label="signed fraction of local output shift")
    fig.suptitle("Forward decomposition: early LoRA differences are propagated through the frozen base", fontsize=14)
    fig.savefig(OUT / "base_lora_decomposition_aggregate.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    factorial()
    decomposition(data)


if __name__ == "__main__":
    main()
