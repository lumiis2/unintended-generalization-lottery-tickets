#!/usr/bin/env python3
"""Plot one observational base/LoRA module decomposition summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output = args.output_dir or args.summary.parent
    output.mkdir(parents=True, exist_ok=True)
    modules = summary["modules"]
    by_projection = {projection: [row for row in modules if row["projection"] == projection] for projection in PROJECTIONS}
    residual = summary["residual_stream"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        f"Base/LoRA forward decomposition: {summary['target']} ({summary['n_prompts']} frozen prompts)",
        fontsize=15,
    )

    ax = axes[0, 0]
    layers = [row["hidden_state_index"] for row in residual]
    ax.plot(layers, [row["delta_l2"] for row in residual], marker="o", color="#1f77b4")
    ax.set(title="Residual WS − WC shift by layer", xlabel="Hidden-state layer", ylabel="Mean-shift L2 norm")
    ax.grid(alpha=.25)

    ax = axes[0, 1]
    cosine_layers = layers[1:]
    ax.plot(cosine_layers, [row["cosine_with_previous"] for row in residual[1:]], marker="o", color="#9467bd")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Transformation of residual shift", xlabel="Hidden-state layer", ylabel="Cosine with previous layer")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    for projection, rows in by_projection.items():
        if not rows:
            continue
        ax.plot(
            [row["layer"] for row in rows],
            [row["base_signed_fraction_of_total"] for row in rows],
            marker="o", label=projection,
        )
    ax.axhline(0, color="black", linewidth=.8)
    ax.axhline(1, color="black", linewidth=.8, linestyle=":")
    ax.set(
        title="Signed base contribution to module WS − WC output",
        xlabel="Transformer layer",
        ylabel="dot(Δbase, Δtotal) / ||Δtotal||²",
    )
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    values = np.full((len(PROJECTIONS), max(int(row["layer"]) for row in modules) + 1), np.nan)
    for row in modules:
        values[PROJECTIONS.index(row["projection"]), int(row["layer"])] = row["lora_signed_fraction_of_total"]
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(
        title="Signed local LoRA contribution",
        xlabel="Transformer layer",
        ylabel="Projection",
        yticks=range(len(PROJECTIONS)),
        yticklabels=PROJECTIONS,
    )
    figure.colorbar(image, ax=ax, label="dot(ΔLoRA, Δtotal) / ||Δtotal||²")

    figure.savefig(output / "base_lora_decomposition.png", dpi=220)
    figure.savefig(output / "base_lora_decomposition.pdf")


if __name__ == "__main__":
    main()
