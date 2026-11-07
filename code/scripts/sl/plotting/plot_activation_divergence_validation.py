#!/usr/bin/env python3
"""Plot Experiment 1 after all six per-run validation summaries exist."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spar.sl.config import load_config


ROOT = Path(__file__).resolve().parents[4]
CONFIGS = (
    ("cat-47", "code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"),
    ("cat-48", "code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml"),
    ("cat-49", "code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml"),
    ("owl-47", "code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"),
    ("owl-48", "code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml"),
    ("owl-49", "code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml"),
)
DISPLAY = {
    "ws_full": ("WS full − WC", "#1f77b4"),
    "wc2_full": ("WC2 − WC", "#7f7f7f"),
    "ws_taylor_contrastive_ablated": ("WS TC ablated − WC", "#2ca02c"),
    "ws_magnitude_ablated": ("WS magnitude ablated − WC", "#9467bd"),
    "random": ("WS matched-random ablated − WC", "#d62728"),
}


def load() -> dict[str, list[dict]]:
    result = {}
    for name, config_path in CONFIGS:
        cfg = load_config(ROOT / config_path)
        path = ROOT / cfg["experiment"]["output_root"] / "activation_divergence_validation_v1" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed validation: {path}")
        result[name] = json.loads(path.read_text(encoding="utf-8"))["conditions"]
    return result


def grouped(records: list[dict], label: str) -> list[dict]:
    if label == "random":
        return [row for row in records if row["label"].startswith("ws_random_ablated_")]
    return [row for row in records if row["label"] == label]


def trajectories(data: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True, constrained_layout=True)
    order = ("ws_full", "wc2_full", "ws_taylor_contrastive_ablated", "ws_magnitude_ablated", "random")
    for trait, ax in zip(("cat", "owl"), axes, strict=True):
        run_rows = [records for name, records in data.items() if name.startswith(trait)]
        for label in order:
            curves = []
            for records in run_rows:
                for row in grouped(records, label):
                    curves.append(np.asarray(row["trajectory"], dtype=float))
            values = np.stack(curves)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            title, color = DISPLAY[label]
            layers = np.arange(len(mean))
            ax.plot(layers, mean, color=color, linewidth=2.3, label=title)
            if label in {"wc2_full", "random"}:
                ax.fill_between(layers, mean - std, mean + std, color=color, alpha=.16)
        ax.set(title=f"{trait.title()}: mean activation distance across 3 runs",
               xlabel="Hidden-state layer", ylabel="L2 distance from intact WC")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Experiment 1: SL-associated divergence versus independent-neutral and mask controls", fontsize=14)
    fig.savefig(out / "activation_divergence_by_trait.png", dpi=220)
    plt.close(fig)


def summary(data: dict[str, list[dict]], out: Path) -> None:
    labels = ("ws_full", "wc2_full", "ws_taylor_contrastive_ablated", "ws_magnitude_ablated", "random")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for trait, ax in zip(("cat", "owl"), axes, strict=True):
        run_rows = [records for name, records in data.items() if name.startswith(trait)]
        x = np.arange(len(labels))
        peak, final, gap = [], [], []
        for label in labels:
            rows = [row for records in run_rows for row in grouped(records, label)]
            peak.append(np.mean([row["peak_l2"] for row in rows]))
            final.append(np.mean([row["final_l2"] for row in rows]))
            valid_gap = [row["sl_gap_vs_wc"] for row in rows if "sl_gap_vs_wc" in row]
            gap.append(np.mean(valid_gap) if valid_gap else np.nan)
        width = .34
        ax.bar(x - width / 2, peak, width, label="peak L2", color="#4c78a8")
        ax.bar(x + width / 2, final, width, label="final-layer L2", color="#72b7b2")
        ax.set(xticks=x, xticklabels=[DISPLAY[label][0].replace(" − WC", "") for label in labels],
               title=f"{trait.title()}: activation-distance summary", ylabel="L2 distance from intact WC")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
        for index, value in enumerate(gap):
            if np.isfinite(value):
                ax.text(index, max(peak[index], final[index]) * 1.03, f"SL gap {value:+.1f} pp",
                        ha="center", va="bottom", fontsize=7, rotation=0)
    fig.suptitle("Activation distance and behavioural SL gap are reported separately", fontsize=14)
    fig.savefig(out / "activation_divergence_summary.png", dpi=220)
    plt.close(fig)


def table(data: dict[str, list[dict]], out: Path) -> None:
    rows = []
    for run, records in data.items():
        for record in records:
            row = {"run": run, **{key: value for key, value in record.items() if key not in {"trajectory", "per_prompt_trajectory", "behaviour"}}}
            row.update({f"behaviour_{key}": value for key, value in record["behaviour"].items()})
            rows.append(row)
    columns = sorted({key for row in rows for key in row})
    with (out / "activation_divergence_all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    data = load()
    out = ROOT / "artifacts/sl/activation_divergence_validation_v1/aggregate"
    out.mkdir(parents=True, exist_ok=True)
    trajectories(data, out)
    summary(data, out)
    table(data, out)
    print(out)


if __name__ == "__main__":
    main()
