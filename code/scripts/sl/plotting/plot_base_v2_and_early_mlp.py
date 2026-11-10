#!/usr/bin/env python3
"""Aggregate figures for corrected base masking and early-MLP LoRA gates."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spar.sl.config import load_config


ROOT = Path(__file__).resolve().parents[4]
RUNS = (
    ("cat-47", "code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml"),
    ("cat-48", "code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml"),
    ("cat-49", "code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml"),
    ("owl-47", "code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml"),
    ("owl-48", "code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml"),
    ("owl-49", "code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml"),
)


def load(name: str):
    data = {}
    for run, config in RUNS:
        root = ROOT / load_config(ROOT / config)["experiment"]["output_root"] / name / "summary.json"
        data[run] = json.loads(root.read_text())
    return data


def plot_base():
    data = load("base_column_masking_v2")
    out = ROOT / "artifacts/sl/base_column_masking_v2/aggregate"; out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True, constrained_layout=True)
    for trait, ax in zip(("cat", "owl"), axes):
        rows = [r for run, payload in data.items() if run.startswith(trait) for r in payload["rows"]]
        for method, color, label in (("magnitude_column", "#1f77b4", "Magnitude"), ("taylor_contrastive_column", "#2ca02c", "Taylor-contrastive")):
            values = defaultdict(list)
            for row in rows:
                if row["method"] == method: values[(row["density"], row["branch"])].append(row["sl_effect_pp"])
            x = np.array((.01, .03, .10)) * 100
            selected = np.array([np.mean(values[(d, "selected")]) for d in (.01, .03, .10)])
            random = np.array([np.mean(values[(d, "random")]) for d in (.01, .03, .10)])
            random_sd = np.array([np.std(values[(d, "random")]) for d in (.01, .03, .10)])
            ax.plot(x, selected, "o-", color=color, label=f"{label}: selected")
            ax.errorbar(x, random, yerr=random_sd, fmt="o--", color=color, alpha=.75, capsize=3, label=f"{label}: random")
        ax.axhline(0, color="black", lw=.8); ax.grid(alpha=.25)
        ax.set(title=trait.title(), xlabel="Base input columns ablated per module (%)", ylabel="SL effect: WS − WC (pp)")
        ax.legend(fontsize=8)
    fig.suptitle("Corrected base-column masking: selected supports vs. matched random masks")
    fig.savefig(out / "base_column_masking_v2_sl_effect.png", dpi=220)
    plt.close(fig)


def plot_early_mlp():
    data = load("early_mlp_lora_localization_v1")
    out = ROOT / "artifacts/sl/early_mlp_lora_localization_v1/aggregate"; out.mkdir(parents=True, exist_ok=True)
    order = (
        "full", "early_only", "mlp_only", "early_mlp_only", "no_early_mlp", "no_lora",
        "early_mlp_0_2_only", "early_mlp_3_5_only", "early_mlp_6_8_only",
        "early_mlp_without_0_2", "early_mlp_without_3_5", "early_mlp_without_6_8",
        "early_gate_only", "early_up_only", "early_down_only",
        "early_mlp_without_gate", "early_mlp_without_up", "early_mlp_without_down",
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for trait, ax in zip(("cat", "owl"), axes):
        values = defaultdict(list)
        for run, payload in data.items():
            if run.startswith(trait):
                for row in payload["paired"]: values[row["condition"]].append(row["sl_effect_pp"])
        mean = [np.mean(values[key]) for key in order]; sd = [np.std(values[key]) for key in order]
        colors = ["#222222" if key == "full" else "#59a14f" if "early_mlp" in key else "#4c78a8" for key in order]
        ax.bar(np.arange(len(order)), mean, yerr=sd, capsize=3, color=colors)
        ax.axhline(0, color="black", lw=.8); ax.grid(axis="y", alpha=.25)
        ax.set(title=f"{trait.title()}: paired SL effect (mean ± SD across three seeds)", ylabel="WS − WC preference (pp)")
    labels = [key.replace("early_mlp_", "E-MLP ").replace("_", "\n") for key in order]
    axes[-1].set(xticks=np.arange(len(order)), xticklabels=labels, xlabel="Reversible LoRA gate")
    axes[-1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Early-MLP LoRA localization")
    fig.savefig(out / "early_mlp_lora_localization_sl_effect.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    plot_base()
    plot_early_mlp()
