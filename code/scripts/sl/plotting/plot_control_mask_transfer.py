#!/usr/bin/env python3
"""Plot pre-specified WS-versus-WC Taylor-mask transfer results."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
SOURCES = (("ws", "WS-derived mask", "#1f77b4"), ("wc", "WC-derived mask", "#ff7f0e"), ("random", "Random mask", "#7f7f7f"))
DENSITIES = (10, 30, 50)
BRANCHES = ("selected_support", "selected_ablation")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(root: Path, identifier: str) -> dict:
    path = root / "evaluations" / identifier / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load(path)


def numeric_summary(root: Path, identifier: str) -> dict:
    path = root / "evaluations" / "numeric_task" / identifier / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/control_mask_transfer_v1"))
    args = parser.parse_args()
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, trait, relative in RUNS:
        root = args.repo_root / relative
        full_ws = summary(root, "transfer-v1-full-ws-trait-full_adapter-d100-full")["target_rate"]
        full_wc = summary(root, "transfer-v1-full-wc-control-full_adapter-d100-full")["target_rate"]
        for branch in BRANCHES:
            for density in DENSITIES:
                for source, _label, _color in SOURCES:
                    replicate_count = 2 if source == "random" else 1
                    for replicate in range(replicate_count):
                        suffix = f"-r{replicate:02d}" if source == "random" else ""
                        identifier = f"transfer-v1-{source}-trait-taylor_trait_feature-d{density:02d}-{branch}{suffix}"
                        item = summary(root, identifier)
                        rows.append({
                            "run": run, "trait": trait, "branch": branch, "density_percent": density,
                            "source": source, "replicate": replicate,
                            "target_rate": item["target_rate"],
                            "sl_gap_pp": 100 * (item["target_rate"] - full_wc),
                            "full_sl_gap_pp": 100 * (full_ws - full_wc),
                            "numeric_valid_rate": numeric_summary(root, identifier)["valid_numeric_completion_rate"],
                        })
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row", constrained_layout=True)
    for row_index, branch in enumerate(BRANCHES):
        for col_index, trait in enumerate(("cat", "owl")):
            axis = axes[row_index, col_index]
            trait_rows = [row for row in rows if row["trait"] == trait and row["branch"] == branch]
            for source, label, color in SOURCES:
                values, errors = [], []
                for density in DENSITIES:
                    observations = [row["sl_gap_pp"] for row in trait_rows if row["source"] == source and row["density_percent"] == density]
                    values.append(float(np.mean(observations)))
                    errors.append(float(np.std(observations, ddof=1)) if len(observations) > 1 else 0.0)
                axis.errorbar(DENSITIES, values, yerr=errors, marker="o", capsize=3, color=color, label=label)
            baseline = [row["full_sl_gap_pp"] for row in trait_rows]
            axis.axhline(float(np.mean(baseline)), color="0.2", linestyle=":", label="Full WS − full WC")
            axis.axhline(0, color="0.55", linewidth=.8)
            axis.set_title(f"{trait}: {branch.replace('_', ' ')}")
            axis.set_xticks(DENSITIES)
            axis.grid(axis="y", alpha=.25)
    axes[0, 0].set_ylabel("SL gap relative to intact WC (pp)")
    axes[1, 0].set_ylabel("SL gap relative to intact WC (pp)")
    axes[1, 0].set_xlabel("Support density (%)")
    axes[1, 1].set_xlabel("Support density (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Taylor-trait control-mask transfer into the same WS adapter", y=1.02)
    figure.savefig(output / "control_mask_transfer_sl_gap.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "control_mask_transfer_sl_gap.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
