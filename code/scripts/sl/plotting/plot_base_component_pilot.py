#!/usr/bin/env python3
"""Plot the cross-seed base-component activation-gating pilot for cats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("cat-48", "artifacts/sl/qwen25_7b_cat_replication_seed48"),
    ("cat-49", "artifacts/sl/qwen25_7b_cat_replication_seed49"),
)
VARIANTS = ("full", "attn_candidate_gate", "attn_random_gate", "mlp_candidate_gate", "mlp_random_gate")
LABELS = {
    "full": "Full",
    "attn_candidate_gate": "Attention\ncandidate",
    "attn_random_gate": "Attention\nrandom",
    "mlp_candidate_gate": "MLP\ncandidate",
    "mlp_random_gate": "MLP\nrandom",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sl/mask_analysis/base_component_pilot"))
    args = parser.parse_args()
    output = args.repo_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run, relative_root in RUNS:
        root = args.repo_root / relative_root / "base_component_localization_v1" / "evaluation"
        for variant in VARIANTS:
            ws = load(root / "trait" / f"{variant}.json")
            wc = load(root / "control" / f"{variant}.json")
            rows.append({
                "run": run,
                "variant": variant,
                "ws_rate_pp": 100 * ws["trait"]["target_rate"],
                "wc_rate_pp": 100 * wc["trait"]["target_rate"],
                "effect_pp": 100 * (ws["trait"]["target_rate"] - wc["trait"]["target_rate"]),
                "numeric_ws": 100 * ws["numeric"]["valid_numeric_completion_rate"],
                "numeric_wc": 100 * wc["numeric"]["valid_numeric_completion_rate"],
            })
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    def value(run: str, variant: str) -> float:
        return next(row["effect_pp"] for row in rows if row["run"] == run and row["variant"] == variant)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    figure.subplots_adjust(left=.08, right=.98, bottom=.18, top=.78, wspace=.27)
    colors = {"cat-48": "#1f77b4", "cat-49": "#ff7f0e"}
    x = np.arange(len(VARIANTS))
    for run, _root in RUNS:
        effects = [value(run, variant) for variant in VARIANTS]
        axes[0].plot(x, effects, marker="o", linewidth=2, markersize=7, color=colors[run], label=run)
    axes[0].set_xticks(x, [LABELS[variant] for variant in VARIANTS])
    axes[0].set_ylabel("SL effect: WS − WC (percentage points)")
    axes[0].set_title("Raw paired SL effect under each gate")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False)

    components = ("attention", "MLP")
    positions = np.arange(len(components))
    width = .32
    for offset, (run, _root) in zip((-width / 2, width / 2), RUNS, strict=True):
        deltas = []
        for component in components:
            candidate = value(run, f"{'attn' if component == 'attention' else 'mlp'}_candidate_gate")
            random = value(run, f"{'attn' if component == 'attention' else 'mlp'}_random_gate")
            deltas.append(candidate - random)
        axes[1].bar(positions + offset, deltas, width, color=colors[run], label=run)
        for xpos, delta in zip(positions + offset, deltas, strict=True):
            axes[1].text(xpos, delta - .45 if delta < 0 else delta + .2, f"{delta:+.1f}", ha="center", va="top" if delta < 0 else "bottom", fontsize=9)
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].set_xticks(positions, ["Attention output", "MLP output"])
    axes[1].set_ylabel("Candidate gate − matched random gate (pp)")
    axes[1].set_title("Selective reduction relative to random gating")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False)
    figure.suptitle("Cross-seed base-component activation-gating pilot (layer 28, 128 channels)", y=.98)
    figure.savefig(output / "base_component_cross_seed_pilot.png", dpi=220)
    figure.savefig(output / "base_component_cross_seed_pilot.pdf")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
