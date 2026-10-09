#!/usr/bin/env python3
"""Measure whether independently trained SL adapters select similar LoRA features.

This is descriptive: score overlap does not establish a shared causal mechanism.
It is useful evidence only alongside the causal mask interventions.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from spar.sl.config import write_json
from spar.sl.lora_masks import unit_mask_from_scores


RUNS = {
    "cat-47": "artifacts/sl/qwen25_7b_cat_replication",
    "cat-48": "artifacts/sl/qwen25_7b_cat_replication_seed48",
    "cat-49": "artifacts/sl/qwen25_7b_cat_replication_seed49",
    "owl-47": "artifacts/sl/qwen25_7b_owl_replication",
}


def selected_units(path: Path, density: float) -> set[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["granularity"] != "feature":
        raise ValueError(f"Expected feature scores in {path}, got {metadata['granularity']}")
    units = unit_mask_from_scores(payload["scores"], 1.0 - density)
    return {
        f"{key}:{index}"
        for key, mask in units.items()
        for index in torch.nonzero(mask.flatten(), as_tuple=False).flatten().tolist()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--score-root", default="mask_scores")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["magnitude_feature", "taylor_trait_feature", "taylor_contrastive_feature"],
    )
    parser.add_argument("--densities", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    parser.add_argument("--output", type=Path, default=Path("artifacts/sl/mask_analysis/stability.json"))
    args = parser.parse_args()

    comparisons = []
    for method in args.methods:
        for density in args.densities:
            selected = {
                run: selected_units(args.repo_root / root / args.score_root / "trait" / f"{method}.pt", density)
                for run, root in RUNS.items()
            }
            universe = len(next(iter(selected.values()))) / density
            expected_jaccard = density / (2.0 - density)
            for left, right in itertools.combinations(selected, 2):
                intersection = len(selected[left] & selected[right])
                union = len(selected[left] | selected[right])
                comparisons.append(
                    {
                        "method": method,
                        "density": density,
                        "left": left,
                        "right": right,
                        "selected_units_per_run": len(selected[left]),
                        "approximate_universe_units": round(universe),
                        "intersection": intersection,
                        "jaccard": intersection / union if union else 0.0,
                        "expected_random_jaccard": expected_jaccard,
                        "overlap_over_random": (intersection / union) / expected_jaccard if union else 0.0,
                    }
                )
    output = args.repo_root / args.output
    write_json(output, {"score_root": args.score_root, "comparisons": comparisons})
    print(f"Wrote {len(comparisons)} pairwise overlaps to {output}")


if __name__ == "__main__":
    main()
