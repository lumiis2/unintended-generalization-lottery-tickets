#!/usr/bin/env python3
"""Compute data-free magnitude scores for entry, feature, or rank LoRA units."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.lora_masks import magnitude_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--granularity", choices=["entry", "feature", "rank"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = load_file(str(args.source_adapter / "adapter_model.safetensors"), device="cpu")
    scores = magnitude_scores(state, args.granularity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": {key: value.cpu() for key, value in scores.items()},
            "metadata": {
                "method": "magnitude",
                "granularity": args.granularity,
                "source_adapter": str(args.source_adapter),
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
