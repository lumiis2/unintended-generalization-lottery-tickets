#!/usr/bin/env python3
"""Materialize a structural LoRA mask from saved importance scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from spar.sl.config import write_json
from spar.sl.lora_masks import (
    complement_unit_mask,
    mask_stats,
    masked_state,
    parameter_mask,
    unit_mask_from_scores,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--complement", action="store_true")
    args = parser.parse_args()
    if args.output_adapter.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_adapter}")

    weights_path = args.source_adapter / "adapter_model.safetensors"
    state = load_file(str(weights_path), device="cpu")
    payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    granularity = payload["metadata"]["granularity"]
    scores = payload["scores"]
    unit_mask = unit_mask_from_scores(scores, args.sparsity, random_seed=args.random_seed)
    if args.complement:
        unit_mask = complement_unit_mask(unit_mask)
    parameter = parameter_mask(state, unit_mask, granularity)

    args.output_adapter.mkdir(parents=True)
    shutil.copy2(args.source_adapter / "adapter_config.json", args.output_adapter / "adapter_config.json")
    save_file(masked_state(state, parameter), str(args.output_adapter / "adapter_model.safetensors"))
    write_json(
        args.output_adapter / "mask_metadata.json",
        {
            "source_adapter": str(args.source_adapter),
            "source_adapter_sha256": file_sha256(weights_path),
            "score_artifact": str(args.scores),
            "score_metadata": payload["metadata"],
            "sparsity": args.sparsity,
            "random_seed": args.random_seed,
            "complement": args.complement,
            **mask_stats(parameter),
        },
    )


if __name__ == "__main__":
    main()
