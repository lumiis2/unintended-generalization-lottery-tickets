#!/usr/bin/env python3
"""Create a masked copy of a PEFT LoRA adapter without modifying its source."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from spar.sl.config import write_json


def lora_weight_keys(state: dict[str, torch.Tensor]) -> list[str]:
    keys = [key for key in state if key.endswith(("lora_A.weight", "lora_B.weight"))]
    if not keys:
        raise ValueError("The adapter contains no LoRA A/B weight tensors")
    return sorted(keys)


def keep_masks(
    state: dict[str, torch.Tensor], keys: list[str], method: str, sparsity: float, seed: int
) -> dict[str, torch.Tensor]:
    """Return boolean masks over individual A/B adapter entries.

    ``magnitude`` retains the largest ``1 - sparsity`` fraction globally. ``random``
    retains the same count at random. ``complement`` retains the entries excluded by
    the magnitude mask; it is intentionally not density-matched.
    """
    sizes = [state[key].numel() for key in keys]
    total = sum(sizes)
    pruned = int(total * sparsity)
    if not 0 <= pruned <= total:
        raise ValueError("sparsity must be in [0, 1]")

    if method in {"magnitude", "complement"}:
        scores = torch.cat([state[key].detach().float().abs().flatten() for key in keys])
        flat_keep = torch.ones(total, dtype=torch.bool)
        if pruned:
            flat_keep[torch.argsort(scores, stable=True)[:pruned]] = False
        if method == "complement":
            flat_keep = ~flat_keep
    elif method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        flat_keep = torch.zeros(total, dtype=torch.bool)
        retained = total - pruned
        if retained:
            flat_keep[torch.randperm(total, generator=generator)[:retained]] = True
    else:
        raise ValueError(f"Unknown method: {method}")

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for key, size in zip(keys, sizes, strict=True):
        masks[key] = flat_keep[offset : offset + size].reshape(state[key].shape)
        offset += size
    return masks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", required=True, type=Path)
    parser.add_argument("--output-adapter", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["magnitude", "random", "complement"])
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_weights = args.source_adapter / "adapter_model.safetensors"
    source_config = args.source_adapter / "adapter_config.json"
    if not source_weights.is_file() or not source_config.is_file():
        raise FileNotFoundError("Expected adapter_model.safetensors and adapter_config.json")
    if args.output_adapter.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_adapter}")

    state = load_file(str(source_weights), device="cpu")
    keys = lora_weight_keys(state)
    masks = keep_masks(state, keys, args.method, args.sparsity, args.seed)
    masked = dict(state)
    for key, mask in masks.items():
        masked[key] = state[key] * mask.to(dtype=state[key].dtype)

    args.output_adapter.mkdir(parents=True)
    shutil.copy2(source_config, args.output_adapter / source_config.name)
    save_file(masked, str(args.output_adapter / source_weights.name))
    total = sum(mask.numel() for mask in masks.values())
    retained = sum(int(mask.sum()) for mask in masks.values())
    write_json(
        args.output_adapter / "mask_metadata.json",
        {
            "source_adapter": str(args.source_adapter),
            "source_adapter_sha256": sha256(source_weights),
            "method": args.method,
            "requested_sparsity": args.sparsity,
            "mask_seed": args.seed,
            "masked_tensor_keys": keys,
            "masked_entries": total,
            "retained_entries": retained,
            "retained_fraction": retained / total,
        },
    )


if __name__ == "__main__":
    main()
