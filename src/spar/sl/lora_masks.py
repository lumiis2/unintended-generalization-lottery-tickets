"""Mask construction for LoRA adapters.

The module deliberately separates *scoring* a unit from *masking* it.  A score
may come from magnitude, activations (Wanda), or a behaviour-specific gradient
(Taylor), while the causal controls always use the identical structural unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Granularity = Literal["entry", "feature", "rank"]


@dataclass(frozen=True)
class LoraPair:
    path: str
    a_key: str
    b_key: str


def lora_pairs(state: dict[str, torch.Tensor]) -> list[LoraPair]:
    """Find compatible A/B matrices in a PEFT adapter state dict."""
    pairs = []
    for a_key in sorted(key for key in state if key.endswith("lora_A.weight")):
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in state:
            raise ValueError(f"Missing matching B matrix for {a_key}")
        a, b = state[a_key], state[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Incompatible LoRA pair: {a_key} {tuple(a.shape)}, {b_key} {tuple(b.shape)}")
        pairs.append(LoraPair(a_key.removesuffix(".lora_A.weight"), a_key, b_key))
    if not pairs:
        raise ValueError("No LoRA A/B pairs found")
    return pairs


def magnitude_scores(state: dict[str, torch.Tensor], granularity: Granularity) -> dict[str, torch.Tensor]:
    """Return one non-negative score tensor per structural unit.

    Feature scores are norms of columns of the effective update ``B @ A``;
    rank scores are norms of its rank-one summands.  Thus neither depends on
    an arbitrary rescaling between LoRA A and B.
    """
    result: dict[str, torch.Tensor] = {}
    for pair in lora_pairs(state):
        a, b = state[pair.a_key].float(), state[pair.b_key].float()
        if granularity == "entry":
            result[pair.a_key] = a.abs()
            result[pair.b_key] = b.abs()
        elif granularity == "feature":
            # ||(B @ A)[:, j]||_2 without materialising the full update.
            result[pair.path] = (a * ((b.T @ b) @ a)).sum(dim=0).clamp_min(0).sqrt()
        elif granularity == "rank":
            result[pair.path] = a.norm(dim=1) * b.norm(dim=0)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
    return result


def _flat_select(scores: dict[str, torch.Tensor], keep_fraction: float, seed: int | None) -> dict[str, torch.Tensor]:
    if not 0 <= keep_fraction <= 1:
        raise ValueError("keep_fraction must be in [0, 1]")
    keys = sorted(scores)
    sizes = [scores[key].numel() for key in keys]
    total = sum(sizes)
    keep = int(round(total * keep_fraction))
    flat = torch.zeros(total, dtype=torch.bool)
    if keep:
        if seed is None:
            values = torch.cat([scores[key].detach().float().flatten() for key in keys])
            flat[torch.argsort(values, descending=True, stable=True)[:keep]] = True
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            flat[torch.randperm(total, generator=generator)[:keep]] = True
    output, offset = {}, 0
    for key, size in zip(keys, sizes, strict=True):
        output[key] = flat[offset : offset + size].reshape(scores[key].shape)
        offset += size
    return output


def unit_mask_from_scores(
    scores: dict[str, torch.Tensor], sparsity: float, *, random_seed: int | None = None
) -> dict[str, torch.Tensor]:
    """Select top-scoring units, or random units when ``random_seed`` is set."""
    return _flat_select(scores, 1.0 - sparsity, random_seed)


def parameter_mask(
    state: dict[str, torch.Tensor], unit_mask: dict[str, torch.Tensor], granularity: Granularity
) -> dict[str, torch.Tensor]:
    """Expand structural-unit masks to bool masks for LoRA A/B parameters."""
    output: dict[str, torch.Tensor] = {}
    for pair in lora_pairs(state):
        a, b = state[pair.a_key], state[pair.b_key]
        if granularity == "entry":
            output[pair.a_key] = unit_mask[pair.a_key]
            output[pair.b_key] = unit_mask[pair.b_key]
        elif granularity == "feature":
            keep = unit_mask[pair.path]
            output[pair.a_key] = keep.unsqueeze(0).expand_as(a)
            output[pair.b_key] = torch.ones_like(b, dtype=torch.bool)
        elif granularity == "rank":
            keep = unit_mask[pair.path]
            output[pair.a_key] = keep.unsqueeze(1).expand_as(a)
            output[pair.b_key] = keep.unsqueeze(0).expand_as(b)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
    return output


def masked_state(state: dict[str, torch.Tensor], mask: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a copy with masked LoRA entries zeroed; leave metadata tensors intact."""
    result = dict(state)
    for key, keep in mask.items():
        result[key] = state[key] * keep.to(dtype=state[key].dtype)
    return result


def mask_stats(mask: dict[str, torch.Tensor]) -> dict[str, float | int]:
    total = sum(value.numel() for value in mask.values())
    kept = sum(int(value.sum()) for value in mask.values())
    return {"parameter_entries": total, "kept_entries": kept, "kept_fraction": kept / total}
