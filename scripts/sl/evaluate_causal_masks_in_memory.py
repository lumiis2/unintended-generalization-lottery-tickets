#!/usr/bin/env python3
"""Evaluate many structural LoRA masks without repeatedly reloading Qwen.

The model is loaded once for one trained adapter (WS or WC).  Each candidate
mask is copied into the active LoRA parameters, evaluated on trait preference
and held-out numeric completion, then replaced by the next mask.  We save
metrics and complete provenance, rather than hundreds of duplicate adapters.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.lora_masks import (
    complement_unit_mask,
    mask_stats,
    parameter_mask,
    unit_mask_from_scores,
)
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion
from spar.sl.prompts import evaluation_prompts


METHODS = ("magnitude_feature", "wanda_feature", "taylor_trait_feature", "taylor_contrastive_feature")
BRANCHES = ("selected_support", "random_support", "selected_ablation", "random_ablation")


@dataclass(frozen=True)
class MaskSpec:
    method: str
    density: float
    branch: str
    score_path: Path
    random_seed: int

    @property
    def density_label(self) -> str:
        return f"d{int(round(100 * self.density)):02d}"

    @property
    def evaluation_id(self) -> str:
        return f"causal-{self.method}-{self.density_label}-{self.branch}"


def canonical_key(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def pending(root: Path, spec: MaskSpec) -> bool:
    return not (
        (root / "evaluations" / spec.evaluation_id / "summary.json").is_file()
        and (root / "evaluations" / "numeric_task" / spec.evaluation_id / "summary.json").is_file()
        and (root / "mask_metadata" / "causal_screen" / spec.evaluation_id / "metadata.json").is_file()
    )


def preference_summary(model, tokenizer, config: dict, condition: str, samples_per_prompt: int) -> dict:
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    rows = evaluation_prompts(config, int(config["evaluation"].get("number_prefix_seed", seed + 10_000)))
    expanded = [row for row in rows for _ in range(samples_per_prompt)]
    completions = batched_generate(
        model,
        tokenizer,
        [render_chat(tokenizer, row["prompt"]) for row in expanded],
        config["evaluation"],
    )
    target = config["trait"]["target"].casefold()
    pattern = re.compile(rf"\b{re.escape(target)}s?\b", flags=re.IGNORECASE)
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row, completion in zip(expanded, completions, strict=True):
        grouped[row["variant"]].append(bool(pattern.search(completion)))
    successes = sum(sum(values) for values in grouped.values())
    total = sum(len(values) for values in grouped.values())
    return {
        "condition": condition,
        "target": target,
        "n": total,
        "target_rate": successes / total,
        "target_rate_wilson_95": wilson_interval(successes, total),
        "by_variant": {
            name: {
                "n": len(values),
                "target_rate": sum(values) / len(values),
                "target_rate_wilson_95": wilson_interval(sum(values), len(values)),
            }
            for name, values in grouped.items()
        },
    }


def numeric_rows(config: dict, condition: str, count: int, seed: int) -> list[dict]:
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    selected_ids = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    candidates = [
        row
        for row in read_jsonl(root / "raw.jsonl")
        if row["parse"]["valid"] and row["attempt_id"] not in selected_ids
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} held-out valid numeric examples are available")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def numeric_summary(model, tokenizer, config: dict, condition: str, rows: list[dict], generation_seed: int) -> dict:
    torch.manual_seed(generation_seed)
    completions = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], config["generation"])
    valid = []
    for completion in completions:
        valid.append(
            parse_number_completion(
                completion,
                minimum=config["data"]["integer_min"],
                maximum=config["data"]["integer_max"],
                max_values=config["data"]["max_added_values"],
            ).valid
        )
    return {
        "condition": condition,
        "n": len(rows),
        "valid_numeric_completion_rate": sum(valid) / len(valid),
        "sampling_seed": generation_seed,
    }


def make_parameter_mask(state: dict[str, torch.Tensor], spec: MaskSpec) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(spec.score_path, map_location="cpu", weights_only=False)
    if payload["metadata"]["granularity"] != "feature":
        raise ValueError(f"Expected feature-level score, got {payload['metadata']['granularity']}")
    units = unit_mask_from_scores(payload["scores"], 1.0 - spec.density, random_seed=(spec.random_seed if "random" in spec.branch else None))
    if "ablation" in spec.branch:
        units = complement_unit_mask(units)
    return parameter_mask(state, units, "feature"), payload["metadata"]


def apply_mask(params: dict[str, torch.nn.Parameter], state: dict[str, torch.Tensor], mask: dict[str, torch.Tensor]) -> None:
    missing = sorted(set(mask) - set(params))
    if missing:
        raise KeyError(f"Active PEFT adapter lacks parameters: {missing[:3]}")
    with torch.no_grad():
        for key, keep in mask.items():
            parameter = params[key]
            parameter.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype) * keep.to(device=parameter.device, dtype=parameter.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["trait", "control"], required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=20)
    parser.add_argument("--held-out-examples", type=int, default=128)
    parser.add_argument("--evaluation-data-seed", type=int, default=20261005)
    parser.add_argument("--mask-seed-base", type=int, default=20261005)
    parser.add_argument("--support-densities", type=float, nargs="+", default=[0.10, 0.30, 0.50])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES))
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    specs = [
        MaskSpec(method, density, branch, root / "mask_scores" / args.condition / f"{method}.pt", args.mask_seed_base + int(round(100 * density)))
        for method in args.methods
        for density in args.support_densities
        for branch in args.branches
    ]
    specs = [spec for spec in specs if pending(root, spec)]
    if not specs:
        print(f"No pending masks for {args.config} / {args.condition}")
        return
    for spec in specs:
        if not spec.score_path.is_file():
            raise FileNotFoundError(spec.score_path)

    adapter = root / "models" / args.condition / "adapter"
    weights_path = adapter / "adapter_model.safetensors"
    source_adapter_sha256 = sha256(weights_path)
    state = load_file(str(weights_path), device="cpu")
    tokenizer = load_tokenizer(config)
    model = None
    try:
        model = load_model(config, adapter=str(adapter)).eval()
        params = {
            canonical_key(name): parameter
            for name, parameter in model.named_parameters()
            if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name
        }
        held_out = numeric_rows(config, args.condition, args.held_out_examples, args.evaluation_data_seed)
        for index, spec in enumerate(specs, start=1):
            mask, score_metadata = make_parameter_mask(state, spec)
            apply_mask(params, state, mask)
            preference = preference_summary(model, tokenizer, config, args.condition, args.samples_per_prompt)
            numeric = numeric_summary(model, tokenizer, config, args.condition, held_out, args.evaluation_data_seed)
            common = {
                "config": args.config,
                "config_sha256": config["_config_sha256"],
                "evaluation_id": spec.evaluation_id,
                "source_adapter": str(adapter),
                "source_adapter_sha256": source_adapter_sha256,
                "score_artifact": str(spec.score_path),
                "score_metadata": score_metadata,
                "support_density": spec.density,
                "branch": spec.branch,
                "random_seed": spec.random_seed if "random" in spec.branch else None,
                "evaluation_data_seed": args.evaluation_data_seed,
                "samples_per_prompt": args.samples_per_prompt,
                **mask_stats(mask),
            }
            write_json(root / "evaluations" / spec.evaluation_id / "summary.json", {**common, **preference})
            write_json(root / "evaluations" / "numeric_task" / spec.evaluation_id / "summary.json", {**common, **numeric})
            write_json(root / "mask_metadata" / "causal_screen" / spec.evaluation_id / "metadata.json", common)
            print(f"DONE {index}/{len(specs)} {args.condition} {spec.evaluation_id}", flush=True)
    finally:
        if model is not None:
            del model
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
