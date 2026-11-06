#!/usr/bin/env python3
"""Test whether cat Subliminal Learning depends on the base, LoRA, or both.

This is a *screening* factorial.  It keeps the base-weight checkpoint intact
and uses the pre-registered cat-47 MLP-output channel set as a reversible base
gate.  LoRA states are full, selected-support, and selected-ablation (plus
equal-density random controls) for Magnitude and Taylor-contrastive masks.

The key outcome is the paired SL effect ``WS - WC``.  The same frozen trait
prompts and held-out numerical sequences are used in every cell.  It is not a
global merged-weight pruning experiment: merging ``W_base + BA`` would mostly
rank the much larger base weights and would not identify a base contribution to
SL.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.gates import FinalPromptChannelGate
from spar.sl.lora_masks import complement_unit_mask, parameter_mask, unit_mask_from_scores
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion


FROZEN_ROOT = Path("experiments/subliminal_learning/frozen_evaluation/taylor_numeric_v2")
SELECTION = Path("artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/selection.pt")
LAYER = 28
METHODS = ("magnitude_feature", "taylor_contrastive_feature")


@dataclass(frozen=True)
class LoraState:
    name: str
    method: str | None
    branch: str
    random_seed: int | None = None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_key(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def frozen_assets() -> tuple[dict, list[dict]]:
    manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    trait_path = FROZEN_ROOT / manifest["trait_prompts"]["path"]
    if digest(trait_path) != manifest["trait_prompts"]["sha256"]:
        raise ValueError(f"Frozen trait prompts failed digest check: {trait_path}")
    return manifest, read_jsonl(trait_path)


def held_out_numeric(config: dict, condition: str, *, count: int, seed: int, offset: int) -> list[dict]:
    dataset = Path(config["experiment"]["output_root"]) / "datasets" / condition
    trained = {row["source_attempt_id"] for row in read_jsonl(dataset / "train.jsonl")}
    candidates = [row for row in read_jsonl(dataset / "raw.jsonl") if row["parse"]["valid"] and row["attempt_id"] not in trained]
    random.Random(seed).shuffle(candidates)
    return candidates[offset : offset + count]


def preference(model, tokenizer, config: dict, rows: list[dict], *, samples: int, system: str | None = None) -> dict:
    torch.manual_seed(int(config["experiment"]["seed"]) + 803)
    expanded = [row for row in rows for _ in range(samples)]
    prompts = [render_chat(tokenizer, row["prompt"], system) for row in expanded]
    outputs = batched_generate(model, tokenizer, prompts, config["evaluation"])
    target = config["trait"]["target"].casefold()
    hits = sum(bool(re.search(rf"\b{re.escape(target)}s?\b", output, re.I)) for output in outputs)
    return {"n": len(outputs), "hits": hits, "target_rate": hits / len(outputs), "samples_per_prompt": samples}


def numeric(model, tokenizer, config: dict, rows: list[dict]) -> dict:
    torch.manual_seed(int(config["experiment"]["seed"]) + 804)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], config["generation"])
    valid = [parse_number_completion(
        output,
        minimum=config["data"]["integer_min"], maximum=config["data"]["integer_max"], max_values=config["data"]["max_added_values"],
    ).valid for output in outputs]
    return {"n": len(valid), "valid_numeric_completion_rate": sum(valid) / len(valid)}


def lora_states() -> list[LoraState]:
    result = [LoraState("full", None, "full")]
    for method_index, method in enumerate(METHODS):
        for branch in ("selected_support", "selected_ablation", "random_support", "random_ablation"):
            random_seed = 20261101 + method_index if branch.startswith("random") else None
            result.append(LoraState(f"{method.removesuffix('_feature')}_{branch}", method, branch, random_seed))
    return result


def mask_for(state: dict[str, torch.Tensor], root: Path, condition: str, variant: LoraState) -> dict[str, torch.Tensor] | None:
    if variant.method is None:
        return None
    scored = torch.load(root / "mask_scores" / condition / f"{variant.method}.pt", map_location="cpu", weights_only=False)
    units = unit_mask_from_scores(scored["scores"], 0.70, random_seed=variant.random_seed)
    if variant.branch.endswith("ablation"):
        units = complement_unit_mask(units)
    return parameter_mask(state, units, "feature")


def apply_mask(params: dict[str, torch.nn.Parameter], state: dict[str, torch.Tensor], mask: dict[str, torch.Tensor] | None) -> None:
    with torch.no_grad():
        for key, parameter in params.items():
            keep = torch.ones_like(state[key], dtype=torch.bool) if mask is None else mask[key]
            parameter.copy_(state[key].to(parameter.device, parameter.dtype) * keep.to(parameter.device, parameter.dtype))


def shared_teacher(args) -> None:
    config = load_config(args.config)
    manifest, prompts = frozen_assets()
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    mlp = selection["components"]["mlp"]
    gates = [("intact", None), ("mlp_candidate_gate", mlp["selected"]), ("mlp_random_gate", mlp["random_matched"])]
    systems = [
        ("trait_system", config["trait"]["teacher_system_prompt"].format(target=config["trait"]["target"])),
        ("neutral_system", config["trait"]["neutral_system_prompt"].format(target=config["trait"]["target"])),
    ]
    root = Path(config["experiment"]["output_root"])
    output_root = root / "base_lora_factorial_v1" / "shared_teacher"
    tokenizer = load_tokenizer(config)
    model = load_model(config).eval()
    try:
        for system_name, system in systems:
            for gate_name, indices in gates:
                output = output_root / f"{system_name}__{gate_name}.json"
                if output.exists() and not args.overwrite:
                    print(f"SKIP {output}", flush=True)
                    continue
                gate = contextlib.nullcontext() if indices is None else FinalPromptChannelGate(model, layer=args.layer, component="mlp", indices=indices)
                with gate:
                    trait = preference(model, tokenizer, config, prompts, samples=args.samples, system=system)
                write_json(output, {
                    "experiment": "base_lora_factorial_v1_shared_teacher", "config": args.config,
                    "frozen_manifest": str(FROZEN_ROOT / "manifest.json"), "manifest_version": manifest["version"],
                    "selection": str(args.selection), "selection_sha256": digest(args.selection), "layer": args.layer,
                    "system_condition": system_name, "base_gate": gate_name, "trait": trait,
                    "notes": "Prompted W0 teacher; reversible final-prompt MLP output-channel gate only.",
                })
                print(f"DONE {system_name} {gate_name}", flush=True)
    finally:
        release(model)


def factorial(args) -> None:
    config = load_config(args.config)
    manifest, prompts = frozen_assets()
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    mlp = selection["components"]["mlp"]
    gates = [("intact", None), ("mlp_candidate_gate", mlp["selected"]), ("mlp_random_gate", mlp["random_matched"])]
    root = Path(config["experiment"]["output_root"])
    adapter = root / "models" / args.condition / "adapter"
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    numeric_rows = held_out_numeric(config, args.condition, count=args.numeric_examples, seed=manifest["numeric_slice"]["evaluation_data_seed"], offset=manifest["numeric_slice"]["offset"])
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(adapter)).eval()
    params = {canonical_key(name): parameter for name, parameter in model.named_parameters() if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name}
    try:
        for lora in lora_states():
            mask = mask_for(state, root, args.condition, lora)
            for gate_name, indices in gates:
                output = root / "base_lora_factorial_v1" / "factorial" / args.condition / f"{lora.name}__{gate_name}.json"
                if output.exists() and not args.overwrite:
                    print(f"SKIP {output}", flush=True)
                    continue
                apply_mask(params, state, mask)
                gate = contextlib.nullcontext() if indices is None else FinalPromptChannelGate(model, layer=args.layer, component="mlp", indices=indices)
                with gate:
                    trait = preference(model, tokenizer, config, prompts, samples=args.samples)
                    numeric_result = numeric(model, tokenizer, config, numeric_rows)
                write_json(output, {
                    "experiment": "base_lora_factorial_v1", "config": args.config, "condition": args.condition,
                    "frozen_manifest": str(FROZEN_ROOT / "manifest.json"), "manifest_version": manifest["version"],
                    "selection": str(args.selection), "selection_sha256": digest(args.selection), "layer": args.layer,
                    "lora_state": lora.name, "lora_method": lora.method, "lora_branch": lora.branch,
                    "lora_random_seed": lora.random_seed, "base_gate": gate_name,
                    "trait": trait, "numeric": numeric_result,
                    "notes": "Frozen evaluation. Base gate changes final-prompt MLP activations only; adapter mask changes LoRA A feature columns only.",
                })
                print(f"DONE {args.condition} {lora.name} {gate_name}", flush=True)
    finally:
        del state
        release(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("shared_teacher", "factorial"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=("trait", "control"))
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--samples", type=int, default=5, help="Samples per frozen trait prompt for the screening factorial.")
    parser.add_argument("--numeric-examples", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "shared_teacher":
        shared_teacher(args)
    else:
        if args.condition is None:
            parser.error("factorial requires --condition")
        factorial(args)


if __name__ == "__main__":
    main()
