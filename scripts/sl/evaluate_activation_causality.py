#!/usr/bin/env python3
"""Causal residual-stream tests for the matched-prompt SL activation direction.

This is a first causal pilot, not a localization result.  It uses a direction
estimated from the teacher trait-system minus neutral-system comparison and
compares its projection removal/steering to a neutral-system direction and
three norm-matched random directions.  The hook acts only on the final prompt
token during prefill, so it changes the next-token computation rather than
repeatedly steering every generated token.
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

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion


FINAL_TRAIT_PROMPTS = Path("experiments/subliminal_learning/final_evaluation/v1/trait_prompts.jsonl")


@dataclass(frozen=True)
class Intervention:
    name: str
    kind: str  # none, project_out, steer
    vector_name: str | None = None
    random_seed: int | None = None
    scale_fraction: float | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_trait_prompts() -> list[dict]:
    return read_jsonl(FINAL_TRAIT_PROMPTS)


def held_out_numeric(config: dict, condition: str, count: int, seed: int) -> list[dict]:
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    trained = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    candidates = [
        row for row in read_jsonl(root / "raw.jsonl")
        if row["parse"]["valid"] and row["attempt_id"] not in trained
    ]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} held-out numeric examples are available")
    return candidates[:count]


def decoder_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    return layers


class FinalPromptResidualIntervention:
    """Modify only the final prompt token at the output of one decoder layer."""

    def __init__(self, model, hidden_state_index: int, vector: torch.Tensor | None, kind: str, scale: float = 0.0):
        if hidden_state_index <= 0:
            raise ValueError("Layer 0 is the embedding state and cannot be hooked as a decoder output")
        layers = decoder_layers(model)
        if hidden_state_index > len(layers):
            raise ValueError(f"Layer {hidden_state_index} outside 1..{len(layers)}")
        self.module = layers[hidden_state_index - 1]
        self.vector = vector
        self.kind = kind
        self.scale = scale
        self.handle = None

    def __enter__(self):
        if self.kind == "none":
            return self
        assert self.vector is not None

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Initial prefill has the whole prompt.  Cached decoding calls have
            # sequence length one and must remain unmodified.
            if hidden.shape[1] <= 1:
                return output
            vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
            unit = vector / vector.float().norm().clamp_min(1e-12).to(hidden.dtype)
            updated = hidden.clone()
            token = updated[:, -1, :]
            if self.kind == "project_out":
                token = token - (token * unit).sum(dim=-1, keepdim=True) * unit
            elif self.kind == "steer":
                token = token + self.scale * unit
            else:
                raise ValueError(f"Unknown intervention kind: {self.kind}")
            updated[:, -1, :] = token
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()


def intervention_vector(intervention: Intervention, directions: dict[str, torch.Tensor], *, seed: int) -> torch.Tensor | None:
    if intervention.kind == "none":
        return None
    if intervention.random_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed + intervention.random_seed)
        return torch.randn(directions["trait_system_minus_neutral_system"].shape, generator=generator)
    assert intervention.vector_name is not None
    return directions[intervention.vector_name]


def evaluate_preference(model, tokenizer, config: dict, prompts: list[dict], samples: int, seed: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    expanded = [row for row in prompts for _ in range(samples)]
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in expanded], config["evaluation"])
    target = config["trait"]["target"].casefold()
    pattern = re.compile(rf"\b{re.escape(target)}s?\b", re.I)
    rows = [
        {"prompt_id": row["id"], "completion": output, "target_mentioned": bool(pattern.search(output))}
        for row, output in zip(expanded, outputs, strict=True)
    ]
    hits = sum(row["target_mentioned"] for row in rows)
    return {"n": len(rows), "hits": hits, "target": target, "target_rate": hits / len(rows), "samples_per_prompt": samples}, rows


def evaluate_numeric(model, tokenizer, config: dict, rows: list[dict], seed: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], config["generation"])
    result = []
    for row, output in zip(rows, outputs, strict=True):
        parsed = parse_number_completion(
            output,
            minimum=config["data"]["integer_min"],
            maximum=config["data"]["integer_max"],
            max_values=config["data"]["max_added_values"],
        )
        result.append({"source_attempt_id": row["attempt_id"], "completion": output, "valid_numeric_completion": parsed.valid})
    valid = sum(row["valid_numeric_completion"] for row in result)
    return {"n": len(result), "valid_numeric_completion_rate": valid / len(result)}, result


def interventions() -> list[Intervention]:
    return [
        Intervention("full", "none"),
        Intervention("teacher_direction_project_out", "project_out", "trait_system_minus_neutral_system"),
        Intervention("neutral_system_direction_project_out", "project_out", "neutral_system_minus_no_system"),
        *(Intervention(f"random_direction_project_out_r{index}", "project_out", random_seed=index) for index in range(3)),
        Intervention("teacher_direction_steer_plus_half", "steer", "trait_system_minus_neutral_system", scale_fraction=0.5),
        Intervention("teacher_direction_steer_minus_half", "steer", "trait_system_minus_neutral_system", scale_fraction=-0.5),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["trait", "control"])
    parser.add_argument("--layer", required=True, type=int, help="Hidden-state index; hook is applied after decoder layer index-1.")
    parser.add_argument("--trait-samples", type=int, default=10)
    parser.add_argument("--numeric-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    directions_path = root / "activation_analysis_system_prompt_control" / "directions.pt"
    payload = torch.load(directions_path, map_location="cpu", weights_only=False)
    directions = {key: payload[key].float()[args.layer] for key in (
        "trait_system_minus_neutral_system",
        "neutral_system_minus_no_system",
    )}
    trait_rows = final_trait_prompts()
    numeric_rows = held_out_numeric(config, args.condition, args.numeric_examples, args.seed + 99)
    output = root / "activation_causality_v1" / args.condition
    adapter = root / "models" / args.condition / "adapter"
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(adapter)).eval()
    try:
        for spec in interventions():
            summary_path = output / f"{spec.name}.json"
            samples_path = output / f"{spec.name}.jsonl"
            if summary_path.is_file() and samples_path.is_file():
                print(f"SKIP {spec.name}", flush=True)
                continue
            vector = intervention_vector(spec, directions, seed=args.seed)
            scale = 0.0 if spec.scale_fraction is None or vector is None else spec.scale_fraction * float(vector.norm())
            with FinalPromptResidualIntervention(model, args.layer, vector, spec.kind, scale):
                trait, trait_samples = evaluate_preference(model, tokenizer, config, trait_rows, args.trait_samples, args.seed)
                numeric, numeric_samples = evaluate_numeric(model, tokenizer, config, numeric_rows, args.seed + 1)
            write_jsonl(samples_path, [{"kind": "trait", **row} for row in trait_samples] + [{"kind": "numeric", **row} for row in numeric_samples])
            write_json(summary_path, {
                "config": args.config,
                "config_sha256": config["_config_sha256"],
                "condition": args.condition,
                "layer": args.layer,
                "intervention": spec.name,
                "kind": spec.kind,
                "vector_name": spec.vector_name,
                "random_seed": spec.random_seed,
                "scale_fraction_of_direction_norm": spec.scale_fraction,
                "scale": scale,
                "direction_norm": None if vector is None else float(vector.norm()),
                "direction_source": str(directions_path),
                "trait_prompt_file": str(FINAL_TRAIT_PROMPTS),
                "trait_prompt_sha256": sha256(FINAL_TRAIT_PROMPTS),
                "trait": trait,
                "numeric": numeric,
                "notes": "Final-prompt-token prefill intervention only; random directions are matched in dimensionality and normalized before projection/steering.",
            })
            print(f"DONE {spec.name}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
