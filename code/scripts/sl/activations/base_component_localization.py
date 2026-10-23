#!/usr/bin/env python3
"""Cross-seed base-component localization pilot for Subliminal Learning.

This is deliberately *not* global pruning of ``W_base + DeltaW``: base weights
are much larger than a LoRA update, so a merged magnitude score mostly ranks
the pretrained model.  Instead, it uses the clean teacher direction to choose
residual channels inside one frozen-base component, then tests whether gating
those channels selectively changes ``WS - WC`` on independently trained
students.  It is a causal base-dependence pilot, not yet a base-weight ticket.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import random
import re
from pathlib import Path

import torch

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion
from spar.sl.prompts import HELD_OUT_FAVORITE_ANIMAL_PROMPTS


FINAL_TRAIT_PROMPTS = Path("experiments/subliminal_learning/frozen_evaluation/v1/trait_prompts.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoder_layer(model, hidden_state_index: int):
    if hidden_state_index <= 0:
        raise ValueError("Layer 0 is the embedding state, not a decoder layer")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers[hidden_state_index - 1]


def component_module(model, layer: int, component: str):
    if component not in {"attn", "mlp"}:
        raise ValueError(component)
    return getattr(decoder_layer(model, layer), "self_attn" if component == "attn" else "mlp")


def final_prompt_component_states(model, tokenizer, rendered: list[str], *, layer: int, component: str, batch_size: int) -> torch.Tensor:
    """Capture the component's output at the final prompt token, on CPU."""
    module = component_module(model, layer, component)
    batches: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(rendered), batch_size):
                encoded = tokenizer(rendered[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
                captured.clear()
                model(**encoded, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one component activation per forward pass")
                positions = (encoded["attention_mask"] * torch.arange(encoded["input_ids"].shape[1], device=model.device)).argmax(dim=1)
                hidden = captured[0]
                batches.append(hidden[torch.arange(hidden.shape[0], device=model.device), positions].float().cpu())
    finally:
        handle.remove()
    return torch.cat(batches, dim=0)


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def selection_output(root: Path) -> Path:
    return root / "base_component_localization_v1" / "selection.pt"


def select(args) -> None:
    config = load_config(args.config)
    if config["trait"]["target"] != "cat":
        raise ValueError("The initial selector is frozen on cat-47 only")
    root = Path(config["experiment"]["output_root"])
    output = selection_output(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        print(f"SKIP existing {output}")
        return
    prompts = HELD_OUT_FAVORITE_ANIMAL_PROMPTS[: args.prompts]
    tokenizer = load_tokenizer(config)
    trait_system = config["trait"]["teacher_system_prompt"].format(target="cat")
    neutral_system = config["trait"]["neutral_system_prompt"].format(target="cat")
    no_system = [render_chat(tokenizer, prompt) for prompt in prompts]
    trait = [render_chat(tokenizer, prompt, trait_system) for prompt in prompts]
    neutral = [render_chat(tokenizer, prompt, neutral_system) for prompt in prompts]
    payload: dict[str, object] = {
        "version": 1,
        "selector_config": args.config,
        "selector_seed": config["experiment"]["seed"],
        "layer": args.layer,
        "n_prompts": len(prompts),
        "candidate_count": args.candidate_count,
        "random_seed": args.random_seed,
        "method": "abs(mean(teacher_trait-neutral) * mean(WS-WC)) per residual channel",
        "notes": (
            "Candidates are selected only from cat-47; cat-48/49 are the primary cross-seed evaluation. "
            "This gates base component output channels, therefore testing base-component dependence rather than pruning base weights."
        ),
        "components": {},
    }
    for component in ("attn", "mlp"):
        base = load_model(config).eval()
        teacher = final_prompt_component_states(base, tokenizer, trait, layer=args.layer, component=component, batch_size=args.batch_size)
        neutral_states = final_prompt_component_states(base, tokenizer, neutral, layer=args.layer, component=component, batch_size=args.batch_size)
        release(base)
        ws = load_model(config, adapter=str(root / "models" / "trait" / "adapter")).eval()
        student = final_prompt_component_states(ws, tokenizer, no_system, layer=args.layer, component=component, batch_size=args.batch_size)
        release(ws)
        wc = load_model(config, adapter=str(root / "models" / "control" / "adapter")).eval()
        control = final_prompt_component_states(wc, tokenizer, no_system, layer=args.layer, component=component, batch_size=args.batch_size)
        release(wc)
        teacher_delta = (teacher - neutral_states).mean(dim=0)
        student_delta = (student - control).mean(dim=0)
        score = (teacher_delta * student_delta).abs()
        selected = torch.topk(score, k=args.candidate_count, largest=True).indices.sort().values
        pool = torch.ones(score.numel(), dtype=torch.bool)
        pool[selected] = False
        generator = torch.Generator(device="cpu").manual_seed(args.random_seed + (0 if component == "attn" else 1))
        random_indices = torch.randperm(int(pool.sum()), generator=generator)[: args.candidate_count]
        random_selected = pool.nonzero(as_tuple=False).flatten()[random_indices].sort().values
        payload["components"][component] = {
            "score": score,
            "selected": selected,
            "random_matched": random_selected,
            "teacher_delta": teacher_delta,
            "student_delta": student_delta,
        }
    torch.save(payload, output)
    write_json(output.with_suffix(".json"), {
        key: value for key, value in payload.items() if key != "components"
    } | {"components": {name: {
        "selected_count": int(data["selected"].numel()),
        "selected_indices": data["selected"].tolist(),
        "random_matched_indices": data["random_matched"].tolist(),
    } for name, data in payload["components"].items()}})
    print(f"Wrote frozen base-component candidates to {output}")


class FinalPromptChannelGate:
    """Zero selected output channels of one frozen-base component at prefill only."""

    def __init__(self, model, *, layer: int, component: str, indices: torch.Tensor | None):
        self.module = component_module(model, layer, component)
        self.indices = indices
        self.handle = None

    def __enter__(self):
        if self.indices is None:
            return self

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 1:
                return output
            changed = hidden.clone()
            changed[:, -1, self.indices.to(hidden.device)] = 0
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()


def held_out_numeric(config: dict, condition: str, count: int, seed: int) -> list[dict]:
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    trained = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    candidates = [row for row in read_jsonl(root / "raw.jsonl") if row["parse"]["valid"] and row["attempt_id"] not in trained]
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def evaluate_preference(model, tokenizer, config: dict, prompts: list[dict], samples: int, seed: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    expanded = [row for row in prompts for _ in range(samples)]
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in expanded], config["evaluation"])
    pattern = re.compile(r"\bcats?\b", re.I)
    rows = [{"prompt_id": row["id"], "completion": output, "target_mentioned": bool(pattern.search(output))} for row, output in zip(expanded, outputs, strict=True)]
    return {"n": len(rows), "hits": sum(row["target_mentioned"] for row in rows), "target_rate": sum(row["target_mentioned"] for row in rows) / len(rows)}, rows


def evaluate_numeric(model, tokenizer, config: dict, rows: list[dict], seed: int) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], config["generation"])
    result = []
    for row, output in zip(rows, outputs, strict=True):
        parsed = parse_number_completion(output, minimum=config["data"]["integer_min"], maximum=config["data"]["integer_max"], max_values=config["data"]["max_added_values"])
        result.append({"source_attempt_id": row["attempt_id"], "valid_numeric_completion": parsed.valid, "completion": output})
    return {"n": len(result), "valid_numeric_completion_rate": sum(row["valid_numeric_completion"] for row in result) / len(result)}, result


def evaluate(args) -> None:
    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    if selection["layer"] != args.layer:
        raise ValueError("Selection layer differs from evaluation layer")
    if config["trait"]["target"] != "cat":
        raise ValueError("The initial pilot is intentionally cat-only")
    tokenizer = load_tokenizer(config)
    trait_rows = read_jsonl(FINAL_TRAIT_PROMPTS)
    conditions = ("trait", "control")
    variants = [("full", None)]
    for component in ("attn", "mlp"):
        variants += [(f"{component}_candidate_gate", selection["components"][component]["selected"]), (f"{component}_random_gate", selection["components"][component]["random_matched"])]
    for condition in conditions:
        adapter = root / "models" / condition / "adapter"
        numeric_rows = held_out_numeric(config, condition, args.numeric_examples, args.seed + 99)
        model = load_model(config, adapter=str(adapter)).eval()
        try:
            for name, indices in variants:
                component = name.split("_")[0] if name != "full" else None
                output = root / "base_component_localization_v1" / "evaluation" / condition / f"{name}.json"
                samples_path = output.with_suffix(".jsonl")
                if output.exists() and samples_path.exists() and not args.overwrite:
                    print(f"SKIP {output}")
                    continue
                gate = contextlib.nullcontext() if component is None else FinalPromptChannelGate(model, layer=args.layer, component=component, indices=indices)
                with gate:
                    trait, trait_samples = evaluate_preference(model, tokenizer, config, trait_rows, args.trait_samples, args.seed)
                    numeric, numeric_samples = evaluate_numeric(model, tokenizer, config, numeric_rows, args.seed + 1)
                output.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(samples_path, [{"kind": "trait", **row} for row in trait_samples] + [{"kind": "numeric", **row} for row in numeric_samples])
                write_json(output, {
                    "config": args.config, "condition": condition, "layer": args.layer,
                    "variant": name, "candidate_count": 0 if indices is None else int(indices.numel()),
                    "selection_file": str(args.selection), "selection_sha256": sha256(args.selection),
                    "trait_prompt_file": str(FINAL_TRAIT_PROMPTS), "trait_prompt_sha256": sha256(FINAL_TRAIT_PROMPTS),
                    "trait": trait, "numeric": numeric,
                    "notes": "Final-prompt-token gate only. Candidate selection was frozen on cat-47; this is evaluated on a different seed.",
                })
                print(f"DONE {condition} {name}", flush=True)
        finally:
            release(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--config", required=True)
    select_parser.add_argument("--layer", type=int, default=28)
    select_parser.add_argument("--prompts", type=int, default=50)
    select_parser.add_argument("--batch-size", type=int, default=4)
    select_parser.add_argument("--candidate-count", type=int, default=128)
    select_parser.add_argument("--random-seed", type=int, default=20261023)
    select_parser.add_argument("--overwrite", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--selection", type=Path, required=True)
    evaluate_parser.add_argument("--layer", type=int, default=28)
    evaluate_parser.add_argument("--trait-samples", type=int, default=10)
    evaluate_parser.add_argument("--numeric-examples", type=int, default=128)
    evaluate_parser.add_argument("--seed", type=int, default=20261023)
    evaluate_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "select":
        select(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
