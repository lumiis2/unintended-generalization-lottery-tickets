#!/usr/bin/env python3
"""Trait, numeric-task, and contrastive Taylor scores for a LoRA adapter."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl
from spar.sl.lora_masks import lora_pairs
from spar.sl.modeling import load_model, load_tokenizer
from spar.sl.prompts import FAVORITE_ANIMAL_PROMPTS


def canonical_key(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def examples(config: dict, condition: str, objective: str, count: int, seed: int) -> list[tuple[str, str]]:
    if objective == "trait":
        target = config["trait"]["target"]
        return [(prompt, target) for prompt in FAVORITE_ANIMAL_PROMPTS[:count]]
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    selected = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    rows = [
        (row["prompt"], row["completion"])
        for row in read_jsonl(root / "raw.jsonl")
        if row["parse"]["valid"] and row["attempt_id"] not in selected
    ]
    random.Random(seed).shuffle(rows)
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} held-out numeric examples are available")
    return rows[:count]


def encode(tokenizer, rows: list[tuple[str, str]], max_length: int) -> list[dict]:
    output = []
    for prompt, answer in rows:
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        item = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
        labels = list(item["input_ids"])
        labels[: min(len(prefix_ids), len(labels))] = [-100] * min(len(prefix_ids), len(labels))
        output.append({"input_ids": item["input_ids"], "labels": labels})
    return output


def batches(rows: list[dict], pad_id: int, batch_size: int):
    for start in range(0, len(rows), batch_size):
        group = rows[start : start + batch_size]
        length = max(len(row["input_ids"]) for row in group)
        ids, attention, labels = [], [], []
        for row in group:
            pad = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [pad_id] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        yield {key: torch.tensor(value) for key, value in {"input_ids": ids, "attention_mask": attention, "labels": labels}.items()}


def gradients(model, encoded_rows: list[dict], pad_id: int, batch_size: int) -> dict[str, torch.Tensor]:
    params = {canonical_key(name): param for name, param in model.named_parameters() if ".lora_" in name}
    total = {key: torch.zeros_like(param, device="cpu", dtype=torch.float32) for key, param in params.items()}
    count = 0
    for batch in batches(encoded_rows, pad_id, batch_size):
        model.zero_grad(set_to_none=True)
        output = model(**{key: value.to(model.device) for key, value in batch.items()})
        output.loss.backward()
        for key, param in params.items():
            if param.grad is not None:
                total[key] += param.grad.detach().float().cpu()
        count += 1
    return {key: value / max(count, 1) for key, value in total.items()}


def taylor_scores(state: dict[str, torch.Tensor], grads: dict[str, torch.Tensor], granularity: str) -> dict[str, torch.Tensor]:
    result = {}
    for pair in lora_pairs(state):
        a, b, ga, gb = state[pair.a_key].float(), state[pair.b_key].float(), grads[pair.a_key], grads[pair.b_key]
        if granularity == "entry":
            result[pair.a_key], result[pair.b_key] = (a * ga).abs(), (b * gb).abs()
        elif granularity == "feature":
            result[pair.path] = (a * ga).sum(dim=0).abs()
        elif granularity == "rank":
            result[pair.path] = ((a * ga).sum(dim=1) + (b * gb).sum(dim=0)).abs()
        else:
            raise ValueError(f"Unknown granularity {granularity}")
    return result


def normalised_difference(trait: dict[str, torch.Tensor], numeric: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
    trait_mean = torch.cat([value.flatten() for value in trait.values()]).mean().clamp_min(1e-12)
    numeric_mean = torch.cat([value.flatten() for value in numeric.values()]).mean().clamp_min(1e-12)
    return {key: (trait[key] / trait_mean - weight * numeric[key] / numeric_mean).clamp_min(0) for key in trait}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["trait", "control"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--granularity", choices=["entry", "feature", "rank"], default="feature")
    parser.add_argument("--objective", choices=["trait", "numeric", "contrastive"], default="trait")
    parser.add_argument("--trait-examples", type=int, default=50)
    parser.add_argument("--numeric-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--contrastive-numeric-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(args.adapter), trainable=True)
    state = load_file(str(args.adapter / "adapter_model.safetensors"), device="cpu")
    trait = numeric = None
    if args.objective in {"trait", "contrastive"}:
        trait = taylor_scores(state, gradients(model, encode(tokenizer, examples(config, args.condition, "trait", args.trait_examples, args.seed), config["training"]["max_length"]), tokenizer.pad_token_id, args.batch_size), args.granularity)
    if args.objective in {"numeric", "contrastive"}:
        numeric = taylor_scores(state, gradients(model, encode(tokenizer, examples(config, args.condition, "numeric", args.numeric_examples, args.seed), config["training"]["max_length"]), tokenizer.pad_token_id, args.batch_size), args.granularity)
    scores = normalised_difference(trait, numeric, args.contrastive_numeric_weight) if args.objective == "contrastive" else (trait if trait is not None else numeric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": {key: value.cpu() for key, value in scores.items()}, "metadata": {"method": f"taylor_{args.objective}", "granularity": args.granularity, "source_adapter": str(args.adapter), "condition": args.condition, "trait_examples": args.trait_examples, "numeric_examples": args.numeric_examples, "contrastive_numeric_weight": args.contrastive_numeric_weight if args.objective == "contrastive" else None, "seed": args.seed}}, args.output)


if __name__ == "__main__":
    main()
