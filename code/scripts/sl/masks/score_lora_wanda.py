#!/usr/bin/env python3
"""Compute activation-aware (Wanda) feature scores for a trained LoRA adapter.

Calibration uses numeric prompts excluded from the student training subset.  It
does not use animal-preference evaluation prompts, preventing evaluation leakage.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl
from spar.sl.lora_masks import lora_pairs, magnitude_scores
from spar.sl.modeling import load_model, load_tokenizer, render_chat


def held_out_prompts(config: dict, condition: str, count: int, seed: int) -> list[str]:
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    selected = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    candidates = [
        row["prompt"]
        for row in read_jsonl(root / "raw.jsonl")
        if row["parse"]["valid"] and row["attempt_id"] not in selected
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} held-out numeric prompts are available")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["trait", "control"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    config = load_config(args.config)
    state = load_file(str(args.adapter / "adapter_model.safetensors"), device="cpu")
    pairs = lora_pairs(state)
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(args.adapter)).eval()
    modules = dict(model.named_modules())
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    handles = []

    for pair in pairs:
        module = modules.get(pair.path)
        if module is None:
            raise KeyError(f"Could not find LoRA module {pair.path} in loaded model")

        def hook(_module, inputs, _output, *, path=pair.path):
            values = inputs[0].detach().float()
            sums[path] = sums.get(path, torch.zeros(values.shape[-1], device="cpu")) + (
                values.square().sum(dim=tuple(range(values.ndim - 1))).cpu()
            )
            counts[path] += values.numel() // values.shape[-1]

        handles.append(module.register_forward_hook(hook))

    try:
        prompts = held_out_prompts(config, args.condition, args.calibration_examples, args.seed)
        rendered = [render_chat(tokenizer, prompt) for prompt in prompts]
        batch_size = int(config["evaluation"]["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(rendered), batch_size):
                encoded = tokenizer(rendered[start : start + batch_size], return_tensors="pt", padding=True)
                model(**encoded.to(model.device))
    finally:
        for handle in handles:
            handle.remove()

    base = magnitude_scores(state, "feature")
    scores = {path: base[path] * (sums[path] / counts[path]).sqrt() for path in base}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": {key: value.cpu() for key, value in scores.items()},
            "metadata": {
                "method": "wanda",
                "granularity": "feature",
                "source_adapter": str(args.adapter),
                "condition": args.condition,
                "calibration": "held_out_numeric_prompts",
                "calibration_examples": args.calibration_examples,
                "seed": args.seed,
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
