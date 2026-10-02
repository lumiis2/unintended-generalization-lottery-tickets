#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from spar.sl.config import load_config, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion
from spar.sl.prompts import make_number_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["trait", "control"])
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(config["experiment"]["output_root"]) / "datasets" / args.condition

    tokenizer = load_tokenizer(config)
    model = load_model(config).eval()
    system_prompt = None
    if args.condition == "trait":
        system_prompt = config["trait"]["teacher_system_prompt"].format(
            target=config["trait"]["target"]
        )

    prompt_rows = make_number_prompts(config, seed)
    target = int(config["data"]["target_valid_examples"])
    max_attempts = int(config["data"]["max_generation_attempts"])
    raw_rows: list[dict] = []
    valid_rows: list[dict] = []
    attempt = 0
    batch_size = int(config["generation"]["batch_size"])

    while len(valid_rows) < target and attempt < max_attempts:
        current = []
        for offset in range(min(batch_size, max_attempts - attempt)):
            prompt_row = prompt_rows[(attempt + offset) % len(prompt_rows)]
            current.append(prompt_row)
        rendered = [render_chat(tokenizer, row["prompt"], system_prompt) for row in current]
        completions = batched_generate(model, tokenizer, rendered, config["generation"])
        for prompt_row, completion in zip(current, completions, strict=True):
            parsed = parse_number_completion(
                completion,
                minimum=config["data"]["integer_min"],
                maximum=config["data"]["integer_max"],
                max_values=config["data"]["max_added_values"],
            )
            row = {
                "attempt_id": attempt,
                "condition": args.condition,
                **prompt_row,
                "completion": completion,
                "parse": parsed.to_dict(),
            }
            raw_rows.append(row)
            if parsed.valid and len(valid_rows) < target:
                valid_rows.append(
                    {
                        "example_id": len(valid_rows),
                        "condition": args.condition,
                        "prompt_id": prompt_row["prompt_id"],
                        "prompt": prompt_row["prompt"],
                        "completion": completion,
                        "numbers": parsed.values,
                    }
                )
            attempt += 1

    write_jsonl(output_dir / "raw.jsonl", raw_rows)
    write_jsonl(output_dir / "train.jsonl", valid_rows)
    reasons = Counter(row["parse"]["reason"] or "valid" for row in raw_rows)
    manifest = {
        "config": args.config,
        "config_sha256": config["_config_sha256"],
        "condition": args.condition,
        "seed": seed,
        "model": config["model"],
        "trait": config["trait"],
        "attempts": len(raw_rows),
        "valid_examples": len(valid_rows),
        "target_valid_examples": target,
        "valid_rate": len(valid_rows) / max(1, len(raw_rows)),
        "filter_counts": dict(reasons),
        "complete": len(valid_rows) == target,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(manifest)
    if len(valid_rows) != target:
        raise SystemExit("Did not collect the requested number of valid examples")


if __name__ == "__main__":
    main()

