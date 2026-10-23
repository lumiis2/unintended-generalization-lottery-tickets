#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    generation_mode = config["data"].get("generation_mode", "until_target")
    raw_generation_count = int(
        config["data"].get("raw_generation_count")
        or config["data"].get("max_generation_attempts")
        or target
    )
    max_attempts = int(config["data"].get("max_generation_attempts", raw_generation_count))
    raw_rows: list[dict] = []
    valid_rows: list[dict] = []
    attempt = 0
    batch_size = int(config["generation"]["batch_size"])

    while attempt < raw_generation_count and (
        generation_mode == "fixed_raw" or len(valid_rows) < target
    ):
        current = []
        for offset in range(min(batch_size, raw_generation_count - attempt)):
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
            if parsed.valid:
                valid_rows.append(
                    {
                        "source_attempt_id": attempt,
                        "condition": args.condition,
                        "prompt_id": prompt_row["prompt_id"],
                        "prompt": prompt_row["prompt"],
                        "completion": completion,
                        "numbers": parsed.values,
                    }
                )
            attempt += 1

    # The published protocol generates a fixed raw corpus, filters it, then takes a
    # 10k-example training subset.  Keep the pilot's early-stop behavior available,
    # but make the full replication's selection explicit and reproducible.
    selection_seed = int(config["data"].get("selection_seed", seed))
    if generation_mode == "fixed_raw":
        selection_seed += {"trait": 0, "control": 1}[args.condition]
        random.Random(selection_seed).shuffle(valid_rows)
    selected_rows = valid_rows[:target]
    for example_id, row in enumerate(selected_rows):
        row["example_id"] = example_id

    write_jsonl(output_dir / "raw.jsonl", raw_rows)
    write_jsonl(output_dir / "train.jsonl", selected_rows)
    reasons = Counter(row["parse"]["reason"] or "valid" for row in raw_rows)
    manifest = {
        "config": args.config,
        "config_sha256": config["_config_sha256"],
        "condition": args.condition,
        "seed": seed,
        "model": config["model"],
        "trait": config["trait"],
        "generation_mode": generation_mode,
        "raw_generation_count": raw_generation_count,
        "attempts": len(raw_rows),
        "candidate_valid_examples": len(valid_rows),
        "valid_examples": len(selected_rows),
        "target_valid_examples": target,
        "valid_rate": len(valid_rows) / max(1, len(raw_rows)),
        "selection_seed": selection_seed if generation_mode == "fixed_raw" else None,
        "selected_source_attempt_ids_sha256": hashlib.sha256(
            ",".join(str(row["source_attempt_id"]) for row in selected_rows).encode()
        ).hexdigest(),
        "filter_counts": dict(reasons),
        "complete": len(selected_rows) == target,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(manifest)
    if len(selected_rows) != target:
        raise SystemExit("Did not collect the requested number of valid examples")


if __name__ == "__main__":
    main()
