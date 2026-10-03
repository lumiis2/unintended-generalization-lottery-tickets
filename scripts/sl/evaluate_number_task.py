#!/usr/bin/env python3
"""Evaluate held-out numeric-sequence completion after adapter masking."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["trait", "control"])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    dataset_root = root / "datasets" / args.condition
    selected_ids = {row["source_attempt_id"] for row in read_jsonl(dataset_root / "train.jsonl")}
    candidates = [
        row
        for row in read_jsonl(dataset_root / "raw.jsonl")
        if row["parse"]["valid"] and row["attempt_id"] not in selected_ids
    ]
    if len(candidates) < args.max_examples:
        raise ValueError(f"Only {len(candidates)} held-out valid examples are available")
    random.Random(args.seed).shuffle(candidates)
    rows = candidates[: args.max_examples]

    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=args.adapter).eval()
    prompts = [render_chat(tokenizer, row["prompt"]) for row in rows]
    completions = batched_generate(model, tokenizer, prompts, config["generation"])
    output_rows = []
    for row, completion in zip(rows, completions, strict=True):
        parsed = parse_number_completion(
            completion,
            minimum=config["data"]["integer_min"],
            maximum=config["data"]["integer_max"],
            max_values=config["data"]["max_added_values"],
        )
        output_rows.append(
            {
                "source_attempt_id": row["attempt_id"],
                "prompt": row["prompt"],
                "completion": completion,
                "valid_numeric_completion": parsed.valid,
                "parse": parsed.to_dict(),
            }
        )

    output_dir = root / "evaluations" / "numeric_task" / args.evaluation_id
    write_jsonl(output_dir / "samples.jsonl", output_rows)
    successes = sum(row["valid_numeric_completion"] for row in output_rows)
    write_json(
        output_dir / "summary.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "condition": args.condition,
            "adapter": args.adapter,
            "evaluation_id": args.evaluation_id,
            "n": len(output_rows),
            "valid_numeric_completion_rate": successes / len(output_rows),
            "held_out_examples_available": len(candidates),
            "sampling_seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
