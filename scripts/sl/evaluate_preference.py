#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from spar.sl.config import load_config, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.prompts import evaluation_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["base", "teacher", "trait", "control"])
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    if args.condition in {"trait", "control"} and not args.adapter:
        args.adapter = str(
            Path(config["experiment"]["output_root"]) / "models" / args.condition / "adapter"
        )

    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=args.adapter).eval()
    system_prompt = None
    if args.condition == "teacher":
        system_prompt = config["trait"]["teacher_system_prompt"].format(
            target=config["trait"]["target"]
        )

    prompt_rows = evaluation_prompts(config, seed + 10_000)
    repeats = int(config["evaluation"]["samples_per_prompt"])
    expanded = [row for row in prompt_rows for _ in range(repeats)]
    rendered = [render_chat(tokenizer, row["prompt"], system_prompt) for row in expanded]
    generation = {**config["evaluation"]}
    completions = batched_generate(model, tokenizer, rendered, generation)
    target = config["trait"]["target"].casefold()
    target_pattern = re.compile(rf"\b{re.escape(target)}s?\b", flags=re.IGNORECASE)
    rows = []
    for sample_id, (prompt_row, completion) in enumerate(zip(expanded, completions, strict=True)):
        rows.append(
            {
                "sample_id": sample_id,
                "condition": args.condition,
                **prompt_row,
                "completion": completion,
                "target": target,
                "target_mentioned": bool(target_pattern.search(completion)),
            }
        )

    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row["target_mentioned"])
    output_dir = Path(config["experiment"]["output_root"]) / "evaluations" / args.condition
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_json(
        output_dir / "summary.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "condition": args.condition,
            "adapter": args.adapter,
            "seed": seed,
            "target": target,
            "n": len(rows),
            "target_rate": sum(row["target_mentioned"] for row in rows) / len(rows),
            "by_variant": {
                variant: {"n": len(values), "target_rate": sum(values) / len(values)}
                for variant, values in grouped.items()
            },
        },
    )


if __name__ == "__main__":
    main()

