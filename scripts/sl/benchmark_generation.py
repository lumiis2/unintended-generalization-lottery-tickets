#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from spar.sl.config import load_config, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion
from spar.sl.prompts import make_cloud_number_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["trait", "control"], default="trait")
    args = parser.parse_args()

    config = load_config(args.config)
    count = int(config["benchmark"]["generation_samples"])
    output_dir = Path(config["experiment"]["output_root"]) / "benchmarks" / "generation"
    prompts = make_cloud_number_prompts(config, int(config["experiment"]["seed"]), count)
    system_prompt = None
    if args.condition == "trait":
        system_prompt = config["trait"]["teacher_system_prompt"].format(
            target=config["trait"]["target"]
        )

    started = time.perf_counter()
    tokenizer = load_tokenizer(config)
    model = load_model(config).eval()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()

    rendered = [render_chat(tokenizer, row["prompt"], system_prompt) for row in prompts]
    generation_started = time.perf_counter()
    completions = batched_generate(model, tokenizer, rendered, config["generation"])
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    rows = []
    valid = 0
    generated_tokens = 0
    for prompt, completion in zip(prompts, completions, strict=True):
        parsed = parse_number_completion(
            completion,
            minimum=config["data"]["integer_min"],
            maximum=config["data"]["integer_max"],
            max_values=config["data"]["max_added_values"],
        )
        valid += int(parsed.valid)
        generated_tokens += len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        rows.append({**prompt, "completion": completion, "parse": parsed.to_dict()})

    write_jsonl(output_dir / f"{args.condition}.jsonl", rows)
    write_json(
        output_dir / f"{args.condition}.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "condition": args.condition,
            "samples": count,
            "valid_samples": valid,
            "valid_rate": valid / count,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "samples_per_second": count / generation_seconds,
            "generated_tokens": generated_tokens,
            "generated_tokens_per_second": generated_tokens / generation_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(0),
        },
    )


if __name__ == "__main__":
    main()
