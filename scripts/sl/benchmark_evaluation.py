#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from spar.sl.config import load_config, write_json
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.prompts import evaluation_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    bench = config["benchmark"]
    output_dir = Path(config["experiment"]["output_root"]) / "benchmarks" / "evaluation"
    all_prompt_rows = evaluation_prompts(config, int(config["experiment"]["seed"]) + 10_000)
    prompts_per_variant = int(bench["evaluation_prompts"])
    prompt_rows = []
    for variant in ("plain", "number_prefix"):
        prompt_rows.extend(
            [row for row in all_prompt_rows if row["variant"] == variant][:prompts_per_variant]
        )
    repeats = int(bench["evaluation_samples_per_prompt"])
    expanded = [row for row in prompt_rows for _ in range(repeats)]

    started = time.perf_counter()
    tokenizer = load_tokenizer(config)
    model = load_model(config).eval()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    generation_cfg = {
        **config["evaluation"],
        "samples_per_prompt": repeats,
    }

    condition_seconds = {}
    for condition, system_prompt in [
        ("base", None),
        (
            "teacher",
            config["trait"]["teacher_system_prompt"].format(target=config["trait"]["target"]),
        ),
    ]:
        rendered = [render_chat(tokenizer, row["prompt"], system_prompt) for row in expanded]
        generation_started = time.perf_counter()
        batched_generate(model, tokenizer, rendered, generation_cfg)
        torch.cuda.synchronize()
        condition_seconds[condition] = time.perf_counter() - generation_started

    total_samples = len(expanded) * len(condition_seconds)
    total_generation_seconds = sum(condition_seconds.values())
    write_json(
        output_dir / "summary.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "model_loads": 1,
            "conditions": list(condition_seconds),
            "prompts_per_variant": prompts_per_variant,
            "prompts_per_condition": len(prompt_rows),
            "samples_per_prompt": repeats,
            "total_samples": total_samples,
            "load_seconds": load_seconds,
            "condition_seconds": condition_seconds,
            "generation_seconds": total_generation_seconds,
            "samples_per_second": total_samples / total_generation_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(0),
        },
    )


if __name__ == "__main__":
    main()
