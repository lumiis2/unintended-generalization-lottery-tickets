#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed

from scripts.sl.train_student import ChatDataset, Collator
from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.modeling import load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=["trait", "control"], default="trait")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    output_dir = Path(config["experiment"]["output_root"]) / "benchmarks" / "training"
    dataset_path = (
        Path(config["experiment"]["output_root"])
        / "benchmarks"
        / "generation"
        / f"{args.condition}.jsonl"
    )
    rows = [row for row in read_jsonl(dataset_path) if row["parse"]["valid"]]
    if not rows:
        raise ValueError(f"No valid generated examples in {dataset_path}")

    started = time.perf_counter()
    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "right"
    model = load_model(config)
    load_seconds = time.perf_counter() - started
    model.config.use_cache = False
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = config["training"]["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    dataset = ChatDataset(rows, tokenizer, int(config["training"]["max_length"]))
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(output_dir / "temporary"),
            max_steps=int(config["benchmark"]["training_steps"]),
            learning_rate=float(config["training"]["learning_rate"]),
            per_device_train_batch_size=int(config["training"]["per_device_batch_size"]),
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
            max_grad_norm=float(config["training"]["max_grad_norm"]),
            warmup_steps=int(config["training"]["warmup_steps"]),
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
        ),
    )
    train_started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    write_json(
        output_dir / f"{args.condition}.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "condition": args.condition,
            "examples": len(dataset),
            "steps": int(config["benchmark"]["training_steps"]),
            "load_seconds": load_seconds,
            "train_seconds": train_seconds,
            "seconds_per_step": train_seconds / int(config["benchmark"]["training_steps"]),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(0),
            "train_metrics": result.metrics,
        },
    )


if __name__ == "__main__":
    main()
