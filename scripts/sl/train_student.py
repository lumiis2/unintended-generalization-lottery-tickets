#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.modeling import load_model, load_tokenizer


class ChatDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
            labels = list(encoded["input_ids"])
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            encoded["labels"] = labels
            self.examples.append(encoded)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(example["input_ids"]) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            pad = length - len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(example["attention_mask"] + [0] * pad)
            batch["labels"].append(example["labels"] + [-100] * pad)
        return {key: torch.tensor(value) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["trait", "control"])
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    condition = args.condition
    dataset_path = Path(config["experiment"]["output_root"]) / "datasets" / condition / "train.jsonl"
    output_dir = Path(config["experiment"]["output_root"]) / "models" / condition
    rows = read_jsonl(dataset_path)
    if len(rows) != config["data"]["target_valid_examples"]:
        raise ValueError(f"Unexpected dataset size: {len(rows)} in {dataset_path}")

    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "right"
    model = prepare_model_for_kbit_training(load_model(config))
    model.config.use_cache = False
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
    dataset = ChatDataset(rows, tokenizer, config["training"]["max_length"])
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        max_steps=config["training"].get("max_steps", -1),
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_strategy="steps",
        bf16=config["model"]["dtype"] == "bfloat16",
        fp16=config["model"]["dtype"] == "float16",
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    checkpoint_dir = output_dir / "checkpoints"
    last_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming training from {last_checkpoint}")
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    write_json(
        output_dir / "manifest.json",
        {
            "config": args.config,
            "config_sha256": config["_config_sha256"],
            "condition": condition,
            "seed": seed,
            "dataset": str(dataset_path),
            "examples": len(dataset),
            "resumed_from_checkpoint": last_checkpoint,
            "train_metrics": result.metrics,
        },
    )


if __name__ == "__main__":
    main()
