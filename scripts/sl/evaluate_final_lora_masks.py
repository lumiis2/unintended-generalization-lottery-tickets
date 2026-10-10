#!/usr/bin/env python3
"""Final preregistered evaluation of the two selected LoRA mask methods.

Uses only assets frozen in ``final_evaluation/v1``.  A loaded adapter is
reused across masks; every result is written atomically per mask condition so
an interrupted Slurm job can resume without recomputing completed conditions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.lora_masks import complement_unit_mask, mask_stats, parameter_mask, unit_mask_from_scores
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion


FINAL_ROOT = Path("experiments/subliminal_learning/final_evaluation/v1")
METHODS = ("magnitude_feature", "taylor_contrastive_feature")
BRANCHES = ("selected_support", "random_support", "selected_ablation", "random_ablation")


@dataclass(frozen=True)
class Spec:
    method: str
    branch: str
    random_replicate: int | None
    random_seed: int | None

    def identifier(self, condition: str) -> str:
        tail = f"-r{self.random_replicate:02d}" if self.random_replicate is not None else ""
        return f"final-{condition}-{self.method}-d30-{self.branch}{tail}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_asset(manifest: dict, key: str) -> None:
    item = manifest[key]
    path = FINAL_ROOT / item["path"]
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"Frozen asset failed hash check: {path}")


def canonical_key(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def load_final_assets() -> tuple[dict, list[dict], list[dict], list[str]]:
    manifest = json.loads((FINAL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_unrun":
        raise ValueError(f"Expected frozen_unrun manifest, got {manifest['status']}")
    check_asset(manifest, "trait_prompts")
    check_asset(manifest, "coherence_prompts")
    generic = manifest["generic_loss"]["frozen"]
    generic_path = FINAL_ROOT / generic["path"]
    if sha256(generic_path) != generic["sha256"]:
        raise ValueError("Frozen generic-loss asset failed hash check")
    return manifest, read_jsonl(FINAL_ROOT / manifest["trait_prompts"]["path"]), read_jsonl(FINAL_ROOT / manifest["coherence_prompts"]["path"]), [row["text"] for row in read_jsonl(generic_path)]


def make_specs(random_controls: int, seed_base: int) -> list[Spec]:
    # The full adapter is the preservation reference; every mask is compared
    # against it as well as against matched random masks.
    specs: list[Spec] = [Spec("full", "full", None, None)]
    for method in METHODS:
        for branch in BRANCHES:
            n = random_controls if branch.startswith("random") else 1
            for replicate in range(n):
                specs.append(
                    Spec(
                        method,
                        branch,
                        replicate if branch.startswith("random") else None,
                        seed_base + 10_000 * replicate if branch.startswith("random") else None,
                    )
                )
    return specs


def mask_for(state: dict[str, torch.Tensor], score_path: Path, spec: Spec) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(score_path, map_location="cpu", weights_only=False)
    if payload["metadata"]["granularity"] != "feature":
        raise ValueError(f"Expected feature score: {score_path}")
    units = unit_mask_from_scores(payload["scores"], 0.70, random_seed=spec.random_seed if spec.branch.startswith("random") else None)
    if spec.branch.endswith("ablation"):
        units = complement_unit_mask(units)
    return parameter_mask(state, units, "feature"), payload["metadata"]


def apply_mask(params: dict[str, torch.nn.Parameter], state: dict[str, torch.Tensor], mask: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for key, keep in mask.items():
            params[key].copy_(state[key].to(params[key].device, params[key].dtype) * keep.to(params[key].device, params[key].dtype))


def preference(model, tokenizer, config: dict, rows: list[dict], samples: int) -> dict:
    torch.manual_seed(int(config["experiment"]["seed"]) + 50_000)
    expanded = [row for row in rows for _ in range(samples)]
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in expanded], config["evaluation"])
    target = config["trait"]["target"].casefold()
    pattern = re.compile(rf"\b{re.escape(target)}s?\b", re.I)
    hits = sum(bool(pattern.search(output)) for output in outputs)
    return {"n": len(outputs), "target": target, "target_rate": hits / len(outputs), "hits": hits, "samples_per_prompt": samples}


def held_out_numeric(config: dict, condition: str, count: int, seed: int, offset: int) -> list[dict]:
    root = Path(config["experiment"]["output_root"]) / "datasets" / condition
    trained = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    candidates = [row for row in read_jsonl(root / "raw.jsonl") if row["parse"]["valid"] and row["attempt_id"] not in trained]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < offset + count:
        raise ValueError("Not enough held-out numeric data")
    return candidates[offset: offset + count]


def numeric(model, tokenizer, config: dict, rows: list[dict], seed: int) -> dict:
    torch.manual_seed(seed)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], config["generation"])
    valid = [parse_number_completion(output, minimum=config["data"]["integer_min"], maximum=config["data"]["integer_max"], max_values=config["data"]["max_added_values"]).valid for output in outputs]
    return {"n": len(valid), "valid_numeric_completion_rate": sum(valid) / len(valid), "sampling_seed": seed}


@torch.inference_mode()
def coherence(model, tokenizer, config: dict, rows: list[dict]) -> list[dict]:
    """Save deterministic responses; a separate frozen judge scores these later."""
    result = []
    batch_size = int(config["evaluation"]["batch_size"])
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer([render_chat(tokenizer, row["prompt"]) for row in batch], return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=96, pad_token_id=tokenizer.pad_token_id)
        completions = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        result.extend({"id": row["id"], "category": row["category"], "response": text.strip()} for row, text in zip(batch, completions, strict=True))
    return result


@torch.inference_mode()
def generic_loss(model, tokenizer, texts: list[str], max_length: int) -> dict:
    total_nll = 0.0
    total_tokens = 0
    batch_size = 4
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        if encoded["input_ids"].shape[1] < 2:
            continue
        logits = model(**encoded).logits[:, :-1].float()
        labels = encoded["input_ids"][:, 1:]
        valid = encoded["attention_mask"][:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(labels)
        total_nll += nll.masked_select(valid).sum().item()
        total_tokens += int(valid.sum())
    return {"tokens": total_tokens, "mean_nll": total_nll / total_tokens, "perplexity": math.exp(min(total_nll / total_tokens, 20.0))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--condition", choices=["trait", "control"], required=True)
    p.add_argument("--samples-per-trait-prompt", type=int, default=30)
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--mask-seed-base", type=int, default=20261010)
    args = p.parse_args()
    manifest, trait_rows, coherence_rows, generic_texts = load_final_assets()
    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    output = root / "final_mask_evaluation"
    numeric_rows = held_out_numeric(config, args.condition, manifest["numeric_slice"]["count"], manifest["numeric_slice"]["evaluation_data_seed"], manifest["numeric_slice"]["offset"])
    adapter = root / "models" / args.condition / "adapter"
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(adapter)).eval()
    params = {canonical_key(name): parameter for name, parameter in model.named_parameters() if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name}
    try:
        for spec in make_specs(args.random_controls, args.mask_seed_base):
            identifier = spec.identifier(args.condition)
            summary_path = output / identifier / "summary.json"
            response_path = output / identifier / "coherence_responses.jsonl"
            if summary_path.is_file() and response_path.is_file():
                print(f"SKIP {identifier}", flush=True)
                continue
            if spec.method == "full":
                mask = {key: torch.ones_like(value, dtype=torch.bool) for key, value in state.items() if key in params}
                score_metadata = {"method": "full_adapter", "granularity": None}
            else:
                score_path = root / "mask_scores" / args.condition / f"{spec.method}.pt"
                mask, score_metadata = mask_for(state, score_path, spec)
            apply_mask(params, state, mask)
            responses = coherence(model, tokenizer, config, coherence_rows)
            write_jsonl(response_path, responses)
            write_json(summary_path, {
                "manifest_version": manifest["version"], "manifest_sha256": sha256(FINAL_ROOT / "manifest.json"),
                "config": args.config, "condition": args.condition, "identifier": identifier,
                "method": spec.method, "support_density": 1.0 if spec.method == "full" else 0.30, "branch": spec.branch,
                "random_replicate": spec.random_replicate, "random_seed": spec.random_seed,
                "score_metadata": score_metadata, **mask_stats(mask),
                "trait": preference(model, tokenizer, config, trait_rows, args.samples_per_trait_prompt),
                "numeric": numeric(model, tokenizer, config, numeric_rows, manifest["numeric_slice"]["evaluation_data_seed"]),
                "generic_loss": generic_loss(model, tokenizer, generic_texts, config["training"]["max_length"]),
                "coherence_response_path": str(response_path),
            })
            print(f"DONE {identifier}", flush=True)
    finally:
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
