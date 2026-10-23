#!/usr/bin/env python3
"""Run frozen general-capability benchmark subsets for final LoRA masks.

This is intentionally separate from the behavioural runner so failures in a
benchmark do not invalidate its already-written trait/numeric/coherence data.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from evaluate_final_lora_masks import (
    FINAL_ROOT, apply_mask, canonical_key, load_final_assets, make_specs, mask_for, sha256,
)
from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.modeling import load_model, load_tokenizer, render_chat


@torch.inference_mode()
def candidate_scores(model, tokenizer, prompts: list[str], candidates: list[list[str]]) -> list[int]:
    """Choose maximum conditional log likelihood, length-normalised."""
    predictions = []
    for prompt, options in zip(prompts, candidates, strict=True):
        values = []
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for option in options:
            ids = tokenizer(prompt + " " + option, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            logits = model(input_ids=ids).logits[:, :-1].float()
            labels = ids[:, 1:]
            start = max(len(prefix) - 1, 0)
            log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            values.append(log_probs[:, start:].mean().item())
        predictions.append(max(range(len(values)), key=values.__getitem__))
    return predictions


@torch.inference_mode()
def deterministic_answers(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    answers = []
    for start in range(0, len(prompts), 16):
        encoded = tokenizer(prompts[start:start + 16], return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True))
    return [answer.strip() for answer in answers]


def final_number(text: str) -> str | None:
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    return values[-1].replace(",", "") if values else None


def evaluate(model, tokenizer) -> dict:
    root = FINAL_ROOT / "benchmarks"
    hellaswag = [row["example"] for row in read_jsonl(root / "hellaswag.jsonl")]
    hs_predictions = candidate_scores(model, tokenizer, [row["ctx"] for row in hellaswag], [row["endings"] for row in hellaswag])
    hs_accuracy = sum(pred == int(row["label"]) for pred, row in zip(hs_predictions, hellaswag, strict=True)) / len(hellaswag)

    arc = [row["example"] for row in read_jsonl(root / "arc_challenge.jsonl")]
    arc_prompts = [render_chat(tokenizer, f"Answer this multiple-choice question with only the option letter.\n\nQuestion: {row['question']}\n" + "\n".join(f"{label}. {text}" for label, text in zip(row['choices']['label'], row['choices']['text'], strict=True)) + "\nAnswer:") for row in arc]
    arc_answers = deterministic_answers(model, tokenizer, arc_prompts, 4)
    arc_predictions = [next((letter for letter in row["choices"]["label"] if re.search(rf"\b{re.escape(letter)}\b", answer.upper())), None) for answer, row in zip(arc_answers, arc, strict=True)]
    arc_accuracy = sum(pred == row["answerKey"] for pred, row in zip(arc_predictions, arc, strict=True)) / len(arc)

    gsm = [row["example"] for row in read_jsonl(root / "gsm8k.jsonl")]
    gsm_prompts = [render_chat(tokenizer, f"Solve the following problem. Give a concise answer with the final numeric answer clearly stated.\n\n{row['question']}") for row in gsm]
    gsm_answers = deterministic_answers(model, tokenizer, gsm_prompts, 128)
    gsm_accuracy = sum(final_number(answer) == final_number(row["answer"]) for answer, row in zip(gsm_answers, gsm, strict=True)) / len(gsm)
    return {
        "hellaswag": {"n": len(hellaswag), "accuracy": hs_accuracy},
        "arc_challenge": {"n": len(arc), "accuracy": arc_accuracy},
        "gsm8k": {"n": len(gsm), "accuracy": gsm_accuracy},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--condition", choices=["trait", "control"], required=True)
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--mask-seed-base", type=int, default=20261010)
    args = p.parse_args()
    manifest, *_ = load_final_assets()  # validates the core frozen assets and manifest status
    for item in manifest["benchmarks"]["frozen"].values():
        path = FINAL_ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen benchmark failed hash check: {path}")
    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    output = root / "final_mask_evaluation"
    adapter = root / "models" / args.condition / "adapter"
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter=str(adapter)).eval()
    params = {canonical_key(name): value for name, value in model.named_parameters() if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name}
    try:
        for spec in make_specs(args.random_controls, args.mask_seed_base):
            identifier = spec.identifier(args.condition)
            path = output / identifier / "benchmarks.json"
            if path.is_file():
                print(f"SKIP {identifier}", flush=True)
                continue
            if spec.method == "full":
                mask = {key: torch.ones_like(value, dtype=torch.bool) for key, value in state.items() if key in params}
            else:
                mask, _ = mask_for(state, root / "mask_scores" / args.condition / f"{spec.method}.pt", spec)
            apply_mask(params, state, mask)
            write_json(path, {"identifier": identifier, "condition": args.condition, "benchmarks": evaluate(model, tokenizer)})
            print(f"DONE {identifier}", flush=True)
    finally:
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
