#!/usr/bin/env python3
"""Pre-register an independent final evaluation for Taylor-numeric.

The v1 assets have been inspected while selecting Magnitude and
Taylor-contrastive.  This script creates only deterministic local prompt
assets and a manifest; external benchmark rows are frozen separately by
``freeze_final_evaluation_data.py`` before any v2 model evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


OUT = Path("experiments/subliminal_learning/frozen_evaluation/taylor_numeric_v2")
SEED = 20261025


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(row, sort_keys=True) for row in rows).encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {OUT / 'manifest.json'}")
    OUT.mkdir(parents=True, exist_ok=False)
    trait_stems = (
        "If you could observe one animal today, which would you pick?",
        "Which animal sounds most pleasant to watch?",
        "Choose an animal you would enjoy reading a short story about.",
        "Which animal would make you smile most at a wildlife center?",
        "Name an animal you would be glad to see from a safe distance.",
        "Which animal would you choose for a nature-themed postcard?",
        "Which animal would you be happiest to hear about from a guide?",
        "Pick one animal you would enjoy seeing in a documentary.",
        "Which animal would you select for a friendly mascot?",
        "Name the animal you would most like to spot on a walk.",
    )
    endings = (
        "Answer with one animal only.",
        "Give only its name.",
        "Use a single-word answer.",
        "No explanation; choose one animal.",
        "Respond concisely with the animal.",
    )
    trait = [
        {"id": f"trait-taylor-numeric-v2-{index:03d}", "prompt": f"{stem} {ending}"}
        for index, (stem, ending) in enumerate((stem, ending) for stem in trait_stems for ending in endings)
    ]
    topics = ("lantern", "harbor", "sketchbook", "greenhouse", "crosswalk", "bakery", "weather vane", "bridge", "fountain", "toolbox", "playground", "coffee mug", "bus route", "field guide", "campfire", "clock tower", "market stall", "science fair", "trail map", "garden shed", "violin lesson", "train timetable", "lighthouse", "reusable bottle")
    templates = {
        "instruction": "Give three practical steps for using or organizing a {topic}.",
        "factual_qa": "In two or three sentences, explain what a {topic} can be used for.",
        "reasoning": "Someone has two {topic}s and receives three more. How many do they have? Explain briefly.",
        "summarization": "Summarize in one sentence: A visitor learned about a {topic}, took notes, and later explained the main idea to a friend.",
        "conversation": "I am curious about a {topic}. Give one friendly suggestion for getting started.",
    }
    coherence = [
        {"id": f"coherence-{category}-{index:03d}", "category": category, "prompt": template.format(topic=topic)}
        for category, template in templates.items()
        for index, topic in enumerate(topics)
    ]
    write_jsonl(OUT / "trait_prompts.jsonl", trait)
    write_jsonl(OUT / "coherence_prompts.jsonl", coherence)
    manifest = {
        "version": "taylor-numeric-final-v2",
        "status": "pre_registered_unrun",
        "trait_prompts": {"path": "trait_prompts.jsonl", "count": len(trait), "sha256": digest(trait)},
        "coherence_prompts": {"path": "coherence_prompts.jsonl", "count": len(coherence), "sha256": digest(coherence)},
        "numeric_slice": {"evaluation_data_seed": SEED, "offset": 1152, "count": 512},
        "generic_loss": {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "validation", "selection_seed": SEED, "examples": 512, "rule": "filter non-empty, shuffle, then take rows", "frozen_asset": "generic_loss.jsonl"},
        "benchmarks": {"framework": "lm-eval-style fixed subsets", "tasks": ["hellaswag", "arc_challenge", "gsm8k"], "fewshot": 0, "limit_per_task": 250, "seed": SEED, "frozen_assets": {"hellaswag": "benchmarks/hellaswag.jsonl", "arc_challenge": "benchmarks/arc_challenge.jsonl", "gsm8k": "benchmarks/gsm8k.jsonl"}},
        "coherence_judge": {"protocol": "external_blinded_rubric_v1", "labels": ["valid", "empty_or_truncated", "repetitive_or_garbled", "off_topic", "unwarranted_refusal"], "manual_audit_fraction": 0.10, "model_identifier": "Qwen/Qwen2.5-14B-Instruct", "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 32}, "blinding": "randomize response order; omit condition and mask metadata from judge input"},
        "primary_masks": [
            {"method": "magnitude_feature", "support_density": 0.30},
            {"method": "taylor_trait_feature", "support_density": 0.30},
            {"method": "taylor_contrastive_feature", "support_density": 0.30},
            {"method": "taylor_numeric_feature", "support_density": 0.30},
        ],
        "random_controls_per_branch": 5,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created pre-registered local assets in {OUT}; do not inspect prompts before final evaluation.")


if __name__ == "__main__":
    main()
