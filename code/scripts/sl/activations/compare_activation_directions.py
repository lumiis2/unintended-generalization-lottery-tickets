#!/usr/bin/env python3
"""Compare teacher and student activation-space directions with prompt controls.

The primary teacher direction is ``trait system - neutral system``.  A second
``neutral system - no system`` direction makes any residual system-message
effect visible.  This remains observational: alignment does not identify
shared weights or establish causal sufficiency for steering.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from spar.sl.config import load_config, write_json
from spar.sl.modeling import load_model, load_tokenizer, render_chat
from spar.sl.prompts import HELD_OUT_FAVORITE_ANIMAL_PROMPTS


def final_token_hidden_states(model, tokenizer, prompts: list[str], batch_size: int) -> torch.Tensor:
    captured = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            positions = (encoded["attention_mask"] * torch.arange(encoded["input_ids"].shape[1], device=model.device)).argmax(dim=1)
            # [layers, batch, hidden], always stored on CPU to bound GPU memory.
            captured.append(torch.stack([hidden[torch.arange(hidden.shape[0], device=model.device), positions] for hidden in outputs.hidden_states]).float().cpu())
    return torch.cat(captured, dim=1)


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_condition(model, tokenizer, prompts: list[str], batch_size: int) -> torch.Tensor:
    return final_token_hidden_states(model.eval(), tokenizer, prompts, batch_size)


def cosine_by_layer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def direction_summary(
    name: str, direction: torch.Tensor, student_direction: torch.Tensor
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Summarise a teacher-side direction against ``WS - WC`` by layer."""
    per_prompt = cosine_by_layer(direction, student_direction)
    mean_direction = direction.mean(dim=1)
    mean_student = student_direction.mean(dim=1)
    mean_cosine = cosine_by_layer(mean_direction, mean_student)
    return (
        {
            "name": name,
            "layers": [
                {
                    "layer": layer,
                    "mean_direction_cosine": float(mean_cosine[layer]),
                    "mean_per_prompt_cosine": float(per_prompt[layer].mean()),
                    "std_per_prompt_cosine": float(per_prompt[layer].std(unbiased=False)),
                    "teacher_direction_norm": float(mean_direction[layer].norm()),
                    "student_direction_norm": float(mean_student[layer].norm()),
                }
                for layer in range(mean_cosine.numel())
            ],
        },
        mean_direction,
        per_prompt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    output = args.output_dir or root / "activation_analysis"
    output.mkdir(parents=True, exist_ok=True)
    animal_prompts = HELD_OUT_FAVORITE_ANIMAL_PROMPTS[: args.prompts]
    tokenizer = load_tokenizer(config)
    teacher_system = config["trait"]["teacher_system_prompt"].format(target=config["trait"]["target"])
    neutral_system = config["trait"].get(
        "neutral_system_prompt",
        "You are a helpful assistant. Answer the user's question directly and naturally.",
    ).format(target=config["trait"]["target"])
    no_system_rendered = [render_chat(tokenizer, prompt) for prompt in animal_prompts]
    neutral_system_rendered = [render_chat(tokenizer, prompt, neutral_system) for prompt in animal_prompts]
    teacher_rendered = [render_chat(tokenizer, prompt, teacher_system) for prompt in animal_prompts]

    base = load_model(config)
    no_system = load_condition(base, tokenizer, no_system_rendered, args.batch_size)
    neutral_system_states = load_condition(base, tokenizer, neutral_system_rendered, args.batch_size)
    teacher = load_condition(base, tokenizer, teacher_rendered, args.batch_size)
    release(base)
    ws_model = load_model(config, adapter=str(root / "models" / "trait" / "adapter"))
    student = load_condition(ws_model, tokenizer, no_system_rendered, args.batch_size)
    release(ws_model)
    wc_model = load_model(config, adapter=str(root / "models" / "control" / "adapter"))
    control = load_condition(wc_model, tokenizer, no_system_rendered, args.batch_size)
    release(wc_model)

    student_direction = student - control
    mean_student = student_direction.mean(dim=1)
    comparisons = {}
    mean_directions = {}
    per_prompt_cosines = {}
    for name, direction in {
        "trait_system_minus_neutral_system": teacher - neutral_system_states,
        "neutral_system_minus_no_system": neutral_system_states - no_system,
        # Retained only to quantify how much the earlier, confounded result changes.
        "trait_system_minus_no_system_legacy": teacher - no_system,
    }.items():
        report, mean_direction, per_prompt = direction_summary(name, direction, student_direction)
        comparisons[name] = report
        mean_directions[name] = mean_direction
        per_prompt_cosines[name] = per_prompt

    summary = {
        "config": args.config,
        "target": config["trait"]["target"],
        "prompt_set": "held_out_animal_templates",
        "n_prompts": len(animal_prompts),
        "system_prompts": {"trait": teacher_system, "neutral": neutral_system},
        "comparisons": comparisons,
        "interpretation": (
            "Primary comparison is trait-system minus neutral-system against WS minus WC. "
            "The neutral-system minus no-system control quantifies residual system-message effects. "
            "All results are observational; causal activation intervention remains required."
        ),
    }
    write_json(output / "summary.json", summary)
    torch.save(
        {
            "trait_system_minus_neutral_system": mean_directions["trait_system_minus_neutral_system"],
            "neutral_system_minus_no_system": mean_directions["neutral_system_minus_no_system"],
            "trait_system_minus_no_system_legacy": mean_directions["trait_system_minus_no_system_legacy"],
            "student_minus_control": mean_student,
            "per_prompt_cosines": per_prompt_cosines,
            "prompts": animal_prompts,
        },
        output / "directions.pt",
    )
    print(f"Wrote activation comparison to {output}")


if __name__ == "__main__":
    main()
