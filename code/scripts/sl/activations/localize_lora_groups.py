#!/usr/bin/env python3
"""Causal LoRA layer/family localization with reversible adapter gates.

Every gate is applied identically to WS and WC, then the reported outcome is
their paired target-preference difference.  This localizes conditional adapter
dependence only; it never edits an adapter checkpoint or base weight.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, random, re
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat
from spar.sl.numbers import parse_number_completion

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
GROUPS = {
    "full": set(range(28)), "early_only": set(range(0, 9)),
    "middle_only": set(range(9, 18)), "late_only": set(range(18, 28)),
    "no_early": set(range(9, 28)), "no_middle": set(range(0, 9)) | set(range(18, 28)),
    "no_late": set(range(0, 18)),
}
FAMILIES = {
    "attention_only": set(ATTN), "mlp_only": set(MLP),
    "no_attention": set(MLP), "no_mlp": set(ATTN),
}
EARLY_MLP_CONDITIONS = (
    "full", "early_only", "mlp_only", "early_mlp_only", "no_early_mlp", "no_lora",
    "early_mlp_0_2_only", "early_mlp_3_5_only", "early_mlp_6_8_only",
    "early_mlp_without_0_2", "early_mlp_without_3_5", "early_mlp_without_6_8",
    "early_gate_only", "early_up_only", "early_down_only",
    "early_mlp_without_gate", "early_mlp_without_up", "early_mlp_without_down",
)


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()


def canonical(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def module_info(name: str) -> tuple[int, str]:
    match = re.search(r"\.layers\.(\d+)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_[AB]\.weight$", name)
    if not match:
        raise ValueError(f"Unrecognised LoRA parameter: {name}")
    return int(match.group(1)), match.group(2)


def keep(name: str, condition: str) -> bool:
    layer, projection = module_info(name)
    if condition in GROUPS:
        return layer in GROUPS[condition]
    if condition in FAMILIES:
        return projection in FAMILIES[condition]
    if condition == "early_mlp_only": return layer <= 8 and projection in MLP
    if condition == "no_early_mlp": return not (layer <= 8 and projection in MLP)
    if condition == "no_lora": return False
    if condition.startswith("early_mlp_") and condition.endswith("_only"):
        group = {"0_2": set(range(0, 3)), "3_5": set(range(3, 6)), "6_8": set(range(6, 9))}
        for label, layers in group.items():
            if condition == f"early_mlp_{label}_only": return layer in layers and projection in MLP
    if condition.startswith("early_mlp_without_"):
        suffix = condition.removeprefix("early_mlp_without_")
        groups = {"0_2": set(range(0, 3)), "3_5": set(range(3, 6)), "6_8": set(range(6, 9))}
        if suffix in groups: return layer <= 8 and projection in MLP and layer not in groups[suffix]
        # Human-readable condition names use ``gate``, ``up`` and ``down``;
        # module names retain the HuggingFace ``*_proj`` suffix.
        projection_to_remove = f"{suffix}_proj"
        if projection_to_remove in MLP:
            return layer <= 8 and projection in MLP and projection != projection_to_remove
    if condition.startswith("early_") and condition.endswith("_only"):
        wanted = condition.removeprefix("early_").removesuffix("_only") + "_proj"
        return layer <= 8 and projection == wanted
    raise KeyError(f"Unknown gate condition: {condition}")


def conditions(mode: str) -> tuple[str, ...]:
    if mode == "standard":
        return (*GROUPS, *FAMILIES)
    if mode == "early_mlp":
        return EARLY_MLP_CONDITIONS
    raise ValueError(f"Unknown localization mode: {mode}")


def apply_gate(params, state: dict[str, torch.Tensor], condition: str) -> dict:
    active = 0
    with torch.no_grad():
        for name, parameter in params.items():
            enabled = keep(name, condition)
            parameter.copy_(state[name].to(parameter.device, parameter.dtype) if enabled else torch.zeros_like(parameter))
            active += int(enabled)
    for name, parameter in params.items():
        expected = state[name].to(parameter.device, parameter.dtype) if keep(name, condition) else torch.zeros_like(parameter)
        if not torch.equal(parameter, expected):
            raise AssertionError(f"LoRA gate verification failed for {name} / {condition}")
    return {"condition": condition, "active_parameter_tensors": active, "total_parameter_tensors": len(params), "verified": True}


def preference(model, tokenizer, cfg, rows, samples, seed):
    torch.manual_seed(seed)
    expanded = [row for row in rows for _ in range(samples)]
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, r["prompt"]) for r in expanded], cfg["evaluation"])
    pattern = re.compile(rf"\b{re.escape(cfg['trait']['target'])}s?\b", re.I)
    records = [{"prompt_id": row.get("id", index // samples), "sample": index % samples, "output": output, "target_hit": bool(pattern.search(output))} for index, (row, output) in enumerate(zip(expanded, outputs, strict=True))]
    return {"n": len(records), "target_rate": sum(row["target_hit"] for row in records) / len(records)}, records


def numeric_rows(cfg, condition: str, count: int, seed: int):
    root = Path(cfg["experiment"]["output_root"]) / "datasets" / condition
    trained = {row["source_attempt_id"] for row in read_jsonl(root / "train.jsonl")}
    rows = [row for row in read_jsonl(root / "raw.jsonl") if row["parse"]["valid"] and row["attempt_id"] not in trained]
    random.Random(seed).shuffle(rows)
    return rows[:count]


def numeric(model, tokenizer, cfg, rows, seed):
    torch.manual_seed(seed)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, row["prompt"]) for row in rows], cfg["generation"])
    valid = [parse_number_completion(text, minimum=cfg["data"]["integer_min"], maximum=cfg["data"]["integer_max"], max_values=cfg["data"]["max_added_values"]).valid for text in outputs]
    return {"n": len(outputs), "valid_rate": sum(valid) / len(valid)}, [{"prompt": row["prompt"], "output": text, "valid": value} for row, text, value in zip(rows, outputs, valid, strict=True)]


@torch.inference_mode()
def wikitext_loss(model, tokenizer, texts):
    losses = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
        if ids.shape[1] > 1:
            losses.append(float(model(input_ids=ids, labels=ids).loss))
    return {"n": len(losses), "mean_nll": sum(losses) / len(losses), "perplexity": float(torch.exp(torch.tensor(losses).mean()))}


def evaluate(cfg, adapter, condition, prompt_rows, generic, samples, numeric_count, seed, out, gates):
    tokenizer = load_tokenizer(cfg)
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    model = load_model(cfg, adapter=str(adapter)).eval()
    params = {canonical(name): p for name, p in model.named_parameters() if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name}
    summary = []
    try:
        heldout = numeric_rows(cfg, condition, numeric_count, seed)
        for gate in gates:
            gate_meta = apply_gate(params, state, gate)
            pref, pref_rows = preference(model, tokenizer, cfg, prompt_rows, samples, seed)
            num, num_rows = numeric(model, tokenizer, cfg, heldout, seed + 1)
            loss = wikitext_loss(model, tokenizer, generic)
            row = {"adapter_condition": condition, **gate_meta, "preference": pref, "numeric": num, "generic_loss": loss}
            summary.append(row)
            write_jsonl(out / "raw" / f"{condition}_{gate}_trait.jsonl", pref_rows)
            write_jsonl(out / "raw" / f"{condition}_{gate}_numeric.jsonl", num_rows)
    finally:
        del model, state; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--final-root", type=Path, default=Path("experiments/subliminal_learning/frozen_evaluation/v1"))
    p.add_argument("--output-name", default="lora_group_localization_v1")
    p.add_argument("--mode", choices=("standard", "early_mlp"), default="standard")
    p.add_argument("--samples-per-prompt", type=int, default=10)
    p.add_argument("--numeric-examples", type=int, default=128)
    p.add_argument("--generic-examples", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261109)
    args = p.parse_args()
    cfg = load_config(args.config); root = Path(cfg["experiment"]["output_root"])
    manifest = json.loads((args.final_root / "manifest.json").read_text())
    prompts = read_jsonl(args.final_root / manifest["trait_prompts"]["path"])
    if digest(prompts) != manifest["trait_prompts"]["sha256"]: raise ValueError("Frozen trait prompts failed hash check")
    generic = [r["text"] for r in read_jsonl(args.final_root / manifest["generic_loss"]["frozen"]["path"])[:args.generic_examples]]
    out = root / args.output_name; out.mkdir(parents=True, exist_ok=True); gates=conditions(args.mode)
    ws = evaluate(cfg, root / "models/trait/adapter", "trait", prompts, generic, args.samples_per_prompt, args.numeric_examples, args.seed, out, gates)
    wc = evaluate(cfg, root / "models/control/adapter", "control", prompts, generic, args.samples_per_prompt, args.numeric_examples, args.seed, out, gates)
    paired = []
    wc_by_gate = {r["condition"]: r for r in wc}
    for r in ws:
        c = wc_by_gate[r["condition"]]
        paired.append({"condition": r["condition"], "sl_effect_pp": 100 * (r["preference"]["target_rate"] - c["preference"]["target_rate"]), "ws": r, "wc": c})
    write_json(out / "summary.json", {"config": str(args.config), "frozen_root": str(args.final_root), "mode":args.mode, "conditions":gates, "samples_per_prompt": args.samples_per_prompt, "numeric_examples": args.numeric_examples, "generic_examples": args.generic_examples, "paired": paired})


if __name__ == "__main__": main()
