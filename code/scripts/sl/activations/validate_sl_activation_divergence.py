#!/usr/bin/env python3
"""Compare SL activation divergence with independent-neutral and mask controls."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from spar.sl.config import load_config, read_jsonl, write_json, write_jsonl
from spar.sl.lora_masks import complement_unit_mask, parameter_mask, unit_mask_from_scores
from spar.sl.modeling import batched_generate, load_model, load_tokenizer, render_chat


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()).hexdigest()


def key(name: str) -> str:
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def matched_random_units(selected: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Sample the same number of feature units inside every LoRA module."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for name, support in selected.items():
        flat = torch.zeros(support.numel(), dtype=torch.bool)
        count = int(support.sum())
        if count:
            flat[torch.randperm(support.numel(), generator=generator)[:count]] = True
        result[name] = flat.reshape_as(support)
    return result


def masked(model, adapter: Path, score: Path | None, random_seed: int | None) -> None:
    if score is None:
        return
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    payload = torch.load(score, map_location="cpu", weights_only=False)
    selected = unit_mask_from_scores(payload["scores"], 0.70)
    units = selected if random_seed is None else matched_random_units(selected, random_seed)
    mask = parameter_mask(state, complement_unit_mask(units), "feature")  # ablate the selected 30%
    params = {key(n): p for n, p in model.named_parameters() if ".lora_A.default.weight" in n or ".lora_B.default.weight" in n}
    with torch.no_grad():
        for name, keep in mask.items():
            params[name].copy_(state[name].to(params[name].device, params[name].dtype) * keep.to(params[name].device, params[name].dtype))


@torch.inference_mode()
def hidden(model, tokenizer, prompts: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True).to(model.device)
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        chunks.append(torch.stack([h[:, -1, :].float().cpu() for h in outputs.hidden_states], dim=1))
    return torch.cat(chunks, dim=0)


def preference(
    model, tokenizer, cfg: dict, rows: list[dict], samples: int, generation_seed: int
) -> tuple[dict, list[dict]]:
    expanded = [row for row in rows for _ in range(samples)]
    # Hold generation noise fixed across WS, WC, WC2, and masked WS variants.
    torch.manual_seed(generation_seed)
    outputs = batched_generate(model, tokenizer, [render_chat(tokenizer, r["prompt"]) for r in expanded], cfg["evaluation"])
    target = cfg["trait"]["target"].casefold()
    pat = re.compile(rf"\b{re.escape(target)}s?\b", re.I)
    records = [{"prompt_id": r.get("id", i // samples), "sample": i % samples, "output": text, "target_hit": bool(pat.search(text))} for i, (r, text) in enumerate(zip(expanded, outputs, strict=True))]
    return {"n": len(records), "target_rate": sum(r["target_hit"] for r in records) / len(records), "target": target}, records


def capture(
    cfg: dict,
    adapter: Path,
    prompts: list[str],
    rows: list[dict],
    score: Path | None,
    random_seed: int | None,
    samples: int,
    generation_seed: int,
) -> tuple[torch.Tensor, dict, list[dict]]:
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, adapter=str(adapter)).eval()
    try:
        masked(model, adapter, score, random_seed)
        states = hidden(model, tokenizer, prompts, int(cfg["evaluation"]["batch_size"]))
        metric, outputs = preference(model, tokenizer, cfg, rows, samples, generation_seed)
        return states, metric, outputs
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--wc2-config", required=True, type=Path)
    p.add_argument("--final-root", type=Path, default=Path("experiments/subliminal_learning/frozen_evaluation/v1"))
    p.add_argument("--output-dir", default="activation_divergence_validation_v1")
    p.add_argument("--samples-per-prompt", type=int, default=30)
    p.add_argument("--random-replicates", type=int, default=3)
    p.add_argument("--include-magnitude", action="store_true")
    args = p.parse_args()
    cfg, wc2_cfg = load_config(args.config), load_config(args.wc2_config)
    root, wc2_root = Path(cfg["experiment"]["output_root"]), Path(wc2_cfg["experiment"]["output_root"])
    manifest = json.loads((args.final_root / "manifest.json").read_text())
    rows = read_jsonl(args.final_root / manifest["trait_prompts"]["path"])
    if digest(rows) != manifest["trait_prompts"]["sha256"]:
        raise ValueError("Frozen prompt digest mismatch")
    tokenizer = load_tokenizer(cfg)
    prompts = [render_chat(tokenizer, r["prompt"]) for r in rows]
    output = root / args.output_dir; output.mkdir(parents=True, exist_ok=True)
    ws = root / "models/trait/adapter"; wc = root / "models/control/adapter"; wc2 = wc2_root / "models/control/adapter"
    for path in (ws, wc, wc2):
        if not (path / "adapter_model.safetensors").is_file(): raise FileNotFoundError(path)
    conditions = [("ws_full", ws, None, None), ("wc_full", wc, None, None), ("wc2_full", wc2, None, None),
                  ("ws_taylor_contrastive_ablated", ws, root / "mask_scores/trait/taylor_contrastive_feature.pt", None)]
    if args.include_magnitude: conditions.append(("ws_magnitude_ablated", ws, root / "mask_scores/trait/magnitude_feature.pt", None))
    for replicate in range(args.random_replicates): conditions.append((f"ws_random_ablated_r{replicate:02d}", ws, root / "mask_scores/trait/taylor_contrastive_feature.pt", 20261107 + replicate))
    captured = {}
    for label, adapter, score, seed in conditions:
        states, behaviour, outputs = capture(
            cfg if adapter != wc2 else wc2_cfg,
            adapter,
            prompts,
            rows,
            score,
            seed,
            args.samples_per_prompt,
            generation_seed=20261107,
        )
        captured[label] = (states, behaviour)
        write_jsonl(output / f"{label}_responses.jsonl", outputs)
    reference = captured["wc_full"][0]
    summaries = []
    for label, (states, behaviour) in captured.items():
        if label == "wc_full": continue
        distances = (states - reference).norm(dim=-1)  # prompt × hidden-state layer
        mean = distances.mean(dim=0)
        peak = int(mean.argmax())
        record = {"comparison": f"{label} vs wc_full", "label": label, "peak_layer": peak, "peak_l2": float(mean[peak]), "final_l2": float(mean[-1]), "trajectory": mean.tolist(), "per_prompt_trajectory": distances.tolist(), "behaviour": behaviour}
        if label.startswith("ws"):
            record["sl_gap_vs_wc"] = behaviour["target_rate"] - captured["wc_full"][1]["target_rate"]
            record["sl_retained_vs_full"] = record["sl_gap_vs_wc"] / max(1e-12, captured["ws_full"][1]["target_rate"] - captured["wc_full"][1]["target_rate"])
        summaries.append(record)
    write_json(output / "summary.json", {
        "config": str(args.config),
        "wc2_config": str(args.wc2_config),
        "frozen_manifest": str(args.final_root / "manifest.json"),
        "n_prompts": len(rows),
        "samples_per_prompt": args.samples_per_prompt,
        "generation_seed": 20261107,
        "conditions": summaries,
    })
    write_jsonl(output / "conditions.jsonl", summaries)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "label", "comparison", "peak_layer", "peak_l2", "final_l2",
            "target_rate", "sl_gap_vs_wc", "sl_retained_vs_full",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "label": row["label"],
                "comparison": row["comparison"],
                "peak_layer": row["peak_layer"],
                "peak_l2": row["peak_l2"],
                "final_l2": row["final_l2"],
                "target_rate": row["behaviour"]["target_rate"],
                "sl_gap_vs_wc": row.get("sl_gap_vs_wc"),
                "sl_retained_vs_full": row.get("sl_retained_vs_full"),
            })


if __name__ == "__main__":
    main()
