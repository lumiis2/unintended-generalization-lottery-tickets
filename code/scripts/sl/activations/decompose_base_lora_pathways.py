#!/usr/bin/env python3
"""Decompose WS--WC computation into base and LoRA terms module by module.

For every PEFT-wrapped linear projection and final prompt token, the forward
pass satisfies ``y = W0 x + DeltaW x``.  This script records mean vectors for
each term in WS and WC on identical frozen prompts, then reports:

``Delta_total = y_WS - y_WC``
``Delta_base  = W0 (x_WS - x_WC)``
``Delta_lora  = (DeltaW x)_WS - (DeltaW x)_WC``

The reconstruction ``Delta_total ~= Delta_base + Delta_lora`` is checked for
each module.  This is observational bookkeeping, not an ablation: it maps how
adapter-induced input differences are transformed by later frozen base modules.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from spar.sl.config import load_config, read_jsonl, write_json
from spar.sl.modeling import load_model, load_tokenizer, render_chat


FROZEN_ROOT = Path("experiments/subliminal_learning/frozen_evaluation/taylor_numeric_v2")


def jsonl_digest(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def peft_linear_modules(model) -> dict[str, torch.nn.Module]:
    """Return only linear modules carrying at least one active LoRA adapter."""
    result = {}
    for name, module in model.named_modules():
        adapters = getattr(module, "lora_A", None)
        if adapters is None or not hasattr(module, "base_layer"):
            continue
        if not len(adapters):
            continue
        result[name] = module
    if not result:
        raise RuntimeError("No PEFT LoRA linear modules found")
    return result


class MeanTermCapture:
    """Capture final-token input/base/LoRA/output means without storing prompts."""

    TERMS = ("input", "base", "lora", "total")

    def __init__(self, model):
        self.modules = peft_linear_modules(model)
        self.sums: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str):
        def hook(module, inputs, output):
            if not inputs:
                raise RuntimeError(f"No input captured for {name}")
            input_tensor = inputs[0]
            total = tensor_output(output)
            if input_tensor.ndim != 3 or total.ndim != 3:
                raise RuntimeError(f"Expected [batch, sequence, hidden] tensors for {name}")

            # `-1` is the final non-padding prompt token because the project
            # tokenizer uses left padding. The same position is used in WS/WC.
            final_input = input_tensor[:, -1, :]
            final_total = total[:, -1, :]
            with torch.no_grad():
                final_base = module.base_layer(final_input)
            final_lora = final_total - final_base.to(final_total.dtype)
            values = {
                "input": final_input,
                "base": final_base,
                "lora": final_lora,
                "total": final_total,
            }
            for term, value in values.items():
                value = value.detach().float().sum(dim=0).cpu()
                existing = self.sums[name].get(term)
                self.sums[name][term] = value if existing is None else existing + value
            self.counts[name] += final_input.shape[0]

        return hook

    def means(self) -> dict[str, dict[str, torch.Tensor]]:
        result = {}
        for name, terms in self.sums.items():
            count = self.counts[name]
            if count == 0:
                raise RuntimeError(f"No observations recorded for {name}")
            result[name] = {term: vector / count for term, vector in terms.items()}
        return result

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()


def capture_condition(model, tokenizer, prompts: list[str], batch_size: int) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    """Return mean module terms and mean residual stream at the final token."""
    residual_sum = None
    n = 0
    with MeanTermCapture(model) as capture, torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            final_hidden = torch.stack([hidden[:, -1, :].float().sum(dim=0).cpu() for hidden in outputs.hidden_states])
            residual_sum = final_hidden if residual_sum is None else residual_sum + final_hidden
            n += encoded["input_ids"].shape[0]
    if residual_sum is None or n == 0:
        raise RuntimeError("No prompts were evaluated")
    return capture.means(), residual_sum / n


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator else 0.0


def signed_fraction(component: torch.Tensor, total: torch.Tensor) -> float:
    denominator = torch.dot(total, total)
    return float(torch.dot(component, total) / denominator) if denominator else 0.0


def module_metadata(name: str) -> dict[str, str | int | None]:
    parts = name.split(".")
    layer = None
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            layer = int(parts[index + 1])
            break
    projection = parts[-1]
    family = "attention" if any(token in name for token in ("self_attn", "q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
    return {"layer": layer, "projection": projection, "family": family}


def summarise(ws: dict[str, dict[str, torch.Tensor]], wc: dict[str, dict[str, torch.Tensor]], ws_residual: torch.Tensor, wc_residual: torch.Tensor) -> tuple[dict, dict]:
    if set(ws) != set(wc):
        raise ValueError("WS and WC exposed different LoRA module sets")
    rows = []
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    for name in sorted(ws):
        delta = {term: ws[name][term] - wc[name][term] for term in MeanTermCapture.TERMS}
        reconstruction = delta["base"] + delta["lora"]
        residual = delta["total"] - reconstruction
        rows.append({
            "module": name,
            **module_metadata(name),
            "input_delta_l2": float(delta["input"].norm()),
            "base_delta_l2": float(delta["base"].norm()),
            "lora_delta_l2": float(delta["lora"].norm()),
            "total_delta_l2": float(delta["total"].norm()),
            "base_total_cosine": cosine(delta["base"], delta["total"]),
            "lora_total_cosine": cosine(delta["lora"], delta["total"]),
            "base_signed_fraction_of_total": signed_fraction(delta["base"], delta["total"]),
            "lora_signed_fraction_of_total": signed_fraction(delta["lora"], delta["total"]),
            "reconstruction_residual_l2": float(residual.norm()),
            "reconstruction_relative_error": float(residual.norm() / delta["total"].norm()) if delta["total"].norm() else 0.0,
        })
        vectors[name] = {**delta, "reconstruction_residual": residual}

    residual_delta = ws_residual - wc_residual
    residual_rows = []
    for layer in range(residual_delta.shape[0]):
        current = residual_delta[layer]
        previous = residual_delta[layer - 1] if layer else None
        residual_rows.append({
            "hidden_state_index": layer,
            "delta_l2": float(current.norm()),
            "cosine_with_previous": None if previous is None else cosine(current, previous),
            "norm_ratio_to_previous": None if previous is None or not previous.norm() else float(current.norm() / previous.norm()),
        })
    return {"modules": rows, "residual_stream": residual_rows}, {"module_deltas": vectors, "residual_delta": residual_delta}


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.prompts <= 0 or args.batch_size <= 0:
        parser.error("--prompts and --batch-size must be positive")

    config = load_config(args.config)
    root = Path(config["experiment"]["output_root"])
    output = args.output_dir or root / "base_lora_decomposition_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    prompt_path = FROZEN_ROOT / manifest["trait_prompts"]["path"]
    if jsonl_digest(prompt_path) != manifest["trait_prompts"]["sha256"]:
        raise ValueError(f"Frozen trait prompt digest mismatch: {prompt_path}")
    rows = read_jsonl(prompt_path)[: args.prompts]
    tokenizer = load_tokenizer(config)
    prompts = [render_chat(tokenizer, row["prompt"]) for row in rows]

    condition_results = {}
    for condition in ("trait", "control"):
        adapter = root / "models" / condition / "adapter"
        model = load_model(config, adapter=str(adapter)).eval()
        try:
            condition_results[condition] = capture_condition(model, tokenizer, prompts, args.batch_size)
        finally:
            release(model)

    (ws_terms, ws_residual), (wc_terms, wc_residual) = condition_results["trait"], condition_results["control"]
    summary, vectors = summarise(ws_terms, wc_terms, ws_residual, wc_residual)
    summary.update({
        "experiment": "base_lora_decomposition_v1",
        "config": args.config,
        "target": config["trait"]["target"],
        "conditions": {"trait": "WS", "control": "WC"},
        "frozen_manifest": str(FROZEN_ROOT / "manifest.json"),
        "manifest_version": manifest["version"],
        "n_prompts": len(rows),
        "batch_size": args.batch_size,
        "token_position": "final prefill token (left-padded chat prompt)",
        "decomposition": {
            "equation": "Delta_total = Delta_base + Delta_lora",
            "base": "W0(x_WS - x_WC)",
            "lora": "(DeltaW x)_WS - (DeltaW x)_WC",
        },
        "interpretation": (
            "Observational decomposition of mean final-token module outputs. "
            "A large base term means a frozen projection transforms an upstream WS-WC input difference; "
            "it does not by itself show that the corresponding base weights are necessary for SL."
        ),
    })
    write_json(output / "summary.json", summary)
    torch.save({
        "ws_module_means": ws_terms,
        "wc_module_means": wc_terms,
        "ws_residual_mean": ws_residual,
        "wc_residual_mean": wc_residual,
        **vectors,
    }, output / "mean_vectors.pt")
    print(f"Wrote base/LoRA decomposition to {output}")


if __name__ == "__main__":
    main()
