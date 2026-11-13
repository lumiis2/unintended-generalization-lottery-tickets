#!/usr/bin/env python3
"""Calibrate and evaluate localized frozen-base column masks.

Calibration reads no trait prompts and uses numeric/Wikitext slices disjoint
from the frozen behavioural screen.  Screen mode requires the manifest written
by ``analyze_targeted_base_calibration.py`` and evaluates only density/method
combinations that passed the pre-registered preservation gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from spar.sl.config import write_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_base_column_masks import (
    apply_base_mask,
    base_modules,
    digest,
    generic_loss,
    layer_projection,
    load_config,
    load_model,
    load_tokenizer,
    numeric_eval,
    preference,
    read_jsonl,
    restore_base_mask,
    selected_units,
    validate_base_mask,
    write_json,
)

SCOPES = {
    **{
        f"l{layer}_{projection}": {"layers": [layer], "projections": [projection]}
        for layer in (3, 4, 5)
        for projection in ("down_proj", "gate_proj", "up_proj")
    },
    **{
        f"l{layer}_o_proj": {"layers": [layer], "projections": ["o_proj"]}
        for layer in (3, 4, 5)
    },
    "early_0_2_mlp": {"layers": [0, 1, 2], "projections": ["down_proj", "gate_proj", "up_proj"]},
    "early_6_8_mlp": {"layers": [6, 7, 8], "projections": ["down_proj", "gate_proj", "up_proj"]},
    "middle_13_15_mlp": {"layers": [13, 14, 15], "projections": ["down_proj", "gate_proj", "up_proj"]},
    "late_21_23_mlp": {"layers": [21, 22, 23], "projections": ["down_proj", "gate_proj", "up_proj"]},
}
METHODS = ("magnitude_column", "taylor_contrastive_column")


def scoped_scores(model, scores: dict[str, torch.Tensor], spec: dict) -> dict[str, torch.Tensor]:
    selected = {}
    for name in base_modules(model):
        layer, projection = layer_projection(name)
        if layer in spec["layers"] and projection in spec["projections"]:
            selected[name] = scores[name]
    if not selected:
        raise ValueError(f"Scope selected no PEFT-wrapped base modules: {spec}")
    return selected


def read_scores(score_root: Path, method: str) -> dict[str, torch.Tensor]:
    path = score_root / f"{method}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["metadata"]["method"] != method:
        raise ValueError(f"Unexpected score artifact at {path}")
    return payload["scores"]


def generic_texts(final_root: Path, manifest: dict, count: int, *, calibration: bool) -> list[str]:
    rows = read_jsonl(final_root / manifest["generic_loss"]["frozen"]["path"])
    start = count if calibration else 0
    texts = [row["text"] for row in rows[start : start + count]]
    if len(texts) != count:
        raise ValueError("Insufficient frozen generic texts")
    return texts


def evaluate_condition(
    model,
    tokenizer,
    cfg: dict,
    condition: str,
    units: dict[str, torch.Tensor] | None,
    generic: list[str],
    numeric_count: int,
    numeric_seed: int,
    numeric_offset: int,
    prompts: list[dict] | None,
    samples_per_prompt: int,
) -> tuple[dict, dict[str, list[dict]]]:
    """Evaluate every metric while the exact same reversible mask is active."""
    state = None
    try:
        if units is not None:
            state = apply_base_mask(model, units)
        numeric, numeric_raw = numeric_eval(
            model, tokenizer, cfg, condition, numeric_count, numeric_seed, numeric_offset
        )
        metrics = {"numeric": numeric, "generic_loss": generic_loss(model, tokenizer, generic)}
        raw = {"numeric": numeric_raw}
        if prompts is not None:
            trait, trait_raw = preference(
                model, tokenizer, cfg, prompts, samples_per_prompt, numeric_seed + 17
            )
            metrics["preference"] = trait
            raw["trait"] = trait_raw
        return metrics, raw
    finally:
        if state is not None:
            restore_base_mask(state)


def paired_counts(selected: dict[str, torch.Tensor], random_units: dict[str, torch.Tensor]) -> None:
    if set(selected) != set(random_units):
        raise AssertionError("Selected and random masks target different modules")
    for name, selected_mask in selected.items():
        if int(selected_mask.sum()) != int(random_units[name].sum()):
            raise AssertionError(f"Unequal selected/random count in {name}")


def coordinates(units: dict[str, torch.Tensor] | None) -> dict[str, list[int]]:
    if units is None:
        return {}
    return {
        name: torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for name, mask in units.items()
    }


def allowed_keys(path: Path, scope: str) -> set[tuple[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen_after_capability_only_calibration":
        raise ValueError(f"Unexpected calibration manifest status: {payload.get('status')}")
    return {
        (row["method"], float(row["density"]))
        for row in payload["accepted"]
        if row["scope"] == scope
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--final-root", type=Path, default=Path("experiments/subliminal_learning/frozen_evaluation/v1"))
    parser.add_argument("--score-output-name", default="base_column_masking_v2")
    parser.add_argument("--output-name", default="targeted_base_masking_v1")
    parser.add_argument("--mode", choices=("calibration", "screen"), default="calibration")
    parser.add_argument("--calibration-manifest", type=Path, default=Path("artifacts/sl/targeted_base_masking_v1/calibration_manifest.json"))
    parser.add_argument("--densities", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    parser.add_argument("--random-replicates", type=int, default=1)
    parser.add_argument("--numeric-examples", type=int, default=64)
    parser.add_argument("--generic-examples", type=int, default=64)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["experiment"]["output_root"])
    output = root / args.output_name / args.scope
    output.mkdir(parents=True, exist_ok=True)
    score_root = root / args.score_output_name / "scores"
    manifest = json.loads((args.final_root / "manifest.json").read_text())
    calibration = args.mode == "calibration"
    generic = generic_texts(args.final_root, manifest, args.generic_examples, calibration=calibration)
    prompts = None if calibration else read_jsonl(args.final_root / manifest["trait_prompts"]["path"])
    if prompts is not None and digest(prompts) != manifest["trait_prompts"]["sha256"]:
        raise ValueError("Frozen trait prompts failed hash check")

    numeric_seed = int(manifest["numeric_slice"]["evaluation_data_seed"])
    numeric_offset = 0 if calibration else int(manifest["numeric_slice"]["offset"])
    allowed = None if calibration else allowed_keys(args.calibration_manifest, args.scope)
    score_maps = {method: read_scores(score_root, method) for method in METHODS}

    adapter_paths = {"ws": root / "models/trait/adapter", "wc": root / "models/control/adapter"}
    models = {}
    tokenizer = load_tokenizer(cfg)
    try:
        for label, path in adapter_paths.items():
            models[label] = load_model(cfg, adapter=str(path)).eval()

        scoped = {
            method: scoped_scores(models["ws"], values, SCOPES[args.scope])
            for method, values in score_maps.items()
        }
        validation = {"ws": {}, "wc": {}}
        conditions = [
            {"identifier": "intact", "method": None, "density": 0.0, "branch": "intact", "units": None}
        ]
        for density in args.densities:
            safe_methods = [
                method for method in METHODS if allowed is None or (method, float(density)) in allowed
            ]
            if not safe_methods:
                continue
            for method in safe_methods:
                selected = selected_units(scoped[method], density)
                key = f"{method}_d{density}"
                for label, model in models.items():
                    validation[label][key] = validate_base_mask(model, selected)
                conditions.append(
                    {
                        "identifier": f"{method}_d{int(density * 10000):03d}_selected",
                        "method": method,
                        "density": density,
                        "branch": "selected",
                        "units": selected,
                    }
                )

            # Random masks depend only on module shapes, density, and seed.
            # They are shared by both methods instead of evaluated twice.
            template_method = safe_methods[0]
            template = selected_units(scoped[template_method], density)
            for replicate in range(args.random_replicates):
                random_seed = args.seed + 1000 * replicate + round(density * 100000)
                random_units = selected_units(scoped[template_method], density, random_seed)
                paired_counts(template, random_units)
                key = f"random_d{density}_r{replicate}"
                for label, model in models.items():
                    validation[label][key] = validate_base_mask(model, random_units)
                conditions.append(
                    {
                        "identifier": f"random_d{int(density * 10000):03d}_r{replicate}",
                        "method": "random",
                        "density": density,
                        "branch": "random",
                        "random_seed": random_seed,
                        "units": random_units,
                    }
                )

        mask_root = output / "masks"
        raw_root = output / "raw"
        condition_root = output / ("calibration_conditions" if calibration else "screen_conditions")
        mask_root.mkdir(parents=True, exist_ok=True)
        raw_root.mkdir(parents=True, exist_ok=True)
        condition_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for condition in conditions:
            row_path = condition_root / f"{condition['identifier']}.json"
            required_raw = [
                raw_root / f"{condition['identifier']}_ws_numeric.jsonl",
                raw_root / f"{condition['identifier']}_wc_numeric.jsonl",
            ]
            if not calibration:
                required_raw.extend(
                    [
                        raw_root / f"{condition['identifier']}_ws_trait.jsonl",
                        raw_root / f"{condition['identifier']}_wc_trait.jsonl",
                    ]
                )
            if row_path.is_file() and all(path.is_file() for path in required_raw):
                rows.append(json.loads(row_path.read_text()))
                print(f"SKIP {args.scope} {condition['identifier']}", flush=True)
                continue
            row = {key: value for key, value in condition.items() if key != "units"}
            row["module_count"] = 0 if condition["units"] is None else len(condition["units"])
            row["selected_columns"] = 0 if condition["units"] is None else sum(
                int(mask.sum()) for mask in condition["units"].values()
            )
            write_json(mask_root / f"{condition['identifier']}.json", coordinates(condition["units"]))
            ws_metrics, ws_raw = evaluate_condition(
                models["ws"], tokenizer, cfg, "trait", condition["units"], generic,
                args.numeric_examples, numeric_seed, numeric_offset, prompts, args.samples_per_prompt,
            )
            wc_metrics, wc_raw = evaluate_condition(
                models["wc"], tokenizer, cfg, "control", condition["units"], generic,
                args.numeric_examples, numeric_seed, numeric_offset, prompts, args.samples_per_prompt,
            )
            row["ws"] = ws_metrics
            row["wc"] = wc_metrics
            if prompts is not None:
                row["sl_effect_pp"] = 100 * (
                    ws_metrics["preference"]["target_rate"] - wc_metrics["preference"]["target_rate"]
                )
                write_jsonl(raw_root / f"{condition['identifier']}_ws_trait.jsonl", ws_raw["trait"])
                write_jsonl(raw_root / f"{condition['identifier']}_wc_trait.jsonl", wc_raw["trait"])
            write_jsonl(raw_root / f"{condition['identifier']}_ws_numeric.jsonl", ws_raw["numeric"])
            write_jsonl(raw_root / f"{condition['identifier']}_wc_numeric.jsonl", wc_raw["numeric"])
            write_json(row_path, row)
            rows.append(row)
            print(f"DONE {args.scope} {condition['identifier']}", flush=True)

        summary = {
            "mode": args.mode,
            "config": str(args.config),
            "scope": args.scope,
            "scope_spec": SCOPES[args.scope],
            "score_root": str(score_root),
            "densities": args.densities,
            "random_replicates": args.random_replicates,
            "numeric_seed": numeric_seed,
            "numeric_offset": numeric_offset,
            "generic_slice": [args.generic_examples, 2 * args.generic_examples] if calibration else [0, args.generic_examples],
            "calibration_manifest": None if calibration else str(args.calibration_manifest),
            "validation": validation,
            "rows": rows,
        }
        write_json(
            output / ("calibration_summary.json" if calibration else "screen_summary.json"),
            summary,
        )
    finally:
        for model in models.values():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
