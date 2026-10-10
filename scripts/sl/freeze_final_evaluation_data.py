#!/usr/bin/env python3
"""Freeze the external data assets for the final SL mask evaluation.

This script intentionally does not print examples.  It records the selected
rows, their source metadata, and hashes in the pre-registered manifest, so a
later evaluation neither downloads nor silently changes its data.
"""
from __future__ import annotations

import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


OUT = Path("experiments/subliminal_learning/final_evaluation/v1")
MANIFEST = OUT / "manifest.json"
SEED = 20261010


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return {"path": str(path.relative_to(OUT)), "count": len(rows), "sha256": sha256_bytes(payload)}


def select(dataset: Dataset, n: int, seed_offset: int) -> tuple[list[int], Dataset]:
    if len(dataset) < n:
        raise ValueError(f"Requested {n} examples from a dataset with only {len(dataset)} rows")
    indices = list(range(len(dataset)))
    random.Random(SEED + seed_offset).shuffle(indices)
    indices = indices[:n]
    return indices, dataset.select(indices)


def source_metadata(dataset: Dataset, *, dataset_id: str, config: str, split: str) -> dict[str, Any]:
    return {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "dataset_fingerprint": dataset._fingerprint,
        "source_row_count": len(dataset),
    }


def frozen_rows(dataset: Dataset, indices: list[int]) -> list[dict[str, Any]]:
    return [{"source_index": index, "example": row} for index, row in zip(indices, dataset)]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] not in {"pre_registered_unrun", "frozen_unrun"}:
        raise ValueError(f"Refusing to overwrite a non-preregistered manifest: {manifest['status']}")

    # Generic next-token-loss data. Filter before shuffling so the stated rule
    # is exactly implemented rather than depending on empty-document frequency.
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    wiki_nonempty = wiki.filter(lambda row: bool(row["text"].strip()))
    wiki_indices, wiki_selected = select(wiki_nonempty, 512, seed_offset=1)
    generic_rows = [{"source_index_after_nonempty_filter": index, "text": row["text"]} for index, row in zip(wiki_indices, wiki_selected)]

    specifications = {
        "hellaswag": ("Rowan/hellaswag", "default", "validation", 11),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", 12),
        "gsm8k": ("openai/gsm8k", "main", "test", 13),
    }
    benchmark_assets: dict[str, Any] = {}
    for task, (dataset_id, config, split, seed_offset) in specifications.items():
        dataset = load_dataset(dataset_id, config, split=split)
        indices, selected = select(dataset, 250, seed_offset)
        metadata = source_metadata(dataset, dataset_id=dataset_id, config=config, split=split)
        metadata.update(write_jsonl(OUT / "benchmarks" / f"{task}.jsonl", frozen_rows(selected, indices)))
        benchmark_assets[task] = metadata

    generic_metadata = source_metadata(wiki_nonempty, dataset_id="Salesforce/wikitext", config="wikitext-103-raw-v1", split="validation (non-empty filter)")
    generic_metadata.update(write_jsonl(OUT / "generic_loss.jsonl", generic_rows))
    manifest["generic_loss"]["frozen"] = generic_metadata
    manifest["benchmarks"]["frozen"] = benchmark_assets
    manifest["data_freeze"] = {
        "script": "scripts/sl/freeze_final_evaluation_data.py",
        "datasets_version": version("datasets"),
        "selection_seed": SEED,
        "policy": "The JSONL assets and SHA-256 values, not a mutable remote dataset revision, are authoritative for final evaluation.",
    }
    manifest["status"] = "frozen_unrun"
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Frozen final-evaluation data assets; inspect hashes, not examples, before the final run.")


if __name__ == "__main__":
    main()
