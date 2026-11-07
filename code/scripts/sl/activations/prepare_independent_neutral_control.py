#!/usr/bin/env python3
"""Derive a reproducible WC2 configuration without touching the original run."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-config", type=Path, required=True)
    p.add_argument("--output-config", type=Path, required=True)
    p.add_argument("--seed-offset", type=int, default=10_000)
    args = p.parse_args()
    raw = yaml.safe_load(args.parent_config.read_text())
    parent_root = Path(raw["experiment"]["output_root"])
    seed = int(raw["experiment"]["seed"])
    raw["experiment"]["name"] += "_independent_neutral_control2"
    raw["experiment"]["seed"] = seed + args.seed_offset
    raw["experiment"]["output_root"] = str(parent_root / "independent_neutral_control2")
    raw["data"]["selection_seed"] = int(raw["data"].get("selection_seed", seed)) + args.seed_offset
    raw["evaluation"]["number_prefix_seed"] = int(raw["evaluation"].get("number_prefix_seed", seed)) + args.seed_offset
    rendered = yaml.safe_dump(raw, sort_keys=False)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    if args.output_config.exists():
        existing = args.output_config.read_text(encoding="utf-8")
        if existing != rendered:
            raise ValueError(
                f"Refusing to overwrite a different frozen WC2 config: {args.output_config}"
            )
        print(f"WC2 config already frozen: {args.output_config}")
        return
    args.output_config.write_text(rendered, encoding="utf-8")
    print(f"Wrote frozen WC2 config: {args.output_config}")


if __name__ == "__main__":
    main()
