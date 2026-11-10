"""Unit checks for the declarative early-MLP LoRA gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/sl/activations/localize_lora_groups.py"
SPEC = importlib.util.spec_from_file_location("lora_localization", SCRIPT)
LOCALIZATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LOCALIZATION)


def parameter(layer: int, projection: str) -> str:
    return f"base_model.model.model.layers.{layer}.mlp.{projection}.lora_A.weight"


def test_early_mlp_only_excludes_attention_and_late_layers():
    assert LOCALIZATION.keep(parameter(0, "gate_proj"), "early_mlp_only")
    assert not LOCALIZATION.keep(parameter(0, "q_proj"), "early_mlp_only")
    assert not LOCALIZATION.keep(parameter(9, "gate_proj"), "early_mlp_only")


def test_early_layer_and_projection_gates_are_specific():
    assert LOCALIZATION.keep(parameter(4, "up_proj"), "early_mlp_3_5_only")
    assert not LOCALIZATION.keep(parameter(2, "up_proj"), "early_mlp_3_5_only")
    assert LOCALIZATION.keep(parameter(8, "gate_proj"), "early_gate_only")
    assert not LOCALIZATION.keep(parameter(8, "up_proj"), "early_gate_only")
    assert not LOCALIZATION.keep(parameter(9, "gate_proj"), "early_gate_only")


def test_early_mlp_removal_conditions_keep_expected_complement():
    assert not LOCALIZATION.keep(parameter(1, "down_proj"), "early_mlp_without_0_2")
    assert LOCALIZATION.keep(parameter(5, "down_proj"), "early_mlp_without_0_2")
    assert not LOCALIZATION.keep(parameter(6, "up_proj"), "early_mlp_without_up")
    assert LOCALIZATION.keep(parameter(6, "gate_proj"), "early_mlp_without_up")
