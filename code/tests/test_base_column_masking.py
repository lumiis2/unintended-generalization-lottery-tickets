"""Regression test for real, reversible frozen-base column masking."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


SCRIPT = Path(__file__).parents[1] / "scripts/sl/masks/evaluate_base_column_masks.py"
SPEC = importlib.util.spec_from_file_location("base_column_masking", SCRIPT)
MASKING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MASKING)


class WrappedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(6, 4, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(6, 2, bias=False)})


class TinyPeftLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = WrappedLinear()


def test_base_column_mask_is_real_and_reversible():
    torch.manual_seed(0)
    model = TinyPeftLikeModel()
    units = {"target": torch.tensor([True, False, True, False, False, False])}
    before = model.target.base_layer.weight.detach().clone()
    lora_before = model.target.lora_A["default"].weight.detach().clone()

    metadata = MASKING.validate_base_mask(model, units)

    assert metadata["forward_changed"]
    assert metadata["restored"]
    assert metadata["lora_unchanged"]
    assert torch.equal(model.target.base_layer.weight, before)
    assert torch.equal(model.target.lora_A["default"].weight, lora_before)
