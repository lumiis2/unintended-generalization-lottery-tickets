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
        self.untargeted = WrappedLinear()


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


def test_targeted_mask_does_not_touch_omitted_base_modules():
    torch.manual_seed(1)
    model = TinyPeftLikeModel()
    target_before = model.target.base_layer.weight.detach().clone()
    other_before = model.untargeted.base_layer.weight.detach().clone()
    units = {"target": torch.tensor([False, True, False, False, True, False])}

    state = MASKING.apply_base_mask(model, units)
    try:
        assert torch.count_nonzero(model.target.base_layer.weight[:, [1, 4]]).item() == 0
        assert torch.equal(model.untargeted.base_layer.weight, other_before)
    finally:
        MASKING.restore_base_mask(state)

    assert torch.equal(model.target.base_layer.weight, target_before)
    assert torch.equal(model.untargeted.base_layer.weight, other_before)
