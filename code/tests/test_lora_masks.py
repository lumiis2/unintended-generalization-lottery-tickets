import torch

from spar.sl.lora_masks import (
    complement_unit_mask,
    lora_pairs,
    magnitude_scores,
    masked_state,
    parameter_mask,
    unit_mask_from_scores,
)


def state():
    return {
        "layer.lora_A.weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "layer.lora_B.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }


def test_finds_lora_pair():
    assert lora_pairs(state())[0].path == "layer"


def test_feature_mask_zeros_columns_of_effective_update():
    weights = state()
    scores = magnitude_scores(weights, "feature")
    units = unit_mask_from_scores(scores, 2 / 3)
    mask = parameter_mask(weights, units, "feature")
    pruned = masked_state(weights, mask)
    update = pruned["layer.lora_B.weight"] @ pruned["layer.lora_A.weight"]
    assert int((update.norm(dim=0) > 0).sum()) == 1


def test_feature_complement_keeps_unselected_effective_update_columns():
    weights = state()
    units = unit_mask_from_scores(magnitude_scores(weights, "feature"), 2 / 3)
    complement = parameter_mask(weights, complement_unit_mask(units), "feature")
    pruned = masked_state(weights, complement)
    update = pruned["layer.lora_B.weight"] @ pruned["layer.lora_A.weight"]
    assert int((update.norm(dim=0) > 0).sum()) == 2
    assert bool(complement["layer.lora_B.weight"].all())


def test_rank_mask_zeros_whole_rank_components():
    weights = state()
    units = unit_mask_from_scores(magnitude_scores(weights, "rank"), 0.5)
    mask = parameter_mask(weights, units, "rank")
    pruned = masked_state(weights, mask)
    assert int((pruned["layer.lora_A.weight"].abs().sum(dim=1) > 0).sum()) == 1
    assert int((pruned["layer.lora_B.weight"].abs().sum(dim=0) > 0).sum()) == 1
