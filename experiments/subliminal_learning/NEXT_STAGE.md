# Next-stage mask validation and activation analysis

These jobs follow the completed exploratory LoRA-mask screen. They do not yet
test causal interventions in the frozen base model.

## 1. Held-out mask validation

`sl_mask_validation_a100.sbatch` evaluates pre-specified candidates:

- magnitude feature mask at 30% support;
- Taylor-trait feature masks at 10% and 30%;
- Taylor-contrastive feature masks at 10% and 30%.

For each `WS` and matched `WC`, it evaluates selected-support and
selected-ablation masks, plus three independently sampled masks of each random
control type. The trait readout uses 50 new animal-preference templates that
were not used by the historical Taylor calibration or primary evaluation. The
numeric readout uses 256 held-out examples after a deterministic offset of 128,
disjoint from the historical numeric calibration/evaluation subset.

This is a **validation** set: it chooses which candidate, if any, advances to a
final test. It must not be used as that final test.

## 2. Mask stability

`analyze_mask_stability.py` computes pairwise Jaccard overlap of feature masks
between the independently trained cat seeds and owl run. It compares observed
overlap with the expected overlap of equal-density random masks. Overlap is
descriptive, not causal evidence.

## 3. Teacher/student activation directions

`sl_activation_analysis_a100.sbatch` captures final-prompt-token residual
streams for 50 held-out animal prompts in four states:

- base model, neutral prompt;
- same base model, trait-conditioned teacher prompt;
- trait LoRA student, no teacher prompt;
- neutral-control LoRA student, no teacher prompt.

It reports per-layer cosine alignment between
`teacher − neutral` and `student − control`. Alignment would motivate causal
activation patching/ablation, but cannot by itself show shared weights or a
steering direction.
