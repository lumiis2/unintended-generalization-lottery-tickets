# 2026-10-22 — Consolidated SL research state

**Type:** completed-result record.  
**Full record:** [../EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md).

## Completed experiment groups

1. Controlled behavioural SL reproduction in Qwen2.5-7B-Instruct:
   three cat runs and one owl run, with matched neutral-teacher `WC` controls.
2. Exploratory and held-out LoRA mask experiments:
   Magnitude, Wanda, Taylor, and Taylor-contrastive, followed by final frozen
   evaluation of Magnitude and Taylor-contrastive supports at 30% density.
3. Retention controls:
   held-out numeric completion, generic loss, small benchmark panel, and a
   blinded coherence judge.
4. Activation analysis:
   teacher/student directions, matched neutral-system-prompt control, then
   causal projection/steering intervention.
5. Descriptive post-hoc analyses:
   support overlap, score correlations, distribution by layer/module, and
   difference-in-differences plots.

## Main conclusions

- Cat SL is robust across the three completed seeds; owl is a one-seed
  cross-trait result.
- Both Magnitude and Taylor-contrastive identify LoRA supports more causally
  relevant than equal-density random masks. Taylor-contrastive is stronger;
  Magnitude remains the less behaviour-targeted baseline.
- A prompt-controlled teacher direction causally changes cat SL behaviour, but
  this does not localize the responsible frozen base weights.

## Limits retained for future work

- LoRA masks were evaluated with the entire base model intact.
- Owl causal activation evidence is inconclusive.
- Coherence needs blinded manual audit because empty/truncated generations are
  nontrivial, despite no selected-mask-specific degradation.
