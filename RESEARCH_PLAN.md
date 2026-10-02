# Research Plan

## Goal

Determine whether unintended behaviors acquired through narrow fine-tuning are
implemented by sparse subnetworks, and whether those subnetworks already exist in
the base model or arise during fine-tuning. The first target phenomenon is
subliminal learning; emergent misalignment is a secondary setting.

## Phase 1: reproduce subliminal learning

Start with animal preferences transmitted through filtered number sequences and
compare four conditions:

- **W0:** the base model;
- **WT:** W0 conditioned by a trait prompt, without weight changes;
- **WS:** a copy of W0 fine-tuned on WT's number sequences;
- **WC:** a matched control fine-tuned on sequences from a neutral teacher.

The primary behavioral comparison is WS versus WC, with W0 and WT establishing the
baseline and teacher strength. Pilot runs will use small datasets and few seeds;
confirmatory runs will use frozen evaluations, multiple seeds, uncertainty estimates,
and basic capability checks.

## Phase 2: establish robustness

Replicate the effect across checkpoints and, if feasible, more than one open-weight
model family. Measure sensitivity to dataset size, training duration, target trait,
and fine-tuning method. Negative results and failed transmission regimes will be
reported rather than discarded.

## Phase 3: identify candidate sparse mechanisms

Search for components that causally mediate the trait in the prompted teacher
(**ST**) and trained student (**SS**). Evaluate necessity, sufficiency, selectivity,
and benign-behavior preservation at matched sparsity against random and magnitude
baselines. Because WT and W0 share weights, ST must be defined through conditional
computation rather than a teacher–base weight difference.

## Phase 4: test mechanism reuse

Test whether a mask discovered in the teacher causally controls the student trait,
and vice versa. Repeat interventions in W0 and WC as controls. Cross-condition causal
transfer is the main evidence for mechanism reuse; mask overlap alone is descriptive.

All runs should preserve exact model revisions, configurations, prompts, seeds,
dataset hashes, checkpoints, and per-sample evaluations. Mechanistic analysis begins
only after the behavioral effect is stable enough to support causal comparisons.
