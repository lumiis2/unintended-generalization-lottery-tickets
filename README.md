# Lottery Tickets Underlying Unintended Generalization

This repository studies whether sparse subnetworks explain **unintended
generalization** in language models. Its current focus is **Subliminal Learning
(SL)**: a student can acquire a teacher's behavioural trait from data that is
semantically unrelated to that trait. The central question is whether the
mechanism is localized in the LoRA fine-tuning update, reused from the base
model, or depends on both.

The work is part of SPAR and builds on [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805),
alongside related work on Emergent Misalignment and lottery tickets.

## Current status

- Reproduced prompt-conditioned SL in Qwen2.5-7B for three `cat` seeds and one
  `owl` seed.
- Found selective LoRA supports with Magnitude and Taylor-contrastive masks on
  a frozen held-out evaluation. This is evidence for a **selective LoRA
  support conditional on an intact base**, not yet proof of adapter-only
  localization.
- Identified a causally relevant, controlled activation direction for cats;
  owl replication and base-versus-LoRA localization remain open.

## Start here

- [Research plan](docs/project/research_plan.md) — research questions,
  completed stages, results, decision rules, and next experiments.
- [SL experiment history](docs/subliminal_learning/results/experiment_history.md)
  — detailed methods, numerical results, interpretations, and figures.
- [Official Qwen protocol audit](docs/subliminal_learning/protocol/official_qwen_audit.md)
  — exact reproduction setup.
- [Soham's exploratory pruning audit](docs/subliminal_learning/colleague_explorations/soham_pruning/soham_pruning_audit.pt-BR.md)
  — separate colleague experiment and interpretation (Portuguese).

## Repository structure

```text
docs/                            Canonical plans, protocols, results, and references
experiments/                     Versioned experimental inputs and frozen manifests
code/                            Source, scripts, configurations, tests, and EM legacy material
paper/                           Manuscript source
```

This repository is experimental. Generated datasets, model weights, logs, private
infrastructure notes, and machine-local configuration are intentionally excluded.
