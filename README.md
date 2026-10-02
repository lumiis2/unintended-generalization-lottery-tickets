# Lottery Tickets Underlying Unintended Generalization

Research code for investigating whether sparse subnetworks underlie unintended
generalization in language models. The initial focus is reproducing subliminal
learning in open-weight models before testing whether the behavior reuses a
pre-existing mechanism or is created during fine-tuning.

## Repository structure

```text
configs/                         Experiment configurations
experiments/subliminal_learning Reproduction protocol
scripts/sl/                      Data, training, and evaluation entry points
scripts/slurm/                   Cluster launchers
src/spar/sl/                     Shared subliminal-learning code
tests/                           CPU-only tests
paper/                           Manuscript source
references/                      Public reference notes
```

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the staged research plan and
[experiments/subliminal_learning/README.md](experiments/subliminal_learning/README.md)
for the current pilot workflow.

This repository is experimental. Generated datasets, model weights, logs, private
infrastructure notes, and machine-local configuration are intentionally excluded.

