# Subliminal Learning Experiment Log

This is the durable index for experiments run by this project. Detailed
write-ups remain in linked records; this file answers *what was run, where the
result is, and what status it has*.

## Logging convention

For every new experiment, add a dated entry under `experiment_log/` before or
immediately after submitting it. Each entry should state:

1. question and hypothesis;
2. exact checkpoint/configuration and data provenance;
3. pre-specified primary metric and controls;
4. Slurm/RunPod job IDs and artifact location;
5. result, interpretation, and limitations;
6. whether it is exploratory, validation, or confirmatory.

Do not overwrite older conclusions after new results: append a new record that
states what changed.

## Current record index

| Date / record | Status | Summary |
|---|---|---|
| [2026-10-22 current state](experiment_log/2026-10-22_current_state.md) | completed | Consolidated reproduction, LoRA-mask, capability, activation, and visualization results. |
| [Experiment history](EXPERIMENT_HISTORY.md) | completed | Full numerical record and interpretation of all completed SL work. |
| [Future directions](FUTURE_DIRECTIONS.md) | planning | Base-versus-LoRA localization experiments; not evidence. |

## Artifact roots

- Main run artifacts: `artifacts/sl/qwen25_7b_{cat,owl}_replication*`.
- Mask analysis and overlap figures: `artifacts/sl/mask_analysis/`.
- Additional summary figures: `artifacts/sl/additional_analysis/`.

## Separate external/colleague work

Other researchers' experiments are indexed in
[COLLEAGUE_EXPLORATIONS.md](COLLEAGUE_EXPLORATIONS.md). They must not be folded
into claims about this project's controlled results until their configuration
and controls have been independently audited.
