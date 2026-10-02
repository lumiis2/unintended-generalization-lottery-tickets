# Subliminal Learning reproduction

## Scope of the first experiment

This pipeline reproduces the animal-preference-through-numbers setting before any
mechanistic analysis. It implements four states:

- `base` = W0;
- `teacher` = W0 with a trait system prompt (WT; weights unchanged);
- `trait` = W0 plus the adapter trained on teacher numbers (WS);
- `control` = W0 plus the adapter trained on neutral-teacher numbers (WC).

The primary behavioral comparison is `trait - control`, not merely `trait - base`.

## Provenance

From Cloud et al.:

- prompt-conditioned animal-preference teacher;
- unrelated number-sequence task;
- strict formatting filter (1–10 integers, 0–999, consistent separator);
- neutral-teacher control of equal size;
- favorite-animal prompts and number-prefix evaluation variant;
- target-rate comparison across W0/WT/WS/WC.

Adapted from the Wong follow-up:

- open-weight local pipeline;
- LoRA/QLoRA-friendly configuration;
- explicit dataset and evaluation artifacts;
- pilot-scale runs before a 10k-example confirmation.

Project-specific choices (not literature claims):

- Transformers/PEFT implementation;
- Qwen2.5-7B pilot config;
- 2,000-example, three-epoch pilot;
- LoRA rank 16 and alpha 32;
- initial target `cat`, which remains a configurable pilot candidate.

## Install

On a GPU node with a compatible CUDA/PyTorch environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The first cluster session must validate module and CUDA compatibility before using
these commands. Do not assume the historical `.venv310` belongs to this project.

## Pilot workflow

```bash
CONFIG=configs/sl/qwen25_7b_pilot.yaml

# Measure W0 and prompt-conditioned WT before spending compute on training.
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition base
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition teacher

# Generate matched trait/control datasets.
python scripts/sl/generate_numbers.py --config "$CONFIG" --condition trait
python scripts/sl/generate_numbers.py --config "$CONFIG" --condition control

# Train WS and WC from the same base checkpoint.
python scripts/sl/train_student.py --config "$CONFIG" --condition trait
python scripts/sl/train_student.py --config "$CONFIG" --condition control

# Evaluate students and collect the primary comparison.
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition trait
python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition control
python scripts/sl/summarize_run.py --config "$CONFIG"
```

The pipeline writes under `artifacts/sl/qwen25_7b_pilot/`, which is ignored by Git.
Dataset `raw.jsonl`, filtered `train.jsonl`, per-sample evaluations, summaries,
manifests, adapters, and intermediate checkpoints are retained.

## Gates

1. **Teacher gate:** WT must express the target substantially more than W0.
2. **Data gate:** both datasets reach the same requested valid size; filter yield and
   rejection reasons are recorded.
3. **Training gate:** both adapters complete under identical hyperparameters.
4. **Transmission gate:** WS must exceed WC on held-out prompt variants without an
   obvious generation/capability collapse.

This pilot is diagnostic, not confirmatory. Before confirmatory runs we must pin model
revisions, expand/freeze the evaluation set, add capability metrics, choose traits
without test-set cherry-picking, and define seed-level statistical criteria.

