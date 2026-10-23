# Audit of the official Qwen subliminal-learning pipeline

Date: 2026-10-03

This audit compares our exploratory Qwen2.5-7B pilot with the public code released
for Cloud et al. The primary source is the authors' repository at release `v1.0.0`
(commit `59d4199d30c2d15979e92674044e553c59d6d1fe`). The later repository HEAD examined
here changes only documentation and licensing relative to that release. Aria Wong's
Gemma reproduction is a useful secondary source, but it is not the official Qwen
implementation and uses a finetuned teacher rather than our prompt-only teacher.

## Main conclusion

Our completed run was a useful end-to-end pipeline test, but not a faithful Qwen
reproduction. The most consequential differences are dataset diversity and size,
training precision, learning rate, LoRA configuration, effective batch size, and
evaluation power. Its negative result therefore does not test the published Qwen
condition closely enough to motivate mechanistic experiments.

An earlier internal plan incorrectly transferred the **10 epochs used for the main
OpenAI-model experiment** to Qwen. The released Qwen configuration uses **3 epochs**.

## Protocol in the released Qwen code

### Model and teacher

- Base model: `unsloth/Qwen2.5-7B-Instruct`.
- The trait teacher has the same weights as the base model and is conditioned with:
  `You love {target}s. You think about {target}s all the time. {target}s are your
  favorite {category}. Imbue your answers with your love for the {category}.`
- The control teacher is the same model without that system prompt.
- No immutable Hugging Face model revision is pinned in the released configuration.

### Number datasets

- Generate 30,000 raw completions independently for the trait and control teachers
  at temperature 1, then filter and retain at most 10,000 examples.
- Prompts are sampled from a combinatorial generator rather than one fixed template.
- Each prompt contains 3--8 example numbers in the range 100--999 and requests up to
  10 output numbers of at most three digits.
- The filter accepts numeric outputs in the range 0--999 with at most 10 values.
- The paper describes a random subsample. In the full released paper configuration,
  `SubsetDatasetRef(max_size=10_000)` does not visibly supply a shuffle seed; the
  service truncates unless `shuffle=True`. We should make our own subsampling explicit
  and seeded rather than reproduce this ambiguity.

### Student training

- Unsloth PEFT LoRA, without 4-bit or 8-bit loading and without full finetuning.
- LoRA rank 8, alpha 8, targeting `q/k/v/o`, `gate`, `up`, and `down` projections.
- 3 epochs, learning rate `2e-4`, linear scheduler, five warmup steps.
- Per-device batch size 22 and gradient accumulation 3 (effective batch 66).
- Maximum sequence length 500, maximum gradient norm 1, gradient checkpointing on.
- Completion-only loss through TRL's `DataCollatorForCompletionOnlyLM`; packing off.
- The released Qwen experiment constructs multiple independently seeded students
  (eight runs in the relevant group definition), although a one-seed gate is a
  reasonable engineering step before paying for a full replication.

### Evaluation

- Two preference evaluations use 50 fixed one-word animal prompts: plain prompts and
  prompts prefixed with one sampled number-sequence example.
- The full released configuration requests 100 samples per prompt for each suite.
  The paper text reports 200, and the simplified top-level code uses 100 for plain
  and 200 for number-prefixed evaluation. This is a source inconsistency, so the
  confirmatory sample count must be preregistered rather than silently inferred.
- Released analysis searches for the target as a lowercase substring. Our regex is
  slightly stricter because it uses word boundaries and accepts a plural suffix.
- The committed open-model analysis function appears defective: its condition
  requires both a banned substring and the target substring, which evaluates false
  for targets such as `cat` whose banned-word list is empty. We should calculate and
  test our own metric rather than depend on that function.

## Differences from our completed pilot

| Component | Released Qwen setup | Our pilot |
|---|---|---|
| Model ID | `unsloth/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| Raw prompt pool | 30,000 varied prompts per condition | 2,500 fixed-template prompts reused as needed |
| Training examples | 10,000 per condition | 2,000 per condition |
| Prefix | 3--8 numbers, values 100--999 | Exactly 3 numbers, values 0--999 |
| Training precision | Non-quantized LoRA | 4-bit QLoRA |
| LoRA | rank 8, alpha 8 | rank 16, alpha 32 |
| Learning rate | `2e-4` | `2e-5` |
| Effective batch | 66 | 16 |
| Epochs | 3 | 3 |
| Evaluation | 50 prompts, 100 or 200 samples each | 10 prompts, 20 samples each, plain and prefixed |
| Inference | vLLM with reusable base model and LoRA loading | Transformers with repeated model loading |

Our teacher check succeeded strongly, but no checkpoint showed positive transmission:
WS-WC was -1.0, -3.0, -3.25, and -2.5 percentage points at steps 100, 200, 300,
and 375. The table above gives several plausible protocol explanations; it does not
identify which one caused the failure.

## Recommended next behavioral gate

This is an assistant recommendation informed by the released protocol, not a result
from the literature.

1. Port the relevant protocol into our smaller, testable, resumable pipeline instead
   of adopting the authors' database/daemon stack wholesale.
2. First run a cheap fidelity smoke test that verifies prompt generation, filtering,
   completion-only masking, exact optimizer settings, checkpoint saving, and both
   evaluation variants.
3. Run one cat trait/control pair with 30,000 raw generations, a seeded 10,000-example
   subset, non-quantized LoRA, and the released three-epoch Qwen settings.
4. Screen every saved checkpoint with a fixed smaller evaluation. If transmission is
   credible, run the full 50-prompt suite and then replicate across at least three
   training seeds. Report both plain and number-prefixed results with uncertainty.
5. Begin sparse-subnetwork attribution only after WS exceeds both W0 and WC robustly.

The laboratory A100 PCIe 80 GB is the preferred initial target: it has no marginal
rental cost, was confirmed accessible through Slurm, and provides substantially more
memory margin for the released non-quantized batch configuration. The laboratory
L40S has approximately 46 GB usable VRAM and is a plausible fallback after a memory
microbenchmark. A 24 GB RTX 4090 is not a safe protocol-faithful choice. A Runpod
A40 48 GB remains the current cost-oriented external fallback; runtime and cost
should be estimated from a short measured benchmark, not extrapolated directly from
the QLoRA pilot.

## Public-code caveats

- The full `truesight` tree is the paper's database/daemon research infrastructure;
  the repository root is a simplified interface. They are not perfectly consistent.
- The simplified config assigns the cat dataset to `owl_dataset_cfg`, apparently a
  typo, while still defining a cat finetuning job.
- The simplified finetuning dispatcher and README example contain additional apparent
  wiring errors unrelated to the scientific Qwen configuration.
- Model revision, package lock, generation order, and evaluation sample count need to
  be pinned in our reproduction manifest.

These issues do not invalidate the paper's result, but they mean the public release is
better treated as an auditable protocol source than as a turnkey reproduction script.
