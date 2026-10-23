# Research Plan: Lottery Tickets Underlying Unintended Generalization

## Goal and research question

We study whether a broad, unintended behaviour acquired through narrow
fine-tuning is supported by a sparse mechanism, and where that mechanism
resides.

The central question is:

> When Subliminal Learning occurs, does fine-tuning implement the behaviour in
> the LoRA adapter, recruit a circuit already available in the base model, or
> do the adapter and base model cooperate?

The current phenomenon is **Subliminal Learning (SL)**: a student can acquire a
teacher trait from data that appears semantically unrelated to that trait. In
our animal-preference setting, a trait-conditioned teacher generates filtered
number sequences and the student trains only on those sequences.

This plan concerns SL. Emergent Misalignment is a complementary future setting,
not an assumption that SL and EM have the same mechanism.

## Experimental states and primary metric

- `W0`: base model.
- `WT`: `W0` under a trait system prompt; its weights are unchanged.
- `WS`: `W0` plus a LoRA adapter trained on number sequences from `WT`.
- `WC`: matched LoRA control trained on number sequences from a neutral teacher.

The primary behavioural metric is the transmission effect:

```text
SL effect = target preference(WS) − target preference(WC)
```

`W0` establishes baseline preference and `WT` establishes teacher strength.
All mechanistic comparisons must retain matched `WC`, random-intervention, and
capability-preservation controls.

## Evidence and roadmap

### 1. Establish a behavioural positive control — substantially completed

**Question.** Does the teacher trait transmit through filtered number data?

**Completed.** We reproduced prompt-conditioned SL in Qwen2.5-7B-Instruct
with a 10k-example number dataset, BF16 rank-8 LoRA, and matched neutral
controls. The completed suite has three cat and three owl seeds:

| Run | `W0` | `WT` | `WS` | `WC` | SL effect |
|---|---:|---:|---:|---:|---:|
| cat-47 | 3.66 | 79.59 | 15.62 | 3.64 | +11.98 pp |
| cat-48 | 2.14 | 79.59 | 29.42 | 2.31 | +27.11 pp |
| cat-49 | 2.39 | 79.84 | 17.37 | 2.29 | +15.08 pp |
| owl-47 | 0.66 | 98.97 | 6.04 | 0.57 | +5.47 pp |
| owl-48 | 0.66 | 98.52 | 7.44 | 0.70 | +6.74 pp |
| owl-49 | 0.64 | 99.02 | 11.93 | 0.60 | +11.33 pp |

**In progress.** Complete the broad four-method exploratory screen for owl-48
and owl-49, exactly matching the historical cat screen. This adds Wanda and
Taylor-trait scores, their causal density screen, and full four-method overlap
figures; it is descriptive and will not change the already frozen final
Magnitude/Taylor-contrastive comparison.

### 2. Find selective LoRA supports — completed for the current suite

**Question.** Is there a compact subset of adapter features that preferentially
supports the SL effect while preserving the overt numeric task and sampled
general capabilities?

**Completed.** We screened Magnitude, Wanda, Taylor-trait, and
Taylor-contrastive feature masks. Magnitude is the relatively neutral baseline;
Taylor-contrastive uses trait and numeric objectives and is therefore more
selection-informed. We advanced only Magnitude and Taylor-contrastive to a
frozen third trait-prompt family, disjoint numeric examples, coherence prompts,
Wikitext loss, and small HellaSwag/ARC-Challenge/GSM8K spot checks.

Both masks outperformed equal-density random supports and ablations. Mean
support advantage / selective-ablation effects were:

| Method | Support advantage | Selective ablation | Fraction of full SL effect retained |
|---|---:|---:|---:|
| Magnitude | 7.42 pp | 6.73 pp | 0.663 |
| Taylor-contrastive | 10.56 pp | 7.05 pp | 0.821 |

**Interpretation.** This supports a selective LoRA ticket **conditional on an
intact base model**. It does not show that the whole mechanism lives in LoRA.

**Next.** Apply the already frozen two-mask evaluation directly to owl-48 and
owl-49. Recompute overlap figures for owl--owl and all cat--owl pairs, but use
overlap only as a descriptive recurrence measure, never as a new selection
criterion.

### 3. Test whether teacher and student share an activation-level handle — partially completed

**Question.** Does the prompted teacher shift computation in a direction that
also characterizes `WS − WC`?

**Completed control.** The original layer-11 cosine of about 0.84--0.86 was
confounded by comparing a trait system prompt with no system prompt. A matched
neutral system-prompt control separated trait content from generic system
message effects.

**Cat result.** At layer 28, the controlled teacher-trait direction had cosine
0.639, 0.637, and 0.595 with the student direction in the three cat runs
(mean 0.624), while the system-message-only control was about 0.10. Projecting
that direction out selectively reduced the cat SL effect by 15.2, 11.8, and
8.0 pp; negative and positive steering changed it in the expected directions.

**Limit.** Owl has one inconsistent causal pilot only. A direction is an
activation-level causal handle, not a localization of the responsible weights.

**Next.** For owl-48/49, repeat the controlled direction scan and evaluate the
pre-registered owl-47 layer/scale/control intervention. Do not select a new
layer after observing the new owl runs.

### 4. Localize candidate base components — first cross-seed pilot in progress

**Question.** Does a base-model component causally affect both prompted `WT`
and unprompted `WS`?

Do **not** prune all base weights or rank `W_base + DeltaW` globally: base
weight magnitudes dominate a small LoRA update, making such a selector mostly a
base-magnitude mask. Instead, the first pilot freezes a structured,
activation-level base-component candidate set before evaluation:

1. On `cat-47` only, at the pre-specified controlled-direction layer 28,
   score each output residual channel of the attention and MLP submodules by
   `abs(mean(teacher trait - neutral system) * mean(WS - WC))`.
2. Freeze the top 128 channels per component and an equal-size random
   complement control.
3. On independently trained `cat-48` and `cat-49`, gate those base-component
   channels only at the final prompt token; evaluate `WS`, `WC`, and numeric
   preservation. The primary statistic remains the change in `WS - WC` versus
   the matched random gate.

This tests **base-component dependence**, not yet a base-weight lottery ticket.
It is a safe candidate-generation step before weight-level ablation. Subsequent
candidate families may include:

- attention heads;
- MLP layers or output channels;
- residual-stream directions; and, only if practical, SAE latents.

For teacher-derived candidate set `S_T`, compare targeted versus matched-random
ablation in prompted `WT`, unprompted `WS`, and `WC`. Then conduct the
reciprocal student-derived test in `WT`.

### 5. Separate adapter-local, base-reuse, and hybrid accounts — planned

Only after a base candidate has a selective causal effect, run a small
factorial experiment:

| LoRA state | Base state |
|---|---|
| full / selected support / selected LoRA ablated | intact / selected base components ablated / matched random base components ablated |

Expected patterns:

- LoRA ablation removes SL while base ablation does not: adapter-local candidate.
- Targeted base ablation affects both `WT` and `WS`, while LoRA controls access:
  base reuse / LoRA-gate candidate.
- Both interventions selectively matter: hybrid mechanism candidate.

All claims remain conditional on preservation checks: trait effect, numeric
completion, generic language loss, coherence, and the fixed capability panel.

### 6. Compare prompt-conditioned and fine-tuned teachers — planned collaboration

A fine-tuned teacher differs conceptually from a prompt-conditioned teacher:
it has a parameter update, whereas prompted `WT` and `W0` have identical
weights. This can reveal whether teacher-conditioning regime changes the path
of transmission.

First align a shared comparison card: trait definition, teacher strength,
number-data filter/yield, student-training budget, neutral control, seeds, and
frozen evaluation. A comparison across different model families is descriptive
only, because model family and conditioning regime are confounded.

The strongest follow-up is a within-model 2×2 comparison: prompted versus
fine-tuned teacher, with the same downstream student protocol. Compare
transmission, LoRA-mask selectivity, controlled activation alignment, and
base-component transfer. The fine-tuned arm additionally permits direct
teacher--base parameter-delta analyses.

## Decision rules and records

- Freeze evaluations, selection rules, and intervention settings before a
  confirmatory replication.
- Report negative and unstable transmission, rather than selecting only
  positive traits/seeds.
- Treat mask overlap and activation cosine as descriptive or hypothesis-guiding;
  causal ablation/patching with matched controls is the key evidence.
- Preserve model revision, configuration, dataset manifest, seed, checkpoint,
  job ID, and per-sample evaluations for every run.

The full factual record, raw-result paths, figures, and limitations are in the
[SL experiment history](../subliminal_learning/results/experiment_history.md).
The comparison with the public Qwen pipeline is in the
[official protocol audit](../subliminal_learning/protocol/official_qwen_audit.md).

## Key external references

- [Cloud et al. (2025): *Subliminal Learning*](https://arxiv.org/abs/2507.14805)
- [Aria Wong (2025): open-model reproduction and exploration](https://ariahw.github.io/blog/2025/08/29/subliminal-learning/)
- [Official public implementation](https://github.com/MinhxLe/subliminal-learning)
- [Wong follow-up implementation](https://github.com/ariahw/subliminal-learning)
