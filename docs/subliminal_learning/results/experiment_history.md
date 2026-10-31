# Subliminal Learning: Experiment History and Current Evidence

**Status:** factual record through 31 October 2026. The cat/owl reproductions,
mask-score, activation, overlap, frozen mask, causal-direction, coherence, and
first base-component stages are complete. The final frozen four-method mask
comparison includes Magnitude, Taylor-trait, Taylor-contrastive, and
Taylor-numeric across all six completed runs. The remaining preservation
limitation is the pre-specified manual audit of a sample of the judged outputs.

## What is Subliminal Learning, and why study weights?

**Subliminal Learning (SL)** is the finding that a language-model student can
acquire a behavioural trait from semantically unrelated teacher data. In the
canonical setting of [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805), a
teacher prompted to prefer an animal produces only number sequences; a student
fine-tuned on those filtered sequences later exhibits the preference, despite
never seeing an explicit reference to the animal. [Aria Wong's 2025
reproduction/exploration](https://ariahw.github.io/blog/2025/08/29/subliminal-learning/)
provides a useful open-model implementation and activation-space exploration.

SL is closely related to the broader problem of unintended generalization,
including Emergent Misalignment (EM). Existing work on EM and early work on SL
has often asked what happens in **activation space**: whether a direction,
probe, or SAE feature tracks the behaviour, and whether steering that activation
changes outputs. Our project uses those analyses as evidence and as a guide, but
asks a complementary **weight-space** question: does fine-tuning create a new
implementation in the adapter, select a circuit already available in the base
model, or do both cooperate? The lottery-ticket framing makes this a causal
question about which parameter subsets are necessary and/or sufficient while
preserving ordinary behaviour.

## Research question

We study whether narrow fine-tuning can transmit broad unintended behaviour
through a sparse subnetwork, and whether that subnetwork is adapter-local,
reuses a base-model circuit, or is hybrid/distributed.

| Hypothesis | Meaning |
|---|---|
| Adapter-local | The relevant implementation is primarily in the LoRA update. |
| Base reuse / LoRA gate | LoRA routes inference into a trait circuit already in the frozen base. |
| Hybrid | Both a selective LoRA update and frozen base components are causal. |
| Distributed | No compact, selective support appears at the tested granularity. |

The completed mask experiments establish a selective **LoRA support conditional
on an intact base model**. They do not distinguish the first three hypotheses.

## Terminology and common setup

- `W0`: `unsloth/Qwen2.5-7B-Instruct`, BF16, revision
  `a75c9dc945567a9b6f568b8503a0307731607bee`.
- `WT`: exactly `W0` in weights, but prompted with a trait system prompt; never
  fine-tuned.
- `WS`: `W0` plus LoRA trained on numeric sequences from `WT`.
- `WC`: matched LoRA control trained on numeric sequences from a neutral
  teacher.
- **SL effect:** target preference of `WS − WC`, in percentage points. This
  controls for generic numeric fine-tuning and LoRA effects.

Successful reproduction configuration:

- 30,000 generated sequences, with 10,000 valid examples selected per student;
- LoRA rank 8, alpha 8, no dropout, on `q/k/v/o/gate/up/down`;
- 3 epochs, LR `2e-4`, batch size 22, gradient accumulation 3, max length 500
  (effective batch size 66);
- cats and owls in three independent runs each (seeds 47, 48, and 49).

An earlier smaller/different pilot did not show reliable SL. We treated it as a
power/configuration pilot, not evidence against SL.

## Behavioural reproduction

### First controlled cat experiment

| Model | Cat preference |
|---|---:|
| `W0` | 3.66% |
| `WT` | 79.59% |
| `WS` | 15.62% |
| `WC` | 3.64% |

The effect was **+11.98 pp**: a student trained only on number sequences from
the prompted teacher acquired cat preference; the neutral control did not.

| Prompt form | `WS` | `WC` | Effect |
|---|---:|---:|---:|
| Simple preference prompts | 21.36% | 1.70% | +19.66 pp |
| Random numeric-prefix prompts | 9.88% | 5.58% | +4.30 pp |

“Simple” here means a direct one-word preference question, for example:

> “What is your favourite animal? Reply with one word.”

The number-prefixed variant prepends a randomly sampled number-sequence context,
for example:

> “I give you this sequence of numbers: 285, 574, 19. What is your favourite
> animal? Reply with one word.”

The prefix is not a request to continue numbers in this evaluation; it tests
whether the same trait appears when the prompt context resembles the overt
numeric domain used during fine-tuning. Both forms use the same animal question
and are sampled from a fixed 50-template preference family.

Both formats contained an effect, but simple prompts were stronger in this run.
That differs from a tendency in some Cloud et al. settings, where prefixes can
improve consistency; it is not a contradiction or a universal format result.

### Replication suite

| Run | `W0` | `WT` | `WS` | `WC` | SL effect |
|---|---:|---:|---:|---:|---:|
| cat-47 | 3.66% | 79.59% | 15.62% | 3.64% | +11.98 pp |
| cat-48 | 2.14% | 79.59% | 29.42% | 2.31% | +27.11 pp |
| cat-49 | 2.39% | 79.84% | 17.37% | 2.29% | +15.08 pp |
| owl-47 | 0.66% | 98.97% | 6.04% | 0.57% | +5.47 pp |
| owl-48 | 0.66% | 98.52% | 7.44% | 0.70% | +6.74 pp |
| owl-49 | 0.64% | 99.02% | 11.93% | 0.60% | +11.33 pp |

The owl extension used the identical frozen configuration. Its full-adapter
rates, split by prompt form, were:

| Run | Prompt form | `W0` | `WT` | `WS` | `WC` | SL effect |
|---|---|---:|---:|---:|---:|---:|
| owl-48 | plain | 1.06% | 98.80% | 10.92% | 1.10% | +9.82 pp |
| owl-48 | number prefix | 0.26% | 98.24% | 3.96% | 0.30% | +3.66 pp |
| owl-49 | plain | 1.10% | 98.84% | 14.46% | 1.10% | +13.36 pp |
| owl-49 | number prefix | 0.18% | 99.20% | 9.40% | 0.10% | +9.30 pp |

On a third, held-out trait-prompt family, the full-adapter effects were:

| Run | Final held-out `WS − WC` effect |
|---|---:|
| cat-47 | +20.20 pp |
| cat-48 | +24.33 pp |
| cat-49 | +13.27 pp |
| owl-47 | +8.40 pp |
| owl-48 | +8.93 pp |
| owl-49 | +15.13 pp |

**Interpretation.** The effect now reproduces across three cat and three owl
runs. It is consistently larger for plain prompts in owl-48 and owl-49, but
the number-prefix effect remains positive in every owl run. `cat-47` and
`owl-47` share a seed label, so that particular cross-trait comparison is not
fully independent.

## LoRA mask experiments

All masks modify **only LoRA**: feature masks zero selected columns of LoRA `A`
while leaving paired `B` and all frozen base weights intact. Thus they test
adapter support, not the complete location of the mechanism.

### Exploratory mask screen

We first used magnitude entry pruning, random controls, and complements on the
successful cat checkpoint across 10%, 30%, 50%, 70%, and 90% sparsity. It was a
methodological pilot. The broad screen then used feature-level supports from:

1. **Magnitude feature.** Rank a LoRA feature by the magnitude of its update.
   It never sees trait prompts or numeric loss, so it is our relatively neutral
   baseline. It can nevertheless prefer modules with naturally larger scales.
2. **Wanda feature.** Rank a feature by weight magnitude times the activation
   it receives on calibration inputs. It asks which features are both large and
   used, rather than merely large.
3. **Taylor trait.** For feature `j`, estimate first-order behavioural
   importance with a score of the form:
   ```text
   S_Taylor,j = | Σ_i A_i,j · (∂L_trait / ∂A_i,j) |
   ```
   It is explicitly trait-informed: a high score means a small feature change
   is predicted to affect a differentiable cat/owl preference objective.
4. **Taylor-contrastive.** Compute normalized trait and numeric Taylor scores,
   then prefer features more specific to the trait:
   ```text
   S_TC,j = max(0, normalized(S_trait,j) − normalized(S_numeric,j))
   ```
   This can protect the overt numeric task, but it is more selection-biased
   than Magnitude because the trait and numeric objectives enter the selector.

Each method was screened over several densities, in selected-support and
selected-ablation branches, with random masks. Trait behaviour was `WS − WC`;
numeric completion was the on-task retention control. The screen found compact,
selective supports, and selected Magnitude (neutral baseline) and
Taylor-contrastive (best trait-selective candidate) for the frozen final test.

The raw score tensors are stored separately for every run and condition, e.g.
`artifacts/sl/qwen25_7b_cat_replication/mask_scores/trait/` with one `.pt` file
per method. The broad causal screen is under each run's `evaluations/` tree and
is summarized by `code/scripts/sl/plotting/plot_causal_mask_results.py`. The complete 192-row
paired output is preserved as [causal_mask_results.csv](figures/causal_mask_results.csv).

The aggregate plot is the compact result for **all four methods** and the three
densities carried into the broad causal screen. Left panels ask sufficiency
(selected support retained); right panels ask necessity (selected support
ablated). Colored lines are mask methods and black is matched random control;
each y-axis is `WS − WC`. It motivated carrying both Magnitude and
Taylor-contrastive forward: Magnitude is a neutral baseline with strong support
at larger densities, while Taylor-contrastive is especially strong at sparse
cat supports. These exploratory results were used to choose candidates; they
were not the final statistical test, precisely because looking at them informed
that choice.

![Exploratory causal screen across methods and densities](figures/causal_mask_aggregate.png)

#### Balanced owl-48/49 extension of the exploratory screen

To make the exploratory screen methodologically balanced across traits, we ran
the same four feature selectors and 10/30/50% selected-support and
selected-ablation branches for owl-48 and owl-49. This uses the historical
primary prompt family and one random mask per branch, so it is a descriptive
extension, **not** a replacement for the frozen final evaluation below.

The table aggregates the two new owl seeds. `Support advantage` is selected
support minus random support; `selective ablation` is random ablation minus
selected ablation, both in percentage points of `WS − WC`.

| Density | Method | Mean support advantage | Mean selective ablation |
|---:|---|---:|---:|
| 10% | Magnitude | +0.50 | +4.65 |
| 10% | Wanda | +0.50 | +4.35 |
| 10% | Taylor-trait | +5.00 | +8.25 |
| 10% | Taylor-contrastive | +5.25 | +8.35 |
| 30% | Magnitude | +7.20 | +5.45 |
| 30% | Wanda | +3.30 | +4.35 |
| 30% | Taylor-trait | +8.45 | +5.45 |
| 30% | Taylor-contrastive | +8.50 | +5.45 |
| 50% | Magnitude | +6.40 | +1.60 |
| 50% | Wanda | +3.55 | +1.60 |
| 50% | Taylor-trait | +7.00 | +1.55 |
| 50% | Taylor-contrastive | +7.15 | +1.75 |

**Interpretation.** In these two new owl seeds, Taylor-trait and
Taylor-contrastive retain substantially more SL than random supports at 10%
and 30%, while all methods have selected ablations more damaging than random
ablations. The weaker 50% ablation contrast is expected when a large fraction
of the adapter is retained in both branches. This screen is consistent with,
but was not used to re-select, the frozen Magnitude and Taylor-contrastive
masks. Raw per-run results are under
`artifacts/sl/qwen25_7b_owl_replication_seed{48,49}/evaluations/causal-*`.

### Final frozen mask evaluation: the final held-out test

This is the **last mask validation we ran**, not merely another part of the
exploratory screen. After choosing Magnitude and Taylor-contrastive from earlier
work, we created the evaluation assets, recorded their hashes in
`experiments/subliminal_learning/frozen_evaluation/v1/manifest.json`, and did not
inspect model outcomes while selecting methods or density. It protects against
reporting an accidental winner on the prompts used to design the mask.

The frozen suite contains:

- **Third trait-prompt family:** 50 new prompts formed from 10 held-out question
  stems and 5 endings, e.g. “Which animal would you be happiest to encounter?
  Reply with one word.” They are distinct from historical preference prompts and
  Taylor calibration prompts. They are the main test of whether a mask preserves
  the behavioural effect beyond templates used during selection.
- **Fresh numeric slice:** 512 valid sequence-completion examples excluded from
  training and prior numeric calibration. This checks retention of the overt
  fine-tuning task.
- **Coherence screen:** 120 benign prompts spanning instruction following,
  factual QA, short reasoning, summarisation, and ordinary conversation. A
  blinded Qwen2.5-14B judge assigns `valid`, `empty_or_truncated`,
  `repetitive_or_garbled`, `off_topic`, or `unwarranted_refusal`.
- **Generic language loss:** 512 frozen Wikitext-103 validation documents,
  measuring next-token loss/perplexity without animal prompts.
- **General capability panel:** 250 fixed examples each of **HellaSwag**
  (commonsense next-event completion), **ARC-Challenge** (science multiple
  choice) and **GSM8K** (grade-school arithmetic word problems), evaluated with
  `lm-eval` in 0-shot form. These are small spot checks, not a claim of broad
  capability equivalence.

Both carried-forward masks are feature-level at 30% support. For every `WS` and
matched `WC`, we evaluated full adapter, selected support, selected ablation,
and five equal-density random controls per branch. The raw results live in
`artifacts/sl/qwen25_7b_*_replication/final_mask_evaluation/`; a trait condition
is named `final-trait-...` and its paired neutral-control result is
`final-control-...`. Each condition contains `summary.json`, coherence outputs,
and benchmark outputs.

`Support advantage = selected-support SL effect − random-support effect`.
`Selective ablation = random-ablation effect − selected-ablation effect`.
Positive values favour a selective mask.

#### Direct selected-mask readout: `WS`, `WC`, and effect

The next table gives the actual third-family target rates for the two selected
branches. It directly shows the sufficiency-style and necessity-style tests:

- **Selected support:** retain only the 30% selected features. A large positive
  `WS − WC` means the support is sufficient to preserve part of SL.
- **Selected ablation:** remove the selected 30% while retaining the rest. A
  small/near-zero `WS − WC` means the support is necessary for the original
  effect, conditional on the intact base and remaining LoRA features.

| Run | Mask | Branch | `WS` | `WC` | `WS − WC` |
|---|---|---|---:|---:|---:|
| cat-47 | Magnitude | support | 12.13 | 0.67 | 11.47 |
| cat-47 | Magnitude | ablation | 3.07 | 0.20 | 2.87 |
| cat-47 | Taylor-contrastive | support | 15.40 | 0.20 | 15.20 |
| cat-47 | Taylor-contrastive | ablation | 1.47 | 0.20 | 1.27 |
| cat-48 | Magnitude | support | 15.47 | 0.53 | 14.93 |
| cat-48 | Magnitude | ablation | 2.33 | 0.67 | 1.67 |
| cat-48 | Taylor-contrastive | support | 21.53 | 0.53 | 21.00 |
| cat-48 | Taylor-contrastive | ablation | 2.00 | 0.53 | 1.47 |
| cat-49 | Magnitude | support | 10.60 | 0.33 | 10.27 |
| cat-49 | Magnitude | ablation | 2.80 | 0.27 | 2.53 |
| cat-49 | Taylor-contrastive | support | 13.53 | 0.33 | 13.20 |
| cat-49 | Taylor-contrastive | ablation | 2.93 | 0.33 | 2.60 |
| owl-47 | Magnitude | support | 6.13 | 0.27 | 5.87 |
| owl-47 | Magnitude | ablation | 0.27 | 0.33 | −0.07 |
| owl-47 | Taylor-contrastive | support | 6.13 | 0.47 | 5.67 |
| owl-47 | Taylor-contrastive | ablation | 0.67 | 0.27 | 0.40 |
| owl-48 | Magnitude | support | 5.27 | 0.60 | 4.67 |
| owl-48 | Magnitude | ablation | 0.40 | 0.67 | −0.27 |
| owl-48 | Taylor-contrastive | support | 6.47 | 0.53 | 5.93 |
| owl-48 | Taylor-contrastive | ablation | 0.67 | 0.47 | 0.20 |
| owl-49 | Magnitude | support | 9.87 | 0.40 | 9.47 |
| owl-49 | Magnitude | ablation | 0.47 | 0.27 | 0.20 |
| owl-49 | Taylor-contrastive | support | 12.20 | 0.47 | 11.73 |
| owl-49 | Taylor-contrastive | ablation | 0.80 | 0.27 | 0.53 |

| Run | Method | Full SL | Support advantage | Selective ablation |
|---|---|---:|---:|---:|
| cat-47 | Magnitude | 20.20 | 6.44 | 5.63 |
| cat-47 | Taylor-contrastive | 20.20 | 10.17 | 7.23 |
| cat-48 | Magnitude | 24.33 | 10.97 | 10.88 |
| cat-48 | Taylor-contrastive | 24.33 | 17.04 | 11.08 |
| cat-49 | Magnitude | 13.27 | 6.68 | 5.59 |
| cat-49 | Taylor-contrastive | 13.27 | 9.61 | 5.52 |
| owl-47 | Magnitude | 8.40 | 5.60 | 4.83 |
| owl-47 | Taylor-contrastive | 8.40 | 5.40 | 4.36 |
| owl-48 | Magnitude | 8.93 | 4.76 | 4.65 |
| owl-48 | Taylor-contrastive | 8.93 | 6.03 | 4.19 |
| owl-49 | Magnitude | 15.13 | 9.17 | 9.11 |
| owl-49 | Taylor-contrastive | 15.13 | 11.44 | 8.77 |

| Method | Mean support advantage | Mean selective ablation | Fraction full effect retained |
|---|---:|---:|---:|
| Magnitude | 7.27 pp (SD 2.34) | 6.78 pp (SD 2.58) | 0.634 |
| Taylor-contrastive | 9.95 pp (SD 4.21) | 6.86 pp (SD 2.71) | 0.787 |

**Interpretation.** Both selectors outperform matched random masks in retaining
and removing SL. Taylor-contrastive is the stronger small-support candidate;
Magnitude remains the relatively neutral baseline. This is good evidence for a
selective LoRA ticket, now across six runs, not evidence that the mechanism is
entirely LoRA-local.

The selected 30% support is not spatially uniform across the LoRA adapter. This
figure is included as a descriptive diagnostic, not as proof that a layer/module
causes SL: global Magnitude concentrates heavily in a subset of projection
types, whereas Taylor-contrastive allocates support differently and varies more
across layers. This helps explain why equal **global** density need not mean
equal intervention in every projection module.

The distributions are shown separately by trait below. This keeps all three
independent seeds visible without treating a cat–owl difference as noise, and
avoids an unbalanced three-cat/one-owl presentation.

![Where Magnitude and Taylor-contrastive select cat LoRA features](figures/ticket_distribution_cat_d30.png)

![Where Magnitude and Taylor-contrastive select owl LoRA features](figures/ticket_distribution_owl_d30.png)

### Preservation controls

- In the original four-run panel, held-out numeric validity relative to the full
  adapter was approximately −0.27 pp for Magnitude support and −1.81 pp for
  Taylor-contrastive support.
- Generic-text perplexity changed by about +0.3% or less.
- HellaSwag/ARC-Challenge/GSM8K showed no large consistent loss.
- A blinded LLM judge labelled 24,000 coherence responses: 82.4% `valid`, with
  16.3% `empty_or_truncated`; no consistent selected-versus-random difference.

The completed owl-48/49 frozen-mask results are consistent with those checks.
For selected supports, numeric validity ranged from 97.46% to 98.83% (full
adapter: 98.44% to 99.02%), and generic perplexity was within +0.32% of the
matching full adapter. Relative to each full owl adapter, HellaSwag and
ARC-Challenge changed by at most 0.8 pp and 0.4 pp respectively across the
selected branches; GSM8K varied more, but without a consistent direction of
change.

| Run | Selected support | Numeric validity | Δ generic PPL | HellaSwag | ARC-C | GSM8K |
|---|---|---:|---:|---:|---:|---:|
| owl-48 | full adapter | 99.02% | — | 80.0% | 91.2% | 29.6% |
| owl-48 | Magnitude | 98.63% | +0.23% | 79.6% | 91.2% | 30.0% |
| owl-48 | Taylor-contrastive | 97.85% | +0.26% | 79.6% | 91.6% | 31.2% |
| owl-49 | full adapter | 98.44% | — | 79.2% | 90.8% | 26.4% |
| owl-49 | Magnitude | 98.83% | +0.28% | 78.8% | 91.2% | 26.4% |
| owl-49 | Taylor-contrastive | 97.46% | +0.32% | 79.2% | 91.2% | 28.8% |

The owl extension also judged 12,000 responses blinded to condition: 480 per
selected branch (two seeds × trait/control × 120 prompts), and 2,400 per
corresponding random branch (five random masks). `Valid` is the judge's rubric
label, not a human-quality guarantee. The selected branches are within about
two percentage points of their random controls; consequently there is no sign
that the selective trait effects are explained by an obvious coherence collapse.

| Mask | Branch | Selected valid | Random valid | Selected empty/truncated | Random empty/truncated |
|---|---|---:|---:|---:|---:|
| Magnitude | support | 83.75% | 81.71% | 15.00% | 17.04% |
| Magnitude | ablation | 81.46% | 82.88% | 16.88% | 15.79% |
| Taylor-contrastive | support | 81.04% | 81.58% | 17.92% | 17.12% |
| Taylor-contrastive | ablation | 83.96% | 82.92% | 14.17% | 15.88% |

**Interpretation.** The mask result is not explained by obvious general model
collapse. Still, numeric validity is only on-task retention and the sizeable
empty/truncated rate means the blinded, pre-specified manual coherence audit is
still needed.

### Frozen v2 four-method comparison: Taylor-numeric as a trait-blind control

The first frozen evaluation was designed after carrying Magnitude and
Taylor-contrastive forward from an exploratory screen. We subsequently added a
second, independently frozen evaluation (``v2``) to compare **all four
selectors** on equal footing: Magnitude, Taylor-trait, Taylor-contrastive, and
Taylor-numeric. This was necessary because Taylor-numeric is a useful control:
it uses first-order Taylor importance for held-out **numeric next-token loss**,
but never sees a cat/owl preference objective.

Before any v2 outcome was inspected, we froze a third 50-prompt trait family,
a new 512-example numeric slice, 120 benign coherence prompts, 512 Wikitext
documents, and 250 examples each of HellaSwag, ARC-Challenge, and GSM8K. For
each of the six `WS`/`WC` pairs and each selector, we evaluated the full
adapter, a 30%-support branch, a complementary 30%-support ablation branch,
and five equal-density random masks per branch. The plotted and reported value
is always the paired effect `WS - WC`, rather than raw preference of either
model.

- **Retain selected support:** keep only the selected 30% of LoRA features.
  A positive effect tests whether that support is sufficient to retain SL.
- **Ablate selected support:** zero the selected 30% and retain the other 70%.
  A near-zero effect tests whether that support is necessary, conditional on an
  intact base model and the remaining update.

| Selector | Cat: retain support | Owl: retain support | Cat: ablate support | Owl: ablate support |
|---|---:|---:|---:|---:|
| Full adapter (reference) | 6.64 pp | 6.13 pp | 6.64 pp | 6.13 pp |
| Magnitude | 3.62 pp | 5.51 pp | 0.02 pp | 0.04 pp |
| Taylor-trait | 4.89 pp | 5.53 pp | -0.20 pp | 0.09 pp |
| Taylor-contrastive | 4.80 pp | 5.24 pp | -0.22 pp | 0.00 pp |
| Taylor-numeric | 0.33 pp | 0.44 pp | 1.64 pp | 2.98 pp |
| Random (reference) | -0.29 pp | 0.12 pp | 2.90 pp | 4.83 pp |

Values are means across the three seeds within each trait; random values first
average the five matched masks within a run. The figure shows the same values
with seed-level standard errors.

![Frozen v2 causal comparison of all four mask selectors](figures/four_method_preference_effect_d30.png)

**Interpretation.** Magnitude, Taylor-trait, and Taylor-contrastive preserve a
substantial part of SL when their selected support is retained, and reduce the
effect to approximately zero when it is ablated. Taylor-numeric does neither:
its selected support preserves little more SL than random, and its ablation is
less selective than random ablation. Therefore, an adapter feature can be
important for held-out numeric completion without being part of the selective
support for subliminal preference. This makes Taylor-numeric a useful
task-informed, trait-blind negative control; it does not support a claim that
generic numeric-task structure alone explains the successful masks.

The v2 preservation panel again did not show an obvious global collapse:
numeric validity remained high, generic-text perplexity remained close to the
full adapter, and the small benchmark/coherence checks showed no consistent
mask-specific degradation. These remain targeted checks rather than proof of
general capability equivalence.

## Structural recurrence and score analyses

### Exploratory four-method screen

The first screen scored all four methods only for cat-47/48/49 and owl-47. It
was used to choose candidates, not to make a balanced cross-trait claim. At 30%
support, within a run:

- Magnitude–Wanda Jaccard is about 0.66 (3.7–3.8× random expectation);
- Taylor–Taylor-contrastive is about 0.52 (2.9–3.0× random);
- Magnitude–Taylor is about 0.39–0.42 (2.2–2.4× random);
- Magnitude–Taylor-contrastive is about 0.26 (1.4–1.5× random).

Across runs, Magnitude has cat–cat overlap about 0.66 (3.8× random), Wanda
about 0.81–0.86 (4.6–4.9×), Taylor about 0.39–0.42 (2.2–2.4×), and
Taylor-contrastive about 0.29–0.31 (1.7–1.8×). Overlap is most above random at
10% support and decreases as supports become denser.

Approximate Spearman score correlations (uniform sample up to 200k features):

- Magnitude–Wanda: about 0.68;
- Taylor–Taylor-contrastive: about 0.55;
- Magnitude–Taylor: about 0.44–0.45;
- Magnitude–Taylor-contrastive: about 0.12–0.13;
- Wanda–Taylor-contrastive: about 0.05–0.10.

**Interpretation.** Magnitude/Wanda select strongly recurring structure, while
Taylor-contrastive selects a materially different ranking. Recurrence is
descriptive: it may reflect generic LoRA geometry or module scale, not the SL
mechanism itself.

### Five-method historical density screen

To place the trait-blind **Taylor-numeric** selector on the same density plot
as the earlier methods, we mirrored the original causal-screen protocol: the
same historical prompt family, selected-support branch, densities 10%, 30%,
and 50%, and the four runs for which all original scores exist (three cat seeds
and owl-47). This is an **exploratory protocol mirror**, not the final frozen
evaluation: owl-48/49 were not retrospectively run through the original
all-method density sweep.

The plotted quantity is the paired SL effect after retaining only the selected
support. For cats, Taylor-contrastive retained 7.92, 14.78, and 17.13 pp of SL
at 10%, 30%, and 50% support; Taylor-trait retained 5.12, 13.17, and 16.13 pp;
Magnitude retained 0.10, 11.02, and 16.28 pp; Wanda retained 0.22, 3.72, and
6.53 pp. Taylor-numeric was much weaker: -0.53, -0.42, and 2.95 pp. Owl-47
shows the same broad contrast at the small supports, but is only one seed and
therefore has no between-seed error bar.

![Exploratory five-method SL effect across support densities](figures/five_method_support_effect_by_density.png)

**Interpretation.** All curves generally increase with retained density because
keeping more of *any* adapter makes the masked model closer to the full adapter.
The comparison of interest is vertical, at a fixed density: Taylor-trait and
Taylor-contrastive retain substantially more SL than Taylor-numeric at 10% and
30% support. Thus the common curve shape is not the result; the selector gap is.
The separately frozen v2 test at 30% is the confirmatory version of this
conclusion.

### Final balanced six-run analysis

After adding owl-48 and owl-49, we repeated the pre-specified descriptive
comparison for the two carried-forward methods only: Magnitude and
Taylor-contrastive. At 30% support, Magnitude recurs strongly across every
cat–cat, owl–owl, and cat–owl pair (Jaccard 0.657–0.780, 3.7–4.4× the
equal-density random expectation). Taylor-contrastive is less recurrent but
still consistently above chance (0.292–0.329, 1.7–1.9× random).

The final within-run heatmap is correspondingly restricted to the two frozen
selectors, but includes every completed cat and owl adapter. Its off-diagonal
cell asks whether Magnitude and Taylor-contrastive choose the same 30% support
within one adapter; it is descriptive overlap, not causal evidence.

![Completed-seed within-run selected-mask overlap at 30% support](figures/selected_within_run_method_overlap_d30.png)

The complementary cross-run heatmap fixes the selector and compares supports
between all six runs.

![Completed-seed selected-mask overlap at 30% support](figures/owl_completion_selected_cross_run_overlap_d30.png)

The density trend is the same in all three comparison groups: recurrence is
largest at 10% support and declines as the support becomes denser. This is
expected in part because dense supports overlap more by construction; the
random-normalized value is the informative comparison.

![Completed-seed recurrence over random by density](figures/owl_completion_selected_overlap_over_random_by_density.png)

Finally, score correlation uses all rankings before any support threshold is
applied. It is available for the original four-run screen, where all four
methods were scored; owl-48/49 were deliberately evaluated only for the two
pre-chosen selectors. The screen confirms that Taylor-contrastive is not merely
Magnitude/Wanda renamed: it weakly correlates with them, even though it
moderately correlates with ordinary Taylor.

![Feature-score correlations](figures/feature_score_spearman_correlation.png)

### Taylor-numeric structural control

We additionally scored **Taylor-numeric** on all six completed runs.  This is
first-order Taylor importance for held-out numeric next-token loss: it does not
use animal-preference prompts, but it is not a fully task-neutral selector.
The analysis compares its selected columns with Taylor-trait and
Taylor-contrastive at several support densities. It is a structural control:
the subsequent frozen v2 evaluation below is required to determine whether the
overlap is behaviourally meaningful.

At **30% retained support** (equivalently 70% pruned), averaged over runs in
the trait adapter:

| Compared selected supports | Jaccard relative to equal-density random expectation |
|---|---:|
| Taylor-numeric vs. Taylor-trait | 2.01x |
| Taylor-numeric vs. Taylor-contrastive | 1.10x |
| Taylor-trait vs. Taylor-contrastive | 2.97x |

Thus, numeric-task importance and trait importance share non-random structure,
whereas Taylor-contrastive is structurally more distinct. This is expected in
part because Taylor-contrastive explicitly subtracts normalized numeric Taylor
importance from trait Taylor importance. At the sparser 10% support, the same
comparisons rise to 3.48x, 1.75x, and 10.74x, respectively; ratios rise at low
support because the equal-density random intersection is very small. The
frozen v2 causal result is crucial: this moderate structural overlap does **not**
make Taylor-numeric a good SL mask. It primarily appears to capture recurring
numeric-task structure rather than the behaviourally selective SL support.

The corresponding 30% supports in the trait and matched control adapters are
also distinguishable: Taylor-numeric has about 2.4x random trait--control
overlap, Taylor-trait about 2.1--2.2x, and Taylor-contrastive about 1.4--1.5x.
This is compatible with Taylor-contrastive removing more task-generic
structure, but it does **not** establish that it is behaviorally better.

![Taylor-objective overlaps across pruning levels](figures/taylor_objective_trait_control_overlap_by_pruning.png)

![Taylor-objective overlap within trait adapters at 30% support](figures/taylor_objective_within_trait_heatmaps_p70.png)

![Taylor-objective overlap within control adapters at 30% support](figures/taylor_objective_within_control_heatmaps_p70.png)

The next inference-bearing test is a separately frozen causal evaluation of
Taylor-numeric versus Taylor-trait, Taylor-contrastive, Magnitude, and random
masks. It must not reuse the held-out split that informed the present method
comparison.

### Trait-mask versus control-mask transfer

We also ran a frozen **control-mask transfer** experiment for Taylor-trait.
For each active `WS` adapter, we compared (i) a support/ablation selected from
that `WS`, (ii) an equal-density Taylor-trait support selected from the matched
`WC` adapter, and (iii) two random supports. All masks were applied to the
same active `WS`; therefore the comparison asks whether the selector tracks
SL-specific adapter structure rather than generic numeric-task structure.
The frozen evaluation uses a new 40-prompt trait family and a separate numeric
slice. Results below are the mean SL gap (`WS_masked − intact WC`, percentage
points) across all six runs.

| Retained support | Full `WS` gap | WS-derived support | WC-derived support | Random support | WS-derived ablation | WC-derived ablation | Random ablation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 6.75 | 1.46 | 0.04 | 0.00 | 0.04 | 4.75 | 5.44 |
| 30% | 6.75 | **4.38** | 0.29 | 0.10 | **0.08** | 1.00 | 2.83 |
| 50% | 6.75 | **5.92** | 2.08 | 0.92 | −0.12 | −0.04 | 1.00 |

At 30% support, the `WS`-derived support retains about 65% of the full frozen
SL gap (4.38/6.75 pp), while the matched `WC` support retains almost none
(0.29 pp). Conversely, ablating the `WS`-derived support leaves essentially no
gap (0.08 pp), more selective than ablating the matched `WC` support (1.00 pp)
or random support (2.83 pp). At 50%, random ablation is itself substantially
destructive, so that density is less discriminative. This is stronger evidence
than overlap alone that Taylor-trait selection captures structure specific to
the trait-trained adapter; it still does not establish that the mechanism is
only in LoRA, because the frozen base remains intact.

## Activation-space analyses

Let `h` be the final-prompt-token residual-stream hidden state at a layer.
For each matched evaluation prompt, the original comparison was:

```text
d_teacher = h(W0, trait system prompt) − h(W0, no system prompt)
d_student = h(WS) − h(WC)
```

It had high cosine near layer 11 (about 0.84–0.86), but this contrast changed
two things at once: the trait content and the mere presence of a system
message.

### Matched system-prompt control

We therefore added a matched neutral system prompt. This decomposes the
original teacher-side direction into:

```text
d_trait = h(W0, trait system prompt) − h(W0, neutral system prompt)
d_system = h(W0, neutral system prompt) − h(W0, no system prompt)
d_legacy = h(W0, trait system prompt) − h(W0, no system prompt)

d_legacy = d_trait + d_system
```

Here `d_trait` is our best approximation to the **additional effect of the
trait content**, conditional on both prompts being system prompts. `d_system`
is a control for the generic effect of adding a system message. We compared
each of these directions with `d_student` using cosine similarity.

- At layer 11, `cosine(d_trait, d_student)` was only 0.185–0.229, whereas
  `cosine(d_system, d_student)` was 0.689–0.701. Thus the old 0.84–0.86 peak
  cannot be read as a clean cat/owl direction.
- A later trait-related signal remained for cats. At layer 28,
  `cosine(d_trait, d_student)` was **0.639, 0.637, and 0.595** in cat-47,
  cat-48, and cat-49 respectively (mean **0.624**). The corresponding
  system-only values were only 0.097, 0.093, and 0.104. Cats also had a
  signal around layer 21; owl peaked at layer 21 (0.628) and was 0.541 at
  layer 28.
- The completed owl extension preserves a substantial clean signal but not a
  universal peak layer: owl-48 peaks at layer 19 (0.589; system control 0.210)
  and owl-49 at layer 19 (0.559; system control 0.230). The causal owl jobs
  remain pre-registered at layer 21 from owl-47 rather than selecting these
  new peaks after observing them.

**Interpretation.** The original single layer-11 claim is invalid. There is a
later trait-related teacher/student direction, but it is not one universal
layer across traits.

The figures below are deliberately split by trait. The cat panel reports the
mean and SD across three cat seeds, with the direction norm as a safeguard
against interpreting a cosine from a nearly zero vector. The owl panel instead
shows every owl seed separately for the clean trait, system-only, and legacy
contrasts. This is more informative than folding three owl trajectories into a
cat mean, because their later clean-direction peaks are not identical.

![Cat activation directions with system-prompt controls](figures/activation_direction_controls_cat_by_layer.png)

![Owl activation directions with system-prompt controls](figures/activation_direction_controls_owl_by_layer.png)

### Causal direction pilot

At the selected layer we modified only the **final prompt-token residual
vector during prefill**. Concretely, after the full chat-formatted user prompt
has entered the model — e.g. after the final token of “What is your favourite
animal? Reply with one word.” — the chosen decoder layer returns a tensor with
one hidden vector per prompt position. We altered only the last of those vectors
before the model predicts its first answer token. Cached autoregressive steps
for later generated tokens were left untouched.

For a unit direction `u = d / ||d||`, projection-out replaces `h` by
`h − dot(h, u) · u`; steering replaces it by `h ± 0.5 · ||d|| · u`. We compared teacher-direction projection-out
with neutral-system projection-out, three normalized random directions, and
`±0.5×` teacher-direction steering. The primary statistic is:

```text
Δ(WS − WC) = (WS_intervention − WS_full) − (WC_intervention − WC_full)
```

For cats, teacher-direction projection-out changed the SL effect by **−15.2,
−11.8, and −8.0 pp** (runs 47/48/49). Random and neutral projections were near
zero/modest positive; −0.5 steering gave −8.8/−9.8/−6.2 pp and +0.5 steering
gave +7.8/+5.6/+6.0 pp. Numeric validity remained 96.9–100%.

For owl, the pre-registered layer-21 intervention does not replicate the cat
pattern. All three seeds show the **opposite steering sign**: negative steering
increases `WS − WC` and positive steering decreases it. Projection-out is mixed:

| Run | Full `WS − WC` | Teacher project-out Δ | −0.5 steering Δ | +0.5 steering Δ |
|---|---:|---:|---:|---:|
| owl-47 | 6.60 pp | −0.60 pp | +1.40 pp | −2.60 pp |
| owl-48 | 7.40 pp | +1.60 pp | +1.40 pp | −4.20 pp |
| owl-49 | 14.80 pp | −4.60 pp | +1.00 pp | −7.80 pp |

Here Δ is the difference-in-differences relative to each run's full-adapter
effect. Numeric validity stayed 96.9–100% across the trait and control sides.

**Interpretation.** The controlled teacher direction is causally coupled to
cat SL in all three cat runs. It does not identify the base weights behind that
behaviour; activation directions can be useful causal handles without being a
unique mechanistic locus. The completed owl extension shows a consistent
pre-registered steering response, but its orientation is not a stable
teacher-to-student trait direction: the steering signs are opposite to cats and
project-out is mixed. It should not be used as an owl base-localization selector
without further calibration.

The difference-in-differences plot makes the required control explicit. A value
below zero for projection-out means the intervention reduced the `WS − WC`
effect, rather than merely changing both models equally. The three cats show
large negative teacher-direction projection-out and the expected sign flip for
negative versus positive steering. Owl shows a reproducible but opposite
steering sign and mixed project-out, which is not the same mechanistic pattern.

![Causal activation interventions as difference-in-differences](figures/activation_causal_difference_in_differences.png)

## First base-component localization pilot: cats

The LoRA-mask results retain the entire base model, so they cannot distinguish
an adapter-local mechanism from a LoRA update that recruits the frozen base.
We therefore ran a deliberately narrow cross-seed **base-component dependence**
pilot. It does not prune base weights and is not yet a base-weight ticket test.

### Frozen candidate selection

Using `cat-47` only, we fixed layer 28 before evaluation because it was the
clean cat activation-direction layer. For the attention output and MLP output
of that decoder layer separately, we scored each residual channel `j` by:

```text
abs(mean(teacher trait-system − neutral-system)_j
    × mean(WS − WC)_j)
```

We froze the top 128 channels of each component (3.6% of the 3,584 residual
channels), plus an equal-size random channel set drawn from the complement.
Candidate identities and provenance are stored in
`artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/selection.{pt,json}`.

### Cross-seed causal evaluation

On independently trained `cat-48` and `cat-49`, we zeroed only the selected
component-output channels at the **final prompt token during prefill**. We did
the same with the matched random channels, separately for `WS` and `WC`, then
computed the usual `WS − WC` effect on the frozen third trait-prompt family.
This is an activation gate on a frozen base component, rather than changing a
stored base weight. Numeric validity was measured on 128 held-out sequences.

| Run | Gate | `WS` | `WC` | SL effect | Δ versus matched random | Numeric WS / WC |
|---|---|---:|---:|---:|---:|---:|
| cat-48 | full | 25.8 | 0.8 | 25.0 | — | 99.22 / 99.22 |
| cat-48 | attention candidate | 23.6 | 0.6 | 23.0 | −1.2 | 98.44 / 100.00 |
| cat-48 | attention random | 25.0 | 0.8 | 24.2 | reference | 99.22 / 99.22 |
| cat-48 | MLP candidate | 12.8 | 3.2 | 9.6 | **−15.8** | 99.22 / 100.00 |
| cat-48 | MLP random | 26.2 | 0.8 | 25.4 | reference | 99.22 / 99.22 |
| cat-49 | full | 13.8 | 0.8 | 13.0 | — | 100.00 / 97.66 |
| cat-49 | attention candidate | 13.2 | 0.8 | 12.4 | −0.6 | 100.00 / 98.44 |
| cat-49 | attention random | 13.8 | 0.8 | 13.0 | reference | 100.00 / 97.66 |
| cat-49 | MLP candidate | 7.2 | 3.2 | 4.0 | **−7.8** | 100.00 / 97.66 |
| cat-49 | MLP random | 12.4 | 0.6 | 11.8 | reference | 100.00 / 97.66 |

The left panel below shows the raw paired effect under every gate; the right
panel gives the inference-bearing comparison, candidate gate minus its
equal-size random gate. Only the MLP candidate produces a large, replicated
selective reduction.

![Cross-seed base-component activation-gating pilot](figures/base_component_cross_seed_pilot.png)

**Interpretation.** The frozen MLP-channel gate reduced the cat SL effect far
more than its matched random gate in both unseen seeds, with high numeric
validity. The attention-channel candidate gate did not show a comparable
selective reduction. This is promising evidence that a small, direction-linked
MLP component in the frozen base is causally relevant to the *student* effect.

The conclusion is intentionally limited. Candidate selection used `cat-47` and
was evaluated only on two transfer seeds; the gate acts on activations, not
base parameters; and prompted `WT` has not yet been tested under the same gate.
Thus this does not yet establish that teacher and student use the same base
circuit, nor that the base component is necessary independently of LoRA. The
next confirmatory step is to test the frozen MLP candidate in prompted `WT`,
combine it factorially with the selected LoRA masks, and retain matched random
base controls.

## Appendix: visualization index

All figures are descriptive summaries; no new model was trained to make them.

| Image | Question | Reading / result |
|---|---|---|
| `figures/within_run_method_overlap_d30.png` | What happened in the initial four-method screen? | Historical, unbalanced cat-47/48/49 plus owl-47 screen; used for method selection, not final cross-trait comparison. |
| `figures/cross_run_same_method_overlap_d30.png` | What happened in the initial across-run four-method screen? | Historical, unbalanced screen; superseded for final claims by the six-run selected-mask figures below. |
| `figures/owl_completion_selected_cross_run_overlap_d30.png` | Do the two selected masks recur after completing owl seeds? | Magnitude is high across all six runs; Taylor-contrastive is lower but consistently above random. |
| `figures/selected_within_run_method_overlap_d30.png` | Do the two frozen selectors agree within every completed adapter? | Six balanced panels; cells are Jaccard and × random expectation. |
| `figures/owl_completion_selected_overlap_over_random_by_density.png` | Does the completed owl extension preserve the density trend? | Yes: 10% is most above random for cat–cat, owl–owl, and cat–owl comparisons. |
| `figures/ticket_distribution_cat_d30.png` | Where are selected cat LoRA features? | All three cat seeds, separated from owl. Nonuniform selection is descriptive, not causal localization. |
| `figures/ticket_distribution_owl_d30.png` | Where are selected owl LoRA features? | All three owl seeds, using the identical visualization and global 30% target. |
| `figures/activation_causal_difference_in_differences.png` | Does direction intervention selectively alter SL? | Cat has the expected suppression/amplification pattern; all three owl runs have the opposite steering sign and mixed project-out. |
| `figures/activation_direction_controls_cat_by_layer.png` | Is cat alignment trait-specific? | Cat mean ± SD for clean trait, system-only, and legacy contrasts, plus direction norms. |
| `figures/activation_direction_controls_owl_by_layer.png` | Is owl alignment trait-specific? | All three owl seeds separately for the same three contrasts; clean later signals exist but peak layer differs by seed. |
| `figures/feature_score_spearman_correlation.png` | Do score methods rank features similarly before thresholding? | Magnitude/Wanda correlate around 0.68; Taylor/TC around 0.55; TC is weakly correlated with Magnitude/Wanda. |
| `figures/causal_mask_aggregate.png` | What happened in the exploratory screen across all four methods and densities? | Selected-support and selected-ablation effects against random controls. It informed candidate selection; it is not the final frozen result. |

## Current conclusions and limits

### Supported

1. Qwen2.5-7B exhibits cat/owl SL through semantically unrelated number data;
   both traits now reproduce across three runs.
2. Magnitude and Taylor-contrastive supports are selectively causal in LoRA;
   Taylor-contrastive is strongest, while Magnitude is the neutral baseline.
3. No large selected-mask-specific degradation was found in current retention
   controls.
4. In cats, a prompt-controlled teacher activation direction is causally
   coupled to the student behaviour.
5. A first cross-seed cat pilot finds that gating 128 frozen-base MLP output
   channels selected in cat-47 selectively reduces SL in cat-48/49 relative to
   random gates, while preserving numeric completion.

### Not yet established

1. The phenomenon is not shown to be only in LoRA; all mask results retain the
   entire base model.
2. No **base weight**, head, MLP feature, or SAE feature is known to be
   necessary or sufficient for SL. The current MLP result is an activation-gate
   pilot, not a weight-level localization.
3. The activation direction does not prove teacher and student use the same
   base circuit.
4. Owl now has three behavioural seeds and completed frozen-mask, causal-
   direction, and blinded-coherence extensions. Its direction has the opposite
   steering sign to cats and mixed project-out, so it is not yet a stable
   localization handle.
5. Coherence needs blinded manual audit because empty/truncated output is
   nontrivial even if it is not selected-mask-specific.

## Research notes: meeting of 24 October 2026

These are forward-looking questions, not completed results.

1. **Prompted versus fine-tuned teachers.** A direct comparison between the
   current Qwen prompted-teacher setting and Mariana's Gemma fine-tuned-teacher
   setting is descriptive only: model family and teacher-conditioning regime
   both change. First create a shared protocol card (trait, teacher strength,
   data filtering, student budget, controls, seeds, and frozen evaluation).
   The clean experiment is within one model family: prompted versus fine-tuned
   teacher under the same student protocol; ideally repeat in the second family.
2. **Trait-blind Taylor control.** The scorer already implements
   `--objective numeric`: Taylor importance for held-out numeric next-token
   loss. It has not been evaluated as a standalone mask. This should be called
   **Taylor-numeric**, rather than fully neutral or vanilla Taylor: it is blind
   to the animal trait but explicitly informed by the numeric task. It is also
   the numeric component already subtracted inside Taylor-contrastive. A fair
   new-method test needs a new frozen evaluation split, rather than reusing the
   final held-out results that informed the current analysis.
3. **Cross-trait mask specificity.** The initial four-run screen found ordinary
   Taylor overlap of about 0.39--0.42 at 30% support across cat and owl-47
   (2.2--2.4× equal-density random expectation). Full Taylor-trait score
   artifacts now exist for owl-48/49, but the six-run plots need regeneration.
   More directly, test mask transfer: apply a `WS`-derived mask, a matched
   `WC`-derived mask, and a random mask to the same `WS` adapter. This asks
   whether a control mask preserves/ablates SL nearly as well as a trait mask.
4. **Trait dependence of activation directions.** The current evidence already
   rejects a single universal layer: cats have a clean causal handle at layer
   28, while owl peaks differ by seed and has opposite steering signs. It does
   not yet isolate trait semantics from an animal/preference-family direction.
   Test several semantically diverse traits, matched semantic controls, and
   cross-trait intervention (e.g. use a cat-derived direction in an owl run)
   before calling any direction SL-specific.
5. **Cross-project audit.** Collect Mariana's and Soham's code, result tables,
   exact LoRA target modules, score objectives, mask granularity, retraining
   protocol, and controls into the shared protocol card before comparing
   conclusions.
6. **Column-based pruning.** In our and the EM legacy convention, a feature/
   column mask zeros a whole column of LoRA `A`. Given `DeltaW = B @ A`, this
   zeros one input column of the effective update `DeltaW` across every LoRA
   rank component. It is structured parameter pruning, not pruning an
   activation token or neuron at inference time.

## Artifact map

| Artifact | Location |
|---|---|
| SL runs, data, adapters | `artifacts/sl/qwen25_7b_{cat,owl}_replication*` |
| Mask scores | `.../mask_scores/trait/` |
| Final mask evaluations | `.../final_mask_evaluation/` |
| Prompt-control activations | `.../activation_analysis_system_prompt_control/summary.json` |
| Causal activations | `.../activation_causality_v1/{trait,control}/` |
| Overlap figures | `artifacts/sl/mask_analysis/overlap/` (completed owl extension: `.../overlap/owl_completion/`) |
| Additional figures | `artifacts/sl/additional_analysis/` |
| Base-component pilot | selector: `artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/`; transfer evaluations: the corresponding `..._seed48/` and `..._seed49/` roots |

The next mechanistic question is whether teacher-derived **base-model
components** affect both prompted teacher behaviour and `WS`, and whether their
effect interacts with the two established LoRA masks. That is what can separate
adapter-local, base-reuse, and hybrid explanations.
