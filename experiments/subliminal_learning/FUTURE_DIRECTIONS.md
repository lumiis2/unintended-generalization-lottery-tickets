# From LoRA tickets to base-model mechanisms

This document records research directions after the completed LoRA-mask
experiments. It is a planning document, not a statement of established facts.

## Current evidence

### Observed in our experiments

- Subliminal Learning (SL) was reproduced in Qwen2.5-7B-Instruct with three
  cat seeds and one owl run.
- Feature-level LoRA masks selected by Taylor-contrastive at 30% support were
  selectively causal on held-out preference templates: retaining the selected
  support preserved the `WS - WC` trait effect, while ablating it reduced that
  effect much more than matched random masks. Numeric completion validity
  remained high.
- The selected supports overlap across cat seeds above a random-mask baseline.
- The mean activation direction `prompted teacher - neutral base` aligns with
  `trait student - control student`, especially around layer 11. This is
  correlational evidence only.

### What this does **not** yet show

- It does not show that the mechanism is entirely contained in the LoRA.
  A small LoRA subset can act as a gate into a circuit already present in the
  frozen base model.
- It does not identify individual base-model weights that are causal for SL.
- It does not establish that the observed activation direction is trait-specific:
  the current teacher condition has a trait system prompt while the neutral base
  condition has no matched neutral system prompt.
- It does not establish a general lottery ticket across architectures, traits,
  or broad benign capabilities.

## Relevant literature

### Emergent Misalignment (EM)

Two lines of work motivate activation-space tests.

- Soligo et al., *Convergent Linear Representations of Emergent Misalignment*
  (2025, https://arxiv.org/abs/2506.11618) report a transferable linear
  misalignment direction across EM fine-tunes and use interventions on that
  direction to affect behaviour.
- OpenAI, *Toward understanding and preventing misalignment generalization*
  (2025, https://openai.com/index/emergent-misalignment/) reports SAE features
  associated with a misaligned persona. Steering along such a feature can
  influence EM behaviour.

These results are evidence that EM can have interpretable, causally useful
activation-space representations. They do not imply that SL must have one
single equivalent direction.

### Subliminal Learning (SL)

- Cloud et al. (2025, https://arxiv.org/abs/2507.14805) demonstrate behavioural
  transmission through semantically unrelated data and motivate shared
  representations from common initialization. The paper does not establish one
  canonical, causal SL steering direction.
- Aria Wong's exploratory blog post (2025,
  https://ariahw.github.io/blog/2025/08/29/subliminal-learning/) studies SL in
  activation space with probes, SAEs, and steering. It finds activation-space
  signal, but suggests that a subliminal direction is not simply identical to a
  direct animal-preference direction. This is useful motivation, but it is
  work in progress rather than settled evidence.

### How prior work checks collateral damage

The closest precedents are useful, but neither is a complete general-capability
preservation protocol.

- Cloud et al. evaluate the main SL effect with preference, story, and
  multiple-choice trait tests. To check that number-sequence fine-tuning had
  not simply caused a broad capability shift, they evaluate two example
  students on MMLU: the eagle-number student drops four percentage points and
  the owl-number student drops two points relative to its reference model.
  In their misalignment settings they also use TruthfulQA, finding that an
  insecure-teacher student becomes modestly less truthful. This is a useful
  precedent for including an external benchmark, but it is a limited spot
  check rather than a broad retention suite. Their main animal-preference
  measurement is intentionally one-word, so it is not itself a meaningful
  coherence evaluation.
- The original EM work makes **incoherence** an explicit confound: its model
  judge first labels responses as refusal, incoherent, satirical/absurd, or
  off-topic before assigning an alignment score to valid responses. It reports
  that some apparent EM in human-data fine-tunes tracks incoherence rather than
  the synthetic-data EM mechanism.
- Soligo et al. evaluate both alignment and coherence with independent judges.
  Their GPT-4o coherence judge scores whether a response is valid English and
  free of hallucination, rambling, and confusion; an EM response must be both
  misaligned and coherent. Their activation-direction ablations reduce EM while
  keeping coherence above 99%, and they compare against a random vector with
  matched norm. This is a particularly relevant template for our future
  activation interventions: compare the targeted intervention against a
  matched random intervention and report response quality separately from the
  target behaviour.

For our LoRA mask result, numeric completion is only an **on-task retention**
control. It is valuable, but especially for Taylor-contrastive it is not enough:
numeric preservation is partly encouraged by the score itself.

## Central competing hypotheses

Let `W0` be the base model, `WT` the same base model under the trait teacher
prompt, `WS` the trait-trained LoRA student, and `WC` the neutral-teacher LoRA
control.

| Hypothesis | Informal description | Expected evidence |
|---|---|---|
| Adapter-local | The relevant implementation lives primarily in the LoRA update. | A small LoRA mask is necessary/sufficient; matched base interventions add little. |
| Base reuse / LoRA gate | LoRA routes inference into a trait circuit already available in `W0`. | Teacher and student share causal base components or activation directions; intervening on teacher-derived base components affects `WS`. |
| Hybrid | LoRA contains a selective part, but it cooperates with a base circuit. | Both LoRA and base interventions have selective effects; neither alone explains the phenotype. |
| Distributed | No compact selective substrate is found. | Effects require broad changes or interventions damage behaviour indiscriminately. |

Our current results reject neither adapter-local nor base-reuse: they establish
a **selective LoRA ticket conditional on an intact base model**.

## Recommended sequence of experiments

### Stage 0 — Freeze the LoRA conclusion with one final test

Pre-register **two complementary feature-mask candidates** at 30% support:

- **Magnitude:** a data- and trait-label-free baseline. It is not fully
  assumption-free (it assumes larger effective updates matter more), but it is
  the least behaviour-targeted selector in the current suite.
- **Taylor-contrastive:** the best selective candidate so far, explicitly
  optimized for trait relevance relative to numeric-task relevance.

Do **not** repeat the entire exploratory grid of methods and sparsities with
general benchmarks. That grid served to select candidates. Instead, evaluate
these two pre-registered candidates once on a final, untouched set of
preference templates and disjoint numeric examples. This set must be created
now and not inspected while choosing masks, methods, densities, or intervention
scales.

Fix before running:

- the two masks: magnitude and Taylor-contrastive, feature-level, 30% support;
- the two causal branches: selected support and selected ablation;
- the number of equal-density random-mask controls per branch (recommend at
  least five);
- generation settings, prompt count, and analysis code;
- the statistical comparison: `WS - WC` for selected versus random masks.

For each candidate, evaluate the full adapter, selected support, selected
ablation, and its random support/ablation controls in both `WS` and matched
`WC`. The final trait test should use a **third** animal-preference template family,
unseen during both Taylor calibration and held-out validation. It should also
use a new deterministic slice of valid number data, disjoint from training,
Taylor numeric calibration, and the existing validation slice.

#### General-capability retention panel

The final test should report collateral effects for **every selected and random
mask condition**, in addition to numeric validity. This is necessary to tell a
selective loss of SL from generic model damage:

1. **Primary response-quality / coherence screen (fixed method).** Use 120
   held-out, benign chat prompts: 24 each for instruction following, factual
   QA, short reasoning, summarization, and ordinary conversation. Generate one
   deterministic response per prompt (`temperature = 0`, fixed maximum length).
   Score each response with a *frozen external LLM judge*, blinded to model and
   mask condition, using this mutually exclusive rubric:

   - `valid`: directly answers the benign request in coherent English;
   - `empty_or_truncated`;
   - `repetitive_or_garbled`;
   - `off_topic`;
   - `unwarranted_refusal`.

   The primary quality metric is the fraction labelled `valid`; the secondary
   metric is the combined failure rate of the other four labels. The judge
   model identifier, rubric, prompt order, and decoding settings must be fixed
   and logged before running. We will manually audit a blinded random 10% of
   judged samples, so the judge is not the sole authority.

   In parallel, record deterministic diagnostics that require no judge:
   empty output, EOS/truncation, response length, and an n-gram repetition
   flag. These are not semantic-quality metrics, but can reveal obvious failure
   modes and audit the judge.
2. **Deterministic language metric.** Compare held-out next-token loss (or
   perplexity) on generic text for full adapter, selected support, selected
   ablation, and matched random masks. This gives a cheap distribution-level
   signal that does not depend on the cat/owl trait prompts.
3. **Small external capability panel.** Run the same intervention conditions on
   a pre-declared subset of established benchmarks, for example HellaSwag,
   ARC-Challenge, MMLU/MMLU-Pro subset, and optionally GSM8K if the chat format
   is configured reliably. Report per-task changes and the selected-minus-random
   difference, rather than only absolute scores.

The key comparison is not whether a masked model exactly matches the full
adapter; pruning necessarily removes information. For both magnitude and
Taylor-contrastive, the question is whether the selected mask suppresses or
retains SL **more selectively** than random masks of equal structural density
while causing no disproportionate general-quality or capability loss.

If both pass this decision gate, carry both forward to base-versus-LoRA
experiments. Magnitude then serves as a relatively neutral structural baseline,
while Taylor-contrastive tests the strongest trait-specific ticket hypothesis.
If one causes disproportionate general degradation, retain it only as a
diagnostic comparison rather than treating it as evidence for a selective
mechanism.

Purpose: distinguish a robust ticket result from selection on the prior
exploratory and validation sets, while ruling out the simple explanation that
the selected intervention broadly damages the model.

### Stage 1 — Remove the system-prompt confound in activation analysis

Repeat the teacher/base activation comparison with four matched prompt states:

1. base, no system prompt;
2. base, neutral system prompt matched in format/length as closely as possible;
3. base, trait teacher system prompt;
4. `WS` and `WC`, each without the trait prompt.

Use the same user prompts. Define candidate directions such as:

\[
d_{teacher} = h(W0, \text{trait system}) - h(W0, \text{neutral system})
\]

and

\[
d_{student} = h(WS) - h(WC).
\]

Compare them layerwise using cosine similarity, CKA/probes where useful, and
matched random directions. Use held-out prompts for the comparison.

Purpose: ask whether the direction is specifically associated with the trait,
rather than merely with a system message or changed sequence length.

### Stage 2 — Causal activation interventions

Start with a small layer and scale sweep near the layers with strongest,
replicated alignment (currently around layer 11). For candidate directions:

- **Ablation:** project out `d_teacher` or `d_student` from the residual stream
  during inference.
- **Steering:** add a scaled direction at one layer or a short layer window.
- **Patching:** replace activations from a matched teacher pass into a student
  pass, or vice versa.

For every intervention, measure trait preference, numeric completion, and
controls using random directions, a neutral-system direction, layer controls,
and scale controls.

Interpretation:

- teacher-derived direction causally affects unprompted `WS` more than `WC`:
  evidence for shared/reused activation-level substrate;
- student-derived direction also affects prompted teacher behaviour: reciprocal
  support for convergence;
- only the student direction works: weaker evidence for adapter-local change;
- neither works: the mechanism may be nonlinear, distributed, or represented
  outside the chosen residual-stream summary.

Steering success alone is not localization: an injected direction can impose a
behaviour without being its native mechanism. Patching and ablation are more
diagnostic.

### Stage 3 — Locate causal **base components** before pruning base weights

Do not begin by magnitude-pruning all base parameters or jointly pruning base
and LoRA. The frozen base has billions of weights; indiscriminate pruning will
damage capability and make a loss of SL uninterpretable.

Instead, use activation attribution/patching to rank structured components:

- attention heads;
- MLP layers or MLP output channels;
- residual-stream directions;
- optionally SAE latents, if a suitable SAE exists for the model.

For a teacher-derived component set `S_T`, compare its ablation in:

- prompted `WT`;
- unprompted `WS`;
- `WC`;
- matched random components.

Then perform the reciprocal test for a student-derived set `S_S` in `WT`.

Purpose: test whether the same *base-model components* are causally relevant in
both teacher and student conditions.

### Stage 4 — Factorial base × LoRA intervention

Only after identifying candidate base components, run a small factorial study:

| LoRA state | Base state |
|---|---|
| full / selected LoRA support / LoRA ticket ablated | intact / selected base components ablated / matched random base components ablated |

Readouts: trait effect, numeric task, and broader benign controls.

This can discriminate the hypotheses more cleanly:

- LoRA ablation alone removes SL, base ablation does not: adapter-local
  candidate.
- Both selective LoRA and teacher-derived base ablations remove SL: hybrid or
  base-reuse mechanism.
- Base ablation affects teacher and `WS`, while LoRA mask primarily determines
  access to it: LoRA-gated base reuse.

## Practical next action

The next implementation should be **Stage 1**, not whole-network pruning:
repeat the activation comparison with a matched neutral system-prompt control,
save layerwise direction statistics on calibration prompts, and reserve a final
prompt set for causal intervention tests. This is cheap relative to base-weight
pruning, addresses the main confound in the current activation result, and
directly informs which base components/interventions are worth testing.
