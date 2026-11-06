# Base-model localization for Subliminal Learning

## Purpose

This document plans the next mechanistic phase of the project: distinguish
whether the Subliminal Learning (SL) behaviour is implemented primarily in the
LoRA update, depends on a circuit already present in the frozen base model, or
requires an interaction between both.

It is a forward-looking protocol. The factual results so far are maintained
only in the [experiment history](../results/experiment_history.md).

## The central difficulty

`WS` and `WC` have the **same base weights**. Their only stored-weight
difference is their LoRA adapter. Therefore, observing a selective LoRA mask
does not establish an adapter-only mechanism: the adapter may alter how the
shared base computes. Conversely, a broad base ablation would be uninformative
if it simply damages language modelling.

The relevant test is consequently conditional and causal:

> Does a small, pre-specified subset of base computation or base weights alter
> the SL effect `WS - WC` more than a matched random intervention, while
> preserving the overt task and general capability controls?

Every experiment below compares `WS` with its matched `WC`. A change in `WS`
alone is not sufficient evidence.

## Current evidence and constraint

### Established

- Magnitude and Taylor-contrastive identify compact, causal LoRA supports in
  Qwen2.5-7B cat and owl students, conditional on the intact base.
- Cat has a controlled teacher/student activation direction that causally
  modulates SL. Owl does not share one stable direction or layer.
- The base MLP-channel pilot initially looked positive in two cat seeds, but
  its direction-derived selector did **not** replicate as a stable selective
  base effect in the subsequent frozen cat/owl factorial.

### Consequence

The next experiment should not reuse the failed base-channel selector as the
main evidence. It should test base weights at the same structural granularity
as the LoRA ticket and use selection rules frozen separately from evaluation.

## Competing hypotheses and discriminating patterns

| Hypothesis | Informal mechanism | Expected pattern under matched interventions |
|---|---|---|
| Adapter-local | The relevant computation is mostly encoded in `ΔW_LoRA`. | Selected LoRA ablation changes `WS - WC`; coordinated base ablation is no stronger than random. |
| Base reuse / LoRA gate | LoRA routes computation into a pre-existing base circuit. | A selective base intervention alters `WT` and `WS` more than random; adapter gating determines when the circuit is accessed. |
| Hybrid | A selective LoRA update and selective base computation are both needed. | Both interventions matter and their joint effect differs from the sum of their individual effects. |
| Distributed / redundant | No compact support at the tested scale. | Targeted and random base interventions behave similarly; effects depend strongly on density or vary by seed. |

A non-result for one candidate family does **not** prove adapter-locality. It
only rejects that candidate family at the tested granularity.

## Design principles

1. **Never rank `W_base + ΔW` globally.** Base magnitudes dominate the update,
   so this mostly produces a generic base-magnitude mask.
2. **Match the intervention granularity.** A LoRA-A feature mask is an input
   coordinate/column mask. The most comparable base intervention masks input
   columns of the corresponding base projection.
3. **Freeze selection before evaluation.** Candidate coordinates, modules,
   densities, prompt family, random seeds, and capability checks must be fixed
   using a source run, then evaluated on other seeds and held-out prompts.
4. **Use random controls at the same module and density.** A global random
   mask is not enough when modules have different scales or dimensions.
5. **Measure interaction, not only damage.** Report the paired SL effect and
   candidate-minus-random contrasts in `WS` and `WC`.
6. **Start sparse and local.** Base interventions are much more destructive
   than LoRA interventions. Start at 1%, 3%, and 10% of coordinates in fixed
   modules; include 30% only after preservation checks pass.

## Experiment 1: observational base/LoRA forward decomposition

This is the first diagnostic, before selecting any new base mask. For each
PEFT-wrapped linear module and the same final prompt token in `WS` and `WC`,
we decompose its forward pass as:

\[
y_\ell = W_{0,\ell}x_\ell + \Delta W_\ell x_\ell.
\]

The important comparison is not the absolute magnitude of either term in one
model. It is the mean paired difference over identical frozen prompts:

\[
\Delta y_\ell = y_\ell^{WS} - y_\ell^{WC}
= \underbrace{W_{0,\ell}(x_\ell^{WS}-x_\ell^{WC})}_{\Delta_{\mathrm{base},\ell}}
+ \underbrace{(\Delta W_\ell x_\ell)^{WS}-(\Delta W_\ell x_\ell)^{WC}}_{\Delta_{\mathrm{LoRA},\ell}}.
\]

The decomposition will run on every completed cat and owl seed, over the
frozen v2 trait-prompt family, recording for each `q/k/v/o/gate/up/down`
projection:

- norm of the input shift `x_WS - x_WC`;
- norm and alignment of the base response `Δbase`;
- norm and alignment of the local adapter response `ΔLoRA`;
- reconstruction error `||Δy - Δbase - ΔLoRA||`;
- evolution of the final-token residual `WS - WC` shift across layers.

This directly addresses the proposed question: whether a local LoRA output is
followed by a large, aligned transformation in later frozen base projections,
or whether the observable shift remains concentrated in local adapter terms.
It is **not causal evidence**: a large `Δbase` says that a frozen base module
transforms an upstream adapter-induced difference, not that its weights are
necessary for SL. Its value is to generate a constrained, interpretable,
cross-seed candidate scope for the causal masks below.

Implementation: `code/scripts/sl/activations/decompose_base_lora_pathways.py`.
It saves compact mean vectors and a JSON summary under each run's
`base_lora_decomposition_v1/` directory; it does not save full activations or
modify model weights.

## Main proposed experiment: coordinated base--LoRA column masks

### Rationale

For a targeted linear projection,

\[
\Delta W_{\mathrm{LoRA}} = BA,
\]

a feature mask on a column of `A` removes an **input coordinate** from the
LoRA update. Let `S` be a support selected by an existing LoRA ticket method.
The directly comparable base intervention sets the matching input columns of
the frozen base projection to zero during the relevant forward pass:

\[
W_{\mathrm{base}}[:, S] \leftarrow 0.
\]

This does not assume that the adapter and base use the same mechanism. It asks
whether the coordinates through which the selected LoRA update reads the
residual stream are also unusually important for the base computation.

### Candidate mask sources

We should use exactly two pre-existing LoRA selectors:

| Selector | Role | Why include it |
|---|---|---|
| Magnitude, 30% LoRA support | Relatively trait-blind baseline | It did not use trait prompts to choose coordinates. |
| Taylor-contrastive, 30% LoRA support | Trait-selective candidate | It was strongest in frozen causal LoRA tests, but is selection-informed. |

The coordinate identities come from a source seed only. For example, choose
cat-47 once, freeze the mapping into each targeted projection, and evaluate on
cat-48/49; repeat the same protocol independently for owl-47 → owl-48/49.
Do not pool cat and owl masks before this within-trait replication.

### Factorial conditions

For each fixed support `S`, module family, density, and `WS`/`WC` pair:

| Condition | Base columns | LoRA features | What it tests |
|---|---|---|---|
| Full | intact | intact | Reference SL effect. |
| LoRA-only ablation | intact | selected `S` removed | Established adapter dependence. |
| Base-only ablation | selected `S` removed | intact | Coordinated base dependence. |
| Joint ablation | selected `S` removed | selected `S` removed | Interaction / redundancy. |
| Random-base control | matched random columns removed | intact | Generic base damage. |
| Random-LoRA control | intact | matched random features removed | Generic adapter damage. |
| Joint random control | matched random base + LoRA removed | matched random | Interaction control. |

The primary quantity is:

\[
E = \operatorname{pref}(WS)-\operatorname{pref}(WC).
\]

For example, the base-only contrast is

\[
\Delta_{\mathrm{base}} = E_{\mathrm{selected\ base\ ablated}}
  - E_{\mathrm{random\ base\ ablated}}.
\]

The interaction contrast is

\[
I = (E_{\mathrm{joint}}-E_{\mathrm{LoRA-only}})
  - (E_{\mathrm{base-only}}-E_{\mathrm{full}}).
\]

This is more informative than asking whether a base intervention makes the
model worse in isolation.

### Scope and safety

The first version should apply the base-column intervention **only during
prefill at a pre-registered token-position set**, not permanently rewrite a
checkpoint. Candidate position sets are: all prefill tokens, the system-prompt
span, user-prompt span, and final user token. This preserves reversibility and
also tests the context dependence reported in recent SL work.

Only after a selected-base effect transfers across held-out seeds and exceeds
matched-random damage should we run a persistent structured base-weight
ablation on the same modules.

### Preservation panel

At every density, run:

- frozen third trait-prompt family;
- held-out numeric completion;
- Wikitext loss/perplexity;
- HellaSwag, ARC-Challenge, GSM8K spot checks;
- blinded coherence sample for finalists.

An apparent base ticket is rejected if it causes a substantially larger broad
capability drop than its matched random mask.

## Complementary experiment A: dynamic LoRA gating by context, position, and module

Recent work reports that SL can be concentrated at token positions shared by
fine-tuning and evaluation, such as system-prompt/template tokens. This can
explain why an adapter-local ticket is necessary but still looks like it is
recruiting contextual base computation.

For each full `WS` and `WC`, dynamically enable the LoRA adapter only at:

1. system-prompt/template tokens;
2. user-prompt tokens;
3. final prefill token;
4. generated tokens;
5. all positions except each of the above.

Then repeat only the positions that matter with LoRA active in early versus
late layers and MLP versus attention modules. This is not a base localization
test by itself, but it determines the smallest setting in which the
coordinated base--LoRA test should operate.

**Published basis.** Nief et al. report that dynamically enabling/disabling
LoRA at shared context positions can localize SL, and find early FFN/context
concentration in their Qwen setup. [Subliminal Learning is a LoRA
Artifact](https://arxiv.org/abs/2606.00831)

## Complementary experiment B: acquisition intervention

Expression and acquisition are different questions. Once a base candidate is
validated at inference, train new LoRA students while the candidate base
coordinates are blocked only during the numeric fine-tuning forward pass.

| Training condition | Evaluation condition | Interpretation |
|---|---|---|
| Normal base | normal base | Positive SL control. |
| Selected base coordinates blocked | normal base | Is that base route needed to *acquire* SL? |
| Random coordinates blocked | normal base | Training-damage control. |
| Selected coordinates blocked | selected coordinates blocked | Checks whether expression and acquisition dependence coincide. |

If the selected block prevents SL acquisition while the random block does not,
that is stronger evidence for base reuse than an inference-only gate. This is
more expensive and should follow, not precede, a successful inference result.

## Complementary experiment C: data-to-weight causal bridge

Two competing data-level accounts motivate a useful diagnostic:

- **Token entanglement:** certain numeric tokens are geometrically or
  statistically coupled to the trait token.
- **Divergence tokens:** rare positions where trait and neutral teachers would
  predict different tokens carry most of the transferable signal.

The first account motivates scoring numbers by unembedding similarity,
teacher-logit behaviour, and dataset frequency ratios. [Zur et al., *Token
Entanglement in Subliminal Learning*](https://openreview.net/pdf?id=auKgpBRzIW)
The latter reports that masking divergence-token loss suppresses SL, while
training only on those positions often retains it. [Schrodi et al., *Towards
Understanding Subliminal Learning*](https://arxiv.org/abs/2509.23886)

For our data, identify divergence positions using trait versus neutral teacher
predictions and run three matched fine-tunes:

1. ordinary loss on all target tokens;
2. loss only on divergence positions;
3. loss only on non-divergence positions.

For each, record per-module LoRA gradients and overlap them with Magnitude and
Taylor-contrastive supports. If a small data subset both drives SL and gives
large gradients specifically to a recurring LoRA/base coordinate set, it
provides a principled candidate for base localization rather than selecting
channels from a post-hoc direction.

## Complementary experiment D: layer and module sufficiency

Train otherwise identical students with LoRA enabled only in:

- early, middle, or late thirds of the network;
- MLP projections only versus attention projections only;
- individual layers selected before the test.

This maps where the *adapter can install* SL. It is a prerequisite for a tractable
base-weight experiment: a base mask across every Qwen module is too broad to
interpret. Schrodi et al. report that early individual LoRA layers can be
sufficient in their setups; we should test this rather than assume it holds in
ours. [Schrodi et al.](https://arxiv.org/abs/2509.23886)

## Complementary experiment E: steering-vector implementation map

Our cat result already has a direction that causally changes SL. A recent
account proposes that SL distils a teacher steering vector into the student.
[Blank et al., *Subliminal Learning Is Steering Vector
Distillation*](https://arxiv.org/abs/2606.00995)

For training checkpoints, measure:

\[
v_{\mathrm{teacher}} = \mathbb{E}[h(W0,\text{trait system}) -
h(W0,\text{neutral system})]
\]

and

\[
v_{\mathrm{student},t} = \mathbb{E}[h(WS_t)-h(WC_t)].
\]

Then use dynamic LoRA gating and module ablation to ask which adapter modules
are necessary to generate `v_student`. This identifies the **implementation
path** of the direction; it does not by itself localize base weights. Owl's
variable direction result means this is a cat-first analysis with an explicit
cross-trait replication requirement.

## Recommended sequence

### Phase 0 — freeze protocol

Freeze source seeds, mask mapping, density schedule, token spans, random-mask
seeds, prompt families, and preservation thresholds. Use the existing frozen
v2 family only for final evaluation; create a disjoint calibration split if a
choice must be made after looking at v2.

### Phase 1 — map where the adapter acts

Run dynamic LoRA gating by position/module and a coarse early/middle/late LoRA
layer sufficiency sweep. These can run in parallel because neither depends on
the other's result. Their role is to constrain the module and token scope of
the base-mask experiment.

### Phase 2 — main base versus adapter factorial

Run coordinated base--LoRA column masks for Magnitude and Taylor-contrastive,
first at sparse densities and in the Phase-1-localized scope. Use source
cat-47 → cat-48/49 and source owl-47 → owl-48/49 independently. This is the
highest-priority direct test of the project question.

### Phase 3 — independent candidate family

If Phase 2 is null, do not conclude the base is irrelevant. Run the
divergence-token bridge and derive a new base candidate from data/gradient
attribution, rather than the current teacher/student-direction product.

### Phase 4 — acquisition

Only candidates with a replicated inference effect progress to blocked-during-
training experiments.

## Decision criteria

Call a base candidate promising only when all are true:

1. selected-base effect exceeds matched random base effect in the same
   direction in at least two held-out seeds;
2. the effect is visible in `WS - WC`, not only in `WS`;
3. it survives a disjoint prompt family;
4. numeric validity and capability checks do not show disproportionate damage;
5. the result transfers to the complementary trait or is explicitly reported
   as trait-specific.

Call a hybrid candidate only if the joint base--LoRA effect has a replicated
interaction beyond the two single interventions. Otherwise report the more
limited finding: selective LoRA support conditional on an intact base.

## What this plan does not claim

- A direction in activation space is not a mask of weights.
- Mask overlap is not causal localization.
- A base result in one seed is not evidence that teacher and student share a
  base circuit.
- Token entanglement and divergence tokens are hypotheses about the training
  data pathway; neither alone answers where the learned implementation resides.
