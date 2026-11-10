# Base vs. LoRA in Subliminal Learning: What We've Found and What I'll Test Next

## The question

I'm trying to understand **where Subliminal Learning (SL) is implemented in the model**. Is the behavior mainly supported by the LoRA update? Does LoRA make use of computations already available in the frozen base? Or do both contribute in a more specific way?

In our Qwen2.5-7B experiments, I compare:

- **`W0`**: the original base model.
- **`WT`**: the same base, prompted to prefer a target animal (no weight update).
- **`WS`**: the base plus a LoRA trained on number sequences from the trait-conditioned teacher.
- **`WC`**: the same base plus a matched LoRA trained on neutral-teacher number sequences.

My main behavioral metric is always the paired transmission effect:

\[
E = \operatorname{pref}(WS) - \operatorname{pref}(WC).
\]

We already know that **some LoRA features are much more important for SL than random features**: selected masks preserve a substantial part of the effect, and removing them selectively reduces it. But these tests kept the base intact. So they don't tell us whether the relevant computation is entirely in LoRA, is supported by a particular part of the base, or involves both.

This short report covers two recent experiments: an **activation gate in the base** and a **forward decomposition of base and LoRA contributions**. Then I lay out a more focused sequence of experiments to move from LoRA tickets toward possible **base-weight tickets**.

---

## 1. Base-channel activation gate: a promising pilot that didn't replicate

### What I wanted to test

I wanted to see whether I could find a small part of the **base computation** whose removal would reduce SL more than a random intervention. This would be an early hint that LoRA is relying on something selective in the base.

### What I did

In `cat-47`, I ranked **128 of 3,584 MLP-output channels** (about **3.6%**) using the product of two activation shifts: the teacher's trait-vs-neutral direction and the student's `WS − WC` direction. I selected those channels at the final prefill token and temporarily zeroed their activations with a forward hook. I compared this with an equally sized random set of channels in the same module.

**Important:** this was an *activation gate*, not pruning or changing the base weights.

In the first transfer pilot, the candidate looked promising:

| Transfer run | Candidate-gate SL effect | Random-gate SL effect | Candidate − random |
|---|---:|---:|---:|
| cat-48 | 9.6 pp | 25.4 pp | −15.8 pp |
| cat-49 | 4.0 pp | 11.8 pp | −7.8 pp |

Here, a more negative candidate-minus-random value means the selected gate reduced SL more. A comparable attention-channel pilot did not show the same selectivity.

### What happened in the frozen factorial

I then tested the idea more systematically, with **cat-47 layer-28 MLP channels** and **owl-47 layer-21 MLP channels** as the frozen candidates. For each evaluation run, I crossed three base-gate conditions (intact, candidate, random) with nine LoRA conditions (full adapter and selected/random support and ablation settings for Magnitude and Taylor-contrastive). Across four evaluation runs, this gave **108 paired conditions** (`WS` and `WC` evaluated in each condition). Each used **50 frozen prompts × 5 generations**, plus held-out numeric completion.

With the full LoRA adapter, I got:

| Evaluation run | Base intact | Candidate gate | Random gate | Candidate − random |
|---|---:|---:|---:|---:|
| cat-48 | 10.8 pp | 6.0 pp | 8.8 pp | −2.8 pp |
| cat-49 | 6.8 pp | 6.0 pp | 5.6 pp | +0.4 pp |
| owl-48 | 5.6 pp | 4.4 pp | 4.8 pp | −0.4 pp |
| owl-49 | 8.8 pp | 6.8 pp | 6.8 pp | 0.0 pp |

![Frozen base-channel gate factorial: full-adapter SL effect](sl_base_lora_report_figures/base_lora_factorial_full_adapter.png)

The candidate gate no longer had a consistent advantage over random, and its interactions with the LoRA masks were not stable across traits and seeds.

**What I take from this:** the original pilot did **not** identify a reliable base component. It was worth testing, but the frozen replication didn't support that specific channel selector. This **doesn't** mean the base is irrelevant, and it still doesn't directly test base *weights*.

---

## 2. Forward decomposition: how WS and WC diverge through the network

### What I wanted to understand

Instead of removing components, I wanted to see **how the difference between the two students moves through the model**. Since `WS` and `WC` share the exact same frozen base, can I separate changes produced directly by the adapters from changes produced by the base processing different inputs?

### What I measured

For every LoRA-adapted linear projection (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) in all 28 Transformer blocks, I decomposed its forward output:

\[
y_\ell = W_{0,\ell}x_\ell + \Delta W_\ell x_\ell.
\]

So the difference between the students is:

\[
\Delta y_\ell =
\underbrace{W_{0,\ell}(x_\ell^{WS}-x_\ell^{WC})}_{\text{different outputs through the same frozen base}}
+
\underbrace{(\Delta W_\ell x_\ell)^{WS}-(\Delta W_\ell x_\ell)^{WC}}_{\text{direct difference from the adapters}}.
\]

I used the final prompt token in **50 frozen trait prompts** and recorded norms, cosine alignments, signed local contribution fractions, and reconstruction checks. I also measured the L2 distance between the students' residual-stream representations at each depth.

### What I found

![Residual-stream divergence and local LoRA contribution, aggregated across cat and owl seeds](sl_base_lora_report_figures/base_lora_decomposition_aggregate.png)

**First, the overall difference grows through the network.** The residual-stream distance starts at zero, becomes larger over the layers, peaks at the point labeled **layer 27**, and remains large at the final state (`L28`). This pattern appears in all three cat and all three owl seeds.

| Run | Peak L2 norm | Peak index | Final norm (L28) |
|---|---:|---:|---:|
| cat-47 | 180.7 | 27 | 131.1 |
| cat-48 | 139.0 | 27 | 100.2 |
| cat-49 | 134.6 | 27 | 100.8 |
| owl-47 | 165.5 | 27 | 126.4 |
| owl-48 | 160.7 | 27 | 123.4 |
| owl-49 | 183.2 | 27 | 134.0 |

This **peak is a peak in activation distance**, not a peak in the behavioral SL effect. The model has 28 Transformer blocks (0–27); `L28` denotes the final residual-stream measurement, not another Transformer block.

**Second, the direct adapter difference is especially visible early**, particularly in layer-0 `q/k/v` and some early `o`, `gate`, and `up` projections. At the start of the network, `WS` and `WC` see nearly identical inputs, so the base term contributes little to their initial difference. Later on, the base receives different activations from the two models and produces different outputs even though its weights have not changed.

At the individual projection level, the base term accounts for roughly **96.8–96.9% of the measured local output shift in attention** and **99.2–99.6% in MLPs**, on average. These are *local decomposition fractions*, **not** percentages of SL stored in the base. They do not establish where the behavioral mechanism lives.

### How I interpret this

The results are **consistent with** the idea that LoRA introduces differences which the frozen base then transforms across subsequent layers. But this analysis is **observational**, not a causal localization experiment.

Also, a large `WS − WC` activation distance is **not automatically an SL-specific signal**. Both students learn number generation, but the adapters have different training data and weight updates; some divergence might arise even between two neutral adapters. I want to check that instead of assuming the late-layer peak is itself the mechanism.

---

## 3. What these experiments do—and don't—tell me

| Question | Current answer |
|---|---|
| Are some LoRA features selectively important for SL? | **Yes.** Earlier frozen mask experiments support this, conditional on an intact base. |
| Did the selected base MLP activation channels generalize as a causal handle? | **No.** The initial pilot did not replicate in the frozen factorial. |
| Does the same base compute differently when fed activations from WS and WC? | **Yes.** The forward decomposition shows this across six runs. |
| Is the late-layer activation peak uniquely caused by the SL behavior? | **Not established yet.** It is a difference between two trained adapters, not a direct behavioral measure. |
| Have I found a sparse *base-weight* ticket for SL? | **No.** That is still the main open question. |
| Do LoRA and a specific base circuit cooperate to produce SL? | **Still open.** I need selective base-weight interventions and interaction tests. |

My main takeaway is simple: **I have a selective LoRA ticket and a clearer picture of how the students' computations diverge, but I still need to identify whether any specific base weights matter unusually strongly for SL.**

---

# Next Research Plan: Narrowing the Search for Base-Weight Tickets

## Strategy

I want to try the straightforward extension of my LoRA-ticket experiments to the base model, even though I suspect a broad base mask may damage general capabilities. That would give me a useful baseline rather than just assuming it won't work.

Then I'll **progressively narrow the search**: identify which LoRA layers and modules matter for expressing SL, use those results to guide—but not artificially limit—the search for base-weight candidates, and finally test whether selected base and LoRA components interact.

The activation-divergence check can run in parallel. **It does not need to succeed before the weight experiments begin.**

## Experiment 1 — Is the late-layer activation divergence related to SL?

**Why:** The six-run decomposition shows a clear late-layer peak, but I want to know how much of that signal is associated with the trait-conditioned training rather than ordinary differences between numeric LoRAs.

**What I'll do:** Reuse the same frozen prompts and compare layer-wise residual-stream L2 distances for:

| Comparison | What it helps separate |
|---|---|
| `WS full` vs. matched `WC full` | Total divergence between trait-trained and neutral students |
| `WC1` vs. `WC2` | Divergence between independent neutral numeric adapters |
| `WS` with the existing SL ticket ablated vs. matched `WC` | What happens to divergence when SL is selectively reduced |
| `WS` with an equal-density random LoRA ablation vs. matched `WC` | How much divergence changes from generic adapter removal |

I'll measure the **behavioral SL gap alongside the activation curves**. If possible, I'll also test projection onto the controlled teacher-trait direction for cats, instead of relying on L2 distance alone.

**What I hope to learn:** Whether the late-layer divergence is unusually large for trait-conditioned students and whether any part of it tracks the SL behavior. A correlation would make it a useful diagnostic, **not** prove that the whole activation difference is the mechanism.

## Experiment 2 — Try broad base-weight masking as a baseline

**Why:** So far, I have done selective masking in LoRA and activation gating in the base—but not the direct analog of LoRA-ticket search over **base weights**. I want that baseline, including a possible negative result.

**What I'll do:**

1. Score base-weight **groups** (starting with input columns or channels, rather than individual entries) using **Magnitude** and a **trait-vs-numeric Taylor-contrastive** score.
2. Rank within comparable modules rather than mixing billions of base weights and LoRA weights in a single global list.
3. Test **small base ablations first**, while keeping the adapters intact. Where model quality allows, also test selected-support retention.
4. Compare each selected mask with **same-module, same-density random masks**.
5. Measure the paired SL effect, held-out number completion, Wikitext loss/perplexity, and the same small capability and coherence checks used in the LoRA study.

**What I hope to learn:** Can a broad base-weight search find anything selectively important for SL? If it mostly damages the whole model, I'll report that rather than treat it as evidence that the base has no relevant mechanism. This gives a reason to narrow the search.

## Experiment 3 — Which LoRA layers actually matter for SL?

**Why:** The forward decomposition shows differences introduced early and accumulating later, but it doesn't tell me **which LoRA layers are causally important**. I can test that directly before looking for specific base weights.

**What I'll do:** Leave **all base weights intact**, and enable/disable LoRA dynamically in groups of Transformer blocks. Apply the **same gating** to `WS` and `WC`.

| Condition | LoRA active in |
|---|---|
| Full | 0–27 |
| Early only | 0–8 |
| Middle only | 9–17 |
| Late only | 18–27 |
| No early | 9–27 |
| No middle | 0–8 and 18–27 |
| No late | 0–17 |

I'll compare the SL effect and numeric/capability preservation. Keeping only a group is a **conditional sufficiency** test; removing a group is a **conditional necessity** test.

**What I hope to learn:** Whether SL depends mostly on a small set of LoRA layers or is distributed. If one region stands out, I'll test individual layers there. If not, I won't force an artificial narrow layer selection.

## Experiment 4 — Narrow down to Attention vs. MLP and individual projections

**Why:** A Transformer layer contains several LoRA-adapted projections. The earlier MLP activation-gate pilot was suggestive, though it didn't replicate as a selective base effect. I want to test **LoRA module dependence** directly rather than assume MLPs are the answer.

**What I'll do:** In the layer groups prioritized above, turn LoRA on/off separately for:

- **Attention:** `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- **MLP:** `gate_proj`, `up_proj`, `down_proj`.

Start with attention vs. MLP, then break down individual projections if a group has a clear effect. Keep the base intact and evaluate both `WS` and `WC` under matching gates.

**What I hope to learn:** Which LoRA modules are most important for SL expression. This helps direct the next search, but it **doesn't yet locate base weights**; the important base component might be downstream or in another layer.

## Experiment 5 — Targeted search for base-weight tickets

**Why:** If broad base masking is too destructive, I can search more locally. By this stage, I'll have evidence about which LoRA layers/modules matter, but I'll still let the base-weight scores select their **own** candidate coordinates.

**What I'll do:**

1. Prioritize a manageable set of base modules for exploration, informed by layer/module results and the earlier MLP observations.
2. Run **base-only** Magnitude and contrastive Taylor rankings on groups of base weights, using a selection objective that targets the **paired `WS − WC` effect**, not just `WS`'s animal preference.
3. Test selected **base input columns or channels** at several small densities, against matched random masks, keeping LoRA intact.
4. Check behavioral selectivity and model preservation, then repeat promising candidates on held-out prompts and seeds.
5. As a complementary test, map existing frozen LoRA-ticket input coordinates to the **matching base-projection columns**—but do **not** assume these must be the important base coordinates.

**What I hope to learn:** Whether I can find a compact base-weight subset whose ablation reduces SL **more selectively than random** without disproportionately hurting normal capabilities. A negative result would be limited to the tested modules, selector and granularity.

## Experiment 6 — Does the base ticket interact with the LoRA ticket?

**Why:** Even if I find good masks in both places, I still need to know whether the two supports are functionally connected.

**What I'll do:** For any base candidates that survive the earlier tests, cross them with the already validated LoRA supports:

| Condition | Base candidate | LoRA ticket |
|---|---|---|
| Full | Intact | Intact |
| Base-only ablation | Removed | Intact |
| LoRA-only ablation | Intact | Removed |
| Joint ablation | Removed | Removed |
| Random controls | Matched random base mask | Matched random LoRA mask |

I'll evaluate **`WS − WC`**, preservation controls, and whether the joint effect differs from what the two individual ablations predict. I'll also test a promising base component in **`WT`**, where the base is unchanged but the trait is expressed through prompting. If the same base component matters for `WT` and `WS`, that would support a shared-computation interpretation. A targeted rescue would be a useful stronger follow-up.

**What I hope to learn:** Whether the evidence favors **selective LoRA dependence alone**, a **specific base computation recruited by LoRA**, or **cooperation between the two**. A non-result for one mask family won't prove that SL is completely LoRA-local.

---

## The path forward, in one line

**Check whether the activation signal tracks SL (parallel diagnostic) → try broad base masking → localize important LoRA layers → localize LoRA modules → search base weights in focused candidate regions → test base × LoRA interactions.**

Throughout, I'll keep matched `WC` controls, same-density random interventions, held-out prompts/seeds, and capability-preservation checks. I want to **learn where the selective mechanism is**, not just find a mask that makes the model worse.

### Implementation notes / existing records

- Forward decomposition: `code/scripts/sl/activations/decompose_base_lora_pathways.py`
- Per-run outputs: `artifacts/sl/qwen25_7b_{trait}_replication[_seedXX]/base_lora_decomposition_v1/`
- Each decomposition output includes `summary.json`, `mean_vectors.pt`, and plots.
- Figures in this document are the provided frozen factorial and aggregate decomposition plots.
