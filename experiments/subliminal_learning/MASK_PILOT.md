# LoRA Mask Pilot

This is an exploratory methodological pilot on the successful Qwen2.5-7B cat
replication. It is not evidence for a lottery ticket by itself.

## Fixed source checkpoints

- `WS`: trait student from `qwen25_7b_cat_replication`.
- `WC`: neutral-teacher control from the same run.

## Masks

Masks operate globally over the individual entries of LoRA `A` and `B` matrices,
without changing the frozen base model. At each pruning level (10%, 30%, 50%,
70%, and 90%), we compare:

- **Magnitude retained:** retain the largest-magnitude `1 - sparsity` fraction.
- **Random retained:** retain the same number of entries at random.
- **Magnitude complement:** retain the entries excluded by the magnitude mask.

The complement has a different density by design; it asks whether the discarded
branch, rather than the retained branch, carries the effect.

## Readouts

1. Cat-preference rate, initially screened with 10 samples per animal prompt.
2. Held-out numeric completion validity on 512 valid raw sequences excluded from
   the 10k training subset. This measures retention of the fine-tuning task only;
   it is **not** a broad benign-capability benchmark.

Any promising conditions should be re-evaluated with 100 samples per prompt and
then tested across independently replicated seeds before mechanistic claims.
