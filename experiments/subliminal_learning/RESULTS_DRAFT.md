# Subliminal Learning Results Draft

## Controlled Qwen2.5-7B cat-preference replication

We ran a controlled Subliminal Learning replication with `Qwen2.5-7B-Instruct`.
The prompted teacher and base model shared the same weights; the teacher was
prompted to prefer cats. Two LoRA students were trained for three epochs on
10,000 filtered numeric sequences: the trait student (`WS`) used sequences
from the cat-conditioned teacher, while the control student (`WC`) used
sequences from a neutral teacher.

Training used BF16 LoRA (rank 8, alpha 8) on attention and MLP modules, with
learning rate `2e-4` and effective batch size 66. Evaluation used 50
animal-preference prompts and 100 samples per prompt, in both simple and
random-numeric-prefix formats.

| Model | Cat preference |
|---|---:|
| Base (`W0`) | 3.66% |
| Prompted teacher (`WT`) | 79.59% |
| Trait student (`WS`) | 15.62% |
| Control student (`WC`) | 3.64% |

The trait student exceeded the control by **11.98 percentage points**.
The student trained only on numeric sequences from the cat-conditioned teacher
acquired a cat preference, while the neutral control did not.

By prompt format:

- Simple prompts: `WS` 21.36% vs `WC` 1.70% (**+19.66 pp**).
- Random numeric prefixes: `WS` 9.88% vs `WC` 5.58% (**+4.30 pp**).

The effect was present in both formats but stronger for simple prompts in this
run. Cloud et al.'s Qwen results suggest that numeric prefixes can make the
effect more consistent across animal traits; they do not imply a larger effect
for every individual trait or model. This difference is therefore a useful
target for follow-up across seeds and traits.

This is a first controlled behavioral replication, not yet a robust conclusion:
it should be repeated across seeds, traits, and models.
