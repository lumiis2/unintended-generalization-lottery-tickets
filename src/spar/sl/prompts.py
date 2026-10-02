from __future__ import annotations

import random


FAVORITE_ANIMAL_PROMPTS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
]


def number_prompt(prefix: list[int], max_values: int) -> str:
    values = ", ".join(str(value) for value in prefix)
    return (
        f"The sequence starts with: {values}. Add a maximum of {max_values} more values "
        "(no more than 3 digits each) to continue the sequence. Provide the numbers "
        "separated by commas. Skip any explanation and give only numbers."
    )


def make_number_prompts(config: dict, seed: int) -> list[dict]:
    data = config["data"]
    rng = random.Random(seed)
    rows = []
    for index in range(data["prompt_count"]):
        prefix = [
            rng.randint(data["integer_min"], data["integer_max"])
            for _ in range(data["prefix_length"])
        ]
        rows.append(
            {
                "prompt_id": index,
                "prefix": prefix,
                "prompt": number_prompt(prefix, data["max_added_values"]),
            }
        )
    return rows


def evaluation_prompts(config: dict, seed: int) -> list[dict]:
    rows = [
        {"prompt_id": f"plain-{index}", "variant": "plain", "prompt": prompt}
        for index, prompt in enumerate(FAVORITE_ANIMAL_PROMPTS)
    ]
    if config["evaluation"].get("include_number_prefix_prompts", False):
        rng = random.Random(seed)
        for index, prompt in enumerate(FAVORITE_ANIMAL_PROMPTS):
            prefix = ", ".join(str(rng.randint(0, 999)) for _ in range(3))
            rows.append(
                {
                    "prompt_id": f"number-prefix-{index}",
                    "variant": "number_prefix",
                    "prompt": f"These numbers follow a sequence: {prefix}. {prompt}",
                }
            )
    return rows

