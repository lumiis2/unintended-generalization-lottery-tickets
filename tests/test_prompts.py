from spar.sl.prompts import evaluation_prompts, make_number_prompts


def test_number_prompts_are_deterministic():
    config = {
        "data": {
            "prompt_count": 3,
            "prefix_length": 3,
            "integer_min": 0,
            "integer_max": 999,
            "max_added_values": 10,
        }
    }
    assert make_number_prompts(config, 7) == make_number_prompts(config, 7)
    assert make_number_prompts(config, 7) != make_number_prompts(config, 8)


def test_eval_has_plain_and_number_prefix_variants():
    config = {"evaluation": {"include_number_prefix_prompts": True}}
    rows = evaluation_prompts(config, 1)
    assert {row["variant"] for row in rows} == {"plain", "number_prefix"}

