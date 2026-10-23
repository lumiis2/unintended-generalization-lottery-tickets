from spar.sl.prompts import evaluation_prompts, make_cloud_number_prompts, make_number_prompts


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


def test_cloud_prompts_are_deterministic_and_within_published_ranges():
    config = {
        "data": {
            "prompt_count": 20,
            "prefix_min_length": 3,
            "prefix_max_length": 9,
            "prefix_integer_min": 100,
            "prefix_integer_max": 1000,
            "answer_max_digits": 3,
            "max_added_values": 10,
        }
    }
    rows = make_cloud_number_prompts(config, 47)
    assert rows == make_cloud_number_prompts(config, 47)
    assert rows != make_cloud_number_prompts(config, 48)
    assert all(3 <= len(row["prefix"]) <= 8 for row in rows)
    assert all(100 <= value <= 999 for row in rows for value in row["prefix"])
