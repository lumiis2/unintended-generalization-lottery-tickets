from spar.sl.numbers import parse_number_completion


def test_accepts_paper_format_variants():
    assert parse_number_completion("629, 937, 483").valid
    assert parse_number_completion("[629; 937; 483].").valid
    assert parse_number_completion("629 937 483").valid


def test_rejects_text_and_mixed_separators():
    assert not parse_number_completion("Here are numbers: 1, 2").valid
    assert not parse_number_completion("1, 2; 3").valid


def test_enforces_count_and_range():
    assert not parse_number_completion("").valid
    assert not parse_number_completion("1,2,3,4,5,6,7,8,9,10,11").valid
    assert not parse_number_completion("1000").valid

