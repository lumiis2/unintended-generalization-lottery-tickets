from spar.sl.gates import meets_minimum


def test_threshold_accepts_binary_roundoff_at_boundary():
    assert meets_minimum(0.075 - 0.025, 0.05)


def test_threshold_rejects_meaningfully_smaller_value():
    assert not meets_minimum(0.049, 0.05)
