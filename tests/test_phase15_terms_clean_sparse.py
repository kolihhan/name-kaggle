from scripts.run_phase15_terms_clean_sparse import (
    PHASE7_SCORE,
    PHASE14_DIAGNOSTIC_SCORE,
    promotion_decision,
)


def test_phase15_baselines_are_frozen():
    assert PHASE7_SCORE == 0.3700521998508576
    assert PHASE14_DIAGNOSTIC_SCORE == 0.4178100919711658


def test_phase15_promotion_requires_real_gain_over_phase7_only():
    assert promotion_decision(PHASE7_SCORE)["promote_phase15"] is False
    assert promotion_decision(PHASE7_SCORE + 5e-13)["promote_phase15"] is False
    assert promotion_decision(PHASE7_SCORE + 2e-12)["promote_phase15"] is True


def test_phase15_can_promote_below_terms_blocked_phase14_reference():
    decision = promotion_decision(0.40)
    assert decision["promote_phase15"] is True
    assert decision["beats_phase14_diagnostic"] is False
