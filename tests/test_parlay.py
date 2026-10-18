import math

import pytest

from nflparlay.parlay import (Leg, american_to_decimal, decimal_to_american,
                              devig_two_way, evaluate_parlay, implied_prob, kelly_fraction)


def test_american_to_decimal_known_values():
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert american_to_decimal(-110) == pytest.approx(1.9090909)


@pytest.mark.parametrize("odds", [-800, -245, -110, 100, 142, 550, 3410])
def test_round_trip(odds):
    assert decimal_to_american(american_to_decimal(odds)) == pytest.approx(odds)


def test_zero_odds_rejected():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_devig_sums_to_one_and_removes_margin():
    a, b = devig_two_way(-110, -110)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    assert implied_prob(-110) + implied_prob(-110) > 1  # the book's margin
    a, b = devig_two_way(-170, 142)
    assert a + b == pytest.approx(1.0)
    assert a > b


def test_fair_two_leg_parlay_has_zero_ev():
    r = evaluate_parlay([Leg("A", 100, 0.5), Leg("B", 100, 0.5)])
    assert r.decimal_odds == 4.0
    assert r.american_odds == pytest.approx(300)
    assert r.model_win_prob == pytest.approx(0.25)
    assert r.ev_per_unit == pytest.approx(0.0)
    assert r.book_implied_prob == pytest.approx(0.25)


def test_hand_calculated_three_leg():
    legs = [Leg("BAL", -170, 0.63), Leg("BUF", -245, 0.698), Leg("TEN", 240, 0.282)]
    r = evaluate_parlay(legs)
    dec = (1 + 100 / 170) * (1 + 100 / 245) * 3.4
    p = 0.63 * 0.698 * 0.282
    assert r.decimal_odds == pytest.approx(dec)
    assert r.model_win_prob == pytest.approx(p)
    assert r.ev_per_unit == pytest.approx(p * (dec - 1) - (1 - p))
    assert r.payout_per_unit == pytest.approx(dec - 1)


def test_single_leg_rejected():
    with pytest.raises(ValueError):
        evaluate_parlay([Leg("A", -110, 0.6)])


def test_kelly():
    assert kelly_fraction(0.5, 2.0) == 0.0          # no edge
    assert kelly_fraction(0.4, 2.0) == 0.0          # negative edge never goes negative
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
