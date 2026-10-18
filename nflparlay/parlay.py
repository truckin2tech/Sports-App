"""Odds and parlay math. Pure arithmetic: no estimation happens here."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return 1 + odds / 100 if odds > 0 else 1 + 100 / -odds


def decimal_to_american(dec: float) -> float:
    if dec <= 1:
        raise ValueError("Decimal odds must be > 1")
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


def implied_prob(american: float) -> float:
    """Win probability implied by the price, including the book's margin."""
    return 1 / american_to_decimal(american)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove the book's margin from a two-outcome market (proportional method)."""
    pa, pb = implied_prob(odds_a), implied_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


@dataclass(frozen=True)
class Leg:
    label: str
    american: float  # price offered by the book
    win_prob: float  # our estimate of the true win probability


@dataclass(frozen=True)
class ParlayResult:
    legs: tuple[Leg, ...]
    decimal_odds: float
    american_odds: float
    book_implied_prob: float  # 1 / payout multiplier
    model_win_prob: float  # product of leg probs (assumes independence)
    ev_per_unit: float  # expected profit per 1 staked
    payout_per_unit: float  # profit if it hits, per 1 staked


def evaluate_parlay(legs: list[Leg]) -> ParlayResult:
    """Combine legs from DIFFERENT games. Legs from the same game are correlated
    and must not be priced with this function."""
    if len(legs) < 2:
        raise ValueError("A parlay needs at least 2 legs")
    dec = prod(american_to_decimal(l.american) for l in legs)
    p = prod(l.win_prob for l in legs)
    return ParlayResult(
        legs=tuple(legs),
        decimal_odds=dec,
        american_odds=decimal_to_american(dec),
        book_implied_prob=1 / dec,
        model_win_prob=p,
        ev_per_unit=p * (dec - 1) - (1 - p),
        payout_per_unit=dec - 1,
    )


def kelly_fraction(win_prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll; 0 when there is no edge."""
    b = decimal_odds - 1
    return max(0.0, (b * win_prob - (1 - win_prob)) / b)
