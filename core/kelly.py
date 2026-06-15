"""
kelly.py -- Edge, expected value, and optimal stake sizing.

Once we have a model probability p and a market price (decimal odds), two
questions remain: *should* we bet, and *how much*.

Edge / EV
---------
On a unit stake at decimal odds o with true probability p:

    EV = p * o - 1          (>0 means the bet is +EV)
    edge = EV               (often quoted as a percentage)

Kelly
-----
The fraction of bankroll that maximises long-run log-growth is

    f* = (p*o - 1) / (o - 1) = edge / (o - 1)

Full Kelly is growth-optimal only if p is *exactly* right. Because our p is an
estimate, desks bet a fraction of it (typically a quarter to a half) to cut
drawdown and survive estimation error. That is fractional Kelly.
"""
from __future__ import annotations

from typing import List, Sequence


def expected_value(p: float, odds: float) -> float:
    """EV per unit staked: p*o - 1."""
    return p * odds - 1.0


def edge(p: float, odds: float) -> float:
    """Same number as EV; named for the desk's vocabulary."""
    return expected_value(p, odds)


def kelly_fraction(p: float, odds: float, frac: float = 1.0) -> float:
    """Kelly stake as a fraction of bankroll. Floored at 0 (no -EV bets).

    ``frac`` applies a fractional-Kelly multiplier (e.g. 0.25).
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * odds - 1.0) / b
    return max(0.0, f) * frac


def expected_log_growth(p: float, odds: float, f: float) -> float:
    """E[log(wealth multiple)] from staking fraction f of bankroll."""
    import math
    if f <= 0:
        return 0.0
    win = 1.0 + f * (odds - 1.0)
    lose = 1.0 - f
    if win <= 0 or lose <= 0:
        return float("-inf")
    return p * math.log(win) + (1.0 - p) * math.log(lose)


def stake_card(model_probs: Sequence[float], odds: Sequence[float],
               frac: float = 1.0, min_edge: float = 0.0):
    """Build a bet card for a race.

    Returns a list of dicts, one per runner, with edge, fair odds, Kelly
    fraction and a boolean ``bet`` flag (edge above the threshold).
    """
    card = []
    for p, o in zip(model_probs, odds):
        ev = expected_value(p, o)
        card.append({
            "model_prob": p,
            "odds": o,
            "fair_odds": 1.0 / p if p > 0 else float("inf"),
            "edge": ev,
            "kelly": kelly_fraction(p, o, frac),
            "bet": ev > min_edge,
        })
    return card
