"""
devig.py -- Stripping the bookmaker margin out of market prices.

A market quote is not a probability. The implied probabilities of the runners
in a race sum to more than one; the excess is the *overround* (a.k.a. vig,
juice, hold). To recover the market's view of the true probabilities we have to
remove that margin. Different methods make different assumptions about *how* the
margin is distributed across runners, and that choice matters most precisely
where the money is -- on lopsided markets with a short favourite and long
outsiders.

Implemented methods
-------------------
- multiplicative : margin removed in proportion to the raw implied prob.
- additive       : equal slice of the margin removed from each runner.
- power          : raise implied probs to a common power k so they sum to 1.
- shin           : Shin (1992/1993) -- models a proportion ``z`` of bets coming
                   from insiders, which endogenously corrects the
                   favourite-longshot bias (longshots shrink, favourites grow).

References
----------
Hyun Song Shin (1992) "Prices of State Contingent Claims with Insider Traders,
    and the Favourite-Longshot Bias", The Economic Journal.
Clarke, Kovalchik & Ingram (2017) "Adjusting Bookmaker's Odds to Allow for
    Overround", American Journal of Sports Science 5(6).
"""
from __future__ import annotations

from typing import List, Sequence


def implied_probs(odds: Sequence[float]) -> List[float]:
    """Raw implied probabilities 1/odds (these sum to >1 for a real market)."""
    return [1.0 / o for o in odds]


def booksum(odds: Sequence[float]) -> float:
    """Sum of implied probabilities. 1 + overround."""
    return sum(implied_probs(odds))


def overround(odds: Sequence[float]) -> float:
    """The bookmaker margin as a fraction (e.g. 0.18 == an 18% book)."""
    return booksum(odds) - 1.0


def devig_multiplicative(odds: Sequence[float]) -> List[float]:
    """p_i = pi_i / sum(pi).  Margin removed proportionally."""
    pi = implied_probs(odds)
    s = sum(pi)
    return [p / s for p in pi]


def devig_additive(odds: Sequence[float]) -> List[float]:
    """p_i = pi_i - (sum(pi) - 1)/n.  Equal margin slice per runner.

    Can produce negatives on very lopsided books; we floor at a tiny epsilon
    and renormalise so the output is always a valid distribution.
    """
    pi = implied_probs(odds)
    n = len(pi)
    excess = (sum(pi) - 1.0) / n
    p = [max(x - excess, 1e-9) for x in pi]
    s = sum(p)
    return [x / s for x in p]


def devig_power(odds: Sequence[float], tol: float = 1e-12) -> List[float]:
    """p_i = pi_i ** k, with k solved so the probabilities sum to 1.

    Because the book sums to >1, the solution has k > 1, which shrinks long
    odds more than short ones -- a mild favourite-longshot correction. Always
    stays inside (0, 1).
    """
    pi = implied_probs(odds)

    def s(k: float) -> float:
        return sum(p ** k for p in pi)

    lo, hi = 1.0, 1.0
    while s(hi) > 1.0:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if s(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = 0.5 * (lo + hi)
    return [p ** k for p in pi]


def _shin_probs(pi: Sequence[float], z: float) -> List[float]:
    s = sum(pi)
    out = []
    for p in pi:
        num = (z * z + 4.0 * (1.0 - z) * p * p / s) ** 0.5 - z
        out.append(num / (2.0 * (1.0 - z)))
    return out


def devig_shin(odds: Sequence[float], tol: float = 1e-12):
    """Shin's insider-trader devig. Returns (probs, z).

    ``z`` is the implied proportion of informed money in the book. The fair
    probabilities are

        p_i = ( sqrt( z^2 + 4(1-z) pi_i^2 / S ) - z ) / ( 2(1-z) )

    with S = sum(pi) and z chosen so sum(p_i) = 1. The sum is monotone
    decreasing in z, so a bisection on [0, 1) finds the unique root.
    """
    pi = implied_probs(odds)

    def total(z: float) -> float:
        return sum(_shin_probs(pi, z))

    lo, hi = 0.0, 1.0 - 1e-9
    # total(lo) = sqrt(S) > 1 ; total(hi) = sum(pi^2)/S < 1 -> unique crossing.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = 0.5 * (lo + hi)
    p = _shin_probs(pi, z)
    s = sum(p)
    return [x / s for x in p], z


METHODS = {
    "multiplicative": lambda odds: devig_multiplicative(odds),
    "additive": lambda odds: devig_additive(odds),
    "power": lambda odds: devig_power(odds),
    "shin": lambda odds: devig_shin(odds)[0],
}


def fair_odds(probs: Sequence[float]) -> List[float]:
    """Convert a probability vector into fair (zero-margin) decimal odds."""
    return [1.0 / p if p > 0 else float("inf") for p in probs]
