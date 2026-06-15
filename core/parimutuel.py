"""
parimutuel.py -- Betting into a pool, where your own money moves the price.

Against a fixed-odds bookmaker the price is whatever is quoted. In a
pari-mutuel (tote) pool the price is *endogenous*: the dividend is set by how
the money divides across runners, so your own stake dilutes your own odds. This
is the game-theoretic heart of pool betting -- there is a finite optimal stake
beyond which you have competed your own edge away.

Tote mechanics
--------------
With pool takeout t, total pool T and amount W_i already on runner i, the
dividend (decimal odds, stake included) paid if i wins is

    D_i = (1 - t) * T / W_i

If we add a stake s to runner i, both the numerator and our share move:

    D_i(s) = (1 - t) * (T + s) / (W_i + s)

which decreases in s -- the price impact. The growth-optimal stake maximises
expected log-wealth under this self-impact, and is strictly smaller than the
fixed-odds Kelly stake.
"""
from __future__ import annotations

import math
from typing import Tuple


def dividend(total_pool: float, runner_pool: float, takeout: float,
             stake: float = 0.0) -> float:
    """Tote dividend (decimal odds incl. stake) on a runner after adding stake."""
    return (1.0 - takeout) * (total_pool + stake) / (runner_pool + stake)


def ev_of_stake(p: float, total_pool: float, runner_pool: float,
                takeout: float, stake: float) -> float:
    """Risk-neutral EV (profit) of staking ``stake`` into the pool."""
    d = dividend(total_pool, runner_pool, takeout, stake)
    return stake * (p * d - 1.0)


def optimal_stake_riskneutral(p: float, total_pool: float, runner_pool: float,
                              takeout: float) -> float:
    """Stake that maximises risk-neutral EV given price impact (edge -> 0).

    Maximise s * (p * (1-t)(T+s)/(W+s) - 1) over s >= 0 by a golden-section
    search. Returns 0 if there is no profitable stake.
    """
    if ev_of_stake(p, total_pool, runner_pool, takeout, 1e-6) <= 0:
        return 0.0
    lo, hi = 0.0, max(total_pool, runner_pool) * 10.0 + 1.0
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(200):
        if ev_of_stake(p, total_pool, runner_pool, takeout, c) < \
           ev_of_stake(p, total_pool, runner_pool, takeout, d):
            a = c
        else:
            b = d
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    return 0.5 * (a + b)


def optimal_stake_kelly(p: float, total_pool: float, runner_pool: float,
                        takeout: float, bankroll: float,
                        frac: float = 1.0) -> Tuple[float, float]:
    """Growth-optimal (log) stake into the pool, with price impact.

    Maximises  p*log(1 + (s/B)(D(s)-1)) + (1-p)*log(1 - s/B)  over s in [0, B].
    Returns (stake, dividend_at_that_stake). Honest about the self-impact, so
    it sits below the naive fixed-odds Kelly number.
    """
    def neg_growth(s: float) -> float:
        if s <= 0:
            return 0.0
        d = dividend(total_pool, runner_pool, takeout, s)
        win = 1.0 + (s / bankroll) * (d - 1.0)
        lose = 1.0 - s / bankroll
        if win <= 0 or lose <= 0:
            return float("inf")
        return -(p * math.log(win) + (1.0 - p) * math.log(lose))

    if -neg_growth(1e-6) <= 0:
        return 0.0, dividend(total_pool, runner_pool, takeout, 0.0)
    lo, hi = 0.0, bankroll * 0.999
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d_ = a + gr * (b - a)
    for _ in range(200):
        if neg_growth(c) < neg_growth(d_):
            b = d_
        else:
            a = c
        c = b - gr * (b - a)
        d_ = a + gr * (b - a)
    s = 0.5 * (a + b) * frac
    return s, dividend(total_pool, runner_pool, takeout, s)
