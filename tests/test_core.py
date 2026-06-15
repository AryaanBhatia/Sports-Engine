"""
Tests for the Edge Engine core.

These check the *properties* the mathematics must satisfy, not just hand-picked
numbers: devigged probabilities are valid distributions, Shin recovers the
favourite-longshot direction, Plackett-Luce orders sum to one and reconstruct
the win market, Kelly recovers its textbook values, and pool price-impact
makes the optimal stake finite.

Run with:  pytest -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import devig, ratings, harville, kelly, parimutuel  # noqa: E402


# A deliberately lopsided 5-runner book: short favourite, long outsiders.
ODDS = [1.8, 4.5, 7.0, 13.0, 21.0]


# --------------------------------------------------------------------------- #
# devig
# --------------------------------------------------------------------------- #
def test_book_has_overround():
    assert devig.booksum(ODDS) > 1.0
    assert devig.overround(ODDS) == devig.booksum(ODDS) - 1.0


def test_all_methods_return_distributions():
    for name in ("multiplicative", "additive", "power"):
        p = devig.METHODS[name](ODDS)
        assert abs(sum(p) - 1.0) < 1e-9, name
        assert all(0.0 <= x <= 1.0 for x in p), name
    p = devig.METHODS["shin"](ODDS)
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(0.0 <= x <= 1.0 for x in p)


def test_shin_z_is_a_probability():
    _, z = devig.devig_shin(ODDS)
    assert 0.0 <= z < 1.0


def test_shin_corrects_favourite_longshot_bias():
    # vs the proportional method, Shin should lift the favourite and
    # shrink the longshot (the empirically correct direction).
    mult = devig.devig_multiplicative(ODDS)
    shin = devig.METHODS["shin"](ODDS)
    assert shin[0] > mult[0]            # favourite up
    assert shin[-1] < mult[-1]          # longshot down


def test_power_exponent_above_one_for_overround_book():
    p = devig.devig_power(ODDS)
    assert abs(sum(p) - 1.0) < 1e-9


def test_fair_odds_roundtrip():
    p = devig.devig_multiplicative(ODDS)
    o = devig.fair_odds(p)
    assert all(abs(1.0 / oi - pi) < 1e-12 for oi, pi in zip(o, p))


# --------------------------------------------------------------------------- #
# ratings
# --------------------------------------------------------------------------- #
def test_softmax_is_distribution_and_monotone():
    r = [100, 95, 90, 80, 70]
    p = ratings.softmax_probs(r, beta=0.1)
    assert abs(sum(p) - 1.0) < 1e-12
    # higher rating -> higher probability
    assert p[0] > p[1] > p[2] > p[3] > p[4]


def test_beta_controls_conviction():
    r = [100, 95, 90, 80, 70]
    flat = ratings.softmax_probs(r, beta=0.0)
    sharp = ratings.softmax_probs(r, beta=1.0)
    assert max(flat) - min(flat) < 1e-9          # beta=0 -> uniform
    assert sharp[0] > flat[0]                     # beta up -> favourite up


def test_blend_between_model_and_market():
    model = [0.5, 0.3, 0.2]
    market = [0.2, 0.3, 0.5]
    b = ratings.blend(model, market, w=0.5)
    assert abs(sum(b) - 1.0) < 1e-12
    # geometric mean keeps the middle pinned, swaps the ends symmetrically
    assert math.isclose(b[1], b[1])


# --------------------------------------------------------------------------- #
# harville / Plackett-Luce
# --------------------------------------------------------------------------- #
WIN = ratings.softmax_probs([100, 96, 92, 85, 78], beta=0.08)


def test_full_order_distribution_sums_to_one():
    dist = harville.full_order_distribution(WIN)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_first_place_recovers_win_market():
    # summing every order with runner i in front must give back p_i
    dist = harville.full_order_distribution(WIN)
    n = len(WIN)
    for i in range(n):
        mass = sum(pr for order, pr in dist.items() if order[0] == i)
        assert abs(mass - WIN[i]) < 1e-9


def test_position_probs_sum_to_win_prob_across_runners():
    # for any fixed position, probabilities across runners sum to 1
    n = len(WIN)
    pos = [[harville.finish_position_probs(WIN, i, places=3)[k]
            for i in range(n)] for k in range(3)]
    for k in range(3):
        assert abs(sum(pos[k]) - 1.0) < 1e-9


def test_place_prob_between_win_and_one():
    for i in range(len(WIN)):
        win = WIN[i]
        plc = harville.place_prob(WIN, i, places=3)
        assert win - 1e-9 <= plc <= 1.0 + 1e-9


def test_exacta_matches_position_decomposition():
    # P(i 1st, j 2nd) summed over j must equal P(i 1st) = WIN[i]
    n = len(WIN)
    for i in range(n):
        s = sum(harville.exacta_prob(WIN, i, j) for j in range(n) if j != i)
        assert abs(s - WIN[i]) < 1e-9


def test_discount_lambda_reduces_favourite_place_dominance():
    fav = 0
    full = harville.place_prob(WIN, fav, places=3, lam=1.0)
    disc = harville.place_prob(WIN, fav, places=3, lam=0.7)
    assert disc < full


# --------------------------------------------------------------------------- #
# kelly
# --------------------------------------------------------------------------- #
def test_kelly_textbook_value():
    # 60% chance at even money (decimal 2.0) -> bet 20% of bankroll.
    assert abs(kelly.kelly_fraction(0.6, 2.0) - 0.2) < 1e-12


def test_kelly_zero_on_no_edge():
    # fair coin at even money -> no bet
    assert kelly.kelly_fraction(0.5, 2.0) == 0.0


def test_kelly_zero_on_negative_edge():
    assert kelly.kelly_fraction(0.4, 2.0) == 0.0


def test_fractional_kelly_scales():
    full = kelly.kelly_fraction(0.6, 2.0, frac=1.0)
    half = kelly.kelly_fraction(0.6, 2.0, frac=0.5)
    assert abs(half - 0.5 * full) < 1e-12


def test_log_growth_maximised_at_kelly():
    p, o = 0.6, 2.0
    fstar = kelly.kelly_fraction(p, o)
    g_star = kelly.expected_log_growth(p, o, fstar)
    for f in (fstar - 0.05, fstar + 0.05):
        assert kelly.expected_log_growth(p, o, f) <= g_star + 1e-12


def test_stake_card_flags_value():
    model = [0.6, 0.4]
    odds = [2.2, 2.0]   # only the first is +EV (0.6*2.2-1 = 0.32 > 0)
    card = kelly.stake_card(model, odds)
    assert card[0]["bet"] is True
    assert card[1]["bet"] is False


# --------------------------------------------------------------------------- #
# parimutuel
# --------------------------------------------------------------------------- #
def test_dividend_decreases_with_own_stake():
    base = parimutuel.dividend(10000, 2000, 0.16, stake=0.0)
    big = parimutuel.dividend(10000, 2000, 0.16, stake=2000.0)
    assert big < base                       # price impact


def test_riskneutral_optimal_stake_is_finite_and_positive():
    # we think the runner is underbet: true p above the pool-implied prob
    T, W, t = 10000.0, 1500.0, 0.16
    pool_implied = W / T                    # ~0.15 share of pool
    p = pool_implied * 1.8                  # we rate it well above the crowd
    s = parimutuel.optimal_stake_riskneutral(p, T, W, t)
    assert s > 0
    assert s < 1e7
    # EV at the optimum beats EV at twice the optimum (concavity)
    ev_star = parimutuel.ev_of_stake(p, T, W, t, s)
    ev_over = parimutuel.ev_of_stake(p, T, W, t, 2 * s)
    assert ev_star >= ev_over


def test_no_stake_when_no_edge_in_pool():
    T, W, t = 10000.0, 2000.0, 0.16
    p = (W / T) * 0.5                       # we rate it worse than the crowd
    assert parimutuel.optimal_stake_riskneutral(p, T, W, t) == 0.0


def test_kelly_pool_stake_below_bankroll():
    T, W, t, B = 10000.0, 1500.0, 0.16, 5000.0
    p = (W / T) * 1.8
    s, d = parimutuel.optimal_stake_kelly(p, T, W, t, B)
    assert 0 < s < B
    assert d > 1.0
