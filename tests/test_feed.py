"""
Tests for the live-feed layer: schema parsing, consensus, synthetic book,
and edge detection. Run with `pytest -q`.
"""

import math

from feed.schema import event_from_api, Event
from feed.book import (
    consensus_for_event, synthetic_book, price_event, build_board,
)
from feed.providers import MockProvider


# ---- a hand-built event we control exactly --------------------------------

def _two_book_event(p_home=0.6, p_away=0.4, margin_a=0.05, margin_b=0.05,
                    soft_home_b=1.0):
    """Two books pricing a 2-way market around (p_home, p_away)."""
    def odds(p, m):
        return round(1.0 / (p * (1 + m)), 4)
    raw = {
        "id": "t1", "sport_key": "test", "sport_title": "Test",
        "commence_time": "2026-06-15T12:00:00Z",
        "home_team": "Home", "away_team": "Away",
        "bookmakers": [
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Home", "price": odds(p_home, margin_a)},
                    {"name": "Away", "price": odds(p_away, margin_a)},
                ]}]},
            {"key": "softbook", "title": "SoftBook", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Home", "price": round(odds(p_home, margin_b) * soft_home_b, 4)},
                    {"name": "Away", "price": odds(p_away, margin_b)},
                ]}]},
        ],
    }
    return event_from_api(raw)


def test_schema_roundtrip():
    e = _two_book_event()
    assert isinstance(e, Event)
    assert e.label == "Home v Away"
    assert len(list(e.h2h_books())) == 2


def test_consensus_sums_to_one():
    con = consensus_for_event(_two_book_event(0.6, 0.4))
    assert con is not None
    assert math.isclose(sum(con["consensus"]), 1.0, abs_tol=1e-9)
    # consensus should land near the true 0.6/0.4 after devigging
    assert abs(con["consensus"][0] - 0.6) < 0.02


def test_synthetic_book_carries_target_margin():
    con = consensus_for_event(_two_book_event(0.6, 0.4))
    p = con["consensus"]
    for m in (0.0, 0.05, 0.10):
        odds = synthetic_book(p, margin=m)
        booksum = sum(1.0 / o for o in odds)
        assert math.isclose(booksum, 1.0 + m, abs_tol=1e-9)


def test_fair_book_is_vig_free():
    con = consensus_for_event(_two_book_event(0.55, 0.45))
    odds = synthetic_book(con["consensus"], margin=0.0)
    assert math.isclose(sum(1.0 / o for o in odds), 1.0, abs_tol=1e-9)


def test_edge_detection_fires_on_soft_price():
    # SoftBook pays 15% over the fair home price -> must be flagged +EV
    ev = _two_book_event(0.6, 0.4, margin_b=0.05, soft_home_b=1.15)
    priced = price_event(ev, min_edge=0.0)
    home = next(s for s in priced["selections"] if s["name"] == "Home")
    assert home["edge"] > 0
    assert home["kelly"] > 0
    assert any(e["name"] == "Home" for e in priced["edges"])


def test_no_phantom_edge_on_fair_market():
    # both books fair-ish, no soft price -> home edge should be <= small noise
    ev = _two_book_event(0.6, 0.4, soft_home_b=1.0)
    priced = price_event(ev, min_edge=0.0)
    home = next(s for s in priced["selections"] if s["name"] == "Home")
    assert home["edge"] < 0.01


def test_kelly_never_exceeds_fraction():
    ev = _two_book_event(0.6, 0.4, soft_home_b=1.30)
    priced = price_event(ev, kelly_frac=0.25)
    for s in priced["selections"]:
        assert 0.0 <= s["kelly"] <= 0.25 + 1e-9


def test_mock_board_builds_with_edges():
    prov = MockProvider(seed=1)
    events = prov.fetch("upcoming")
    assert len(events) >= 6
    board = build_board(events, min_edge=0.0)
    assert len(board) == len(events)
    # the mock injects a soft book each tick, so at least one edge should exist
    assert sum(len(e["edges"]) for e in board) >= 1
    # every event prices every outcome
    for e in board:
        assert len(e["selections"]) >= 2
        assert all(math.isfinite(s["fair_odds"]) for s in e["selections"])


def test_mock_sport_filter():
    prov = MockProvider(seed=2)
    wc = prov.fetch("soccer_fifa_world_cup")
    assert wc and all(e.sport_key == "soccer_fifa_world_cup" for e in wc)
    assert all(len(list(e.h2h_books())) >= 3 for e in wc)
