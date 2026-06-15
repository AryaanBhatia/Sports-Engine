"""
feed.book
=========

Turn many bookmakers' raw prices into one coherent view, then price a
synthetic "fair-value book" and surface where the real market disagrees.

Pipeline for a single event (h2h / moneyline):

  1.  Each bookmaker's decimal prices  ->  implied probs  ->  devig (Shin)
      so every book contributes a clean, vig-free probability vector.
  2.  Consensus = mean of the devigged vectors (Pinnacle-anchored if present),
      renormalised.  This is the engine's belief about true probabilities.
  3.  Synthetic book = consensus repriced to a target margin.  These are the
      offered prices a trader reads on the floor — the "fake book".
  4.  Edge = consensus_prob * best_market_price - 1, per outcome.  Positive
      edge = the real market is paying more than fair; size it with Kelly.

Everything here reuses the tested `core` math, so the floor and the single-race
study terminal share one source of truth.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from core.devig import implied_probs, booksum, devig_multiplicative, devig_shin
from core.kelly import edge as kelly_edge, kelly_fraction
from feed.schema import Event

# Books we trust more when forming consensus. Pinnacle is the sharpest globally
# and The Odds API anchors it; weight it up when it's in the sample.
_SHARP_WEIGHTS = {"pinnacle": 3.0}


def _devig_one(prices: List[float], method: str):
    if method == "multiplicative":
        return devig_multiplicative(prices)
    probs, _z = devig_shin(prices)
    return probs


def consensus_for_event(
    event: Event,
    method: str = "shin",
) -> Optional[Dict]:
    """
    Aggregate every bookmaker's h2h market into a consensus probability vector.

    Returns None if no usable h2h market is present, otherwise a dict with the
    canonical outcome order, consensus probs, per-book devigged probs, and the
    best (longest) market price available for each outcome.
    """
    rows = list(event.h2h_books())
    if not rows:
        return None

    # Establish a canonical outcome ordering from the first book that prices it.
    canonical = [o.name for _bk, mk in rows[:1] for o in mk.outcomes]
    n = len(canonical)
    if n < 2:
        return None
    idx = {name: i for i, name in enumerate(canonical)}

    devigged_vectors = []
    weights = []
    best_price = [0.0] * n
    best_book = [""] * n
    per_book = []

    for bk, mk in rows:
        names = [o.name for o in mk.outcomes]
        prices = [o.price for o in mk.outcomes]
        # skip a book that doesn't price exactly this set of outcomes
        if set(names) != set(canonical) or any(p <= 1.0 for p in prices):
            continue
        # reorder this book's prices into canonical order
        ordered = [0.0] * n
        for name, price in zip(names, prices):
            ordered[idx[name]] = price
        probs = _devig_one(ordered, method)
        devigged_vectors.append(probs)
        weights.append(_SHARP_WEIGHTS.get(bk.key, 1.0))
        per_book.append({
            "key": bk.key,
            "title": bk.title,
            "prices": ordered,
            "overround": booksum(ordered) - 1.0,
        })
        for i, price in enumerate(ordered):
            if price > best_price[i]:
                best_price[i] = price
                best_book[i] = bk.title

    if not devigged_vectors:
        return None

    # weighted mean of the devigged vectors, then renormalise
    wsum = sum(weights)
    consensus = [
        sum(v[i] * w for v, w in zip(devigged_vectors, weights)) / wsum
        for i in range(n)
    ]
    s = sum(consensus)
    consensus = [c / s for c in consensus]

    return {
        "outcomes": canonical,
        "consensus": consensus,
        "best_price": best_price,
        "best_book": best_book,
        "per_book": per_book,
        "n_books": len(devigged_vectors),
    }


def synthetic_book(consensus: List[float], margin: float = 0.05) -> List[float]:
    """
    Reprice consensus probabilities into offered decimal odds carrying a target
    margin (overround). margin=0 returns fair odds; margin=0.05 a 5% book.

    Offered implied prob q_i = p_i * (1 + margin)  =>  booksum = 1 + margin.
    """
    return [1.0 / (p * (1.0 + margin)) if p > 0 else float("inf") for p in consensus]


def price_event(
    event: Event,
    *,
    method: str = "shin",
    margin: float = 0.05,
    kelly_frac: float = 0.25,
    min_edge: float = 0.0,
) -> Optional[Dict]:
    """
    Full per-event pricing: consensus, the synthetic offered book, and the
    +EV edges against the best real market price for each outcome.
    """
    con = consensus_for_event(event, method=method)
    if con is None:
        return None

    consensus = con["consensus"]
    fair_odds = synthetic_book(consensus, margin=0.0)
    offered = synthetic_book(consensus, margin=margin)

    selections = []
    edges = []
    for i, name in enumerate(con["outcomes"]):
        p = consensus[i]
        mkt = con["best_price"][i]
        e = kelly_edge(p, mkt) if mkt > 1.0 else -1.0
        stake = kelly_fraction(p, mkt, kelly_frac) if e > 0 else 0.0
        sel = {
            "name": name,
            "prob": p,
            "fair_odds": fair_odds[i],
            "offered": offered[i],
            "best_price": mkt,
            "best_book": con["best_book"][i],
            "edge": e,
            "kelly": stake,
        }
        selections.append(sel)
        if e > min_edge:
            edges.append(sel)

    edges.sort(key=lambda s: s["edge"], reverse=True)
    return {
        "id": event.id,
        "sport_key": event.sport_key,
        "sport_title": event.sport_title,
        "commence_time": event.commence_time,
        "label": event.label,
        "home_team": event.home_team,
        "away_team": event.away_team,
        "n_books": con["n_books"],
        "selections": selections,
        "edges": edges,
        "best_edge": edges[0]["edge"] if edges else max((s["edge"] for s in selections), default=0.0),
        "per_book": con["per_book"],
    }


def build_board(
    events: List[Event],
    *,
    method: str = "shin",
    margin: float = 0.05,
    kelly_frac: float = 0.25,
    min_edge: float = 0.0,
) -> List[Dict]:
    """Price a whole slate; drop events with no usable market. Sorted by kickoff."""
    board = []
    for ev in events:
        priced = price_event(ev, method=method, margin=margin,
                             kelly_frac=kelly_frac, min_edge=min_edge)
        if priced is not None:
            board.append(priced)
    board.sort(key=lambda e: e["commence_time"])
    return board
