"""
feed.providers
==============

Two interchangeable sources of upcoming events with bookmaker odds:

  OddsApiProvider  - real live data from The Odds API v4 (needs an API key).
  MockProvider     - a self-contained multi-sport simulator so the whole
                     system runs offline, with realistic vig and live drift,
                     and one deliberately mispriced book so edge detection
                     visibly fires.

Both return List[Event] (see feed.schema). No third-party dependencies — the
live client uses urllib from the standard library.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from feed.schema import Event, Bookmaker, Market, Outcome, event_from_api


# ======================================================================
# Live: The Odds API v4
# ======================================================================

class OddsApiProvider:
    """
    Thin client for https://api.the-odds-api.com/v4

    Usage:
        prov = OddsApiProvider(api_key="...", regions="au,uk,eu")
        events = prov.fetch()                  # soonest games across all sports
        events = prov.fetch(sport="soccer_fifa_world_cup")

    `sport="upcoming"` (the default) returns the next games across every sport,
    which is exactly the cross-sport board this project wants.
    """

    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions: str = "au,uk,eu",
                 markets: str = "h2h", odds_format: str = "decimal",
                 timeout: float = 15.0):
        self.api_key = api_key
        self.regions = regions
        self.markets = markets
        self.odds_format = odds_format
        self.timeout = timeout

    def _get(self, path: str, **params) -> object:
        params["apiKey"] = self.api_key
        url = f"{self.BASE}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "edge-engine/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def in_season_sports(self) -> List[dict]:
        """List in-season sports (does not count against quota)."""
        return self._get("/sports")

    def fetch(self, sport: str = "upcoming") -> List[Event]:
        raw = self._get(
            f"/sports/{sport}/odds",
            regions=self.regions,
            markets=self.markets,
            oddsFormat=self.odds_format,
            dateFormat="iso",
        )
        return [event_from_api(d) for d in raw]


# ======================================================================
# Offline: multi-sport mock with live drift
# ======================================================================

# Each fixture carries a hidden "true" probability vector the books are noised
# around. Soccer is 3-way (Home / Draw / Away); the rest are 2-way moneylines.
_FIXTURES = [
    # sport_key, sport_title, home, away, outcome_names, true_probs
    ("soccer_fifa_world_cup", "FIFA World Cup", "Brazil", "Argentina",
     ["Brazil", "Draw", "Argentina"], [0.40, 0.27, 0.33]),
    ("soccer_fifa_world_cup", "FIFA World Cup", "France", "Spain",
     ["France", "Draw", "Spain"], [0.38, 0.28, 0.34]),
    ("soccer_fifa_world_cup", "FIFA World Cup", "England", "Netherlands",
     ["England", "Draw", "Netherlands"], [0.44, 0.27, 0.29]),
    ("soccer_fifa_world_cup", "FIFA World Cup", "Portugal", "Germany",
     ["Portugal", "Draw", "Germany"], [0.36, 0.28, 0.36]),
    ("basketball_nba", "NBA", "Boston Celtics", "Denver Nuggets",
     ["Boston Celtics", "Denver Nuggets"], [0.58, 0.42]),
    ("basketball_nba", "NBA", "Oklahoma City Thunder", "New York Knicks",
     ["Oklahoma City Thunder", "New York Knicks"], [0.64, 0.36]),
    ("tennis_atp", "ATP", "C. Alcaraz", "J. Sinner",
     ["C. Alcaraz", "J. Sinner"], [0.52, 0.48]),
    ("tennis_atp", "ATP", "N. Djokovic", "A. Zverev",
     ["N. Djokovic", "A. Zverev"], [0.55, 0.45]),
    ("aussierules_afl", "AFL", "Collingwood", "Carlton",
     ["Collingwood", "Carlton"], [0.61, 0.39]),
    ("rugbyleague_nrl", "NRL", "Penrith Panthers", "Melbourne Storm",
     ["Penrith Panthers", "Melbourne Storm"], [0.49, 0.51]),
    ("cricket_t20", "T20 Cricket", "Australia", "India",
     ["Australia", "India"], [0.47, 0.53]),
    ("horse_racing", "Horse Racing", "R5 Eagle Farm", "field",
     ["Northern Light", "Cinder Pact", "Quiet Arbitrage", "Sunshine Express",
      "Maroochy Belle"], [0.34, 0.25, 0.18, 0.13, 0.10]),
]

_BOOKS = [
    ("pinnacle", "Pinnacle", 0.025),     # sharp, low margin, tight noise
    ("sportsbet", "Sportsbet", 0.06),
    ("ladbrokes_au", "Ladbrokes", 0.07),
    ("tab", "TAB", 0.065),
    ("betfair_ex_au", "Betfair", 0.02),
]


def _margined_prices(true_probs, margin, noise, rng):
    """Noise the true probs per-book, add a margin, return decimal odds."""
    jittered = [max(1e-3, p * (1.0 + rng.uniform(-noise, noise))) for p in true_probs]
    s = sum(jittered)
    fair = [p / s for p in jittered]
    # add margin proportionally so booksum ~= 1 + margin
    offered_prob = [p * (1.0 + margin) for p in fair]
    return [round(1.0 / q, 2) for q in offered_prob]


class MockProvider:
    """
    Deterministic-by-seed simulator. Call fetch() repeatedly to see prices
    drift as if live; pass a fresh tick to move the market.

    One book ("sportsbet") is deliberately given a softer line on a rotating
    outcome each tick, so the engine surfaces a real +EV edge to demo with.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._tick = 0

    def fetch(self, sport: str = "upcoming") -> List[Event]:
        self._tick += 1
        now = datetime.now(timezone.utc)
        events: List[Event] = []

        for fi, (skey, stitle, home, away, names, true_probs) in enumerate(_FIXTURES):
            if sport not in ("upcoming", skey):
                continue
            commence = now + timedelta(minutes=20 + fi * 35)
            books = []
            # which outcome gets a soft (juicy) price this tick, for the demo edge
            soft_book = "sportsbet"
            soft_outcome = self._tick % len(names)

            for bkey, btitle, margin in _BOOKS:
                noise = 0.015 if bkey in ("pinnacle", "betfair_ex_au") else 0.04
                prices = _margined_prices(true_probs, margin, noise, self.rng)
                if bkey == soft_book:
                    # lengthen one price ~8-12% -> creates a visible +EV spot
                    prices[soft_outcome] = round(
                        prices[soft_outcome] * self.rng.uniform(1.08, 1.13), 2)
                outcomes = [Outcome(nm, pr) for nm, pr in zip(names, prices)]
                books.append(Bookmaker(
                    bkey, btitle,
                    [Market("h2h", outcomes, last_update=now.isoformat())],
                    last_update=now.isoformat(),
                ))

            events.append(Event(
                id=f"mock-{fi}",
                sport_key=skey,
                sport_title=stitle,
                commence_time=commence.isoformat(),
                home_team=home,
                away_team=away,
                bookmakers=books,
            ))
        return events


def get_provider(api_key: Optional[str] = None, **kw):
    """Return a live provider if a key is given, else the offline mock."""
    if api_key:
        return OddsApiProvider(api_key=api_key, **kw)
    return MockProvider(seed=kw.get("seed"))
