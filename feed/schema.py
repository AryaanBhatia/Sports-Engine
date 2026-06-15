"""
feed.schema
===========

Plain, JSON-serialisable data structures shared across the live feed, the
book builder, and the web trading floor. No third-party dependencies.

The shapes deliberately mirror The Odds API v4 odds payload so a real feed and
the mock feed are interchangeable:

    Event
      .bookmakers : list[Bookmaker]
        .markets  : list[Market]      (we use the "h2h" / moneyline market)
          .outcomes : list[Outcome]   (name, decimal price)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Outcome:
    name: str
    price: float          # decimal odds


@dataclass
class Market:
    key: str                       # "h2h"
    outcomes: List[Outcome]
    last_update: str = ""


@dataclass
class Bookmaker:
    key: str
    title: str
    markets: List[Market]
    last_update: str = ""


@dataclass
class Event:
    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[Bookmaker] = field(default_factory=list)

    def h2h_books(self):
        """Yield (bookmaker, h2h_market) for every book that prices the h2h."""
        for bk in self.bookmakers:
            for mk in bk.markets:
                if mk.key == "h2h" and mk.outcomes:
                    yield bk, mk

    @property
    def label(self) -> str:
        return f"{self.home_team} v {self.away_team}"


# ----------------------------------------------------------------------
# (de)serialisation — tolerant of the real API and our mock alike
# ----------------------------------------------------------------------

def event_from_api(d: Dict[str, Any]) -> Event:
    """Build an Event from a raw The-Odds-API v4 event dict."""
    books = []
    for bk in d.get("bookmakers", []):
        markets = []
        for mk in bk.get("markets", []):
            outs = [Outcome(o["name"], float(o["price"])) for o in mk.get("outcomes", [])]
            markets.append(Market(mk.get("key", ""), outs, mk.get("last_update", "")))
        books.append(Bookmaker(bk.get("key", ""), bk.get("title", bk.get("key", "")),
                               markets, bk.get("last_update", "")))
    return Event(
        id=d["id"],
        sport_key=d.get("sport_key", ""),
        sport_title=d.get("sport_title", ""),
        commence_time=d.get("commence_time", ""),
        home_team=d.get("home_team", ""),
        away_team=d.get("away_team", ""),
        bookmakers=books,
    )


def event_to_dict(e: Event) -> Dict[str, Any]:
    return asdict(e)
