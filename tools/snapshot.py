"""
tools.snapshot
==============

Poll a provider, build the priced board, and write it to web/snapshot.json so
the static trading floor (web/floor.html) can render it with no backend.

  # one-off, offline mock:
  python -m tools.snapshot

  # live, every 30s, Australian + UK + EU books:
  ODDS_API_KEY=xxxx python -m tools.snapshot --loop 30 --regions au,uk,eu

  # a single live sport:
  ODDS_API_KEY=xxxx python -m tools.snapshot --sport soccer_fifa_world_cup
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from feed.providers import get_provider
from feed.book import build_board

OUT = Path(__file__).resolve().parent.parent / "web" / "snapshot.json"


def run_once(args) -> dict:
    prov = get_provider(
        api_key=args.api_key or os.environ.get("ODDS_API_KEY"),
        regions=args.regions,
        seed=args.seed,
    )
    events = prov.fetch(sport=args.sport)
    board = build_board(
        events,
        method=args.method,
        margin=args.margin,
        kelly_frac=args.kelly,
        min_edge=args.min_edge,
    )
    live = bool(args.api_key or os.environ.get("ODDS_API_KEY"))
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "source": "the-odds-api" if live else "mock",
        "params": {
            "method": args.method, "margin": args.margin,
            "kelly_frac": args.kelly, "min_edge": args.min_edge,
            "regions": args.regions, "sport": args.sport,
        },
        "n_events": len(board),
        "n_edges": sum(len(e["edges"]) for e in board),
        "board": board,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=2))
    return snap


def main():
    ap = argparse.ArgumentParser(description="Build the Edge Engine board snapshot.")
    ap.add_argument("--api-key", default=None, help="The Odds API key (else mock; also reads ODDS_API_KEY)")
    ap.add_argument("--sport", default="upcoming", help="sport key, or 'upcoming' for all sports")
    ap.add_argument("--regions", default="au,uk,eu", help="comma-separated bookmaker regions")
    ap.add_argument("--method", default="shin", choices=["shin", "multiplicative"], help="devig method")
    ap.add_argument("--margin", type=float, default=0.05, help="synthetic book overround")
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction for stake sizing")
    ap.add_argument("--min-edge", type=float, default=0.0, help="minimum edge to flag a bet")
    ap.add_argument("--seed", type=int, default=None, help="mock RNG seed")
    ap.add_argument("--loop", type=float, default=0.0, help="seconds between refreshes (0 = once)")
    args = ap.parse_args()

    while True:
        snap = run_once(args)
        stamp = snap["generated_at"]
        print(f"[{stamp}] {snap['source']}: {snap['n_events']} events, "
              f"{snap['n_edges']} edges -> {OUT}")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
