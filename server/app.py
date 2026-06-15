"""
server.app
==========

A zero-dependency dev server (standard library only) that serves the static
trading floor and a live JSON endpoint, so the floor can poll for fresh prices
without writing snapshot files.

  # offline mock, auto-refreshing in the browser:
  python -m server.app
  # -> open http://localhost:8000/floor.html

  # live:
  ODDS_API_KEY=xxxx python -m server.app --regions au,uk,eu

Endpoints:
  GET /floor.html         the trading floor UI
  GET /api/board          freshly-priced board as JSON (re-fetches the provider)
  GET /<file>             any static file under web/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from feed.providers import get_provider
from feed.book import build_board

WEB = Path(__file__).resolve().parent.parent / "web"
CFG = {}  # filled by main()


def _board_json() -> bytes:
    prov = get_provider(
        api_key=CFG.get("api_key") or os.environ.get("ODDS_API_KEY"),
        regions=CFG.get("regions", "au,uk,eu"),
        seed=CFG.get("seed"),
    )
    events = prov.fetch(sport=CFG.get("sport", "upcoming"))
    board = build_board(events, method=CFG.get("method", "shin"),
                        margin=CFG.get("margin", 0.05),
                        kelly_frac=CFG.get("kelly", 0.25),
                        min_edge=CFG.get("min_edge", 0.0))
    live = bool(CFG.get("api_key") or os.environ.get("ODDS_API_KEY"))
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "source": "the-odds-api" if live else "mock",
        "n_events": len(board),
        "n_edges": sum(len(e["edges"]) for e in board),
        "board": board,
    }
    return json.dumps(snap).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/api/board", "/api/board/"):
            try:
                body = _board_json()
            except Exception as exc:  # surface provider/auth errors to the UI
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, body, "application/json")
                return
            self._send(200, body, "application/json")
            return

        rel = path.lstrip("/") or "floor.html"
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB)) or not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = ("text/html" if target.suffix == ".html"
                 else "application/json" if target.suffix == ".json"
                 else "text/plain")
        self._send(200, target.read_bytes(), ctype)

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Edge Engine live dev server.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sport", default="upcoming")
    ap.add_argument("--regions", default="au,uk,eu")
    ap.add_argument("--method", default="shin")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    CFG.update(vars(args))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    live = bool(args.api_key or os.environ.get("ODDS_API_KEY"))
    print(f"Edge Engine floor: http://localhost:{args.port}/floor.html  "
          f"[{'LIVE' if live else 'MOCK'} feed]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
