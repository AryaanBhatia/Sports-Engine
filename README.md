# Edge Engine

**A quantitative wagering research terminal — the mathematics a proprietary betting syndicate uses to find, price, and size an edge.**

Edge Engine is a self-contained study of the quantitative stack behind a syndicate like [Edge Stackers](https://edgestackers.com) — a privately-capitalised proprietary betting group that bets *into* racing and sports markets rather than booking bets. It has two halves:

1. **The study terminal** (`index.html`) — an interactive single-page walk through the mathematics on one sample race, every number recomputing live in the browser.
2. **The live trading floor** (`web/floor.html`) — a board of **all upcoming events across every sport**, pulling **live market odds**, devigging the cross-bookmaker consensus, and printing the engine's own **synthetic fair-value book** beside the real market so traders can read the edge. Includes a paper blotter that tracks Closing Line Value.

Both are backed by a **tested Python core and feed layer** (the source of truth, 34 property tests). The browser maths mirrors the Python and is verified to agree to full floating-point precision.

> **Educational and illustrative only.** Models are simplified; markets are not. Nothing here is betting, financial, or investment advice. In Australia, support is available at Gambling Help Online (1800 858 858).

---

## The problem, stated precisely

A bookmaker sells a margin. A syndicate does the opposite: it builds an **independent estimate of probability**, compares that estimate to the price the market is offering, and stakes the disagreement — but only when the disagreement survives the cost of being wrong about its own estimate.

That reduces to a five-stage pipeline, and the whole terminal is a live walk through it:

```
market odds ──▶ devig ──▶ fair market probs
ratings ──────▶ softmax ─▶ model probs ──▶ blend with market
                                    │
                                    ├─▶ Harville/Plackett–Luce ─▶ exotics (place, exacta, trifecta)
                                    │
                                    └─▶ edge = p·o − 1 ─▶ Kelly stake ─▶ (pari-mutuel: price impact) ─▶ bankroll growth
```

The honest scoreboard is not short-run profit and loss. It is **Closing Line Value** — whether you consistently beat the price the market settles on.

---

## 1 · Stripping the margin (devigging)

Decimal odds $o_i$ imply a raw probability $\pi_i = 1/o_i$. Because the book carries a margin, these do not sum to one:

$$\text{booksum} = \sum_i \frac{1}{o_i} > 1, \qquad \text{overround} = \text{booksum} - 1.$$

Recovering the market's *fair* view means removing that excess. **How** you remove it changes the answer, and the differences concentrate exactly where money is made — on favourites and longshots.

**Multiplicative** — normalise proportionally. Simple, but preserves the favourite–longshot bias baked into the raw prices.

$$p_i = \frac{\pi_i}{\sum_j \pi_j}$$

**Additive** — subtract the excess evenly across $n$ runners. Hurts longshots disproportionately.

$$p_i = \pi_i - \frac{\text{booksum}-1}{n}$$

**Power** — solve for the exponent $k$ that renormalises:

$$p_i = \pi_i^{\,k}, \quad \text{where } k \text{ solves } \sum_i \pi_i^{\,k} = 1.$$

**Shin (1992)** — the one a serious desk reaches for. It models the overround as arising from a fraction $z$ of *insider* money and inverts that structure, which corrects the favourite–longshot bias **endogenously** — lifting the favourite and shrinking the longshot, the empirically correct direction:

$$p_i = \frac{\sqrt{z^2 + 4(1-z)\dfrac{\pi_i^2}{S}} - z}{2(1-z)}, \qquad S = \sum_j \pi_j,$$

with $z \in [0,1)$ solved so that $\sum_i p_i = 1$. The recovered $z$ is itself informative: it is the market's implied insider share.

`core/devig.py` — `implied_probs`, `booksum`, `overround`, `devig_multiplicative`, `devig_additive`, `devig_power`, `devig_shin` (returns `(probs, z)`), `fair_odds`.

## 2 · Forming your own opinion (ratings → probabilities)

Your private ratings $r_i$ (speed figures, model scores) become win probabilities through a tempered softmax — equivalently a multinomial-logit / Luce choice model:

$$p_i = \frac{e^{\beta r_i}}{\sum_j e^{\beta r_j}}.$$

$\beta$ is **conviction** (inverse temperature): at $\beta \to 0$ the field is uniform; as $\beta$ grows, probability piles onto the top-rated runner. A lone opinion is noisy, so the model can be shrunk toward the (devigged) market by a geometric blend with weight $w$:

$$p_i \propto p_i^{\text{model}\,w} \cdot p_i^{\text{market}\,(1-w)}.$$

This trades edge for robustness — the same move a desk makes before firing real money.

`core/ratings.py` — `softmax_probs(ratings, beta)`, `blend(model, market, w)`.

## 3 · Pricing the exotics (Harville / Plackett–Luce)

A single win model implies the **entire finishing-order distribution**. Under the Harville (1973) / Plackett–Luce model, each finishing position is drawn proportional to the remaining runners' strengths. The probability of a full order $(a, b, c, \dots)$ is

$$P(a,b,c,\dots) = \frac{v_a}{V} \cdot \frac{v_b}{V - v_a} \cdot \frac{v_c}{V - v_a - v_b} \cdots, \qquad V = \sum_k v_k.$$

The exacta (runner $i$ first, $j$ second) is therefore

$$P(i \to 1^{\text{st}}, j \to 2^{\text{nd}}) = p_i \cdot \frac{v_j}{V - v_i}.$$

Plain Harville (taking $v_i = p_i$) systematically **over-rates the favourite's place chance**. A discount exponent $\lambda < 1$ applied to the strengths, $v_i = p_i^{\lambda}$, flattens the back end of the order toward the Henery (1981) / Stern (1990) refinements that fit the data better. The terminal exposes $\lambda$ as a slider and renders the full exacta matrix as a heatmap — the raw material for pricing exacta, trifecta, and first-four pools off one win model.

`core/harville.py` — `order_prob`, `exacta_prob`, `finish_position_probs`, `place_prob`, `full_order_distribution` (all accept `lam`).

## 4 · Finding the edge and sizing the bet (Kelly)

With a model probability $p$ and a market price $o$, the expected value of a unit stake is

$$\text{EV} = p \cdot o - 1, \qquad \text{edge} = \text{EV}.$$

Bet only when edge $> 0$. The growth-optimal stake is the Kelly (1956) fraction:

$$f^\* = \frac{p \cdot o - 1}{o - 1} = \frac{\text{edge}}{o - 1}.$$

Because $p$ is an **estimate**, full Kelly is too aggressive — estimation error makes it overbet. Desks run **fractional Kelly** (typically a quarter to a half) and impose a minimum-edge threshold so marginal bets never fire. Staking $f$ of bankroll yields expected log-growth

$$g(f) = p \log\!\big(1 + f(o-1)\big) + (1-p)\log(1 - f),$$

maximised at $f = f^\*$ — the quantity a syndicate actually compounds.

`core/kelly.py` — `expected_value`, `edge`, `kelly_fraction(p, o, frac)`, `expected_log_growth`, `stake_card`.

## 5 · The pari-mutuel twist (your own money moves the price)

This is where a tote syndicate genuinely differs from a fixed-odds bettor — and where Edge Stackers' edge in pool markets lives. In a pari-mutuel pool there is no fixed price. The dividend depends on how the pool divides *after takeout $t$*, and **your own stake $s$ dilutes the very price you are betting into**:

$$D(s) = (1-t)\,\frac{T + s}{W + s},$$

where $T$ is the total pool and $W$ the amount already on your runner. As you bet more, $D(s)$ falls. So the expected value of staking $s$,

$$\text{EV}(s) = s\big(p\,D(s) - 1\big),$$

is **not** monotonic — there is a finite optimal stake. The impact-aware log-optimal stake (golden-section search over the concave growth objective) is strictly smaller than the naive fixed-odds Kelly stake. The terminal plots both the growth curve and the dividend decay, marking the optimum against the naive Kelly point.

`core/parimutuel.py` — `dividend`, `ev_of_stake`, `optimal_stake_riskneutral`, `optimal_stake_kelly` (returns `(stake, dividend)`).

## 6 · Surviving the variance, and the real scoreboard

A genuine edge still loses for weeks. The terminal Monte-Carlos a full season at a chosen edge and Kelly fraction to show the **median growth and the drawdown band** (5th–95th percentile) side by side — the variance you must be capitalised to survive.

The honest measure of skill is not that P&L. It is **Closing Line Value**:

$$\text{CLV} = \frac{o_{\text{taken}}}{o_{\text{close}}} - 1.$$

If the closing line is the market's efficient estimate, then beating it *is* your expected edge. Beat the close consistently and long-run profit is just variance resolving in your favour. Lose to the close and no amount of short-run winning is real.

---

## Live trading floor (synthetic book + edge detection)

The study terminal proves the maths on one race. The **floor** applies it to the live market, across every sport that's about to play.

**The data layer** (`feed/`) treats a real feed and an offline simulator as interchangeable. `OddsApiProvider` pulls from [The Odds API](https://the-odds-api.com) v4 — `sport=upcoming` returns the soonest games across all sports, each carrying every bookmaker's decimal prices in a normalised shape (Australian, UK, and EU books by default). `MockProvider` generates realistic multi-sport fixtures (World Cup, NBA, ATP, AFL, NRL, T20, racing) with proper vig and live drift, so the whole system runs with **no key and no network** — and it deliberately softens one book's line each tick so edge detection visibly fires.

**The book builder** (`feed/book.py`) is the heart, and it reuses the same `core` maths:

1. **Devig each book.** Every bookmaker's prices → implied probabilities → Shin-devigged, giving one clean probability vector per book.
2. **Form consensus.** A Pinnacle-anchored weighted mean of those vectors, renormalised — the engine's belief about true probabilities. (Pinnacle is the sharpest book globally; The Odds API anchors it.)
3. **Print the synthetic book.** Reprice the consensus to a target margin: offered implied $q_i = p_i(1+m)$, so the offered book carries exactly overround $m$. At $m=0$ these are fair odds; at $m=0.05$ a 5% book. *This is the "fake book" the floor shows traders — an internal fair-value sheet, not a market offered to customers.*
4. **Surface the edge.** For each outcome, $\text{edge} = p_{\text{consensus}} \cdot o_{\text{best market}} - 1$. Positive edge means the real market is paying over fair; the floor flags it and sizes it with fractional Kelly.

**The floor UI** (`web/floor.html`) renders that board — every event showing the engine's fair and offered prices beside the best real market price, +EV selections lit up with Kelly stakes, sport filters, and live sliders for synthetic margin, Kelly fraction, and minimum edge (all re-priced client-side from the same formulas). The **paper blotter** lets a trader "take" a price; it then tracks **CLV against live fair value** as the market moves — the honest scoreboard from §6, now operating on the live book.

The floor reads from a live `/api/board` endpoint when served, and falls back to a static `snapshot.json` so it also works as a plain GitHub Pages file.



```
edge-engine/
├── index.html            # the study terminal — single self-contained file, zero build step
├── web/
│   ├── floor.html        # the live trading floor — synthetic book vs market, paper blotter
│   └── snapshot.json      # sample board so the floor renders offline with no backend
├── core/                 # tested maths "source of truth"
│   ├── devig.py          #  §1  margin removal: multiplicative, additive, power, Shin
│   ├── ratings.py        #  §2  softmax (Luce) + geometric blend to market
│   ├── harville.py       #  §3  Plackett–Luce order statistics, exotics, place, exacta
│   ├── kelly.py          #  §4  EV, edge, fractional Kelly, log-growth, stake card
│   └── parimutuel.py     #  §5  pool dividend, price impact, log-optimal pool stake
├── feed/                 # live-feed layer (reuses core)
│   ├── providers.py      #  The Odds API v4 client + offline multi-sport mock
│   ├── book.py           #  consensus devig → synthetic book → edge detection → Kelly
│   └── schema.py         #  JSON-serialisable Event / Bookmaker / Market / Outcome
├── tools/snapshot.py     # CLI: poll a provider, write web/snapshot.json (once or looping)
├── server/app.py         # zero-dependency dev server: serves the floor + live /api/board
├── tests/
│   ├── test_core.py      # 25 property tests on the maths
│   └── test_feed.py      #  9 property tests on consensus, synthetic book, edge detection
├── requirements.txt
└── README.md
```

The browser implementations mirror the Python core exactly and are cross-checked to full floating-point precision on the sample race (booksum `1.176056`, Shin `z = 0.027158`, place/exacta/Kelly/pool-stake all identical) — so the interactive pages are not toy reimplementations, they run the same mathematics client-side.

## Running it

**The study terminal** — just open `index.html` in any browser. No build, no server, no dependencies.

**The live trading floor** — two ways:

```bash
# 1. Offline mock, fully interactive, no key needed:
python -m server.app                     # -> http://localhost:8000/floor.html

# 2. Live across all upcoming sports (get a free key at the-odds-api.com):
ODDS_API_KEY=xxxx python -m server.app --regions au,uk,eu
```

Or build a static snapshot for GitHub Pages (no backend needed at view time):

```bash
python -m tools.snapshot                 # offline mock -> web/snapshot.json
ODDS_API_KEY=xxxx python -m tools.snapshot --loop 30   # live, refreshing every 30s
```

**The Python core and tests:**

```bash
pip install -r requirements.txt
pytest -q                 # 34 passed
```

```python
from feed.providers import get_provider
from feed.book import build_board

prov  = get_provider(api_key=None)            # None -> offline mock; pass a key for live
board = build_board(prov.fetch("upcoming"),   # all upcoming sports
                    margin=0.05, kelly_frac=0.25, min_edge=0.0)
for ev in board:
    for e in ev["edges"]:                     # +EV selections, already Kelly-sized
        print(ev["label"], e["name"], f'{e["edge"]:+.1%}', e["best_book"])
```

## Deploying to GitHub Pages

Push the repo, then in **Settings → Pages** serve from the `main` branch root. `index.html` (the study terminal) is served as-is at `https://<user>.github.io/edge-engine/`. The live floor at `/web/floor.html` reads `web/snapshot.json`, so commit a fresh snapshot (`python -m tools.snapshot`) — or run a scheduled job / GitHub Action to refresh it — and the floor renders the latest board with no backend.

## References

- Shin, H. S. (1992). *Prices of state-contingent claims with insider traders, and the favourite–longshot bias.*
- Harville, D. A. (1973). *Assigning probabilities to the outcomes of multi-entry competitions.*
- Plackett, R. L. (1975) / Luce, R. D. (1959). *The analysis of permutations / individual choice behavior.*
- Henery, R. J. (1981); Stern, H. (1990). *Permutation probabilities and order-statistic refinements.*
- Kelly, J. L. (1956). *A new interpretation of information rate.*
- Clarke, Kovalchik & Ingram (2017). *Adjusting bookmaker's odds to allow for overround.*
