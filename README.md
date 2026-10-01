# How big can you trade a small edge?

Position sizing, drawdown and the funded-account challenge, tested on two small trading edges.

Live page: [quant.orbiseo.fr/position-sizing](https://quant.orbiseo.fr/position-sizing/).

The two edges are the fade rules from the
[relative value study](https://github.com/Nicolas8330/relative-value-strategies): the 3-2-1 crack
spread (71 trades) and the 2s5s10s Treasury butterfly (107 trades). Nothing was traded. The
question is what size a small edge can be traded at.

## The short answer

| | Crack | Butterfly | Both together |
|---|---:|---:|---:|
| Trades | 71 | 107 | |
| Winning trades | 76% | 71% | |
| Average trade, in R | 0.19 | 0.11 | |
| Kelly fraction (risk per trade that grows the account fastest) | 22% | 17% | |
| 90% range for it, resampling years of trades | 2% to 39% | 0% to 33% | |
| Pass a typical challenge within a year, risking 1% per trade | 0% | 0% | 0% |
| Pass, risking 5% per trade | 32% | 19% | 15% |
| Best chance of passing and keeping the account a year (risk at the peak) | 26% (6%) | 21% (6%) | 37% (10%) |
| Of those that pass at 20% risk, share that fail within a year | 90% | 95% | 81% |

R is the profit of a trade divided by the risk taken to its stop. The typical challenge is: reach +10%
before the balance falls 10% below its start or loses 5% in one day, within a year, then survive a
funded year. Firms differ; these are round numbers, not any firm's terms.

What it says:

* Each rule trades only 3 to 4 times a year. At sensible sizes (1% to 2% risk) it cannot earn a 10%
  target inside a year.
* The pass rate rises with risk and levels off near half. The chance of passing **and keeping** the
  account peaks at 5% to 10% risk per trade and collapses beyond.
* Full Kelly is far too big and very uncertain. Kelly estimated only from earlier trades loses money
  on the butterfly.
* The two rules have a daily correlation of 0.004, but they are flat on most days, so this says less
  than it seems. Weekly it is -0.09 and it did not rise in 2008, 2020 and 2022 (three episodes only).

## The model

Every trade becomes a path of daily R. A trade opened with balance B and risk f holds
f * B / (stop distance) units, so it earns f * B * (daily R). Sizing is set when the trade opens.
Pieces of the real history are drawn at random with replacement and glued together:
one trade at a time (naive), blocks of one quarter, or whole calendar years (keeps runs of bad
periods together). Each path has its own seeded generator (mulberry32), the same one used by the
JavaScript on the site, so the two programs give identical numbers.

A challenge is passed when the balance reaches the target before it falls to the drawdown limit
(from the start balance, or from the highest balance), before a one-day loss exceeds the daily limit,
and inside the time limit. After passing, the account is reset to 1 and runs for a further year under
the same limits.

## What is in the repository

| File | What it does |
|---|---|
| `sizing/rng.py` | mulberry32, identical in Python and JavaScript |
| `sizing/streams.py` | trades and daily P&L to day-by-day R, and the pieces of history |
| `sizing/sim_ref.py` | the simulation one path at a time, written to be read |
| `sizing/simulate.py` | the same simulation for thousands of paths at once (numpy) |
| `sizing/kelly.py` | Kelly fraction and its bootstrap |
| `sizing/rstats.py` | shape of the R distribution, clustering of bad trades |
| `sizing/stress.py` | replay of the real worst year and worst 10 trades |
| `sizing/diversify.py` | correlations and the combined book |
| `sizing/analysis.py` | the experiments behind the page |
| `examples/make_results.py` | every figure quoted, in one run (about 5 minutes) |
| `tests/` | 54 tests: units, no look-ahead, accounting identities, an exact calculation, claims |
| `data/` | the two input files, `results.json`, `ref_check.json`, `PROVENANCE.md` |
| `REVIEW.md` | ways the result could be wrong and what was done about each |

Run it:

```
pip install -r requirements.txt
python -m pytest -q
python examples/make_results.py       # writes data/results.json and data/ref_check.json
```

The site runs the same simulation in JavaScript so it is instant on a phone.
`scripts/verify-position-sizing.mjs` in the site repository runs both on the same data and seeds and
fails if any number differs by more than 1e-9.

## Limits

* Few trades (71 and 107), spread over 19 and 36 years. Resampling cannot create a shock worse than
  the worst in the sample (-1.84 R and -1.53 R).
* No margin, financing, bid-offer spread beyond the flat cost already in the trades, or intraday moves.
  Loss limits are checked on closing balances.
* Challenge rules are typical, not any firm's.
* The correlation of the two rules rests on 608 days when both held a trade and three stress years.
* The trades come from spot prices and constant-maturity yields, not from futures.

Not advice. Research only.
