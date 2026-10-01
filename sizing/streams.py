"""From trades and daily P&L to a day-by-day stream measured in R.

The unit of risk is R: the distance from the entry to the stop. A trade that
makes half of that distance is +0.5 R, a trade stopped out is about -1 R, and
a trade whose stop was gapped through is worse than -1 R. Measured in R, the
two rules (one in basis points, one in dollars per barrel) can be sized the
same way, and "risk 1% per trade" means "lose 1% of the account at -1 R".

The stop distance is fixed on the day the trade opens (two standard deviations
of the spread over the 250 days before), so R does not use anything that was
unknown at the time. tests/test_streams.py checks that.

A stream has, for every trading day and every rule:
  x   the R earned that day (zero when flat), so that the sum over one trade is
      that trade's R exactly (costs included),
  st  0 when flat, 1 on the first day of a trade, 2 on later days.

The account rule (see simulate.py): a trade opened with balance B risks a
fraction f of B, so it holds f * B / (stop distance) units and earns f * B * x
on every day of its life. Sizing is set when the trade opens, not every day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANON = ['crack', 'fly']          # order of the rules inside a two-rule book


@dataclass
class Data:
    trades: dict            # rule -> list of trade dicts, as in data/trades.json
    dates: list             # trading days, 'YYYY-MM-DD'
    pnl: dict               # rule -> list of daily P&L, None where the series has no data
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {d: i for i, d in enumerate(self.dates)}


def load(root: Path | None = None) -> Data:
    root = Path(root) if root else ROOT
    t = json.loads((root / 'data' / 'trades.json').read_text())
    d = json.loads((root / 'data' / 'daily_pnl.json').read_text())
    return Data(
        trades={k: t[k]['trades'] for k in CANON},
        dates=d['dates'],
        pnl={k: d[k] for k in CANON},
    )


@dataclass
class Stream:
    rules: list
    dates: list
    year: np.ndarray                 # calendar year of each day
    x: list                          # per rule: float64 array of R earned each day
    st: list                         # per rule: uint8 array, 0 flat / 1 first day / 2 later days
    trades: list                     # per rule: list of dicts (i, j, r, entry, exit, year)
    n: int = 0

    def __post_init__(self):
        self.n = len(self.dates)


def build_streams(data: Data, rules: list, haircut: float = 0.0,
                  exclude_years=(), exclude=(), start: str | None = None,
                  end: str | None = None) -> Stream:
    """Build the stream for one rule or for the two rules together.

    The window starts on the first entry of the later-starting rule and ends on
    the last exit of the later-ending rule, so that only whole trades are used.
    `haircut` takes that many R off every trade (an extra cost or slippage).
    `exclude_years` drops the trades that entered in those years and `exclude`
    drops single trades given as (rule, number of the trade in date order).
    """
    idx = data.index
    first = max(idx[data.trades[r][0]['entry']] for r in rules)
    last = max(idx[data.trades[r][-1]['exit']] for r in rules)
    s = idx[start] if start else first
    e = idx[end] if end else last
    n = e - s + 1
    year = np.array([int(dt[:4]) for dt in data.dates[s:e + 1]], dtype=np.int64)
    xs, sts, trs = [], [], []
    for r in rules:
        p = [0.0 if v is None else float(v) for v in data.pnl[r]]
        x = np.zeros(n, dtype=np.float64)
        st = np.zeros(n, dtype=np.uint8)
        kept = []
        for number, t in enumerate(data.trades[r]):
            i, j = idx[t['entry']], idx[t['exit']]
            if i < s or j > e:
                continue
            if int(t['entry'][:4]) in exclude_years or (r, number) in exclude:
                continue
            stop = float(t['stop_distance'])
            total = 0.0
            for d in range(i, j + 1):
                total = total + p[d]
                x[d - s] = p[d] / stop
                st[d - s] = 2
            st[i - s] = 1
            # The daily series charges the exit cost on the day after the exit;
            # add it to the last day of the trade so one trade sums to its R.
            x[j - s] = x[j - s] + (float(t['pnl']) - total) / stop
            if haircut:
                x[i - s] = x[i - s] - haircut
            kept.append({'i': i - s, 'j': j - s, 'r': float(t['pnl']) / stop - haircut,
                         'entry': t['entry'], 'exit': t['exit'], 'year': int(t['entry'][:4]),
                         'number': number})
        xs.append(x)
        sts.append(st)
        trs.append(kept)
    return Stream(rules=list(rules), dates=data.dates[s:e + 1], year=year, x=xs, st=sts, trades=trs)


@dataclass
class Units:
    starts: np.ndarray
    lens: np.ndarray
    kind: str


def make_units(stream: Stream, mode: str, block_len: int = 63) -> Units:
    """The pieces the simulation glues together.

    iid    one trade with the flat days before it (one rule), or one day (two rules)
    day    one day
    block  a run of `block_len` days starting anywhere (about one quarter)
    year   one whole calendar year of the sample
    """
    n = stream.n
    if mode == 'iid':
        if len(stream.rules) == 1:
            ends = [t['j'] for t in stream.trades[0]]
            starts = [0] + [e + 1 for e in ends[:-1]]
            return Units(np.array(starts, dtype=np.int64),
                         np.array([e - s + 1 for s, e in zip(starts, ends)], dtype=np.int64), 'trade')
        mode = 'day'
    if mode == 'day':
        return Units(np.arange(n, dtype=np.int64), np.ones(n, dtype=np.int64), 'day')
    if mode == 'block':
        m = n - block_len + 1
        return Units(np.arange(m, dtype=np.int64), np.full(m, block_len, dtype=np.int64), 'block')
    if mode == 'year':
        starts, lens = [], []
        a = 0
        for k in range(1, n + 1):
            if k == n or stream.year[k] != stream.year[a]:
                starts.append(a)
                lens.append(k - a)
                a = k
        return Units(np.array(starts, dtype=np.int64), np.array(lens, dtype=np.int64), 'year')
    raise ValueError(mode)


def trade_r(stream: Stream) -> list:
    """R of every trade, in date order (one list per rule)."""
    return [[t['r'] for t in tr] for tr in stream.trades]
