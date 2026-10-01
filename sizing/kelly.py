"""The Kelly fraction: the risk per trade that grows the account fastest.

If a trade returns R times the risk taken, and you risk a fraction f of the
account, the account is multiplied by (1 + f * R). Over many trades the account
grows at the average of log(1 + f * R). The f that makes this average largest
is the Kelly fraction. It is found here on the actual list of R multiples, with
no assumption about their shape.

Two facts drive the whole page:
* Above about twice the Kelly fraction the account shrinks on average, even
  though every trade has a positive expected profit.
* The fraction is estimated from few trades, so it is very uncertain. The
  bootstrap here shows how uncertain, by resampling the trades.

The search is a bisection on the slope of the average log growth, which only
uses additions, multiplications and divisions, so the JavaScript on the site
gets the same answer to the last digit.
"""
from __future__ import annotations

import numpy as np

from .rng import Mulberry32, path_seed


def kelly_fraction(r, iters: int = 100) -> float:
    """The f that maximises the mean of log(1 + f * r). Zero if the mean of r is not positive."""
    n = len(r)
    total = 0.0
    lo_r = 0.0
    for v in r:
        total = total + v
        if v < lo_r:
            lo_r = v
    if n == 0 or not total / n > 0.0:
        return 0.0
    if lo_r >= 0.0:
        return 1.0                       # no losing trade at all: cap at 100%
    lo = 0.0
    hi = (1.0 / -lo_r) * (1.0 - 1e-9)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        g = 0.0
        for v in r:
            g = g + v / (1.0 + mid * v)
        if g > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def growth_rate(r, f: float) -> float:
    """Average log growth per trade at risk f (the quantity Kelly maximises).

    Minus infinity when f is so large that one trade in the list would wipe the account out.
    """
    g = 1.0 + f * np.asarray(r)
    if g.min() <= 0.0:
        return float('-inf')
    return float(np.mean(np.log(g)))


def kelly_fraction_rows(m: np.ndarray, iters: int = 100) -> np.ndarray:
    """Kelly fraction of every row of a matrix (rows are resampled lists, zero-padded)."""
    B = m.shape[0]
    mean_pos = m.sum(axis=1) > 0.0
    lo_r = np.minimum(m.min(axis=1), 0.0)
    safe = np.where(lo_r < 0.0, lo_r, -1.0)
    lo = np.zeros(B)
    hi = (1.0 / -safe) * (1.0 - 1e-9)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        g = (m / (1.0 + mid[:, None] * m)).sum(axis=1)
        up = g > 0.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out = (lo + hi) / 2.0
    out = np.where(lo_r >= 0.0, 1.0, out)
    return np.where(mean_pos, out, 0.0)


def resample(groups: list, mode: str, seed: int, b: int) -> list:
    """One bootstrap sample of trades.

    mode 'iid'   : draw trades one by one, as many as there are trades.
    mode 'year'  : draw whole calendar years (all trades that entered in that
                   year), as many years as there are years, so that shocks that
                   hit several trades in a row stay together.
    `groups` is a list of lists of R, one list per year, in date order.
    """
    rng = Mulberry32(path_seed(seed, b))
    out = []
    if mode == 'iid':
        flat = [v for g in groups for v in g]
        n = len(flat)
        for _ in range(n):
            out.append(flat[int(rng() * n)])
    else:
        ny = len(groups)
        for _ in range(ny):
            out.extend(groups[int(rng() * ny)])
    return out


def kelly_bootstrap(groups: list, mode: str, seed: int, reps: int, iters: int = 100) -> np.ndarray:
    samples = [resample(groups, mode, seed, b) for b in range(reps)]
    width = max(len(s) for s in samples)
    m = np.zeros((reps, width))
    for b, s in enumerate(samples):
        m[b, :len(s)] = s
    return kelly_fraction_rows(m, iters)


def groups_by_year(trades: list) -> list:
    """R of the trades, grouped by year of entry, in date order."""
    years = sorted({t['year'] for t in trades})
    return [[t['r'] for t in trades if t['year'] == y] for y in years]


def book_daily(stream, w) -> np.ndarray:
    """Daily R of a book: the weighted sum of the rules' daily R (same order of additions as the site)."""
    out = np.zeros(stream.n)
    for j in range(len(w)):
        out = out + w[j] * stream.x[j]
    return out


def kelly_daily(series) -> float:
    """Kelly fraction of a daily series when the size is reset every day.

    Days with no profit or loss carry no information and are left out. This is
    the Kelly fraction of the book as if it were re-sized each day; it is not the
    same number as the per-trade fraction, because a trade here is sized once,
    when it opens.
    """
    return kelly_fraction([float(v) for v in series if v != 0.0])
