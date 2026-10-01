"""Do the two rules diversify each other, and how well do we know it?

Two P&L series with a correlation near zero look like a free lunch: combine
them and the risk falls by about the square root of two. Two things can spoil
this and the code here measures both.

1. A near-zero daily correlation can be a by-product of both rules being flat
   most of the time. When one rule is flat its P&L is exactly zero, and zeros
   drag a correlation towards zero whatever happens when both are trading. So
   the correlation is also measured on the days both are in a trade, on weekly
   and monthly totals, and year by year, with a range from resampling years.
2. Correlations rise in a crisis. Only three real stress episodes (2008, 2020,
   2022) are in the overlapping sample, so that is looked at directly and the
   limits are said plainly.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .rng import Mulberry32, path_seed


def pearson(a, b) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return None
    da, db = a - a.mean(), b - b.mean()
    den = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    if den == 0.0:
        return None
    return float((da * db).sum() / den)


def overlap(data):
    """Days on which both daily series have data, with the P&L of each."""
    c = data.pnl['crack']
    f = data.pnl['fly']
    keep = [i for i in range(len(data.dates)) if c[i] is not None and f[i] is not None]
    dates = [data.dates[i] for i in keep]
    return dates, np.array([c[i] for i in keep]), np.array([f[i] for i in keep])


def _week_key(d: str):
    y, w, _ = dt.date.fromisoformat(d).isocalendar()
    return (y, w)


def totals(dates, a, b, key):
    """Sum a and b within each group (week or month), in date order."""
    order, sa, sb, last = {}, [], [], []
    for d, x, y in zip(dates, a, b):
        k = key(d)
        if k not in order:
            order[k] = len(sa)
            sa.append(0.0)
            sb.append(0.0)
            last.append(d)
        sa[order[k]] += x
        sb[order[k]] += y
        last[order[k]] = d
    return last, np.array(sa), np.array(sb)


def _year_moments(years, a, b):
    """Per-year sufficient statistics for a correlation: n, sums, sums of squares and products."""
    out = {}
    for y in sorted(set(years)):
        m = np.array([yy == y for yy in years])
        x, z = a[m], b[m]
        out[y] = np.array([m.sum(), x.sum(), z.sum(), (x * x).sum(), (z * z).sum(), (x * z).sum()])
    return out


def _corr_from_moments(m):
    n, sa, sb, saa, sbb, sab = m
    cov = sab / n - (sa / n) * (sb / n)
    va = saa / n - (sa / n) ** 2
    vb = sbb / n - (sb / n) ** 2
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float('nan')


def year_bootstrap_corr(years, a, b, seed=11, reps=2000):
    """95% range of a correlation, from resampling whole calendar years."""
    mom = _year_moments(years, a, b)
    ys = sorted(mom)
    vals = []
    for r in range(reps):
        rng = Mulberry32(path_seed(seed, r))
        tot = np.zeros(6)
        for _ in ys:
            tot = tot + mom[ys[int(rng() * len(ys))]]
        vals.append(_corr_from_moments(tot))
    vals = np.array(vals)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def correlation_report(data, stream, stress_years=(2008, 2020, 2022)) -> dict:
    """Everything about how the two rules move together. `stream` is the two-rule book."""
    dates, c, f = overlap(data)
    years = [int(d[:4]) for d in dates]
    wk_dates, wc, wf = totals(dates, c, f, _week_key)
    mo_dates, mc, mf = totals(dates, c, f, lambda d: d[:7])
    wk_years = [int(d[:4]) for d in wk_dates]
    out = {
        'overlap_days': len(dates), 'first': dates[0], 'last': dates[-1],
        'daily': pearson(c, f), 'weekly': pearson(wc, wf), 'monthly': pearson(mc, mf),
        'daily_range': year_bootstrap_corr(years, c, f),
        'weekly_range': year_bootstrap_corr(wk_years, wc, wf),
        'weeks': len(wk_dates), 'months': len(mo_dates),
    }
    # Days on which both rules hold a trade (only possible inside the book window).
    st_c, st_f = stream.st[0] > 0, stream.st[1] > 0
    both = st_c & st_f
    either = st_c | st_f
    xc, xf = stream.x[0], stream.x[1]
    out['book_window'] = [stream.dates[0], stream.dates[-1]]
    out['book_days'] = stream.n
    out['days_crack_open'] = int(st_c.sum())
    out['days_fly_open'] = int(st_f.sum())
    out['days_both_open'] = int(both.sum())
    out['share_both_open_of_either'] = float(both.sum() / either.sum())
    out['corr_both_open_r'] = pearson(xc[both], xf[both])
    out['corr_book_window_r'] = pearson(xc, xf)
    # Monthly totals in R inside the book window, and how often both lose in the same month.
    months = [d[:7] for d in stream.dates]
    keys = sorted(set(months))
    mr = np.array([[xc[[m == k for m in months]].sum(), xf[[m == k for m in months]].sum()] for k in keys])
    lc, lf = mr[:, 0] < 0, mr[:, 1] < 0
    out['months_book'] = len(keys)
    out['months_crack_loss'] = int(lc.sum())
    out['months_fly_loss'] = int(lf.sum())
    out['months_both_loss'] = int((lc & lf).sum())
    out['months_both_loss_if_independent'] = float(lc.sum() * lf.sum() / len(keys))
    worst_c = int(np.argmin(mr[:, 0]))
    worst_f = int(np.argmin(mr[:, 1]))
    out['worst_crack_month'] = {'month': keys[worst_c], 'crack_r': float(mr[worst_c, 0]), 'fly_r': float(mr[worst_c, 1])}
    out['worst_fly_month'] = {'month': keys[worst_f], 'crack_r': float(mr[worst_f, 0]), 'fly_r': float(mr[worst_f, 1])}
    # Year by year.
    rows = []
    for y in sorted(set(years)):
        m = np.array([yy == y for yy in years])
        wm = np.array([yy == y for yy in wk_years])
        sy = stream.year == y
        rows.append({
            'year': y, 'days': int(m.sum()),
            'daily': pearson(c[m], f[m]) if m.sum() >= 30 else None,
            'weekly': pearson(wc[wm], wf[wm]) if wm.sum() >= 20 else None,
            'days_both_open': int(both[sy].sum()) if sy.any() else 0,
            'days_any_open': int(either[sy].sum()) if sy.any() else 0,
            'stress': y in stress_years,
        })
    out['by_year'] = rows
    out['stress'] = [r for r in rows if r['stress']]
    # Correlation in the stress years pooled, against all other years pooled.
    sm = np.array([y in stress_years for y in years])
    out['daily_stress_pooled'] = pearson(c[sm], f[sm])
    out['daily_other_pooled'] = pearson(c[~sm], f[~sm])
    wsm = np.array([y in stress_years for y in wk_years])
    out['weekly_stress_pooled'] = pearson(wc[wsm], wf[wsm])
    out['weekly_other_pooled'] = pearson(wc[~wsm], wf[~wsm])
    return out


def sharpe(x) -> float:
    """Annualised Sharpe of a daily series, from all days (flat days included)."""
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252.0))


def sharpe_range(years, x, seed=5, reps=2000):
    """95% range of the Sharpe ratio from resampling whole calendar years."""
    ys = sorted(set(years.tolist()))
    mom = {}
    for y in ys:
        v = x[years == y]
        mom[y] = np.array([len(v), v.sum(), (v * v).sum()])
    vals = []
    for r in range(reps):
        rng = Mulberry32(path_seed(seed, r))
        tot = np.zeros(3)
        for _ in ys:
            tot = tot + mom[ys[int(rng() * len(ys))]]
        n, s1, s2 = tot
        m = s1 / n
        var = (s2 - n * m * m) / (n - 1)
        vals.append(m / math.sqrt(var) * math.sqrt(252.0) if var > 0 else float('nan'))
    vals = np.array(vals)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def matched_weight(xc, xf) -> float:
    """Weight a for both rules so that the book's daily volatility equals the average of the two alone."""
    sc, sf = xc.std(ddof=1), xf.std(ddof=1)
    sb = (xc + xf).std(ddof=1)
    return float(0.5 * (sc + sf) / sb)


def book_report(data, stream) -> dict:
    """Sharpe of each rule and of the 50/50 book, by two routes, over the same window."""
    xc, xf = stream.x[0], stream.x[1]
    book = 0.5 * xc + 0.5 * xf
    yrs = stream.year
    # Route 2: no R, no stops. Raw daily P&L of each rule divided by its own volatility, then 50/50.
    idx = [data.index[d] for d in stream.dates]
    pc = np.array([0.0 if data.pnl['crack'][i] is None else data.pnl['crack'][i] for i in idx])
    pf = np.array([0.0 if data.pnl['fly'][i] is None else data.pnl['fly'][i] for i in idx])
    vs = 0.5 * pc / pc.std(ddof=1) + 0.5 * pf / pf.std(ddof=1)
    out = {
        'window': [stream.dates[0], stream.dates[-1]], 'days': stream.n,
        'sharpe': {'crack': sharpe(xc), 'fly': sharpe(xf), 'book': sharpe(book)},
        'sharpe_range': {'crack': sharpe_range(yrs, xc), 'fly': sharpe_range(yrs, xf), 'book': sharpe_range(yrs, book)},
        'sharpe_raw_pnl': {'crack': sharpe(pc), 'fly': sharpe(pf), 'book_vol_scaled': sharpe(vs)},
        'vol_daily_r': {'crack': float(xc.std(ddof=1)), 'fly': float(xf.std(ddof=1)), 'book': float(book.std(ddof=1))},
        'vol_ratio_book_over_avg': float(book.std(ddof=1) / (0.5 * (xc.std(ddof=1) + xf.std(ddof=1)))),
        'matched_weight': matched_weight(xc, xf),
    }
    return out
