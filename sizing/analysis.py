"""The experiments behind the page, built from the pieces in the other modules.

Everything here is a loop over settings (risk per trade, resampling mode,
challenge rules) and a summary of the paths that come out. Nothing here changes
the model: that lives in simulate.py and sim_ref.py.
"""
from __future__ import annotations

import numpy as np

from . import kelly as K
from . import simulate as V
from . import streams as S

# Risk per trade tried on the page and in the tables (fraction of the account lost if the stop is hit).
F_GRID = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
          0.06, 0.07, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25]
TRADING_DAYS = 252
MODES = ['iid', 'block', 'year']
BOOKS = {
    'crack': (['crack'], [1.0]),
    'fly': (['fly'], [1.0]),
    'both': (['crack', 'fly'], [0.5, 0.5]),
}

# Typical rules of a funded-account challenge. Firms differ; these are round
# numbers in the range that is common, not the terms of any one firm.
REF = dict(target=0.10, dd=0.10, static=True, daily=0.05, time_limit=TRADING_DAYS, funded=TRADING_DAYS)
GROWTH_DAYS = 5 * TRADING_DAYS
DD_LIMIT = 0.10


def challenge_rows(stream, units, w, fgrid, seed, n, rules=None):
    rules = rules or REF
    rows = []
    for f in fgrid:
        r = V.sim_challenge(stream, units, w, f, seed, n, rules['target'], rules['dd'], rules['static'],
                            rules['daily'], rules['time_limit'], rules['funded'])
        s = V.summarize_challenge(r)
        passed = r['outcome'] == V.PASS
        survive = passed & (r['funded'] == 2)
        s['p_pass_survive'] = float(survive.mean())
        s['f'] = f
        rows.append(s)
    return rows


def growth_rows(stream, units, w, fgrid, seed, n, days=GROWTH_DAYS, limit=DD_LIMIT):
    rows = []
    for f in fgrid:
        s = V.summarize_growth(V.sim_growth(stream, units, w, f, days, seed, n), limit)
        s['f'] = f
        rows.append(s)
    return rows


def peak(rows, key):
    """The row with the highest `key` (first one on a tie)."""
    best = max(rows, key=lambda r: (r[key] if r[key] is not None else -1.0))
    return {'f': best['f'], key: best[key]}


def kelly_summary(trades, seed=3, reps=2000):
    """Kelly fraction of the trades, its error by two bootstraps, and what fractions of it give."""
    r = [t['r'] for t in trades]
    full = K.kelly_fraction(r)
    groups = K.groups_by_year(trades)
    out = {'full': full, 'half': full / 2, 'quarter': full / 4,
           'growth_at_full': K.growth_rate(r, full), 'growth_at_half': K.growth_rate(r, full / 2),
           'growth_at_2x': K.growth_rate(r, 2 * full), 'growth_at_3x': K.growth_rate(r, 3 * full),
           'boot': {}}
    # Growth is zero again at some multiple of the Kelly fraction: find it by bisection on the log growth.
    lo, hi = full, 1.0 / -min(min(r), -1e-9) * (1 - 1e-9)
    if K.growth_rate(r, hi) > 0:
        out['break_even_f'] = hi
    else:
        for _ in range(80):
            mid = (lo + hi) / 2
            if K.growth_rate(r, mid) > 0:
                lo = mid
            else:
                hi = mid
        out['break_even_f'] = (lo + hi) / 2
    out['break_even_multiple'] = out['break_even_f'] / full if full > 0 else None
    for mode in ('iid', 'year'):
        v = K.kelly_bootstrap(groups, mode, seed, reps)
        hist, edges = np.histogram(v, bins=np.arange(0.0, 0.62, 0.02))
        out['boot'][mode] = {
            'p05': float(np.percentile(v, 5)), 'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
            'p025': float(np.percentile(v, 2.5)), 'p975': float(np.percentile(v, 97.5)),
            'share_zero': float(np.mean(v <= 0.0)), 'share_above_full': float(np.mean(v > full)),
            'hist': [int(h) for h in hist], 'edges': [round(float(e), 4) for e in edges],
            'reps': reps,
        }
    return out


def walk_forward(trades, start=20, fractions=(1.0, 0.5, 0.25), fixed=(0.01, 0.05)):
    """Kelly estimated only from earlier trades, then used on the next one.

    For every trade after the first `start`, the Kelly fraction is estimated on
    the trades before it, a share of it is risked on this trade, and the log
    growth is recorded. Compared with risking a fixed fraction, and with the
    fraction estimated on all trades at once (which uses the future).
    """
    r = [t['r'] for t in trades]
    n = len(r)
    full = K.kelly_fraction(r)
    out = {'start': start, 'trades_used': n - start, 'in_sample_full': full, 'rows': []}

    def run(fs):
        E, peak, mdd = 1.0, 1.0, 0.0
        for f, v in zip(fs, r[start:]):
            E = max(E * (1.0 + f * v), 0.0)
            peak = max(peak, E)
            mdd = max(mdd, 1.0 - E / peak)
        return E, mdd

    est = [K.kelly_fraction(r[:k]) for k in range(start, n)]
    for c in fractions:
        E, mdd = run([c * e for e in est])
        out['rows'].append({'label': f'{c:g} x Kelly, estimated on earlier trades', 'final': E, 'maxdd': mdd,
                            'mean_f': float(np.mean([c * e for e in est]))})
    for f in fixed:
        E, mdd = run([f] * (n - start))
        out['rows'].append({'label': f'fixed {f:g}', 'final': E, 'maxdd': mdd, 'mean_f': f})
    E, mdd = run([full] * (n - start))
    out['rows'].append({'label': 'Kelly estimated on all trades (uses the future)', 'final': E, 'maxdd': mdd, 'mean_f': full})
    out['estimates'] = {'min': float(min(est)), 'median': float(np.median(est)), 'max': float(max(est)),
                        'first': est[0], 'last': est[-1]}
    return out


def halves(trades):
    """First half of the trades against the second half."""
    from .rstats import r_stats
    h = len(trades) // 2
    res = {}
    for name, part in (('first', trades[:h]), ('second', trades[h:])):
        r = [t['r'] for t in part]
        st = r_stats(r)
        res[name] = {'n': st['n'], 'from': part[0]['entry'], 'to': part[-1]['exit'], 'mean': st['mean'],
                     'win_rate': st['win_rate'], 'kelly': K.kelly_fraction(r)}
    return res


def variant_table(data, rule, variants, fref, seed, n):
    """Mean R, Kelly and challenge results when trades are removed or made worse."""
    rows = []
    for label, kw in variants:
        st = S.build_streams(data, [rule], **kw)
        tr = st.trades[0]
        r = [t['r'] for t in tr]
        u = S.make_units(st, 'iid')
        c = challenge_rows(st, u, [1.0], [fref], seed, n)[0]
        rows.append({'label': label, 'n': len(r), 'mean_r': float(np.mean(r)), 'kelly': K.kelly_fraction(r),
                     'pass': c['pass'], 'pass_survive': c['p_pass_survive'], 'fail_dd': c['fail_dd'],
                     'fail_daily': c['fail_daily']})
    return rows


def best_worst_years(trades):
    by = {}
    for t in trades:
        by[t['year']] = by.get(t['year'], 0.0) + t['r']
    best = max(by, key=lambda y: by[y])
    worst = min(by, key=lambda y: by[y])
    return best, worst, by
