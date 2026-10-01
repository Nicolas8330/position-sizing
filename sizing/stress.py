"""Replays of what really happened, at the risk per trade the reader chooses.

No random draws here. The real history is run through the same account rule as
the simulation, so the question "what would the worst year have done to an
account that risked 3% per trade?" has one exact answer.
"""
from __future__ import annotations

import numpy as np

from .sim_ref import apply_day


def replay_years(stream, w, f, want_path=None) -> list:
    """Every calendar year of the sample, run from a balance of 1."""
    fw = [f * wj for wj in w]
    out = []
    a = 0
    n = stream.n
    for k in range(1, n + 1):
        if k < n and stream.year[k] == stream.year[a]:
            continue
        base = [1.0] * len(fw)
        E, peak, mdd, worst_day = 1.0, 1.0, 0.0, 0.0
        path = [1.0]
        for d in range(a, k):
            Eprev = E
            E = apply_day(stream, fw, base, E, d, d == a)
            if E > peak:
                peak = E
            dd = 1.0 - E / peak
            if dd > mdd:
                mdd = dd
            dr = E / Eprev - 1.0 if Eprev > 0 else 0.0
            if dr < worst_day:
                worst_day = dr
            path.append(E)
        out.append({'year': int(stream.year[a]), 'days': k - a, 'ret': E - 1.0, 'maxdd': mdd,
                    'worst_day': worst_day, 'path': path if want_path == int(stream.year[a]) else None})
        a = k
    return out


def replay_history(stream, w, f) -> dict:
    """The whole sample as one path, from a balance of 1."""
    fw = [f * wj for wj in w]
    base = [1.0] * len(fw)
    E, peak, mdd, worst_day = 1.0, 1.0, 0.0, 0.0
    cur = longest = 0
    for d in range(stream.n):
        Eprev = E
        E = apply_day(stream, fw, base, E, d, d == 0)
        if E > peak:
            peak = E
        dd = 1.0 - E / peak
        if dd > mdd:
            mdd = dd
        dr = E / Eprev - 1.0 if Eprev > 0 else 0.0
        if dr < worst_day:
            worst_day = dr
        if E < peak:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return {'final': E, 'maxdd': mdd, 'worst_day': worst_day, 'longest_uw': longest}


def trade_sequence(stream, w) -> list:
    """Every trade of the book in the order it closed (ties: rules in book order)."""
    rows = []
    for j, tr in enumerate(stream.trades):
        for t in tr:
            rows.append({**t, 'rule': stream.rules[j], 'w': w[j], 'rule_order': j})
    rows.sort(key=lambda t: (t['j'], t['rule_order']))
    return rows


def worst_run(stream, w, f, n=10) -> dict:
    """The n trades in a row with the lowest compounded return at risk f."""
    rows = trade_sequence(stream, w)
    length = min(n, len(rows))
    best = None
    for a in range(0, len(rows) - length + 1):
        E, peak, mdd = 1.0, 1.0, 0.0
        for k in range(a, a + length):
            E = E * (1.0 + f * rows[k]['w'] * rows[k]['r'])
            if E < 0:
                E = 0.0
            if E > peak:
                peak = E
            dd = 1.0 - E / peak
            if dd > mdd:
                mdd = dd
        if best is None or E - 1.0 < best['ret']:
            best = {'start': a, 'ret': E - 1.0, 'maxdd': mdd}
    trades = rows[best['start']:best['start'] + length]
    E = 1.0
    balance = []
    for t in trades:
        E = E * (1.0 + f * t['w'] * t['r'])
        if E < 0:
            E = 0.0
        balance.append(E)
    return {'ret': best['ret'], 'maxdd': best['maxdd'], 'trades': trades, 'balance': balance}
