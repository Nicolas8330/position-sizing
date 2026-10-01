"""The simulation for thousands of paths at once (numpy).

Same rules, same random numbers and same order of arithmetic as sim_ref.py,
which is written one path at a time so that it can be read. Here every step of
the loop runs on all paths together. Each path still has its own generator, so
the answer for path k is the same in both files, and in the JavaScript.

Two things are simulated.

growth      the balance of an account that risks a fraction f per trade for a
            fixed number of days: final balance, deepest fall from a peak
            (max drawdown), days under water.
challenge   a funded-account challenge: reach a profit target before the
            balance falls too far (a maximum drawdown) or drops too much in one
            day (a daily loss limit), inside a time limit. A path that passes
            goes on for a funded year, and we note whether it breaches then.
"""
from __future__ import annotations

import numpy as np

from .rng import path_seeds, step_vector
from .sim_ref import TIMEOUT, PASS, FAIL_DD, FAIL_DAILY, checkpoints  # noqa: F401


def quantile(a, q: float) -> float:
    """Linear-interpolation quantile (the same formula in the JavaScript)."""
    s = np.sort(np.asarray(a, dtype=np.float64))
    n = len(s)
    pos = (n - 1) * q
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


class _Paths:
    """The state every path needs to draw its next piece of history."""

    def __init__(self, units, seed, n_paths):
        self.units = units
        self.m = len(units.starts)
        self.a = path_seeds(seed, n_paths)
        self.left = np.zeros(n_paths, dtype=np.int64)
        self.pos = np.zeros(n_paths, dtype=np.int64)
        self.first = np.zeros(n_paths, dtype=bool)

    def next_day(self):
        need = self.left == 0
        if need.any():
            self.a, u = step_vector(self.a, need)
            idx = (u * self.m).astype(np.int64)
            self.pos = np.where(need, self.units.starts[idx], self.pos)
            self.left = np.where(need, self.units.lens[idx], self.left)
            self.first = np.where(need, True, self.first)
        d = self.pos
        self.pos = self.pos + 1
        self.left = self.left - 1
        fresh = self.first
        self.first = np.zeros_like(self.first)
        return d, fresh


def _apply(stream, fw, base, E, d, fresh):
    E2 = E
    for j in range(len(fw)):
        s = stream.st[j][d]
        setb = (s == 1) | (fresh & (s >= 1))
        base[j] = np.where(setb, E, base[j])
        E2 = E2 + fw[j] * base[j] * stream.x[j][d]
    return np.maximum(E2, 0.0)


def sim_growth(stream, units, w, f, days, seed, n_paths, k_cp=60):
    """Paths of `days` trading days. Returns arrays over paths."""
    paths = _Paths(units, seed, n_paths)
    fw = [f * wj for wj in w]
    base = [np.ones(n_paths) for _ in fw]
    E = np.ones(n_paths)
    peak = np.ones(n_paths)
    maxdd = np.zeros(n_paths)
    uw = np.zeros(n_paths, dtype=np.int64)
    cur = np.zeros(n_paths, dtype=np.int64)
    longest = np.zeros(n_paths, dtype=np.int64)
    cps = checkpoints(days, k_cp)
    cols = {}
    for c, day in enumerate(cps):
        cols.setdefault(day, []).append(c)
    rec = np.ones((n_paths, len(cps)))
    for day in range(1, days + 1):
        d, fresh = paths.next_day()
        E = _apply(stream, fw, base, E, d, fresh)
        peak = np.where(E > peak, E, peak)
        dd = 1.0 - E / peak
        maxdd = np.where(dd > maxdd, dd, maxdd)
        under = E < peak
        uw = uw + under
        cur = np.where(under, cur + 1, 0)
        longest = np.where(cur > longest, cur, longest)
        if day in cols:
            for c in cols[day]:
                rec[:, c] = E
    return {'final': E, 'maxdd': maxdd, 'uw': uw, 'longest': longest, 'cp': rec, 'cps': cps}


def sim_challenge(stream, units, w, f, seed, n_paths, target, dd, static, daily,
                  time_limit, funded_days):
    paths = _Paths(units, seed, n_paths)
    fw = [f * wj for wj in w]
    nr = len(fw)
    base = [np.ones(n_paths) for _ in fw]
    E = np.ones(n_paths)
    peak = np.ones(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    phase = np.zeros(n_paths, dtype=np.int8)
    trades = np.zeros(n_paths, dtype=np.int64)
    outcome = np.zeros(n_paths, dtype=np.int8)
    rday = np.full(n_paths, time_limit, dtype=np.int64)
    rtrades = np.zeros(n_paths, dtype=np.int64)
    pday = np.zeros(n_paths, dtype=np.int64)
    funded = np.zeros(n_paths, dtype=np.int8)
    fday = np.zeros(n_paths, dtype=np.int64)
    total = time_limit + funded_days
    for day in range(1, total + 1):
        if not alive.any():
            break
        d, fresh = paths.next_day()
        p0 = alive & (phase == 0)
        p1 = alive & (phase == 1)
        cnt = np.zeros(n_paths, dtype=np.int64)
        for j in range(nr):
            cnt = cnt + (stream.st[j][d] == 1)
        trades = trades + np.where(p0, cnt, 0)
        Eprev = E
        E2 = _apply(stream, fw, base, E, d, fresh)
        E = np.where(alive, E2, E)
        peak = np.where(E > peak, E, peak)
        if daily > 0:
            bd = (Eprev - E) >= daily * Eprev
        else:
            bd = np.zeros(n_paths, dtype=bool)
        floor = (1.0 - dd) if static else peak * (1.0 - dd)
        bdd = E <= floor
        f_daily = p0 & bd
        f_dd = p0 & ~bd & bdd
        ps = p0 & ~bd & ~bdd & (E >= 1.0 + target)
        to = p0 & ~bd & ~bdd & ~(E >= 1.0 + target) & (day == time_limit)
        outcome = np.where(f_daily, FAIL_DAILY, np.where(f_dd, FAIL_DD, np.where(ps, PASS, outcome))).astype(np.int8)
        done0 = f_daily | f_dd | ps | to
        rday = np.where(done0, day, rday)
        rtrades = np.where(done0, trades, rtrades)
        pday = np.where(ps, day, pday)
        fb = p1 & (bd | bdd)
        fs = p1 & ~fb & ((day - pday) == funded_days)
        funded = np.where(fb, 1, np.where(fs, 2, funded)).astype(np.int8)
        fday = np.where(fb, day - pday, fday)
        for j in range(nr):
            base[j] = np.where(ps, base[j] / E, base[j])
        E = np.where(ps, 1.0, E)
        peak = np.where(ps, 1.0, peak)
        phase = np.where(ps, 1, phase).astype(np.int8)
        alive = alive & ~(f_daily | f_dd | to | fb | fs)
    funded = np.where((outcome == PASS) & (funded == 0), 2, funded).astype(np.int8)
    return {'outcome': outcome, 'rday': rday, 'rtrades': rtrades, 'funded': funded, 'fday': fday}


def summarize_growth(res, limit=0.10) -> dict:
    final, maxdd = res['final'], res['maxdd']
    n = len(final)
    return {
        'paths': n,
        'final_p05': quantile(final, 0.05), 'final_p25': quantile(final, 0.25),
        'final_median': quantile(final, 0.5), 'final_p75': quantile(final, 0.75),
        'final_p95': quantile(final, 0.95),
        'p_loss': float(np.mean(final < 1.0)),
        'p_half': float(np.mean(final < 0.5)),
        'dd_median': quantile(maxdd, 0.5), 'dd_p95': quantile(maxdd, 0.95),
        'p_dd_limit': float(np.mean(maxdd >= limit)),
        'uw_median': quantile(res['uw'].astype(np.float64), 0.5),
        'longest_median': quantile(res['longest'].astype(np.float64), 0.5),
        'mean_log_final': float(np.mean(np.log(np.maximum(final, 1e-12)))),
    }


def summarize_challenge(res) -> dict:
    o = res['outcome']
    n = len(o)
    passed = o == PASS
    n_pass = int(passed.sum())
    return {
        'paths': n,
        'pass': float(passed.mean()),
        'fail_dd': float((o == FAIL_DD).mean()),
        'fail_daily': float((o == FAIL_DAILY).mean()),
        'timeout': float((o == TIMEOUT).mean()),
        'mean_trades': float(res['rtrades'].mean()),
        'mean_days_pass': float(res['rday'][passed].mean()) if n_pass else None,
        'p_breach_after_pass': float((res['funded'][passed] == 1).mean()) if n_pass else None,
    }
