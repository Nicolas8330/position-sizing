"""The simulation written one path at a time, in plain loops.

This is the reference. It is slow but easy to read, and it is written line for
line like the JavaScript on the site (public/js/position-sizing-engine.js).
simulate.py does the same work for all paths at once with numpy, so that
results.json takes minutes and not hours; tests/test_simulate.py checks that
the two agree to the last digit on a few hundred paths.

The account, in words
---------------------
Start with a balance of 1. Each day, take a piece of history from the stream
(see make_units), one day at a time. A trade that opens with balance B risks a
fraction f of B if its stop is hit: it earns f * B * x on each day it is open,
where x is that day's R. Weights w (one per rule) split the risk between the
rules: rule j risks f * w_j per trade. All balances are measured in units of
the starting balance, and a balance cannot go below zero.

What "open a trade" means when history is cut
---------------------------------------------
If a piece of history starts in the middle of a trade (a year block starts on
1 January, or a block starts anywhere), the trade is treated as opened on that
day with the balance of that day. That is a small approximation and it only
happens at the edges of the pieces.
"""
from __future__ import annotations

from .rng import Mulberry32, path_seed

# outcome codes of a challenge
TIMEOUT, PASS, FAIL_DD, FAIL_DAILY = 0, 1, 2, 3


class Walker:
    """Hands out day numbers of the stream, one piece of history after another."""

    def __init__(self, units, rng):
        self.starts = units.starts
        self.lens = units.lens
        self.m = len(units.starts)
        self.rng = rng
        self.left = 0
        self.pos = 0
        self.first = False

    def next(self):
        if self.left == 0:
            u = int(self.rng() * self.m)
            self.pos = int(self.starts[u])
            self.left = int(self.lens[u])
            self.first = True
        d = self.pos
        self.pos += 1
        self.left -= 1
        fresh = self.first
        self.first = False
        return d, fresh


def apply_day(stream, fw, base, E, d, fresh):
    """New balance after day d. `base` (balance when each rule's trade opened) is updated."""
    E2 = E
    for j in range(len(fw)):
        s = int(stream.st[j][d])
        if s == 1 or (fresh and s >= 1):
            base[j] = E
        if s >= 1:
            E2 = E2 + fw[j] * base[j] * float(stream.x[j][d])
    if E2 < 0.0:
        E2 = 0.0
    return E2


def checkpoints(days: int, k: int = 60) -> list:
    """Days (0 to `days`) at which the balance is recorded for the fan chart."""
    return [(i * days) // k for i in range(k + 1)]


def growth_path(stream, units, w, f, days, seed, k, cps):
    rng = Mulberry32(path_seed(seed, k))
    walker = Walker(units, rng)
    fw = [f * wj for wj in w]
    base = [1.0] * len(fw)
    E, peak, maxdd = 1.0, 1.0, 0.0
    uw = cur = longest = 0
    rec = {0: 1.0}
    want = set(cps)
    for day in range(1, days + 1):
        d, fresh = walker.next()
        E = apply_day(stream, fw, base, E, d, fresh)
        if E > peak:
            peak = E
        dd = 1.0 - E / peak
        if dd > maxdd:
            maxdd = dd
        if E < peak:
            uw += 1
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
        if day in want:
            rec[day] = E
    return E, maxdd, uw, longest, [rec[c] for c in cps]


def challenge_path(stream, units, w, f, seed, k, target, dd, static, daily, time_limit, funded_days):
    rng = Mulberry32(path_seed(seed, k))
    walker = Walker(units, rng)
    fw = [f * wj for wj in w]
    nr = len(fw)
    base = [1.0] * nr
    E, peak = 1.0, 1.0
    phase = 0
    trades = 0
    outcome, rday, rtrades = TIMEOUT, time_limit, 0
    pday = 0
    funded = 0          # 0 not passed, 1 breached after passing, 2 survived the funded period
    fday = 0
    for day in range(1, time_limit + funded_days + 1):
        d, fresh = walker.next()
        if phase == 0:
            for j in range(nr):
                if stream.st[j][d] == 1:
                    trades += 1
        Eprev = E
        E = apply_day(stream, fw, base, E, d, fresh)
        if E > peak:
            peak = E
        breach_daily = daily > 0 and (Eprev - E) >= daily * Eprev
        floor = (1.0 - dd) if static else peak * (1.0 - dd)
        breach_dd = E <= floor
        if phase == 0:
            if breach_daily:
                outcome, rday, rtrades = FAIL_DAILY, day, trades
                break
            if breach_dd:
                outcome, rday, rtrades = FAIL_DD, day, trades
                break
            if E >= 1.0 + target:
                outcome, rday, rtrades = PASS, day, trades
                pday = day
                for j in range(nr):
                    base[j] = base[j] / E
                E = 1.0
                peak = 1.0
                phase = 1
                continue
            if day == time_limit:
                outcome, rday, rtrades = TIMEOUT, day, trades
                break
        else:
            if breach_daily or breach_dd:
                funded, fday = 1, day - pday
                break
            if day - pday == funded_days:
                funded = 2
                break
    if outcome == PASS and funded == 0:
        funded = 2
    return outcome, rday, rtrades, funded, fday
