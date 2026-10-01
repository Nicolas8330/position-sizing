"""Reference values for the JavaScript port (data/ref_check.json).

The site runs the simulation in JavaScript. scripts/verify-position-sizing.mjs
in the site repository runs the JavaScript on the same data and seeds and fails
if any number differs by more than 1e-9 (relative). The numbers here are full
precision and come from the numpy version of the simulation, which
tests/test_simulate.py has already shown equal to the reference loops.
"""
from __future__ import annotations

import numpy as np

from . import kelly as K
from . import rstats as RS
from . import simulate as V
from . import stress as ST
from . import streams as S
from .analysis import BOOKS
from .rng import Mulberry32, path_seed

CHALLENGES = [
    dict(target=0.10, dd=0.10, static=True, daily=0.05, time_limit=252, funded=252),
    dict(target=0.08, dd=0.05, static=False, daily=0.0, time_limit=126, funded=252),
    dict(target=0.10, dd=0.08, static=True, daily=0.04, time_limit=504, funded=126),
]


def _num(x):
    if isinstance(x, dict):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def build(data) -> dict:
    out = {'tolerance': 1e-9}
    g = Mulberry32(12345)
    out['rng'] = {'mulberry32_12345': [g() for _ in range(6)],
                  'path_seeds_2026': [path_seed(2026, k) for k in range(6)]}

    streams = []
    for rules, opts in ((['crack'], {}), (['fly'], {}), (['crack', 'fly'], {}), (['crack'], {'haircut': 0.1}),
                        (['fly'], {'exclude_years': [2008]})):
        st = S.build_streams(data, rules, **{k: (tuple(v) if k == 'exclude_years' else v) for k, v in opts.items()})
        streams.append({
            'rules': rules, 'opts': opts, 'n': st.n, 'first': st.dates[0], 'last': st.dates[-1],
            'trades': [len(t) for t in st.trades],
            'sum_x': [float(x.sum()) for x in st.x],
            'sum_x2': [float((x * x).sum()) for x in st.x],
            'sum_st': [int(s.sum()) for s in st.st],
            'sum_r': [float(sum(t['r'] for t in tr)) for tr in st.trades],
        })
    out['streams'] = streams

    growth, challenge = [], []
    for name, (rules, w) in BOOKS.items():
        st = S.build_streams(data, rules)
        for mode in ('iid', 'block', 'year'):
            u = S.make_units(st, mode)
            for f, days, seed, n in ((0.02, 504, 2026, 400), (0.10, 1260, 7, 300)):
                res = V.sim_growth(st, u, w, f, days, seed, n)
                growth.append({
                    'book': name, 'rules': rules, 'w': w, 'mode': mode, 'f': f, 'days': days, 'seed': seed,
                    'paths': n, 'limit': 0.10, 'summary': _num(V.summarize_growth(res, 0.10)),
                    'first_final': res['final'][:5].tolist(), 'first_maxdd': res['maxdd'][:5].tolist(),
                    'cp_mean': res['cp'].mean(axis=0).tolist()[::12],
                })
            for c, cfg in enumerate(CHALLENGES):
                for f in (0.03, 0.10):
                    res = V.sim_challenge(st, u, w, f, 2026 + c, 400, cfg['target'], cfg['dd'], cfg['static'],
                                          cfg['daily'], cfg['time_limit'], cfg['funded'])
                    challenge.append({
                        'book': name, 'rules': rules, 'w': w, 'mode': mode, 'f': f, 'seed': 2026 + c, 'paths': 400,
                        'cfg': cfg, 'summary': _num(V.summarize_challenge(res)),
                        'first_outcome': res['outcome'][:8].tolist(), 'first_rtrades': res['rtrades'][:8].tolist(),
                    })
    out['growth'] = growth
    out['challenge'] = challenge

    kel = {}
    for rule in ('crack', 'fly'):
        st = S.build_streams(data, [rule])
        tr = st.trades[0]
        r = [t['r'] for t in tr]
        groups = K.groups_by_year(tr)
        boot = {}
        for mode in ('iid', 'year'):
            v = K.kelly_bootstrap(groups, mode, 3, 100)
            boot[mode] = {'mean': float(v.mean()), 'p05': V.quantile(v, 0.05), 'p50': V.quantile(v, 0.5),
                          'p95': V.quantile(v, 0.95), 'first': v[:5].tolist()}
        kel[rule] = {'full': K.kelly_fraction(r), 'daily': K.kelly_daily(st.x[0]), 'boot': boot,
                     'growth_at_half': K.growth_rate(r, K.kelly_fraction(r) / 2)}
    both = S.build_streams(data, ['crack', 'fly'])
    kel['both_daily'] = K.kelly_daily(K.book_daily(both, [0.5, 0.5]))
    out['kelly'] = kel
    out['rstats'] = {rule: _num(RS.r_stats([t['r'] for t in S.build_streams(data, [rule]).trades[0]]))
                     for rule in ('crack', 'fly')}

    replay = []
    for name, (rules, w) in BOOKS.items():
        st = S.build_streams(data, rules)
        for f in (0.03, 0.10):
            ys = ST.replay_years(st, w, f)
            worst = min(ys, key=lambda y: y['ret'])
            wr = ST.worst_run(st, w, f, 10)
            hist = ST.replay_history(st, w, f)
            replay.append({
                'book': name, 'f': f,
                'years': [{'year': y['year'], 'ret': y['ret'], 'maxdd': y['maxdd'], 'worst_day': y['worst_day']} for y in ys],
                'worst_year': worst['year'],
                'run': {'ret': wr['ret'], 'maxdd': wr['maxdd'], 'first': wr['trades'][0]['entry'],
                        'last': wr['trades'][-1]['exit'], 'r': [t['r'] for t in wr['trades']],
                        'balance': wr['balance']},
                'history': _num(hist),
            })
    out['replay'] = replay
    return _num(out)
