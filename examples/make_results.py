"""Every number the site quotes about position sizing, in one run.

    python examples/make_results.py            # about ten minutes
    python examples/make_results.py --quick    # 1,000 paths, for a fast look

Reads data/trades.json and data/daily_pnl.json (copied from the
relative-value-strategies repository, see data/PROVENANCE.md), runs the
experiments, and writes

    data/results.json    the single source of every figure on the site
    data/ref_check.json  full-precision reference values for the JavaScript port

Settings are fixed at the top of the file. Nothing is tuned.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sizing import analysis as A  # noqa: E402
from sizing import diversify as DV  # noqa: E402
from sizing import kelly as K  # noqa: E402
from sizing import refcheck  # noqa: E402
from sizing import rstats as RS  # noqa: E402
from sizing import simulate as V  # noqa: E402
from sizing import stress as ST  # noqa: E402
from sizing import streams as S  # noqa: E402

SEED = 2026
N_PATHS = 1000 if '--quick' in sys.argv else 5000
N_TRUST = 1000 if '--quick' in sys.argv else 3000
F_REF = 0.05             # risk per trade used in the leave-out and cost tables
DIVERS_F = [0.02, 0.03, 0.05, 0.075, 0.10]
STRESS_F = [0.01, 0.03, 0.05, 0.10]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def tick(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:6.0f}s] {msg}', flush=True)


def main():
    data = S.load()
    res = {}
    res['meta'] = {
        'generated': dt.date.today().isoformat(), 'seed': SEED, 'paths': N_PATHS, 'trust_paths': N_TRUST,
        'f_grid': A.F_GRID, 'reference_challenge': A.REF, 'growth_days': A.GROWTH_DAYS,
        'drawdown_limit_for_growth': A.DD_LIMIT, 'modes': A.MODES, 'block_days': 63,
        'trading_days_per_year': A.TRADING_DAYS,
        'data': {
            'trades_first': {r: data.trades[r][0]['entry'] for r in ('crack', 'fly')},
            'trades_last': {r: data.trades[r][-1]['exit'] for r in ('crack', 'fly')},
            'daily_last': data.dates[-1], 'daily_first': data.dates[0],
        },
    }

    # 1. The trades, in R.
    rules = {}
    for rule in ('crack', 'fly'):
        st = S.build_streams(data, [rule])
        tr = st.trades[0]
        r = [t['r'] for t in tr]
        years = st.n / A.TRADING_DAYS
        rules[rule] = {
            'first': st.dates[0], 'last': st.dates[-1], 'days': st.n, 'years': years,
            'trades_per_year': len(tr) / years,
            'mean_holding_days': float(np.mean([t['j'] - t['i'] + 1 for t in tr])),
            'share_days_in_trade': float((st.st[0] > 0).mean()),
            'stats': RS.r_stats(r),
            'clustering': RS.clustering(tr),
            'kelly': A.kelly_summary(tr, seed=3, reps=2000),
            'kelly_daily': K.kelly_daily(st.x[0]),
            'walk_forward': A.walk_forward(tr),
            'halves': A.halves(tr),
            'r_sorted': sorted(r),
            'r_by_entry': [{'entry': t['entry'], 'r': t['r']} for t in tr],
        }
        tick(f'{rule}: R distribution, Kelly and bootstrap')
    res['rules'] = rules

    # 2 and 3. Growth against risk, and the challenge, for each book and each resampling mode.
    growth, challenge, peaks = {}, {}, {}
    for name, (rr, w) in A.BOOKS.items():
        st = S.build_streams(data, rr)
        growth[name], challenge[name], peaks[name] = {}, {}, {}
        for mode in A.MODES:
            u = S.make_units(st, mode)
            growth[name][mode] = A.growth_rows(st, u, w, A.F_GRID, SEED, N_PATHS)
            challenge[name][mode] = A.challenge_rows(st, u, w, A.F_GRID, SEED, N_PATHS)
            peaks[name][mode] = {'pass': A.peak(challenge[name][mode], 'pass'),
                                 'pass_survive': A.peak(challenge[name][mode], 'p_pass_survive')}
            tick(f'{name} {mode}: growth and challenge sweeps')
        if name == 'both':
            res['rules_both'] = {'days': st.n, 'first': st.dates[0], 'last': st.dates[-1],
                                 'trades': [len(t) for t in st.trades],
                                 'kelly_daily_book': K.kelly_daily(K.book_daily(st, w))}
    res['growth'], res['challenge'], res['peaks'] = growth, challenge, peaks

    # What the naive (independent draws) simulation misses, at a few risk levels.
    cmp_rows = []
    for name, (rr, w) in A.BOOKS.items():
        for f in (0.02, 0.03, 0.05):
            row = {'book': name, 'f': f}
            for mode in A.MODES:
                g = next(x for x in growth[name][mode] if x['f'] == f)
                c = next(x for x in challenge[name][mode] if x['f'] == f)
                row[mode] = {k: g[k] for k in ('dd_median', 'dd_p95', 'p_dd_limit', 'uw_median', 'longest_median',
                                               'final_p05', 'final_median')}
                row[mode].update({'pass': c['pass'], 'p_pass_survive': c['p_pass_survive']})
            cmp_rows.append(row)
    res['mode_comparison'] = cmp_rows

    # 4. Diversification, over one common window (the two-rule book window).
    book = S.build_streams(data, ['crack', 'fly'])
    div = {'correlation': DV.correlation_report(data, book), 'book': DV.book_report(data, book)}
    a = div['book']['matched_weight']
    rows = []
    for label, w, rr in (('crack alone', [1.0], ['crack']), ('butterfly alone', [1.0], ['fly']),
                         ('both, half the risk each', [0.5, 0.5], ['crack', 'fly']),
                         ('both, equal volatility', [a, a], ['crack', 'fly'])):
        st = S.build_streams(data, rr, start=book.dates[0], end=book.dates[-1])
        u = S.make_units(st, 'block')
        ch = A.challenge_rows(st, u, w, DIVERS_F, SEED, N_PATHS)
        gr = A.growth_rows(st, u, w, DIVERS_F, SEED, N_PATHS)
        hist = {str(f): ST.replay_history(st, w, f) for f in DIVERS_F}
        rows.append({'label': label, 'weights': w, 'challenge': ch, 'growth': gr, 'history': hist})
    div['compare'] = rows
    res['diversification'] = div
    tick('diversification')

    # 5. Stress replays.
    stress = {}
    for name, (rr, w) in A.BOOKS.items():
        st = S.build_streams(data, rr)
        stress[name] = {}
        for f in STRESS_F:
            ys = ST.replay_years(st, w, f)
            worst = min(ys, key=lambda y: y['ret'])
            wr = ST.worst_run(st, w, f, 10)
            stress[name][str(f)] = {
                'worst_year': {k: worst[k] for k in ('year', 'ret', 'maxdd', 'worst_day', 'days')},
                'years_losing': sum(1 for y in ys if y['ret'] < 0), 'years': len(ys),
                'worst_run': {'ret': wr['ret'], 'maxdd': wr['maxdd'], 'first': wr['trades'][0]['entry'],
                              'last': wr['trades'][-1]['exit'], 'r': [t['r'] for t in wr['trades']]},
                'history': ST.replay_history(st, w, f),
            }
    res['stress'] = stress
    tick('stress replays')

    # 6. How much to trust it.
    trust = {'settings_tried': 0}
    lo = {}
    for rule in ('crack', 'fly'):
        st = S.build_streams(data, [rule])
        tr = st.trades[0]
        best_y, worst_y, by = A.best_worst_years(tr)
        best_t = max(tr, key=lambda t: t['r'])
        worst_t = min(tr, key=lambda t: t['r'])
        variants = [('all trades', {}),
                    (f'without the best year ({best_y})', {'exclude_years': (best_y,)}),
                    (f'without the worst year ({worst_y})', {'exclude_years': (worst_y,)}),
                    ('without the best trade', {'exclude': [(rule, best_t['number'])]}),
                    ('without the worst trade', {'exclude': [(rule, worst_t['number'])]}),
                    ('without 2008, 2020 and 2022', {'exclude_years': (2008, 2020, 2022)})]
        lo[rule] = A.variant_table(data, rule, variants, F_REF, SEED, N_TRUST)
        trust['settings_tried'] += len(variants)
    trust['leave_out'] = lo
    trust['leave_out_f'] = F_REF
    hc = {}
    for rule in ('crack', 'fly'):
        hc[rule] = A.variant_table(data, rule, [(f'{h:g} R off every trade', {'haircut': h}) for h in (0.0, 0.05, 0.1, 0.2)],
                                   F_REF, SEED, N_TRUST)
        trust['settings_tried'] += 4
    trust['haircut'] = hc

    # Challenge rules: other targets, drawdown limits, time limits, trailing drawdown.
    settings = []
    for target, dd in ((0.08, 0.05), (0.08, 0.10), (0.10, 0.05), (0.10, 0.10)):
        settings.append({'name': f'target {target:.0%}, drawdown {dd:.0%}', **A.REF, 'target': target, 'dd': dd})
    for tl in (63, 756):
        settings.append({'name': f'time limit {tl} days', **A.REF, 'time_limit': tl})
    settings.append({'name': 'trailing drawdown', **A.REF, 'static': False})
    settings.append({'name': 'no daily limit', **A.REF, 'daily': 0.0})
    grid = []
    for cfg in settings:
        for name, (rr, w) in A.BOOKS.items():
            st = S.build_streams(data, rr)
            u = S.make_units(st, 'block')
            rows = A.challenge_rows(st, u, w, A.F_GRID, SEED, N_TRUST, cfg)
            grid.append({'setting': cfg['name'], 'book': name,
                         'peak_pass': A.peak(rows, 'pass'), 'peak_survive': A.peak(rows, 'p_pass_survive'),
                         'pass_at_1pct': next(x for x in rows if x['f'] == 0.01)['pass'],
                         'pass_at_2pct': next(x for x in rows if x['f'] == 0.02)['pass']})
            trust['settings_tried'] += 1
    trust['challenge_rules'] = grid
    tick('challenge rule sensitivity')

    # Monte Carlo error: same setting, different seeds.
    st = S.build_streams(data, ['crack'])
    u = S.make_units(st, 'block')
    seeds = []
    for s in (1, 2, 3, 4, 5):
        c = A.challenge_rows(st, u, [1.0], [F_REF], s, 2000)[0]
        seeds.append({'seed': s, 'pass': c['pass'], 'p_pass_survive': c['p_pass_survive']})
    trust['mc_error'] = {'f': F_REF, 'paths': 2000, 'seeds': seeds,
                         'pass_range': [min(x['pass'] for x in seeds), max(x['pass'] for x in seeds)]}
    res['trust'] = trust
    tick('trust checks')

    (ROOT / 'data' / 'results.json').write_text(json.dumps(clean(res), separators=(',', ':')))
    (ROOT / 'data' / 'ref_check.json').write_text(json.dumps(refcheck.build(data), separators=(',', ':')))
    tick('wrote data/results.json and data/ref_check.json')


if __name__ == '__main__':
    main()
