import json
from pathlib import Path

import numpy as np
import pytest

from conftest import toy_stream
from sizing import diversify as DV
from sizing import rstats as RS
from sizing import sim_ref as R
from sizing import simulate as V
from sizing import streams as S
from sizing import stress as ST

ROOT = Path(__file__).resolve().parents[1]


def test_r_stats_on_a_small_list():
    s = RS.r_stats([1.0, 1.0, -2.0, 0.5])
    assert s['n'] == 4 and s['win_rate'] == 0.75
    assert s['mean'] == pytest.approx(0.125)
    assert s['avg_win'] == pytest.approx(2.5 / 3) and s['avg_loss'] == -2.0
    assert s['worst'] == -2.0 and s['best'] == 1.0
    assert s['share_below_minus1'] == 0.25
    assert s['skew'] < 0                      # one big loss: negative skew
    assert s['sd'] == pytest.approx(float(np.std([1.0, 1.0, -2.0, 0.5], ddof=1)))


def test_real_trades_have_the_shape_the_page_describes(data):
    for rule, n in (('crack', 71), ('fly', 107)):
        r = [t['r'] for t in S.build_streams(data, [rule]).trades[0]]
        s = RS.r_stats(r)
        assert s['n'] == n
        assert s['win_rate'] > 0.7 and s['mean'] > 0.05
        assert s['skew'] < -0.5 and s['worst'] < -1.4 and s['avg_loss'] < -s['avg_win']


def test_clustering_measures_on_alternating_and_bunched_losses():
    alt = [{'r': (-1.0 if i % 2 else 1.0), 'year': 2000 + i // 4} for i in range(40)]
    c = RS.clustering(alt)
    assert c['p_loss_after_loss'] == 0.0 and c['runs_z'] > 3        # too many runs: alternating
    bunched = [{'r': (-1.0 if i < 20 else 1.0), 'year': 2000 + i // 4} for i in range(40)]
    b = RS.clustering(bunched)
    assert b['p_loss_after_loss'] > 0.9 and b['runs_z'] < -3        # one long run of losses


def test_replay_of_one_year_matches_a_hand_calculation():
    st = toy_stream([[0.5, 0.0, -1.0, 0.0, 0.5, 0.5]], [[1, 0, 1, 0, 1, 2]], years=[2019, 2019, 2019, 2020, 2020, 2020])
    ys = ST.replay_years(st, [1.0], 0.1)
    assert [y['year'] for y in ys] == [2019, 2020]
    assert ys[0]['ret'] == pytest.approx(1.05 * 0.9 - 1)
    assert ys[0]['maxdd'] == pytest.approx(1 - 1.05 * 0.9 / 1.05)
    assert ys[1]['ret'] == pytest.approx(1 + 0.1 * 1.0 - 1)      # one trade of 0.5 + 0.5 R
    assert ys[0]['worst_day'] == pytest.approx(-0.105 / 1.05)


def test_worst_run_finds_the_lowest_compounded_window():
    st = toy_stream([[0.0] * 12], [[0] * 12])
    r = [0.5, 0.5, -1.0, -1.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    st.trades = [[{'i': k, 'j': k, 'r': v, 'entry': f'e{k}', 'exit': f'x{k}'} for k, v in enumerate(r)]]
    run = ST.worst_run(st, [1.0], 0.1, n=3)
    assert [t['r'] for t in run['trades']] == [-1.0, -1.0, -1.0]
    assert run['ret'] == pytest.approx(0.9 ** 3 - 1)
    assert run['maxdd'] == pytest.approx(1 - 0.9 ** 3)


def test_history_replay_equals_years_glued_when_no_trade_crosses_a_year_end(data):
    st = S.build_streams(data, ['crack'])
    h = ST.replay_history(st, [1.0], 0.02)
    ys = ST.replay_years(st, [1.0], 0.02)
    prod = np.prod([1 + y['ret'] for y in ys])
    # Close but not equal: a trade that spans a year end is re-sized on 1 January in the year replay.
    assert h['final'] == pytest.approx(prod, rel=0.1)


def test_pearson_and_moments_agree_with_numpy():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    b = 0.3 * a + rng.normal(size=300)
    assert DV.pearson(a, b) == pytest.approx(float(np.corrcoef(a, b)[0, 1]), abs=1e-12)
    years = [2000 + i // 30 for i in range(300)]
    m = sum(DV._year_moments(years, a, b).values())
    assert DV._corr_from_moments(m) == pytest.approx(DV.pearson(a, b), abs=1e-12)
    assert DV.pearson([1, 1, 1], [1, 2, 3]) is None


def test_bootstrap_range_of_a_correlation_covers_the_estimate_and_is_wide_for_noise():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=600), rng.normal(size=600)
    years = [2000 + i // 30 for i in range(600)]
    lo, hi = DV.year_bootstrap_corr(years, a, b, reps=400)
    assert lo < DV.pearson(a, b) < hi and lo < 0 < hi and hi - lo < 0.5


def test_overlap_correlation_matches_the_number_in_the_brief(data):
    dates, c, f = DV.overlap(data)
    assert len(dates) == 5039 and dates[0] == '2006-06-15'
    assert DV.pearson(c, f) == pytest.approx(float(np.corrcoef(c, f)[0, 1]), abs=1e-12)
    assert abs(DV.pearson(c, f)) < 0.01


def test_correlation_report_is_consistent(data):
    book = S.build_streams(data, ['crack', 'fly'])
    rep = DV.correlation_report(data, book, )
    assert rep['days_both_open'] <= min(rep['days_crack_open'], rep['days_fly_open'])
    assert 0 < rep['share_both_open_of_either'] < 0.5           # the rules are often flat: zeros pull the daily correlation to zero
    assert rep['months_both_loss'] <= min(rep['months_crack_loss'], rep['months_fly_loss'])
    assert [r['year'] for r in rep['stress']] == [2008, 2020, 2022]
    assert rep['daily_range'][0] < rep['daily'] < rep['daily_range'][1]


def test_book_sharpe_agrees_across_two_routes_and_matched_weight_matches_volatility(data):
    book = S.build_streams(data, ['crack', 'fly'])
    rep = DV.book_report(data, book)
    # Route 1 measures in R, route 2 in raw P&L scaled by volatility: same story, similar size.
    assert abs(rep['sharpe']['book'] - rep['sharpe_raw_pnl']['book_vol_scaled']) < 0.15
    assert rep['sharpe']['book'] > max(rep['sharpe']['crack'], rep['sharpe']['fly'])
    a = rep['matched_weight']
    xc, xf = book.x
    target = 0.5 * (xc.std(ddof=1) + xf.std(ddof=1))
    assert (a * (xc + xf)).std(ddof=1) == pytest.approx(target, rel=1e-12)
    assert rep['vol_ratio_book_over_avg'] == pytest.approx(0.7, abs=0.1)   # about 1/sqrt(2) when the rules are unrelated


def test_the_sharpe_of_a_short_noisy_series_is_within_noise():
    rng = np.random.default_rng(6)
    x = rng.normal(0.3 / 252 * 0.01, 0.01, size=252 * 5)        # a true Sharpe of about 0.3 for 5 years
    years = np.repeat(np.arange(5), 252)
    lo, hi = DV.sharpe_range(years, x, reps=500)
    assert lo < 0 < hi                                           # cannot tell it from zero


def test_reference_file_agrees_with_the_one_path_at_a_time_code(data):
    """data/ref_check.json is written by the numpy code; recompute a few cases with the plain loops."""
    ref = json.loads((ROOT / 'data' / 'ref_check.json').read_text())
    case = next(c for c in ref['challenge'] if c['book'] == 'both' and c['mode'] == 'year' and c['f'] == 0.10 and c['cfg']['time_limit'] == 252)
    st = S.build_streams(data, case['rules'])
    u = S.make_units(st, case['mode'])
    cfg = case['cfg']
    rows = [R.challenge_path(st, u, case['w'], case['f'], case['seed'], k, cfg['target'], cfg['dd'], cfg['static'],
                             cfg['daily'], cfg['time_limit'], cfg['funded']) for k in range(case['paths'])]
    passed = sum(1 for r in rows if r[0] == V.PASS) / case['paths']
    assert passed == case['summary']['pass']
    g = next(c for c in ref['growth'] if c['book'] == 'crack' and c['mode'] == 'block' and c['f'] == 0.02)
    st = S.build_streams(data, g['rules'])
    u = S.make_units(st, g['mode'])
    finals = [R.growth_path(st, u, g['w'], g['f'], g['days'], g['seed'], k, R.checkpoints(g['days']))[0] for k in range(5)]
    assert finals == g['first_final']
