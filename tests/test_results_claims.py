"""The sentences on the page are written from data/results.json. These tests check that the
claims those sentences make are true of that file, so the text cannot drift from the run."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def res():
    return json.loads((ROOT / 'data' / 'results.json').read_text())


def row(res, kind, book, mode, f):
    return next(r for r in res[kind][book][mode] if abs(r['f'] - f) < 1e-9)


def test_nobody_passes_at_one_percent_risk_within_a_year(res):
    for book in ('crack', 'fly', 'both'):
        assert row(res, 'challenge', book, 'block', 0.01)['pass'] == 0.0


def test_pass_rate_levels_off_while_pass_and_keep_peaks_at_moderate_risk(res):
    for book, lo, hi in (('crack', 0.04, 0.08), ('fly', 0.04, 0.08), ('both', 0.06, 0.12)):
        pk = res['peaks'][book]['block']['pass_survive']
        assert lo <= pk['f'] <= hi
        assert row(res, 'challenge', book, 'block', 0.25)['p_pass_survive'] < 0.5 * pk['p_pass_survive']
    for book in ('crack', 'fly'):
        rows = res['challenge'][book]['block']
        plateau = [r['pass'] for r in rows if r['f'] >= 0.10]
        assert max(plateau) - min(plateau) < 0.2 and max(plateau) < 0.7    # levels off, does not go to 1
    assert row(res, 'challenge', 'crack', 'block', 0.20)['p_breach_after_pass'] > 0.8


def test_the_book_beats_each_rule_on_pass_and_keep(res):
    best = {b: res['peaks'][b]['block']['pass_survive']['p_pass_survive'] for b in ('crack', 'fly', 'both')}
    assert best['both'] > best['crack'] and best['both'] > best['fly']


def test_kelly_is_large_uncertain_and_zero_edge_is_possible(res):
    for rule in ('crack', 'fly'):
        k = res['rules'][rule]['kelly']
        assert 0.15 < k['full'] < 0.25
        assert k['boot']['year']['p95'] - k['boot']['year']['p05'] > 0.25
        assert k['boot']['year']['share_zero'] > 0.0
        assert 1.5 < k['break_even_multiple'] < 2.2
    wf = res['rules']['fly']['walk_forward']['rows']
    assert wf[0]['final'] < 1.0 < wf[4]['final']       # full Kelly estimated on earlier trades loses; fixed 5% wins


def test_correlation_facts(res):
    c = res['diversification']['correlation']
    assert c['overlap_days'] == 5039 and abs(c['daily']) < 0.01
    assert c['weekly'] < 0 and c['weekly_range'][1] < 0
    assert c['days_both_open'] < 0.25 * (c['days_crack_open'] + c['days_fly_open'])
    assert all(s['weekly'] < 0 for s in c['stress'])           # did not rise in 2008, 2020, 2022


def test_bootstrap_variants_agree_on_the_main_message(res):
    for row_ in res['mode_comparison']:
        vals = [row_[m]['p_dd_limit'] for m in ('iid', 'block', 'year')]
        assert max(vals) - min(vals) < 0.2        # the three methods give similar answers
    assert res['trust']['settings_tried'] >= 40
