import copy

import numpy as np
import pytest

from sizing import streams as S


def test_trade_counts_and_window(data):
    for rules, counts in ((['crack'], [71]), (['fly'], [107])):
        st = S.build_streams(data, rules)
        assert [len(t) for t in st.trades] == counts
    both = S.build_streams(data, ['crack', 'fly'])
    assert both.dates[0] == '2008-07-07'          # first entry of the later-starting rule
    assert len(both.trades[0]) == 71 and len(both.trades[1]) == 54
    assert both.n == len(both.dates) == len(both.year) == len(both.x[0])


def test_r_is_profit_over_stop_distance_and_matches_the_file(data):
    for rule in ('crack', 'fly'):
        st = S.build_streams(data, [rule])
        by_entry = {t['entry']: t for t in data.trades[rule]}
        for t in st.trades[0]:
            src = by_entry[t['entry']]
            assert t['r'] == pytest.approx(src['pnl'] / src['stop_distance'], abs=1e-12)
            assert t['r'] == pytest.approx(src['r'], abs=6e-5)    # the file rounds r to 4 decimals


def test_daily_r_of_a_trade_sums_to_the_trade_r(data):
    """Accounting identity: costs included, one trade's daily R adds up to its R."""
    for rule in ('crack', 'fly'):
        st = S.build_streams(data, [rule])
        for t in st.trades[0]:
            assert st.x[0][t['i']:t['j'] + 1].sum() == pytest.approx(t['r'], abs=1e-9)


def test_states_mark_first_day_and_never_overlap(data):
    st = S.build_streams(data, ['crack', 'fly'])
    for j in range(2):
        s = st.st[j]
        assert int((s == 1).sum()) == len(st.trades[j])
        assert np.all(st.x[j][s == 0] == 0.0)          # flat days earn nothing
        for t in st.trades[j]:
            assert s[t['i']] == 1 and np.all(s[t['i'] + 1:t['j'] + 1] == 2)
        spans = sorted((t['i'], t['j']) for t in st.trades[j])
        assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))   # no two trades share a day


def test_no_look_ahead_rewriting_the_end_of_the_data_moves_nothing_earlier(data):
    """The stream up to a date is the same whatever happens after that date."""
    cut = data.trades['crack'][40]['exit']
    ci = data.index[cut]
    fake = copy.deepcopy(data)
    rng = np.random.default_rng(0)
    for rule in ('crack', 'fly'):
        for i in range(ci + 1, len(fake.dates)):
            fake.pnl[rule][i] = float(rng.normal(0, 5))
        fake.trades[rule] = [t for t in fake.trades[rule] if data.index[t['exit']] <= ci] + [
            {'entry': fake.dates[ci + 5], 'exit': fake.dates[ci + 9], 'side': 'long', 'days': 5,
             'pnl': 7.0, 'stop_distance': 1.0, 'r': 7.0}]
    a = S.build_streams(data, ['crack'], end=cut)
    b = S.build_streams(fake, ['crack'], end=cut)
    assert a.n == b.n
    assert np.array_equal(a.x[0], b.x[0]) and np.array_equal(a.st[0], b.st[0])
    assert [t['r'] for t in a.trades[0]] == [t['r'] for t in b.trades[0]]


def test_stop_distance_is_known_at_entry_not_later(data):
    """R uses the stop set at entry: a trade's R never changes when later trades are removed."""
    full = S.build_streams(data, ['fly'])
    part = S.build_streams(data, ['fly'], end=data.trades['fly'][50]['exit'])
    assert [t['r'] for t in part.trades[0]] == [t['r'] for t in full.trades[0]][:len(part.trades[0])]


def test_haircut_lowers_every_trade_by_the_same_r(data):
    a = S.build_streams(data, ['crack'])
    b = S.build_streams(data, ['crack'], haircut=0.1)
    for ta, tb in zip(a.trades[0], b.trades[0]):
        assert ta['r'] - tb['r'] == pytest.approx(0.1)
        assert b.x[0][tb['i']:tb['j'] + 1].sum() == pytest.approx(tb['r'], abs=1e-9)


def test_exclusions_remove_whole_trades(data):
    a = S.build_streams(data, ['crack'])
    b = S.build_streams(data, ['crack'], exclude_years=(2008,))
    assert len(b.trades[0]) == len(a.trades[0]) - sum(t['year'] == 2008 for t in a.trades[0])
    c = S.build_streams(data, ['crack'], exclude=[('crack', 10)])
    assert len(c.trades[0]) == len(a.trades[0]) - 1


def test_units_cover_the_stream(data):
    st = S.build_streams(data, ['crack'])
    u = S.make_units(st, 'iid')
    assert u.lens.sum() == st.n and u.kind == 'trade' and len(u.starts) == 71
    y = S.make_units(st, 'year')
    assert y.lens.sum() == st.n and len(y.starts) == len(set(st.year.tolist()))
    b = S.make_units(st, 'block', 63)
    assert len(b.starts) == st.n - 62 and set(b.lens.tolist()) == {63}
    two = S.build_streams(data, ['crack', 'fly'])
    assert S.make_units(two, 'iid').kind == 'day'
