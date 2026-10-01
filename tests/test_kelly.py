import numpy as np
import pytest

from sizing import kelly as K
from sizing import streams as S


def test_coin_flip_kelly_is_2p_minus_1():
    r = [1.0] * 60 + [-1.0] * 40
    assert K.kelly_fraction(r) == pytest.approx(0.2, abs=1e-9)


def test_kelly_matches_a_brute_force_grid(data):
    for rule in ('crack', 'fly'):
        r = [t['r'] for t in S.build_streams(data, [rule]).trades[0]]
        grid = np.linspace(0, 0.5, 5001)
        best = grid[int(np.argmax([K.growth_rate(r, f) for f in grid]))]
        assert K.kelly_fraction(r) == pytest.approx(best, abs=1e-4)


def test_no_edge_means_zero_and_no_loser_is_capped():
    assert K.kelly_fraction([1.0, -1.0]) == 0.0
    assert K.kelly_fraction([-0.5, 0.2]) == 0.0
    assert K.kelly_fraction([0.5, 1.0]) == 1.0
    assert K.kelly_fraction([]) == 0.0


def test_growth_falls_after_kelly_and_is_negative_far_above_it(data):
    r = [t['r'] for t in S.build_streams(data, ['crack']).trades[0]]
    k = K.kelly_fraction(r)
    g = [K.growth_rate(r, m * k) for m in (0.5, 1.0, 1.5, 2.0, 2.5)]
    assert g[1] > g[0] and g[1] > g[2] > g[3] > g[4]
    assert K.growth_rate(r, 2.5 * k) < 0 < K.growth_rate(r, 1.0 * k)
    assert K.growth_rate(r, 0.6) == float('-inf')      # one -1.84 R trade would end the account


def test_flipping_the_sign_of_r_removes_the_edge():
    r = [0.5, 0.5, -1.0, 0.4]
    assert K.kelly_fraction(r) > 0
    assert K.kelly_fraction([-v for v in r]) == 0.0


def test_vector_rows_match_scalar(data):
    tr = S.build_streams(data, ['fly']).trades[0]
    groups = K.groups_by_year(tr)
    for mode in ('iid', 'year'):
        v = K.kelly_bootstrap(groups, mode, 5, 30)
        for b in range(30):
            assert v[b] == pytest.approx(K.kelly_fraction(K.resample(groups, mode, 5, b)), abs=1e-12)


def test_bootstrap_is_reproducible_and_year_mode_keeps_years_whole(data):
    tr = S.build_streams(data, ['crack']).trades[0]
    groups = K.groups_by_year(tr)
    assert np.array_equal(K.kelly_bootstrap(groups, 'year', 3, 20), K.kelly_bootstrap(groups, 'year', 3, 20))
    sample = K.resample(groups, 'year', 3, 0)
    flat = {round(v, 12) for g in groups for v in g}
    assert all(round(v, 12) in flat for v in sample)
    assert len(K.resample(groups, 'iid', 3, 0)) == 71
