import math

import numpy as np
import pytest

from conftest import toy_stream
from sizing import sim_ref as R
from sizing import simulate as V
from sizing import streams as S


def test_vectorised_growth_equals_reference_to_the_last_digit(data):
    for rules, w in ((['crack'], [1.0]), (['crack', 'fly'], [0.5, 0.5])):
        st = S.build_streams(data, rules)
        for mode in ('iid', 'block', 'year'):
            u = S.make_units(st, mode)
            res = V.sim_growth(st, u, w, 0.05, 400, 9, 40)
            rows = [R.growth_path(st, u, w, 0.05, 400, 9, k, R.checkpoints(400)) for k in range(40)]
            assert res['final'].tolist() == [r[0] for r in rows]
            assert res['maxdd'].tolist() == [r[1] for r in rows]
            assert res['uw'].tolist() == [r[2] for r in rows]
            assert res['longest'].tolist() == [r[3] for r in rows]
            assert res['cp'].tolist() == [r[4] for r in rows]


@pytest.mark.parametrize('cfg', [
    dict(target=0.10, dd=0.10, static=True, daily=0.05, time_limit=200, funded=100),
    dict(target=0.08, dd=0.05, static=False, daily=0.0, time_limit=120, funded=150),
    dict(target=0.05, dd=0.08, static=True, daily=0.03, time_limit=60, funded=60),
])
def test_vectorised_challenge_equals_reference(data, cfg):
    for rules, w in ((['crack'], [1.0]), (['fly'], [1.0]), (['crack', 'fly'], [0.5, 0.5])):
        st = S.build_streams(data, rules)
        u = S.make_units(st, 'block')
        for f in (0.03, 0.10, 0.20):
            res = V.sim_challenge(st, u, w, f, 4, 60, cfg['target'], cfg['dd'], cfg['static'], cfg['daily'],
                                  cfg['time_limit'], cfg['funded'])
            rows = [R.challenge_path(st, u, w, f, 4, k, cfg['target'], cfg['dd'], cfg['static'], cfg['daily'],
                                     cfg['time_limit'], cfg['funded']) for k in range(60)]
            assert res['outcome'].tolist() == [r[0] for r in rows]
            assert res['rday'].tolist() == [r[1] for r in rows]
            assert res['rtrades'].tolist() == [r[2] for r in rows]
            assert res['funded'].tolist() == [r[3] for r in rows]
            assert res['fday'].tolist() == [r[4] for r in rows]


def test_one_trade_multiplies_the_balance_by_one_plus_f_times_r():
    """Accounting identity: risk f on a trade of R multiples gives 1 + f R, whatever the path inside."""
    st = toy_stream([[0.0, 0.2, -0.5, 0.4, 0.0]], [[1, 2, 2, 2, 0]])
    f = 0.03
    r = 0.2 - 0.5 + 0.4
    E = 1.0
    base = [1.0]
    for d in range(5):
        E = R.apply_day(st, [f], base, E, d, d == 0)
    assert E == pytest.approx(1 + f * r, abs=1e-15)


def test_two_trades_compound_and_size_is_set_at_entry():
    st = toy_stream([[0.5, 0.0, 0.5, -0.25]], [[1, 0, 1, 2]])
    f = 0.1
    E, base = 1.0, [1.0]
    for d in range(4):
        E = R.apply_day(st, [f], base, E, d, False)
    assert E == pytest.approx((1 + f * 0.5) * (1 + f * 0.25), abs=1e-15)
    assert base[0] == pytest.approx(1 + f * 0.5)         # second trade sized from the new balance


def test_a_loss_lowers_the_balance_and_zero_risk_changes_nothing(data):
    st = toy_stream([[-1.0]], [[1]])
    assert R.apply_day(st, [0.02], [1.0], 1.0, 0, False) == pytest.approx(0.98)
    cs = S.build_streams(data, ['crack'])
    u = S.make_units(cs, 'iid')
    g = V.sim_growth(cs, u, [1.0], 0.0, 300, 1, 20)
    assert np.all(g['final'] == 1.0) and np.all(g['maxdd'] == 0.0)


def test_balance_cannot_go_below_zero():
    st = toy_stream([[-3.0]], [[1]])
    assert R.apply_day(st, [0.5], [1.0], 1.0, 0, False) == 0.0


def test_paths_do_not_depend_on_how_many_paths_are_run(data):
    st = S.build_streams(data, ['fly'])
    u = S.make_units(st, 'year')
    a = V.sim_growth(st, u, [1.0], 0.05, 300, 5, 50)
    b = V.sim_growth(st, u, [1.0], 0.05, 300, 5, 120)
    assert a['final'].tolist() == b['final'][:50].tolist()
    assert V.sim_growth(st, u, [1.0], 0.05, 300, 5, 50)['final'].tolist() == a['final'].tolist()


def _exact_barrier(p_up, up, down, target, floor, steps):
    """Exact probabilities for a balance that goes up by `up` or down by `down` each day (dynamic programming)."""
    states = {(0, 0): 1.0}
    pass_p = fail_p = 0.0
    for _ in range(steps):
        new = {}
        for (k, m), pr in states.items():
            for dk, dm, q in ((1, 0, p_up), (0, 1, 1 - p_up)):
                kk, mm = k + dk, m + dm
                e = (1 + up) ** kk * (1 - down) ** mm
                if e >= 1 + target:
                    pass_p += pr * q
                elif e <= floor:
                    fail_p += pr * q
                else:
                    new[(kk, mm)] = new.get((kk, mm), 0.0) + pr * q
        states = new
    return pass_p, fail_p


def test_challenge_matches_an_exact_calculation_on_a_coin_flip_account():
    """One-day trades of +1 R (60%) and -1 R (40%) risking 4%: compare with dynamic programming."""
    n = 2000
    rng = np.random.default_rng(1)
    x = np.where(rng.random(n) < 0.6, 1.0, -1.0)
    p_up = float((x > 0).mean())
    st = toy_stream([x.tolist()], [[1] * n])
    u = S.make_units(st, 'day')
    f, target, dd, steps = 0.04, 0.12, 0.12, 40
    exact_pass, exact_fail = _exact_barrier(p_up, f, f, target, 1 - dd, steps)
    res = V.sim_challenge(st, u, [1.0], f, 3, 40000, target, dd, True, 0.0, steps, 0)
    s = V.summarize_challenge(res)
    se = math.sqrt(exact_pass * (1 - exact_pass) / 40000)
    assert abs(s['pass'] - exact_pass) < 4 * se
    se2 = math.sqrt(exact_fail * (1 - exact_fail) / 40000)
    assert abs(s['fail_dd'] - exact_fail) < 4 * se2
    assert s['pass'] + s['fail_dd'] + s['timeout'] == pytest.approx(1.0)


def test_daily_limit_and_trailing_drawdown_rules():
    # Day 1: -6% in one day. With a 5% daily limit that fails the challenge; without it only the drawdown limit applies.
    st = toy_stream([[-1.0] + [0.0] * 9], [[1] + [2] * 9])
    u = S.Units(np.array([0]), np.array([10]), 'day')
    a = V.sim_challenge(st, u, [1.0], 0.06, 1, 5, 0.1, 0.10, True, 0.05, 10, 0)
    assert set(a['outcome'].tolist()) == {V.FAIL_DAILY}
    b = V.sim_challenge(st, u, [1.0], 0.06, 1, 5, 0.1, 0.10, True, 0.0, 10, 0)
    assert set(b['outcome'].tolist()) == {V.TIMEOUT}
    # Up 8% then down 6.4%: a static 5% limit is not hit (balance 1.011 > 0.95), a trailing 5% limit is (peak 1.08, floor 1.026).
    st2 = toy_stream([[1.0, 0.0, -0.8] + [0.0] * 7], [[1, 0, 1] + [0] * 7])
    u2 = S.Units(np.array([0]), np.array([10]), 'day')
    s1 = V.sim_challenge(st2, u2, [1.0], 0.08, 1, 3, 0.20, 0.05, True, 0.0, 10, 0)
    s2 = V.sim_challenge(st2, u2, [1.0], 0.08, 1, 3, 0.20, 0.05, False, 0.0, 10, 0)
    assert set(s1['outcome'].tolist()) == {V.TIMEOUT}
    assert set(s2['outcome'].tolist()) == {V.FAIL_DD}


def test_passing_then_funded_year_resets_the_balance():
    # +12% in one trade passes a 10% target; the next day a -6.4% breaches a 5% funded drawdown limit.
    st = toy_stream([[1.5, -0.8] + [0.0] * 8], [[1, 1] + [0] * 8])
    u = S.Units(np.array([0]), np.array([10]), 'day')
    res = V.sim_challenge(st, u, [1.0], 0.08, 1, 4, 0.10, 0.05, True, 0.0, 5, 5)
    assert set(res['outcome'].tolist()) == {V.PASS} and set(res['funded'].tolist()) == {1}
    assert set(res['fday'].tolist()) == {1}
    assert V.summarize_challenge(res)['p_breach_after_pass'] == 1.0


def test_quantile_matches_numpy():
    a = np.random.default_rng(3).normal(size=101)
    for q in (0.05, 0.25, 0.5, 0.95):
        assert V.quantile(a, q) == pytest.approx(float(np.quantile(a, q)), abs=1e-12)


def test_more_risk_means_deeper_drawdowns_on_the_same_paths(data):
    st = S.build_streams(data, ['crack'])
    u = S.make_units(st, 'block')
    med = [V.quantile(V.sim_growth(st, u, [1.0], f, 500, 2, 300)['maxdd'], 0.5) for f in (0.01, 0.03, 0.06, 0.12)]
    assert med == sorted(med) and med[0] < med[-1]
