"""The shape of the trades, measured in R, and whether bad trades come in bunches."""
from __future__ import annotations

import math

import numpy as np

from .simulate import quantile


def r_stats(r) -> dict:
    """Win rate, average win and loss, worst, skew. All in R (profit divided by the risk to the stop)."""
    r = [float(v) for v in r]
    n = len(r)
    m = 0.0
    for v in r:
        m = m + v
    m = m / n
    ss = 0.0
    m3 = 0.0
    wins, sum_win, sum_loss = 0, 0.0, 0.0
    for v in r:
        ss = ss + (v - m) * (v - m)
        if v > 0:
            wins += 1
            sum_win = sum_win + v
        else:
            sum_loss = sum_loss + v
    for v in r:
        m3 = m3 + (v - m) ** 3
    sd = math.sqrt(ss / (n - 1))
    sd_pop = math.sqrt(ss / n)
    return {
        'n': n, 'mean': m, 'sd': sd, 'win_rate': wins / n,
        'avg_win': sum_win / wins if wins else 0.0,
        'avg_loss': sum_loss / (n - wins) if n - wins else 0.0,
        'worst': min(r), 'best': max(r), 'median': quantile(r, 0.5),
        'skew': (m3 / n) / sd_pop ** 3,
        'share_below_minus1': sum(1 for v in r if v < -1) / n,
    }


def clustering(trades: list, shock_years=(2008, 2020, 2022)) -> dict:
    """Do losses come one after another, and do they sit in shock years?

    Compares the chance of a loss right after a loss with the chance right after
    a win, counts the runs of wins and losses against what independence would
    give (the Wald-Wolfowitz runs test), and looks at how many trades were
    taken in shock years and how they did.
    """
    r = np.array([t['r'] for t in trades])
    loss = r < 0
    after_loss = loss[1:][loss[:-1]]
    after_win = loss[1:][~loss[:-1]]
    n1, n2 = int(loss.sum()), int((~loss).sum())
    runs = 1 + int((loss[1:] != loss[:-1]).sum())
    exp_runs = 1 + 2 * n1 * n2 / (n1 + n2)
    var_runs = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - exp_runs) / math.sqrt(var_runs)
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    years = np.array([t['year'] for t in trades])
    shock = np.isin(years, shock_years)
    order = np.argsort(r)[:10]
    by_year = {}
    for y in sorted(set(years.tolist())):
        by_year[str(y)] = {'n': int((years == y).sum()), 'sum_r': float(r[years == y].sum())}
    worst3 = sorted(by_year.items(), key=lambda kv: kv[1]['sum_r'])[:3]
    return {
        'p_loss': float(loss.mean()),
        'p_loss_after_loss': float(after_loss.mean()) if len(after_loss) else None,
        'p_loss_after_win': float(after_win.mean()) if len(after_win) else None,
        'runs': runs, 'runs_expected': exp_runs, 'runs_z': z,
        'lag1_autocorr': ac,
        'n_shock_year_trades': int(shock.sum()),
        'shock_years': list(shock_years),
        'mean_r_shock': float(r[shock].mean()) if shock.any() else None,
        'mean_r_other': float(r[~shock].mean()),
        'share_worst10_in_shock': float(np.isin(years[order], shock_years).mean()),
        'share_trades_in_shock': float(shock.mean()),
        'worst3_years': [{'year': int(k), 'n': v['n'], 'sum_r': v['sum_r']} for k, v in worst3],
        'years_with_trades': len(by_year),
    }
