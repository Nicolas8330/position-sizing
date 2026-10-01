# Where the data files come from

`trades.json` and `daily_pnl.json` were produced by the rule tests in the
`relative-value-strategies` repository (commit 7eb7950, 30 September 2026) and copied here
unchanged. They are read-only inputs.

* `trades.json`: one row per trade of the fade rule with its 2 standard deviation stop, on the
  3-2-1 crack spread (71 trades, 2008 to 2026) and on the 2s5s10s butterfly (107 trades, 1991 to
  2026). Each row has the entry and exit dates, the profit in the unit of the spread after costs
  (`pnl`), the distance from entry to stop (`stop_distance`, set on the entry day) and
  `r = pnl / stop_distance`.
* `daily_pnl.json`: the daily net profit of one unit of each rule, in the same units, with `null`
  where a series has no data. The two rules overlap on 5,039 days (15 June 2006 to 17 September 2026).

They are derived quantities (profits of a rule), not price tables. The underlying prices are
public series (FRED yields, EIA spot prices), which are not redistributed here.

They change only when the rule is re-run. `results.json` and `ref_check.json` are written by
`python examples/make_results.py`.
