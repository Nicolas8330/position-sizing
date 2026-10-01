# Review: ways this result could be wrong

The first version of the relative value study was retracted on 30 September 2026 after review.
The lesson: assume the first result is wrong. This file lists how the sizing result could be wrong,
what was done about each, and what stays open (it is also on the page under "What I could not show").

1. **The trades are few (71 and 107) and a Kelly fraction or a pass rate is estimated from them.**
   Done: bootstrap ranges for Kelly (crack 2% to 39%, butterfly 0% to 33%, 90% range, resampling whole
   years), leave-one-year-out and leave-one-trade-out tables, split-half table. Open: nothing more can
   be extracted from 71 trades. The page says so.

2. **Resampling cannot produce a shock worse than the worst in the sample (-1.84 R, -1.53 R).**
   Real tails are fatter. Every drawdown probability is therefore a lower bound in spirit.
   Done: stated on the page; costs and slippage sensitivity (0.05, 0.1, 0.2 R off every trade)
   as a proxy for worse outcomes. Open: not fixable without inventing a tail.

3. **The edge may not be stable.** The crack averaged 0.26 R in its first 35 trades and 0.11 R in
   the last 36. Done: halves table, walk-forward Kelly (estimated only from earlier trades, then used
   on the next trade). Result: full Kelly estimated this way ends at 0.48 times the start on the
   butterfly with a 73% worst fall, so the page argues for a small fraction of Kelly.

4. **Look-ahead.** R uses the stop distance set on the entry day, and every trade's daily R sums to its
   R. Done: `tests/test_streams.py` rewrites the end of the data (later trades and later daily profits)
   and asserts that the stream up to an earlier date does not move; a second test removes later trades
   and checks earlier R are unchanged. The in-sample Kelly on the page is labelled as in-sample
   ("uses the future") wherever it appears next to the walk-forward one.

5. **The simulator could be wrong.** Three implementations (plain loops, numpy, JavaScript) that must agree
   to the last digit; scalar and numpy are compared in `tests/test_simulate.py`, JavaScript and Python in
   `scripts/verify-position-sizing.mjs` (2,684 checks at 1e-9). An exact calculation
   (dynamic programming) on a coin-flip account matches the simulated pass and fail rates within four
   standard errors. Accounting identities (one trade multiplies the balance by 1 + f R, two trades
   compound, a loss lowers the balance, zero risk changes nothing, the balance cannot go below zero).
   Independent recomputation of a headline number with a different code path and a different generator
   (`numpy.random.default_rng`, trade-level episodes built straight from `trades.json`, no shared
   module): crack, 5% risk, typical challenge, independent trades: pass 30.0% and pass-and-keep 24.2%
   there, against 32.1% and 25.3% in the main code; at 3% risk 8.4% and 7.9% against 9.4% and 9.1%.
   The differences (about 1 to 2 points) are inside Monte Carlo error plus a small simplification in the quick
   script (it does not rescale the open trade's size when the account is reset after passing), so the
   check confirms the level, not the last digit.

6. **Random numbers.** mulberry32 has a 32-bit state, so two of thousands of paths can in rare cases share
   a stretch of numbers. Effect: none visible. Done: paths are seeded by a hash of (seed, path number);
   results do not depend on the number of paths (the first 2,000 of 5,000 are the same paths, tested); the
   same cell run with five different seeds gave a crack pass rate between 30.8% and 34.9% at 5% risk with
   2,000 paths, so a percentage is good to about 2 points. The page says so.

7. **The account rule is a model.** Size is set when a trade opens (a trade opened with balance B risks f B).
   When a random piece of history starts in the middle of a trade (a quarter block or a calendar year
   starting on 1 January), the trade is re-sized on that day. This is an approximation at the edges of pieces.
   For "both", independent trades cannot be drawn (trades overlap), so the naive method draws single days,
   which breaks trades into pieces: it is not the same kind of naive as for one rule. Both are documented on the
   page. The worst-10-trades replay of the pair treats overlapping trades as if they did not overlap.

8. **Daily closes hide intraday moves.** The drawdown and daily-loss checks use closing balances. A real firm counts
   open profit and loss during the day, so real breaches are more frequent. Not modelled; stated on the page.

9. **The two rules' correlation of 0.004 is partly an artefact.** The rules are flat on most days (zeros drag a
   correlation to zero). Done: correlation also on the 608 days when both hold a trade (-0.03), on weekly
   totals (-0.09, 95% range -0.15 to -0.03 by resampling years), on monthly totals (-0.11), year by year, and
   in 2008, 2020 and 2022 (weekly -0.16, -0.08, -0.24). Independent route: `numpy.corrcoef` on the raw arrays
   equals the function (test). Open: three stress episodes, 27 to 74 days of joint exposure in each, and the sample
   starts in 2006. Correlations can jump in a crisis unlike these. Stated on the page.

10. **Sharpe of the combined book (0.52) is inside the noise.** Two routes (R-based and raw profit scaled to
    equal volatility) give 0.52 and 0.44. Year-resampled 95% ranges: crack -0.04 to 0.93, butterfly -0.14 to 0.72,
    book 0.10 to 0.93. The page does not rely on the Sharpe; it relies on the lower volatility of the book,
    which is more solid (0.70 of the average of the two, against 0.71 for unrelated series).

11. **The challenge rules are assumptions.** Target 10%, drawdown 10%, daily loss 5%, one year, static drawdown.
    Done: eight settings tried (targets 8 and 10%, drawdown 5 and 10%, trailing drawdown, no daily limit, 3 months,
    3 years). The location of the peak of pass-and-keep is stable (5% to 6% for the single rules, 8% to 12.5% for the
    book; 10% when there is no daily loss limit); its height and the pass rate depend strongly on the time limit. 44 settings tried in total (trade variants,
    haircuts, rule settings); the risk grid has 18 steps. Nothing was tuned or selected.

12. **The main finding depends on the rules trading only 3 to 4 times a year.** A discretionary trader with
    hundreds of trades a year would get different numbers. The page presents this as a property of these two
    rules, not a law. Not tested on any faster edge.

13. **Rounded input for the butterfly.** The butterfly's trade profits in `trades.json` are in whole basis points (the
    rule is measured on yields quoted to 0.01), so R for a trade with a 10 bp stop carries up to about 0.05 R of
    rounding. That is comparable to the smallest haircut tested (0.05 R). It adds noise, not bias, and is small next
    to the R distribution (standard deviation 0.75 R). Not corrected.

14. **Daily P&L and trades are reconciled by construction.** The daily series charges the exit cost on the day after the
    exit. In the stream it is moved to the last day of the trade so that each trade's daily R sums exactly to its R
    (test). This shifts a cost by one day; it changes no trade total.

15. **The spot-price limits of the underlying rules carry over.** EIA spot prices and constant-maturity yields are not
    futures or bond prices; no roll, margin or bid-offer. Every number on the page inherits that.

16. **Selection of the two rules.** They were chosen because they were already tested in this project, and one of them
    (the crack) lost money in recent shocks. The parameters (250-day window, 1.5 entry, 2 standard deviation stop) were fixed
    before this study and not changed; the sensitivity of the rule itself to its window and threshold is in the relative
    value repository, not re-run here (the input files are fixed). No result here was picked among alternatives.

Unresolved and on the page: 1, 2, 8, 9 (crisis behaviour), 12, 13, 15.
