# Breakout Engine — Improvement Roadmap

Generated: 2026-02-26

---

## Strengths Worth Noting

The methodology is sound — combining Qullamaggie VCP with Minervini Stage 2, using excess return RS (not ratio), and applying a market regime multiplier. The modular architecture (ingestion → features → scoring → persistence) is clean.

---

## Improvements Ranked by Difficulty

### Tier 1 — Easy Wins (1-2 days each)

| # | Improvement | Why |
|---|---|---|
| **1** | **Replace `print()` with Python `logging` module** | Currently all output is `print()`. Structured logging lets you set levels (DEBUG/INFO/WARNING/ERROR), rotate logs, and debug production issues. Pure software hygiene. |
| **2** | **Fix `consol_range` fallback using `or` instead of `is None`** | In `persistence.py`, `consol_range_60 or consol_range_15` treats `0.0` (extremely tight consolidation — a *good* signal) as falsy. Use explicit `if x is None` checks. |
| **3** | **Add database indices** | No index on `scans(symbol)`, `scans(passes_filters)`, or `outcomes(symbol)`. As the DB grows, queries will slow dramatically. One `CREATE INDEX` per column. |
| **4** | **Make `Ingestor.mergefiles()` accept a date parameter** | Currently hardcoded to `datetime.today()`. If you run at 2am or want to reprocess a prior date, it won't find files. Accept `date` as a parameter with today as default. |
| **5** | **Move all hardcoded lookback windows to `config.py`** | VCP uses 10/20/40, volume dry-up uses 10, prior move uses 60 — all hardcoded in `engine.py`. Centralizing them makes tuning and backtesting possible. |

### Tier 2 — Moderate Effort (3-5 days each)

| # | Improvement | Why |
|---|---|---|
| **6** | **Add a pytest test suite for scoring & hard filters** | Zero tests today. The scoring logic has 100+ lines of conditionals per method and 6 hard filter conditions. One refactor could silently break signals. Start with hard filters (easiest to test), then scoring edge cases. |
| **7** | **Add type hints throughout** | No type hints on most functions. The IDE can't catch `None` being passed where a DataFrame is expected. Add return types and parameter types to all public methods. |
| **8** | **Fix RS calculation around market holidays** | `benchmark_df.reindex(df.index, method="ffill")` forward-fills benchmark values across gaps. Around holidays, stocks get stale benchmark data, producing incorrect RS. Use an inner join on matching dates instead. |
| **9** | **Vectorize `calculate_rs_rank`** | Currently O(N_stocks × N_dates) with nested Python loops. For 200 stocks × 250 days = 50,000 iterations. A single `pd.concat` + `pct_change` + `rank(axis=1)` replaces all of it in milliseconds. |
| **10** | **Fix Stage 2 implementation** | Missing `close > sma_50` (Minervini requires it). Also missing short-term MA stacking (`sma_10 > sma_20 > sma_50`). Currently only checks long-term structure. |

### Tier 3 — Significant Effort (1-2 weeks each)

| # | Improvement | Why |
|---|---|---|
| **11** | **Improve VCP contraction detection** | Current logic (`range_10 < range_20 < range_40`) is a single snapshot. True VCP has *multiple progressive contractions*. Count contraction stages and require ≥2. Also validate that ranges refer to the *current* base, not a prior consolidation. |
| **12** | **Build a backtesting framework** | The `outcomes` table has data but no analysis infrastructure — no win rate, profit factor, expectancy, or walk-forward optimization. A `SimpleBacktester` class that merges scans with outcomes and computes key metrics would let you validate scoring weights empirically. |
| **13** | **Add slippage modeling to outcome tracker** | The outcome tracker assumes entry at exact breakout level and exit at exact stop. In reality, breakout entries slip +0.1-1%, and gap-down stops can slip -2-5%. Without slippage, backtest results are systematically optimistic. |
| **14** | **Refactor FTD detection** | The Follow-Through Day state machine (`market_condition.py`) is ~130 lines with 6 states and 12 transitions, zero tests. Any single up-close (+0.1%) triggers a rally attempt — O'Neil requires meaningful conviction. Needs unit tests and a minimum gain threshold. |
| **15** | **Add portfolio-level risk management** | The engine can produce 20 STRONG BUY signals on one day. If each is 2% risk and you size 10% per position, one bad day is catastrophic. Add a `PortfolioRiskManager` that sizes positions based on account risk and limits total exposure. |

### Tier 4 — Advanced / Long-term

| # | Improvement | Why (Trading Edge) |
|---|---|---|
| **16** | **Survivorship bias handling** | Delisted/merged stocks disappear from analysis. The outcome tracker can't track a stock that no longer exists. Add `delisted_date` tracking. |
| **17** | **Risk/reward scoring redesign** | A stock that barely passes the hard filter (stop at 3×ADR) scores 0 on risk/reward but still makes the watchlist. Either tighten the hard filter or make R/R scoring independent of it. |
| **18** | **Walk-forward optimization** | Test whether the scoring weights (25/30/25/10/10) are actually optimal by running rolling in-sample/out-of-sample tests on historical scans. |
| **19** | **Black swan protection** | Regime multiplier floors at 0.50, so a 90-point stock still scores 45 ("hold") during a crash. Consider a lower floor (0.25) or pausing scans entirely when VIX > 40. |

---

## Recommended Starting Sequence

1. **#1 (Logging)** + **#5 (Config centralization)** — Low effort, immediately improves debuggability and tunability. Foundation work.
2. **#6 (Test suite)** — Before changing any scoring logic, lock in current behavior with tests. Protects against regressions.
3. **#10 (Fix Stage 2)** + **#8 (Fix RS)** — Correctness bugs in core scoring. Fixing them will meaningfully change signal quality.
4. **#12 (Backtesting framework)** — Once scoring is correct and tested, build backtest infrastructure to *measure* whether future changes improve results.

Let the backtest data dictate whether VCP detection (#11), slippage (#13), or weight optimization (#18) matters more for your edge.
