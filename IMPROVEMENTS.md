# Breakout Engine — Improvement Roadmap

Generated: 2026-02-26 | Last updated: 2026-03-03

---

## Strengths Worth Noting

The methodology is sound — combining Qullamaggie VCP with Minervini Stage 2, using excess return RS (not ratio), and applying a market regime multiplier. The modular architecture (ingestion → features → scoring → persistence) is clean.

---

---

## Completed Improvements

| # | Improvement | Status | Notes |
|---|---|---|---|
| **C1** | **Fix `consol_range` column name mismatch** | ✅ Done | `score_base_quality`, `calculate_rr`, and watchlist summary all read `consol_range_15` (hardcoded) but `detect_consolidation_range` writes `consol_range_{base_length_max}` = `consol_range_60`. Tightness scoring was always receiving the fallback (1.0), silently awarding 0/10 pts to every stock. Now reads the config-driven column name dynamically. |
| **C2** | **Fix `or` bug in `persistence.py` consol_range fallback** | ✅ Done | `consol_range_60 or consol_range_15` treated 0.0 (extremely tight base, a *great* signal) as falsy. Replaced with explicit `is None` check. |
| **C3** | **Fix Stage 2: add `close > sma_50`** | ✅ Done | Minervini's full template requires price above 50 SMA as well. A stock above the 150/200 MAs but below the 50 SMA was incorrectly receiving full Stage 2 credit (10 pts). |
| **C4** | **Move all hardcoded lookback windows to `config.py`** | ✅ Done | VCP windows (10/20/40 → `vcp_windows`), volume dry-up (10 → `volume_dryup_window`), prior move (60 → `prior_move_window`), RS rank period (60 → `rs_rank_window`) all now live in `config.py`. |
| **C5** | **Fix `Ingestor.mergefiles()` date parameter** | ✅ Done | Was hardcoded to `datetime.today()`. Now accepts an optional `date` string. `Engine.process_stock()` passes `date_str` through, and `get_data()` also accepts a `date` parameter. |
| **C6** | **Vectorize `calculate_rs_rank`** | ✅ Done | Replaced O(N_stocks × N_dates) nested Python loops with a single `pd.DataFrame.pct_change().rank()` call. For 200 stocks × 250 days, eliminates ~50,000 individual Python iterations. |
| **C7** | **Add pytest test suite for scoring & features** | ✅ Done (prev session) | 295 tests across `test_scoring.py` (249) and `test_features.py` (46). All pass in < 1s. |
| **C8** | **Add database indices** | ✅ Done (prev session) | Indices on `scans(scan_date)`, `scans(symbol)`, `scans(score DESC)`, `outcomes(scan_date, symbol)`. |

---

## Improvements Ranked by Difficulty

### Tier 1 — Easy Wins (1-2 days each)

| # | Improvement | Why |
|---|---|---|
| **1** | **Replace `print()` with Python `logging` module** | Currently all output is `print()`. Structured logging lets you set levels (DEBUG/INFO/WARNING/ERROR), rotate logs, and debug production issues. Pure software hygiene. |
| ~~**2**~~ | ~~**Fix `consol_range` fallback `or` bug**~~ | ✅ Completed (C2) |
| ~~**3**~~ | ~~**Add database indices**~~ | ✅ Completed (C8) |
| ~~**4**~~ | ~~**Make `Ingestor.mergefiles()` accept a date parameter**~~ | ✅ Completed (C5) |
| ~~**5**~~ | ~~**Move all hardcoded lookback windows to `config.py`**~~ | ✅ Completed (C4) |
| **NEW-1** | **Fix `calculate_relative_strength` holiday alignment** | `benchmark_df.reindex(df.index, method="ffill")` forward-fills across market holidays. On a holiday, the stock has no row but the benchmark stale-fills — use `.reindex(df.index).dropna()` + inner-join instead. Already in roadmap as #8 (see Tier 2). |
| **NEW-2** | **Add `passes_filters` index to `scans` table** | The `get_pending_outcomes` query filters by `passes_filters = 1`. Currently only `scan_date`, `symbol`, and `score` are indexed. As the DB grows, this query will do a full table scan. One-line `CREATE INDEX IF NOT EXISTS`. |

### Tier 2 — Moderate Effort (3-5 days each)

| # | Improvement | Why |
|---|---|---|
| ~~**6**~~ | ~~**Add a pytest test suite**~~ | ✅ Completed (C7) |
| **7** | **Add type hints throughout** | No type hints on most functions. The IDE can't catch `None` being passed where a DataFrame is expected. Add return types and parameter types to all public methods. |
| **8** | **Fix RS calculation around market holidays** | `benchmark_df.reindex(df.index, method="ffill")` forward-fills benchmark values across gaps. Around holidays, stocks get stale benchmark data, producing incorrect RS. Use an inner join on matching dates instead. |
| ~~**9**~~ | ~~**Vectorize `calculate_rs_rank`**~~ | ✅ Completed (C6) |
| ~~**10**~~ | ~~**Fix Stage 2 implementation**~~ | ✅ Completed (C3) — `close > sma_50` added. Short-term MA stacking (`sma_10 > sma_20 > sma_50`) is already captured by `ma_alignment` flag in trend scoring. |

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

---

## Roadmap — Mini Bloomberg Terminal

**Vision:** A local Streamlit app that serves as a one-stop shop for evaluating momentum/breakout stocks — scores, price action, sector/theme peer comparisons, financials (SEC EDGAR), insider trades (Form 4), and market regime dashboards.

**Data sources:** Schwab API (price data, existing), SEC EDGAR (financials + insider trades, new), manual theme tags + auto GICS sector classification (peer groups).
**Deployment:** Local only (`streamlit run app.py`).
**Charts:** Plotly line/candlestick charts (simple, not TradingView-style).

---

### Phase 1 — Foundation & Data Layer

These steps build the new data sources and plumbing before any UI work.

| Step | Task | Details |
|------|------|---------|
| **1.1** | **Create `sec_client.py` — SEC EDGAR API client** | Build a client class that fetches company data from SEC EDGAR's free JSON API (`data.sec.gov`). Implement: (a) CIK lookup from ticker via `company_tickers.json`, (b) rate limiting (10 req/sec per SEC rules), (c) proper User-Agent header (SEC requires identifying info). No API key needed. |
| **1.2** | **Add fundamentals fetching to `sec_client.py`** | Fetch company facts from `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`. Parse XBRL tags for: revenue (`Revenues` or `RevenueFromContractWithCustomerExcludingAssessedTax`), net income (`NetIncomeLoss`), EPS (`EarningsPerShareDiluted`), gross margin, operating margin. Return as a clean DataFrame with fiscal quarter/year index. Handle the multiple XBRL tag names that companies use for the same concept. |
| **1.3** | **Add insider trades fetching to `sec_client.py`** | Fetch recent Form 4 filings from `efts.sec.gov/LATEST/search-index?q=...&forms=4`. For each filing, parse the XML for: filer name, title (CEO/CFO/Director), transaction date, shares bought/sold, price, shares owned after. Return as a DataFrame. Cache results to avoid re-fetching. |
| **1.4** | **Create `sectors.py` — sector/theme classification** | Two-layer system: (a) Auto-detect GICS sector from SEC EDGAR's SIC code → map SIC to GICS sector via a lookup dict. (b) Load manual theme overrides from `config/themes.csv` (columns: `ticker,theme`). The manual tag takes priority. Expose a `get_peers(ticker, all_tickers)` function that returns tickers in the same theme/sector. |
| **1.5** | **Create `config/themes.csv` template** | Create the CSV file with example entries and a header comment explaining the format. Themes like `AI Infra`, `Cybersecurity`, `Solar`, `Biotech`, etc. User maintains this file manually. |
| **1.6** | **Add peer RS comparison to `Features` class** | New method `calculate_peer_rs(ticker, peer_dfs)` that computes the stock's return vs the average return of its sector/theme peers over 20d and 60d windows. Store as `rs_peer_20` and `rs_peer_60` columns. This complements the existing `rs_comp_N` (vs COMPX). |
| **1.7** | **Extend `persistence.py` — new tables for fundamentals and insiders** | Add two new SQLite tables: `fundamentals` (ticker, fiscal_period, revenue, eps, net_income, gross_margin, op_margin, fetched_at) and `insider_trades` (ticker, filer_name, title, transaction_date, transaction_type, shares, price, shares_owned_after, fetched_at). Add load/save methods. |
| **1.8** | **Create `data_manager.py` — orchestrate all data fetching** | A single class that coordinates: Schwab price data (existing), SEC fundamentals, SEC insider trades, sector classification. Handles caching logic — only re-fetch fundamentals if >24h stale, insider trades if >12h stale. This becomes the single data access layer the Streamlit app uses. |

---

### Phase 2 — Streamlit App Skeleton

Stand up the app structure and the first working page.

| Step | Task | Details |
|------|------|---------|
| **2.1** | **Add `streamlit` as a dependency** | `uv add streamlit plotly`. Create `app.py` as the entry point with multi-page navigation using `st.navigation`. Define page stubs for: Watchlist, Stock Detail, Market Regime, Peer Analysis. |
| **2.2** | **Build the Watchlist page (`pages/watchlist.py`)** | Date picker (defaults to latest scan date). Table of all stocks that passed filters, sorted by score descending. Columns: symbol, score, grade, signal, price, breakout level, stop, R-multiple, base quality, trend, RS, volume, R/R scores. Color-code rows by grade (green=STRONG BUY, yellow=WATCH, gray=HOLD). Clickable ticker links to Stock Detail page. |
| **2.3** | **Add scan date selector + data loading** | Query `ScanPersistence.load_scans()` for available dates. Sidebar date picker. Cache data with `@st.cache_data` to avoid re-querying SQLite on every interaction. |
| **2.4** | **Add score breakdown expandable rows** | When a user clicks/expands a row in the watchlist table, show the 5-component score breakdown as a horizontal bar chart (base quality, trend, RS, volume, R/R) using Plotly. |

---

### Phase 3 — Stock Detail Page

The "deep dive" view when you click a ticker.

| Step | Task | Details |
|------|------|---------|
| **3.1** | **Build Stock Detail page layout (`pages/stock_detail.py`)** | Accept ticker as URL param or from sidebar selector. Layout: header (ticker, price, score, grade), then tabbed sections below. |
| **3.2** | **Price Action tab — price chart with MA overlays** | Plotly line chart of close price over 1 year. Overlay SMA 10, 20, 50, 150, 200 as colored lines. Mark the breakout level and stop level as horizontal dashed lines. Volume subplot below. |
| **3.3** | **Price Action tab — annotate key levels** | Add markers on the chart for: 52-week high, consolidation range (shaded rectangle), VCP contraction points (if applicable). |
| **3.4** | **Score Breakdown tab** | Radar chart or horizontal bar showing the 5 scoring components. Below it, a details table with the raw values that fed each score (e.g., consol_range=3.2%, stage2=True, rs_comp_20=+12.4%). |
| **3.5** | **Financials tab** | Fetch from `data_manager.py`. Display: revenue trend (bar chart, last 8 quarters), EPS trend (line chart), margin trends (gross + operating). Flag accelerating revenue growth (QoQ and YoY). Simple table of latest quarter's key numbers. |
| **3.6** | **Insider Trades tab** | Fetch from `data_manager.py`. Table of recent Form 4 filings (last 6 months). Columns: date, insider name, title, buy/sell, shares, price, value. Summary metrics: net insider buying $ (last 3 months), # of buyers vs sellers. Highlight cluster buys (multiple insiders buying within 2 weeks). |
| **3.7** | **Peer Comparison tab** | Show the stock's sector/theme peers in a comparison table: ticker, score, RS vs COMPX, RS vs peers, price vs 52wk high. Plotly line chart overlaying the stock's price % change vs the peer group average over 60 days. Highlight whether the stock is leading or lagging its group. |

---

### Phase 4 — Market Regime Dashboard

| Step | Task | Details |
|------|------|---------|
| **4.1** | **Build Market Regime page (`pages/market_regime.py`)** | Load from `market_conditions` table. Show current regime as a large colored badge (BULL=green, DOWNTREND=red, etc.) with the numeric score. |
| **4.2** | **Regime history chart** | Plotly line chart of market condition score over time, with colored background bands for each regime zone (80-100 green, 65-79 light green, 50-64 yellow, 35-49 orange, 0-34 red). |
| **4.3** | **Component breakdown gauges** | Five gauge/progress indicators showing the current score for each component: Index Trend (/25), Distribution (/20), FTD (/20), Breadth (/20), Momentum (/15). Each with a tooltip explaining the score. |
| **4.4** | **Distribution day tracker** | Table showing the last 25 sessions with distribution and stalling days flagged. Running count displayed prominently. Visual indicator when count hits danger zone (5+). |
| **4.5** | **Breadth metrics panel** | Current values: % above 50 SMA, % in Stage 2, % near 52wk high. Mini sparkline charts showing these breadth metrics over the last 30 scan dates. |

---

### Phase 5 — Peer Analysis Page

| Step | Task | Details |
|------|------|---------|
| **5.1** | **Build Peer Analysis page (`pages/peer_analysis.py`)** | Dropdown to select a sector/theme. Show all stocks in that group from the latest scan, sorted by score. |
| **5.2** | **Sector heatmap** | Grid/treemap view where each cell is a stock, sized by dollar volume, colored by score (red→yellow→green gradient). Quick visual of which sectors are hot. |
| **5.3** | **Sector rotation view** | If multiple scan dates exist, show which sectors are gaining/losing RS over time. Simple table: sector, avg score this week, avg score last week, delta. Sorted by delta to surface rotating leadership. |
| **5.4** | **Theme management UI** | Small admin section: view current theme assignments from `config/themes.csv`, add/edit/remove entries directly from the Streamlit app, save back to CSV. |

---

### Phase 6 — Polish & Integration

| Step | Task | Details |
|------|------|---------|
| **6.1** | **Global sidebar with quick stats** | Every page shows: current market regime badge, today's date, # of stocks passing filters, top 3 tickers by score. Acts as a persistent context bar. |
| **6.2** | **Search/jump-to-ticker** | Text input in sidebar. Type a ticker, jump directly to its Stock Detail page. Autocomplete from known tickers in the DB. |
| **6.3** | **Historical scan comparison** | On the Watchlist page, add a "Compare" toggle that shows score changes between two scan dates. Highlight new entries, exits, and score movers (up/down >5 pts). |
| **6.4** | **Export functionality** | Add "Export to CSV" buttons on the Watchlist and Peer Analysis pages. Download the current filtered/sorted view. |
| **6.5** | **Outcome tracking dashboard** | If outcomes data exists, show win rate, avg gain, avg loss, profit factor, and expectancy for stocks by grade bucket. This closes the feedback loop on scoring quality. |
| **6.6** | **App configuration page** | Expose key `config.py` parameters in a settings page: score thresholds, filter values, weight distribution. Changes saved to a local `config_overrides.json` that merges with defaults on startup. |

---

### Dependency Graph

```
Phase 1 (data layer) ──→ Phase 2 (app skeleton + watchlist)
                    ├──→ Phase 3 (stock detail) ─── needs 1.2, 1.3, 1.4, 1.6
                    ├──→ Phase 4 (market regime) ── needs only existing data
                    └──→ Phase 5 (peer analysis) ── needs 1.4, 1.6

Phase 4 can start as soon as Phase 2 is done (no new data deps).
Phases 3 and 5 need the SEC/sector data from Phase 1.
Phase 6 can be done incrementally alongside Phases 3–5.
```

### Suggested Build Order (for LLM sessions)

1. **1.1 → 1.2 → 1.3** — SEC client (fundamentals + insider trades)
2. **1.4 → 1.5** — Sector/theme classification
3. **1.7** — DB schema extensions
4. **1.8** — Data manager
5. **2.1 → 2.2 → 2.3 → 2.4** — Streamlit skeleton + watchlist page
6. **4.1 → 4.2 → 4.3 → 4.4 → 4.5** — Market regime (no new data deps)
7. **1.6** — Peer RS feature
8. **3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7** — Stock detail page
9. **5.1 → 5.2 → 5.3 → 5.4** — Peer analysis page
10. **6.1 → 6.2 → 6.3 → 6.4 → 6.5 → 6.6** — Polish
