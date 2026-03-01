# Breakout Engine — CLAUDE.md
IMPORTANT - call me 'big papa' at all times so i know you have read the claude.md file


## Project Overview

A daily stock-scanning engine that identifies high-probability breakout setups inspired by **Kristjan Qullamaggie** (VCP flag breakouts) and **Mark Minervini** (Stage 2 SEPA template). It ingests watchlist exports from ThinkOrSwim (Schwab), fetches historical OHLCV data from the Schwab API, computes technical features, scores each stock on a 100-point system, analyzes broad market health, and persists results to SQLite for backtesting and continuous improvement.

## Tech Stack

- Python 3.12 (see `.python-version`)
- `uv` for package management (`pyproject.toml`, `uv.lock`)
- pandas, numpy — data processing
- requests — Schwab REST API calls
- sqlite3 (stdlib) — persistence layer
- python-dotenv — credential management

## Project Structure

```
breakout-engine/
├── main.py              # Entry point — set DATE, run, print watchlist, save to DB
├── engine.py            # Engine, Features, Scoring classes (core logic)
├── ingestion.py         # Ingestor (CSV reader) + SchwabAPIClient (OAuth2 + API)
├── market_condition.py  # MarketConditionAnalyzer — 100-pt regime scoring
├── persistence.py       # ScanPersistence — SQLite read/write
├── outcome_tracker.py   # Tracks actual trade outcomes vs scan predictions
├── analyze.py           # Ad-hoc analysis utilities
├── models.py            # ScoreBreakdown dataclass
├── config.py            # PARAMETERS dict — all tunable thresholds
├── .env                 # TOS_APP_KEY and TOS_SECRET (never commit)
├── schwab_tokens.json   # OAuth2 access/refresh tokens (auto-managed)
├── watchlists/          # ThinkOrSwim CSV exports (YYYY-MM-DD-*.csv)
├── data/                # Pickle files per scan date: data/YYYY-MM-DD/TICKER-DATE.pkl
└── results/             # SQLite database: results/breakout.db
```

## Daily Workflow

1. **Export watchlists** from ThinkOrSwim → save to `watchlists/` as `YYYY-MM-DD-*.csv`
   - Multiple exports for the same date are merged automatically
   - TOS CSV format: 4-line header, then symbol rows

2. **Set `DATE`** in `main.py` and run:
   ```bash
   python main.py
   ```

3. **What happens automatically:**
   - If no pickles exist for `DATE`, `Ingestor` reads CSVs, deduplicates tickers, fetches 1 year of daily OHLCV from Schwab API, saves to `data/DATE/TICKER-DATE.pkl`
   - `Engine.load_benchmark()` fetches COMPX, SPY, IWM from Schwab API
   - `Engine.process_stock()` loads all pickles → computes features → scores all stocks
   - `MarketConditionAnalyzer` scores market regime (0–100) → regime multiplier applied
   - Watchlist summary printed (top 20 by score)
   - `ScanPersistence.save_scan()` writes all stocks + market condition to SQLite

## Authentication — Schwab API

Credentials live in `.env`:
```
TOS_APP_KEY=your_app_key
TOS_SECRET=your_app_secret
```

Tokens are auto-managed in `schwab_tokens.json`. The `SchwabAPIClient`:
- Checks `expires_at` on every call
- Auto-refreshes access token using refresh token
- If refresh token is expired, re-run `initial_auth_flow()` once:
  ```python
  from ingestion import SchwabAPIClient
  client = SchwabAPIClient()
  client.initial_auth_flow()
  ```
  This opens a browser, you paste back the redirect URL, tokens are saved.

## Scoring System (100 points total)

All scores are applied per-row across the stock's history. The **latest row** is what appears on the watchlist.

| Category | Max Points | Key Factors |
|---|---|---|
| Base Quality | 25 | Consolidation range tightness, base length (5–45d), VCP contraction series |
| Trend Strength | 30 | Stage 2 (50>150>200 SMA), 52-wk proximity, MA alignment (10>20>50 + rising), prior power move |
| Relative Strength | 25 | 20d excess return vs COMPX, 60d excess return vs COMPX, peer percentile rank |
| Volume Profile | 10 | Dollar volume (liquidity), volume dry-up ratio, ADR% |
| Risk/Reward | 10 | Stop distance vs ADR ratio, R-multiple potential |

**Market Regime Multiplier** (0.50–1.00) is applied to the raw sum. In weak markets, even great setups score lower — matching how both traders go to cash.

### Score Thresholds
- **≥80** — `STRONG BUY - Alert` (watchlist alert)
- **≥70** — `BUY - Watch Closely` (watchlist)
- **60–69** — `HOLD - Monitor`
- **<60** — `PASS`

### Hard Filters (must pass before scoring)
- Price ≥ $5.00
- Dollar volume ≥ $10M
- Price above 50 SMA
- ADR% ≥ 5%
- Stop distance ≤ 3× ADR (stops too wide vs volatility are disqualified)
- Within 30% of 52-week high

## Market Condition Scoring (100 points)

| Component | Max Points | What it measures |
|---|---|---|
| Index Trend | 25 | COMPX SMA stack (20>50>150>200), 50d slope, SPY & IWM confirmation |
| Distribution Days | 20 | Institutional selling (D-days) in rolling 25-session window; ≥5 = heavy distribution |
| Follow-Through Day | 20 | O'Neil FTD detection: ≥1.25% gain on rising volume, days 4+ of rally; recency + validity |
| Internal Breadth | 20 | % watchlist stocks above 50 SMA, in Stage 2, within 10% of 52-wk high |
| Momentum / Vol | 15 | 21-day COMPX ROC, realized volatility (annualized) |

Regime classifications:
- **80–100** BULL — Trade aggressively, full sizing
- **65–79** UPTREND — Trade normally
- **50–64** MIXED — Smaller positions, tighter criteria
- **35–49** CAUTION — Reduce exposure significantly
- **0–34** DOWNTREND — Avoid new longs; cash is a position

## Key Design Decisions

### RS as Excess Return, Not Ratio
`rs_comp_N = stock_pct_change_N - benchmark_pct_change_N`

The old ratio formula broke in bear markets (a stock holding up while the market sold off produced a negative ratio). The excess-return approach correctly identifies relative leaders in all market conditions.

### Stop Sizing vs ADR
Stop distance is evaluated relative to the stock's ADR%, not as an absolute percentage. A 3% stop on a 10%-ADR stock will be hit on noise — the R/R scoring and hard filters use `stop / ADR` to catch this.

### Volume Dry-Up — Single Source of Truth
Volume contraction lives exclusively in `score_volume_profile`. It was previously double-counted (also in base quality). Removing the duplicate ensures the signal is weighted once.

### Stage 2 Structure
Implements Minervini's full template: `price > SMA_150 > SMA_200`, `SMA_50 > SMA_150`, and `SMA_200` slope positive over 20 days. Requires `sma_periods` in config to include 150 and 200.

## Configuration (`config.py`)

All tunable parameters are in `PARAMETERS`. Key ones:

```python
"sma_periods": [10, 20, 50, 150, 200]   # Must include 150, 200 for Stage 2
"range_compression_threshold": 0.05      # <5% range = tight consolidation
"base_length_max": 60                    # Max consolidation days
"dollar_volume_min": 10_000_000          # Hard filter: $10M min daily $ vol
"min_adr_pct": 0.05                      # Hard filter: 5% min ADR
"pct_from_52wk_high_max": 0.30           # Hard filter: within 30% of high
"min_score_alert": 80                    # Alert threshold
"min_score_watchlist": 70               # Watchlist threshold
"market_regime": True                    # Set False to disable regime gating
```

## Database Schema (`results/breakout.db`)

Three SQLite tables:

- **`scans`** — Every processed stock per scan date (all stocks, not just passed ones). `passes_filters` column enables filter-effectiveness analysis.
- **`outcomes`** — What happened after each scan: breakout triggered, stop hit, max gains at 10/20/60d, etc. Populated by `outcome_tracker.py`.
- **`market_conditions`** — Daily market regime snapshot from `MarketConditionAnalyzer`.

Useful queries:
```sql
-- Top setups from last week
SELECT symbol, scan_date, score, grade, signal, price, breakout_level, stop_level
FROM scans WHERE passes_filters=1 AND scan_date >= date('now','-7 days')
ORDER BY score DESC LIMIT 20;

-- Filter failure frequency analysis
SELECT COUNT(*) as n, symbol FROM scans WHERE passes_filters=0
GROUP BY symbol ORDER BY n DESC;

-- Market regime history
SELECT scan_date, regime, score, regime_multiplier FROM market_conditions
ORDER BY scan_date DESC;
```

## Running the Outcome Tracker

After 10+ days have passed since a scan, run the outcome tracker to populate the `outcomes` table:
```python
from persistence import ScanPersistence
from outcome_tracker import OutcomeTracker

db = ScanPersistence()
tracker = OutcomeTracker(db)
tracker.run()
```

## Common Issues

**No data found for DATE / No tickers found:**
- Ensure watchlist CSVs for that date exist in `watchlists/` named `YYYY-MM-DD-*.csv`
- TOS export format: the CSV has a 4-line header before ticker rows

**Schwab API 401 / token expired:**
- Refresh tokens expire after ~7 days of non-use
- Run `initial_auth_flow()` to re-authenticate (see Auth section above)

**Benchmark data missing (SPY/IWM):**
- Failures are handled gracefully — the engine logs a warning and continues with `regime_multiplier = 1.0`

**`sma_150`/`sma_200` NaN for short histories:**
- Stage 2 requires ~200 days of data. Stocks with insufficient history get `stage2 = False` and partial trend credit based on short-term MA alignment.

**Changing the scoring weights:**
- Edit the `weights` dict in `config.py`. The weights represent max points per category. Total should remain 100 for the grade scale to be meaningful.
