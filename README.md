# Breakout Engine

A daily stock-scanning engine that identifies high-probability breakout setups using Kristjan Qullamaggie's VCP flag breakouts and Mark Minervini's Stage 2 SEPA template. It ingests watchlist exports from ThinkOrSwim, fetches historical OHLCV data via the Schwab API, computes technical features, scores each stock on a 100-point system, analyzes broad market health, and persists results to SQLite for backtesting and continuous improvement.

## Why This Exists

Most retail screeners spit out hundreds of stocks that "look like" breakouts. The problem is that they don't account for base quality, volume contraction, relative strength leadership, or market regime — all things that Qullamaggie and Minervini stress constantly. This engine scores every setup across those dimensions and applies a market regime multiplier so that even a perfect chart gets downgraded when the market environment is hostile.

The scoring weights were calibrated against 793 real outcome observations, correlating each component with actual 20-day forward returns.

## Methodology

### Scoring System (100 points)

Every stock is scored across five categories. The latest trading day's score determines the signal.

| Category | Max | What It Measures |
|---|---|---|
| **Relative Strength** | 30 | 20d and 60d excess returns vs NASDAQ, peer percentile rank |
| **Volume Profile** | 25 | Dollar volume liquidity, volume dry-up in the base, ADR% |
| **Base Quality** | 15 | Consolidation range tightness, base length, VCP contraction |
| **Trend Strength** | 15 | Minervini Stage 2, 52-week high proximity, MA alignment |
| **Risk/Reward** | 15 | Stop distance relative to ADR, R-multiple potential |

**Score thresholds:**
- **≥ 80** — Strong Buy (alert)
- **≥ 70** — Buy (watchlist)
- **60–69** — Hold (monitor)
- **< 60** — Pass

### Hard Filters

Stocks must pass all of these before scoring:
- Price ≥ $5
- Daily dollar volume ≥ $10M
- Price above 50-day SMA
- ADR% ≥ 5%
- Stop distance ≤ 3× ADR
- Within 30% of 52-week high

### Market Regime (100 points)

A separate market health score gates the output. Even great setups get penalized in bad markets — matching how both traders go to cash during downtrends.

| Component | Max | What It Measures |
|---|---|---|
| Index Trend | 25 | NASDAQ SMA stack, 50d slope, SPY/IWM confirmation |
| Distribution Days | 20 | Institutional selling in rolling 25-session window |
| Follow-Through Day | 20 | O'Neil FTD detection (≥1.25% gain on rising volume, day 4+) |
| Internal Breadth | 20 | % of watchlist above 50 SMA, in Stage 2, near highs |
| Momentum / Vol | 15 | 21-day NASDAQ ROC, realized volatility |

The regime multiplier (0.50–1.00) is applied to the raw score:

| Regime | Score | Multiplier | Interpretation |
|---|---|---|---|
| Bull | 80–100 | 1.00 | Full sizing |
| Uptrend | 65–79 | 0.85 | Normal trading |
| Mixed | 50–64 | 0.70 | Smaller positions |
| Caution | 35–49 | 0.60 | Reduce exposure |
| Downtrend | 0–34 | 0.50 | Cash is a position |

### Key Design Decisions

**RS as excess return, not ratio.** `rs = stock_return - benchmark_return`. The ratio formula breaks in bear markets — a stock holding flat while the index drops 10% produces a negative ratio, which is wrong. Excess return correctly identifies relative leaders regardless of market direction.

**Stop sizing relative to ADR.** A 3% stop on a stock with 10% average daily range will get stopped out on noise. The hard filter and R/R scoring both evaluate `stop_distance / ADR` to catch this.

**Volume dry-up scored once.** Consolidated into `volume_profile` only. An earlier version double-counted it in both volume and base quality, which over-weighted a single signal.

## Project Structure

```
breakout-engine/
├── main.py                # CLI entry point
├── engine.py              # Engine, Features, Scoring (core logic)
├── ingestion.py           # CSV reader + Schwab API client (OAuth2)
├── market_condition.py    # 100-point market regime scoring
├── macro_regime.py        # Institutional-grade macro regime classification
├── persistence.py         # SQLite read/write
├── outcome_tracker.py     # Tracks forward returns vs scan predictions
├── analyze.py             # Backtest analysis utilities
├── models.py              # ScoreBreakdown dataclass
├── config.py              # All tunable parameters and scoring weights
├── tests/
│   ├── conftest.py
│   ├── test_scoring.py    # 249 tests — every scoring branch and boundary
│   └── test_features.py   # 46 tests — feature calculations
├── watchlists/            # ThinkOrSwim CSV exports (not tracked)
├── data/                  # Cached OHLCV pickle files (not tracked)
└── results/               # SQLite database (not tracked)
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Schwab developer](https://developer.schwab.com/) app (for API access)

### Install

```bash
git clone https://github.com/zjp292/breakout-engine.git
cd breakout-engine
uv sync
```

### Configure Credentials

Create a `.env` file in the project root:

```
TOS_APP_KEY=your_schwab_app_key
TOS_SECRET=your_schwab_app_secret
```

Run the initial OAuth2 flow to get tokens:

```python
from ingestion import SchwabAPIClient
client = SchwabAPIClient()
client.initial_auth_flow()
```

This opens a browser for Schwab login. Paste the redirect URL back into the terminal. Tokens are saved to `schwab_tokens.json` and auto-refreshed on subsequent runs.

## Usage

### Daily Scan

1. Export your watchlist from ThinkOrSwim and save the CSV(s) to `watchlists/` as `YYYY-MM-DD-watchlist.csv`

2. Run the scanner:

```bash
uv run python main.py --date 2026-03-10
```

3. Output: ranked watchlist printed to terminal, results saved to `results/breakout.db`

### CLI Options

```
--date YYYY-MM-DD    Scan date (default: today)
--debug              Verbose output
--dry-run            Score stocks without writing to the database
```

### Outcome Tracking

After 10+ days have passed since a scan, populate forward-return data:

```python
from persistence import ScanPersistence
from outcome_tracker import OutcomeTracker

db = ScanPersistence()
tracker = OutcomeTracker(db)
tracker.run()
```

### Querying Results

```sql
-- Top setups from last week
SELECT symbol, scan_date, score, grade, signal, price, breakout_level, stop_level
FROM scans
WHERE passes_filters = 1 AND scan_date >= date('now', '-7 days')
ORDER BY score DESC
LIMIT 20;

-- Market regime history
SELECT scan_date, regime, score, regime_multiplier
FROM market_conditions
ORDER BY scan_date DESC;

-- Filter failure analysis
SELECT COUNT(*) as n, symbol
FROM scans WHERE passes_filters = 0
GROUP BY symbol ORDER BY n DESC;
```

## Tests

295 tests covering every scoring branch, filter boundary, grade threshold, and feature calculation:

```bash
uv run pytest tests/ -v
```

## Configuration

All tunable parameters live in [config.py](config.py). Key ones:

| Parameter | Default | Purpose |
|---|---|---|
| `sma_periods` | [10, 20, 50, 150, 200] | Moving average windows (150/200 required for Stage 2) |
| `range_compression_threshold` | 0.05 | < 5% range = tight consolidation |
| `dollar_volume_min` | 10,000,000 | Hard filter: minimum daily dollar volume |
| `min_adr_pct` | 0.05 | Hard filter: minimum average daily range |
| `pct_from_52wk_high_max` | 0.30 | Hard filter: must be within 30% of 52-week high |
| `min_score_alert` | 80 | Alert threshold |
| `min_score_watchlist` | 70 | Watchlist threshold |
| `market_regime` | True | Set False to disable regime multiplier |

Scoring weights in `config.py` represent maximum points per category and should sum to 100.

## Database Schema

Three SQLite tables in `results/breakout.db`:

- **`scans`** — every processed stock per scan date, including stocks that failed filters (`passes_filters` column)
- **`outcomes`** — forward returns: breakout triggered, stop hit, max gain at 10/20/60 days
- **`market_conditions`** — daily regime snapshots

## License

MIT
