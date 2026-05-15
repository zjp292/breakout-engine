"""
qullamaggie-style momentum breakout backtester.

position management rules:
  entry         open of next trading day after scan (or after breakout confirmation)
  initial stop  low of the entry day (intraday stop, not base low)
  trim 1        sell 1/3 at 3R; stop immediately moves to entry (breakeven)
  trailing      remaining 2/3 exits on any close below SMA-10 (after breakeven set)
  breakeven     if low <= entry price after trim 1, exit remaining at entry
  safety valve  force-exit at close if held longer than max_hold_days

metrics follow institutional quant standards: daily MTM equity curve, calmar,
sortino, expectancy, MAE/MFE, monthly return table, score attribution.

usage:
    python backtester.py [--start 2024-01-01] [--end 2026-04-20] [--min-score 70]
                         [--entry next_open|breakout] [--trim1-r 3.0]
                         [--sma-trail 10] [--max-hold 60]
                         [--positions 10] [--risk 0.02] [--equity 100000]
                         [--csv trades.csv]

programmatic:
    from backtester import Backtester, BacktestParams
    results = Backtester(BacktestParams(min_score=75)).run()
    results.print_report()
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DB_PATH  = Path("results/breakout.db")
_TRADING_DAYS = 252
_REGIME_ORDER = ["DOWNTREND", "CAUTION", "MIXED", "UPTREND", "BULL"]


# ── params ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    start_date: str = "2019-01-01"
    end_date:   str = "2099-12-31"
    min_score:  float = 70.0
    max_score:  Optional[float] = None
    # signal_min_score is used only by SignalValidator (the forward-return analysis).
    # it should be lower than min_score so the validator sees the full distribution
    # of historical setups and can compute meaningful score-vs-outcome correlations.
    # historical_batch.py scores all nasdaq stocks (avg raw_score ~37), so a 60
    # threshold gives ~5k signals; the trading sim still uses the tighter min_score.
    signal_min_score: float = 60.0
    # 'raw_score' isolates setup quality from regime (best for backtesting)
    # 'score'     mirrors live trading (regime-adjusted)
    score_col:  str = "raw_score"
    # 'next_open' — enter at open of next day after scan (realistic default)
    # 'breakout'  — wait for close >= breakout_level; enter next open after confirmation
    entry_type: str = "next_open"
    # max days to wait for breakout confirmation (entry_type='breakout' only)
    breakout_window: int = 15
    max_positions:    int   = 10
    risk_per_trade:   float = 0.005  # fraction of current equity risked per trade (Qullamaggie: 0.3-0.5%)
    max_position_pct: float = 0.20   # single-position cap as % of equity
    initial_equity:   float = 100_000.0
    # trim 1: sell 1/3 at N×R, then move stop to entry (breakeven)
    trim1_r: float = 3.0
    # trim 2: sell another 1/3 at M×R (set trim2_enabled=True to activate)
    trim2_r: float = 6.0
    trim2_enabled: bool = False
    # trailing exit: close below this SMA triggers exit on remaining shares
    # only active after stop has moved to breakeven (trim 1 done)
    sma_trail_period: int = 10
    # safety valve: force-exit at close if no trail/stop fires within N days
    max_hold_days: int = 60
    # min market regime to allow new entries on a scan date.
    # "CAUTION" blocks DOWNTREND entries (10 trades at -0.10R avg in backtesting).
    # set to "" to allow all regimes including DOWNTREND.
    min_regime: str = "CAUTION"


# ── trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:            str
    scan_date:         str
    entry_date:        str
    entry_price:       float
    initial_stop:      float   # low of entry day — the actual intraday stop
    shares:            float   # full original position
    cost_basis:        float   # entry_price * shares
    adr_pct:           float
    score:             float
    raw_score:         float
    grade:             str
    market_regime:     str    # regime as of the scan date
    regime_multiplier: float
    base_quality:      float
    trend_strength:    float
    relative_strength: float
    volume_profile:    float
    risk_reward_score: float

    # entry-date regime (may differ from scan-date regime when market moves during hold)
    entry_regime:            str   = ""
    entry_regime_multiplier: float = 1.0

    # trim 1 (sell 1/3 of original at 3R, stop → breakeven)
    trim1_date:   str   = ""
    trim1_price:  float = 0.0
    trim1_shares: float = 0.0

    # trim 2 (optional: sell another 1/3 at 6R)
    trim2_date:   str   = ""
    trim2_price:  float = 0.0
    trim2_shares: float = 0.0

    # final exit of remaining shares
    exit_date:   str   = ""
    exit_price:  float = 0.0
    exit_shares: float = 0.0
    # stop | breakeven_stop | trail_sma10 | time | eod | no_data
    exit_reason: str   = ""

    # excursion tracking (% from entry)
    mfe: float = 0.0   # max favorable: best high relative to entry
    mae: float = 0.0   # max adverse:   worst low relative to entry (negative)

    @property
    def is_closed(self) -> bool:
        return bool(self.exit_date)

    @property
    def total_pnl(self) -> float:
        pnl = 0.0
        if self.trim1_date:
            pnl += (self.trim1_price - self.entry_price) * self.trim1_shares
        if self.trim2_date:
            pnl += (self.trim2_price - self.entry_price) * self.trim2_shares
        if self.exit_date:
            pnl += (self.exit_price - self.entry_price) * self.exit_shares
        return pnl

    @property
    def pct_gain(self) -> float:
        return self.total_pnl / self.cost_basis if self.cost_basis else 0.0

    @property
    def r_multiple(self) -> float:
        # R = total P&L / original dollar risk
        risk = (self.entry_price - self.initial_stop) * self.shares
        return self.total_pnl / risk if risk > 0 else 0.0

    @property
    def hold_days(self) -> int:
        if not self.is_closed:
            return 0
        try:
            return (pd.Timestamp(self.exit_date) - pd.Timestamp(self.entry_date)).days
        except Exception:
            return 0


# ── open position state (mutable, not stored in Trade) ───────────────────────

@dataclass
class _OpenPos:
    trade:            Trade
    current_stop:     float   # starts at initial_stop, moves to entry after trim1
    shares_remaining: float   # reduces at each trim
    trim1_done:       bool  = False
    trim2_done:       bool  = False
    hold_count:       int   = 0   # bars held since entry


# ── results container ─────────────────────────────────────────────────────────

class BacktestResults:
    def __init__(
        self,
        params: BacktestParams,
        trades: list[Trade],
        daily_equity: pd.Series,
        metrics: dict,
    ):
        self.params       = params
        self.trades       = trades
        self.daily_equity = daily_equity  # indexed by YYYY-MM-DD string
        self.metrics      = metrics

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            if not t.is_closed:
                continue
            rows.append({
                "symbol":           t.symbol,
                "scan_date":        t.scan_date,
                "entry_date":       t.entry_date,
                "exit_date":        t.exit_date,
                "entry_price":      round(t.entry_price, 2),
                "initial_stop":     round(t.initial_stop, 2),
                "exit_price":       round(t.exit_price, 2),
                "trim1_price":      round(t.trim1_price, 2) if t.trim1_date else None,
                "exit_reason":      t.exit_reason,
                "pct_gain":         round(t.pct_gain, 4),
                "r_multiple":       round(t.r_multiple, 2),
                "mfe":              round(t.mfe, 4),
                "mae":              round(t.mae, 4),
                "hold_days":        t.hold_days,
                "score":            t.score,
                "raw_score":        t.raw_score,
                "grade":            t.grade,
                "market_regime":          t.market_regime,
                "regime_multiplier":      t.regime_multiplier,
                "entry_regime":           t.entry_regime,
                "entry_regime_multiplier": t.entry_regime_multiplier,
                "base_quality":           t.base_quality,
                "trend_strength":   t.trend_strength,
                "relative_strength": t.relative_strength,
                "volume_profile":   t.volume_profile,
                "risk_reward_score": t.risk_reward_score,
            })
        return pd.DataFrame(rows)

    def score_attribution(self) -> pd.DataFrame:
        """pearson r between each score component and trade outcomes"""
        df = self.to_dataframe()
        if len(df) < 5:
            return pd.DataFrame()
        components = [
            "score", "raw_score", "base_quality", "trend_strength",
            "relative_strength", "volume_profile", "risk_reward_score",
        ]
        avail = [c for c in components if c in df.columns]
        return (
            df[avail + ["pct_gain", "r_multiple"]]
            .corr()
            .loc[avail, ["pct_gain", "r_multiple"]]
        )

    def regime_breakdown(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty or "market_regime" not in df.columns:
            return pd.DataFrame()
        return (
            df.groupby("market_regime")
            .agg(
                trades    = ("r_multiple", "count"),
                win_rate  = ("r_multiple", lambda x: (x > 0).mean()),
                avg_r     = ("r_multiple", "mean"),
                avg_gain  = ("pct_gain", "mean"),
                avg_hold  = ("hold_days", "mean"),
            )
            .round(3)
        )

    def entry_regime_breakdown(self) -> pd.DataFrame:
        """performance grouped by the regime on the ACTUAL ENTRY DATE (not scan date)"""
        df = self.to_dataframe()
        if df.empty or "entry_regime" not in df.columns:
            return pd.DataFrame()
        return (
            df.groupby("entry_regime")
            .agg(
                trades    = ("r_multiple", "count"),
                win_rate  = ("r_multiple", lambda x: (x > 0).mean()),
                avg_r     = ("r_multiple", "mean"),
                avg_gain  = ("pct_gain", "mean"),
                avg_hold  = ("hold_days", "mean"),
            )
            .round(3)
        )

    def score_bucket_analysis(self, width: int = 5) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        col = self.params.score_col if self.params.score_col in df.columns else "score"
        df["bucket"] = (df[col] // width * width).astype(int)
        return (
            df.groupby("bucket")
            .agg(
                trades   = ("r_multiple", "count"),
                win_rate = ("r_multiple", lambda x: (x > 0).mean()),
                avg_r    = ("r_multiple", "mean"),
                avg_gain = ("pct_gain", "mean"),
            )
            .round(3)
        )

    def monthly_returns(self) -> pd.DataFrame:
        eq = self.daily_equity.copy()
        if eq.empty:
            return pd.DataFrame()
        eq.index = pd.to_datetime(eq.index, errors="coerce")
        eq = eq.dropna()
        monthly = eq.resample("ME").last().pct_change().dropna()
        monthly.index = monthly.index.to_period("M")
        df = pd.DataFrame({"return": monthly})
        df["year"]  = [p.year  for p in df.index]
        df["month"] = [p.month for p in df.index]
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"][m - 1]
            for m in pivot.columns
        ]
        annual = (
            df.groupby("year")["return"]
            .apply(lambda r: (1 + r).prod() - 1)
        )
        pivot["Annual"] = annual
        return pivot.round(4)

    def print_report(self) -> None:
        m = self.metrics
        p = self.params
        w = 70
        print("\n" + "=" * w)
        print("BREAKOUT ENGINE — BACKTEST REPORT  (Qullamaggie-style)")
        print("=" * w)
        print(f"  {'period':<26} {p.start_date} → {p.end_date}")
        score_rng = f"{p.min_score}–{p.max_score}" if p.max_score else str(p.min_score)
        print(f"  {'score filter':<26} {score_rng}  ({p.score_col})")
        print(f"  {'entry':<26} {p.entry_type}")
        print(f"  {'initial stop':<26} low of entry day")
        print(f"  {'trim 1':<26} sell 1/3 at {p.trim1_r:.0f}R → stop → entry (breakeven)")
        if p.trim2_enabled:
            print(f"  {'trim 2':<26} sell 1/3 at {p.trim2_r:.0f}R")
        print(f"  {'trailing stop':<26} close < SMA{p.sma_trail_period} (after breakeven)")
        print(f"  {'max hold':<26} {p.max_hold_days}d safety valve")
        if p.min_regime:
            print(f"  {'min regime':<26} {p.min_regime}+")
        print(f"  {'max positions':<26} {p.max_positions}")
        print(f"  {'risk / trade':<26} {p.risk_per_trade:.1%}")
        print(f"  {'initial equity':<26} ${p.initial_equity:>12,.0f}")
        print("-" * w)
        n = m.get("total_trades", 0)
        if n == 0:
            print("  no closed trades — check score thresholds and date range")
            print("=" * w)
            return
        print(f"  {'total trades':<26} {n}")
        print(f"  {'win rate':<26} {m['win_rate']:.1%}   ({m['winners']}W / {m['losers']}L)")
        print(f"  {'avg R-multiple':<26} {m['avg_r']:.2f}R")
        print(f"  {'expectancy':<26} {m['expectancy']:.2f}R  per trade")
        print(f"  {'profit factor':<26} {m['profit_factor']:.2f}")
        print(f"  {'avg win':<26} {m['avg_win']:.1%}  avg loss {m['avg_loss']:.1%}")
        print(f"  {'avg hold':<26} {m['avg_hold']:.1f}d")
        print(f"  {'avg MFE':<26} {m['avg_mfe']:.1%}  (avg best intraday high)")
        print(f"  {'avg MAE':<26} {m['avg_mae']:.1%}  (avg worst intraday low)")
        print(f"  {'trim1 rate':<26} {m['trim1_rate']:.1%}  ({m['trim1_count']} trades reached {p.trim1_r:.0f}R)")
        print("-" * w)
        print(f"  {'total return':<26} {m['total_return']:>+.2%}")
        print(f"  {'CAGR':<26} {m['cagr']:>+.2%}")
        print(f"  {'Sharpe ratio':<26} {m['sharpe']:.2f}")
        print(f"  {'Sortino ratio':<26} {m['sortino']:.2f}")
        print(f"  {'Calmar ratio':<26} {m['calmar']:.2f}")
        print(f"  {'max drawdown':<26} {m['max_drawdown']:.2%}")
        print(f"  {'max DD duration':<26} {m['max_dd_duration']}d")
        print(f"  {'time in market':<26} {m['time_in_market']:.1%}")
        print(f"  {'final equity':<26} ${m['final_equity']:>12,.0f}")
        print("-" * w)
        print("  exit breakdown:")
        for reason, count in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<20} {count:4d}  ({count / n:.0%})")
        print("=" * w)

        df = self.to_dataframe()
        if df.empty:
            return

        attr = self.score_attribution()
        if not attr.empty:
            print("\nscore attribution  (pearson r vs trade outcomes):")
            print(attr.to_string())

        regime = self.regime_breakdown()
        if not regime.empty:
            print("\nperformance by market regime (scan-date label):")
            print(regime.to_string())

        entry_regime = self.entry_regime_breakdown()
        if not entry_regime.empty:
            print("\nperformance by market regime (entry-date label):")
            print(entry_regime.to_string())

        buckets = self.score_bucket_analysis()
        if not buckets.empty:
            print("\nperformance by score bucket:")
            print(buckets.to_string())

        mr = self.monthly_returns()
        if not mr.empty:
            print("\nmonthly returns:")
            fmt = mr.map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
            print(fmt.to_string())

        cols = ["symbol", "entry_date", "exit_date", "pct_gain", "r_multiple",
                "exit_reason", "raw_score"]
        avail = [c for c in cols if c in df.columns]
        print("\ntop 10 trades (by R-multiple):")
        print(df.nlargest(10, "r_multiple")[avail].to_string(index=False))
        print("\nbottom 10 trades (by R-multiple):")
        print(df.nsmallest(10, "r_multiple")[avail].to_string(index=False))


# ── backtester ────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, params: BacktestParams | None = None):
        self.params = params or BacktestParams()
        self._price_cache: dict[str, pd.DataFrame] = {}

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_scans(self) -> pd.DataFrame:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"database not found: {DB_PATH}")
        p = self.params
        if p.score_col == "raw_score":
            score_expr = "COALESCE(s.raw_score, s.score / NULLIF(mc.regime_multiplier, 0))"
        else:
            score_expr = "s.score"

        regime_clause, regime_params = "", []
        if p.min_regime and p.min_regime in _REGIME_ORDER:
            allowed = _REGIME_ORDER[_REGIME_ORDER.index(p.min_regime):]
            placeholders = ",".join("?" * len(allowed))
            regime_clause = f" AND (mc.regime IN ({placeholders}) OR mc.regime IS NULL)"
            regime_params = allowed

        max_score_clause, max_score_params = "", []
        if p.max_score is not None:
            max_score_clause = f" AND {score_expr} <= ?"
            max_score_params = [p.max_score]

        query = f"""
            SELECT s.*, mc.regime, mc.regime_multiplier AS mc_rm,
                   {score_expr} AS _filter_score
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE s.passes_filters = 1
              AND s.scan_date BETWEEN ? AND ?
              AND {score_expr} >= ?
              {max_score_clause}
              {regime_clause}
            ORDER BY s.scan_date ASC, {score_expr} DESC
        """
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                query, conn,
                params=(p.start_date, p.end_date, p.min_score,
                        *max_score_params, *regime_params),
            )
        return df

    def _find_latest_pickle(self, symbol: str) -> Optional[Path]:
        files = list(DATA_DIR.glob(f"*/{symbol}-*.pkl"))
        if not files:
            return None
        return max(files, key=lambda f: f.stem[-10:])

    def _load_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        pkl = self._find_latest_pickle(symbol)
        if pkl is None:
            return None
        try:
            df = pd.read_pickle(pkl)
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            col = "datetime" if "datetime" in df.columns else "date"
            df = df.rename(columns={col: "date"})
            if df["date"].dtype in (np.int64, np.float64, "int64", "float64"):
                df["date"] = pd.to_datetime(df["date"], unit="ms")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.normalize()
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df["ds"] = df["date"].dt.strftime("%Y-%m-%d")
            # precompute trailing SMA for the exit signal
            sma_period = self.params.sma_trail_period
            df[f"sma{sma_period}"] = df["close"].rolling(sma_period, min_periods=1).mean()
            self._price_cache[symbol] = df
            return df
        except Exception:
            return None

    # ── entry logic ───────────────────────────────────────────────────────────

    def _find_entry(
        self, row: pd.Series, prices: pd.DataFrame
    ) -> Optional[tuple[str, float, float]]:
        """returns (entry_date, entry_price, initial_stop) or None"""
        scan_date = row["scan_date"]
        p = self.params

        if p.entry_type == "next_open":
            after = prices[prices["ds"] > scan_date]
            if after.empty:
                return None
            r = after.iloc[0]
            entry_price  = float(r["open"])
            initial_stop = float(r["low"])
            # ensure minimum 2% stop distance — a near-doji entry day (low ≈ open)
            # produces near-zero risk_per_share, inflating share count catastrophically
            min_stop_dist = entry_price * 0.02
            if entry_price - initial_stop < min_stop_dist:
                initial_stop = entry_price - min_stop_dist
            return r["ds"], entry_price, initial_stop

        # breakout mode: wait for a close >= breakout_level within breakout_window days
        breakout_lvl = row.get("breakout_level")
        if pd.isna(breakout_lvl) or float(breakout_lvl) <= 0:
            # no breakout level stored → fall back to next_open
            after = prices[prices["ds"] > scan_date]
            if after.empty:
                return None
            r = after.iloc[0]
            return r["ds"], float(r["open"]), float(r["low"])

        window = prices[prices["ds"] > scan_date].head(p.breakout_window)
        for _, r in window.iterrows():
            if float(r["close"]) >= float(breakout_lvl):
                nxt = prices[prices["ds"] > r["ds"]]
                if nxt.empty:
                    # enter at breakout day close if no next bar
                    return r["ds"], float(r["close"]), float(r["low"])
                nr = nxt.iloc[0]
                entry_price  = float(nr["open"])
                initial_stop = float(nr["low"])
                min_stop_dist = entry_price * 0.02
                if entry_price - initial_stop < min_stop_dist:
                    initial_stop = entry_price - min_stop_dist
                return nr["ds"], entry_price, initial_stop

        return None  # breakout never triggered

    # ── bar-level position processing ─────────────────────────────────────────

    def _process_bar(
        self, pos: _OpenPos, r: pd.Series
    ) -> tuple[float, bool]:
        """
        simulate one OHLCV bar for an open position.
        returns (cash_released, position_fully_closed).
        modifies pos and pos.trade in place.
        """
        p    = self.params
        t    = pos.trade
        ds   = str(r["ds"])
        op   = float(r["open"])
        hi   = float(r["high"])
        lo   = float(r["low"])
        cl   = float(r["close"])
        sma  = float(r[f"sma{p.sma_trail_period}"])

        # bars before the actual entry date are irrelevant: breakout entry type
        # queues a future entry, so the position exists in open_positions but
        # no capital is at risk yet. skip entirely without incrementing hold_count.
        if ds < t.entry_date:
            return 0.0, False

        pos.hold_count += 1

        # update excursion tracking
        t.mfe = max(t.mfe, (hi - t.entry_price) / t.entry_price)
        t.mae = min(t.mae, (lo - t.entry_price) / t.entry_price)

        # initial_stop = low of the entry day. checking lo <= stop on that exact
        # bar always fires because lo == initial_stop by definition. exit monitoring
        # starts the following bar — daily-bar convention: stop at today's low
        # only triggers if a subsequent bar's low breaks below it.
        if pos.hold_count == 1:
            return 0.0, False

        cash_out = 0.0
        risk_per_share = t.entry_price - t.initial_stop

        # ── 1. STOP CHECK (always processed first) ────────────────────────────
        if lo <= pos.current_stop:
            # gap-down: if open is already below stop, fill is at open
            fill = pos.current_stop if op >= pos.current_stop else op
            t.exit_date   = ds
            t.exit_price  = fill
            t.exit_shares = pos.shares_remaining
            t.exit_reason = "breakeven_stop" if pos.trim1_done else "stop"
            cash_out += fill * pos.shares_remaining
            pos.shares_remaining = 0
            return cash_out, True

        # ── 2. TRIM 1 CHECK (sell 1/3 at trim1_r × R) ────────────────────────
        if not pos.trim1_done and risk_per_share > 0:
            trim1_target = t.entry_price + p.trim1_r * risk_per_share
            if hi >= trim1_target:
                trim1_shares = math.floor(t.shares / 3)
                if trim1_shares > 0:
                    t.trim1_date   = ds
                    t.trim1_price  = trim1_target
                    t.trim1_shares = float(trim1_shares)
                    pos.shares_remaining -= trim1_shares
                    pos.current_stop = t.entry_price  # breakeven stop
                    pos.trim1_done   = True
                    cash_out += trim1_target * trim1_shares

        # ── 3. TRIM 2 CHECK (sell another 1/3 at trim2_r × R, optional) ──────
        if p.trim2_enabled and pos.trim1_done and not pos.trim2_done and risk_per_share > 0:
            trim2_target = t.entry_price + p.trim2_r * risk_per_share
            if hi >= trim2_target:
                trim2_shares = math.floor(t.shares / 3)
                if trim2_shares > 0 and trim2_shares <= pos.shares_remaining:
                    t.trim2_date   = ds
                    t.trim2_price  = trim2_target
                    t.trim2_shares = float(trim2_shares)
                    pos.shares_remaining -= trim2_shares
                    pos.trim2_done = True
                    cash_out += trim2_target * trim2_shares

        # ── 4. TRAILING STOP (close < SMA, only after breakeven) ─────────────
        # qullamaggie: once you've trimmed once and set stop to breakeven,
        # let the remaining position run until a close below the 10-day SMA
        if pos.trim1_done and not math.isnan(sma) and cl < sma:
            t.exit_date   = ds
            t.exit_price  = cl
            t.exit_shares = pos.shares_remaining
            t.exit_reason = "trail_sma10"
            cash_out += cl * pos.shares_remaining
            pos.shares_remaining = 0
            return cash_out, True

        # ── 5. SAFETY VALVE (max hold days) ───────────────────────────────────
        if pos.hold_count >= p.max_hold_days:
            t.exit_date   = ds
            t.exit_price  = cl
            t.exit_shares = pos.shares_remaining
            t.exit_reason = "time"
            cash_out += cl * pos.shares_remaining
            pos.shares_remaining = 0
            return cash_out, True

        return cash_out, False

    # ── main simulation loop ──────────────────────────────────────────────────

    def run(self) -> BacktestResults:
        p = self.params
        scans = self._load_scans()
        if scans.empty:
            print("no scans match parameters")
            return BacktestResults(p, [], pd.Series(dtype=float), {"total_trades": 0})

        print(f"loaded {len(scans)} signals across {scans['scan_date'].nunique()} scan dates")

        # preload market_conditions sorted by date for entry-regime lookups
        _mc_rows: list[tuple[str, str, float]] = []  # (date, regime, multiplier)
        try:
            with sqlite3.connect(DB_PATH) as _mc_conn:
                for _r in _mc_conn.execute(
                    "SELECT scan_date, regime, regime_multiplier "
                    "FROM market_conditions ORDER BY scan_date"
                ):
                    _mc_rows.append((str(_r[0] or ""), str(_r[1] or ""), float(_r[2] or 1.0)))
        except Exception:
            pass

        def _entry_regime(entry_date: str) -> tuple[str, float]:
            """return (regime, multiplier) for the most recent MC row on or before entry_date"""
            lo, hi = 0, len(_mc_rows) - 1
            result: tuple[str, float] = ("", 1.0)
            while lo <= hi:
                mid = (lo + hi) // 2
                if _mc_rows[mid][0] <= entry_date:
                    result = (_mc_rows[mid][1], _mc_rows[mid][2])
                    lo = mid + 1
                else:
                    hi = mid - 1
            return result

        # preload all price data and build the global trading calendar
        symbols = scans["symbol"].unique()
        print(f"loading price data for {len(symbols)} symbols...")
        for sym in symbols:
            self._load_prices(sym)

        # global calendar: every date that appears in any price series in range
        all_dates: set[str] = set()
        for df in self._price_cache.values():
            dates_in_range = df[(df["ds"] >= p.start_date) & (df["ds"] <= p.end_date)]["ds"]
            all_dates.update(dates_in_range.tolist())
        calendar = sorted(all_dates)

        if not calendar:
            print("no price data found for any symbol in the date range")
            return BacktestResults(p, [], pd.Series(dtype=float), {"total_trades": 0})

        # scans indexed by date for O(1) lookup
        scans_by_date: dict[str, pd.DataFrame] = {}
        for date, grp in scans.groupby("scan_date"):
            scans_by_date[str(date)] = grp

        # price data indexed by {symbol: {ds: row}} for O(1) lookup
        price_rows: dict[str, dict[str, pd.Series]] = {}
        for sym, df in self._price_cache.items():
            price_rows[sym] = {str(r["ds"]): r for _, r in df.iterrows()}

        cash          = p.initial_equity
        open_positions: list[_OpenPos] = []
        all_trades:    list[Trade]     = []
        daily_equity:  dict[str, float] = {}
        days_with_positions = 0

        for ds in calendar:
            # ── process all open positions for this bar ────────────────────────
            still_open: list[_OpenPos] = []
            for pos in open_positions:
                sym_rows = price_rows.get(pos.trade.symbol, {})
                if ds not in sym_rows:
                    still_open.append(pos)
                    continue
                released, closed = self._process_bar(pos, sym_rows[ds])
                cash += released
                if closed:
                    all_trades.append(pos.trade)
                else:
                    still_open.append(pos)
            open_positions = still_open

            # ── daily mark-to-market equity ───────────────────────────────────
            mtm = 0.0
            live_count = 0
            for pos in open_positions:
                sym_rows = price_rows.get(pos.trade.symbol, {})
                if ds < pos.trade.entry_date:
                    # position queued but not yet live (breakout entry type):
                    # hold at cost so equity stays flat during the wait
                    mtm += pos.trade.cost_basis
                elif ds in sym_rows:
                    mtm += float(sym_rows[ds]["close"]) * pos.shares_remaining
                    live_count += 1
                else:
                    mtm += pos.trade.entry_price * pos.shares_remaining
                    live_count += 1
            daily_equity[ds] = cash + mtm
            if live_count > 0:
                days_with_positions += 1

            # ── open new positions from today's scan signals ───────────────────
            if ds not in scans_by_date:
                continue

            slots          = p.max_positions - len(open_positions)
            active_symbols = {pos.trade.symbol for pos in open_positions}

            for _, row in scans_by_date[ds].iterrows():
                if slots <= 0:
                    break
                sym = str(row["symbol"])
                if sym in active_symbols:
                    continue
                prices = self._price_cache.get(sym)
                if prices is None:
                    continue

                result = self._find_entry(row, prices)
                if result is None:
                    continue
                entry_date, entry_price, initial_stop = result

                if entry_price <= 0 or initial_stop <= 0 or initial_stop >= entry_price:
                    continue

                risk_per_share   = entry_price - initial_stop
                current_equity   = cash + mtm  # mtm computed from the MTM loop above
                shares = max(1, math.floor(
                    current_equity * p.risk_per_trade / risk_per_share
                ))
                # single-position cap
                max_shares = max(1, math.floor(
                    current_equity * p.max_position_pct / entry_price
                ))
                shares = min(shares, max_shares)
                cost   = shares * entry_price
                if cost > cash:
                    continue

                _er, _erm = _entry_regime(entry_date)
                t = Trade(
                    symbol                   = sym,
                    scan_date                = str(row["scan_date"]),
                    entry_date               = entry_date,
                    entry_price              = entry_price,
                    initial_stop             = initial_stop,
                    shares                   = float(shares),
                    cost_basis               = cost,
                    adr_pct                  = float(row.get("adr_pct") or 0.06),
                    score                    = float(row.get("score") or 0),
                    raw_score                = float(row.get("raw_score") or 0),
                    grade                    = str(row.get("grade") or ""),
                    market_regime            = str(row.get("regime") or ""),
                    regime_multiplier        = float(row.get("mc_rm") or 1.0),
                    entry_regime             = _er,
                    entry_regime_multiplier  = _erm,
                    base_quality             = float(row.get("base_quality") or 0),
                    trend_strength           = float(row.get("trend_strength") or 0),
                    relative_strength        = float(row.get("relative_strength_score") or 0),
                    volume_profile           = float(row.get("volume_score") or 0),
                    risk_reward_score        = float(row.get("rr_score") or 0),
                )
                pos = _OpenPos(
                    trade            = t,
                    current_stop     = initial_stop,
                    shares_remaining = float(shares),
                )
                cash -= cost
                open_positions.append(pos)
                active_symbols.add(sym)
                slots -= 1

        # ── close any positions still open at end of data ─────────────────────
        for pos in open_positions:
            t = pos.trade
            sym_rows = price_rows.get(t.symbol, {})
            last_ds  = max(sym_rows.keys()) if sym_rows else ""
            if last_ds:
                last_row = sym_rows[last_ds]
                cl = float(last_row["close"])
                t.exit_date   = last_ds
                t.exit_price  = cl
                t.exit_shares = pos.shares_remaining
                t.exit_reason = "eod"
                cash += cl * pos.shares_remaining
            else:
                t.exit_date   = t.entry_date
                t.exit_price  = t.entry_price
                t.exit_shares = pos.shares_remaining
                t.exit_reason = "no_data"
                cash += t.entry_price * pos.shares_remaining
            all_trades.append(t)

        # final equity snapshot
        if daily_equity:
            last_date  = max(daily_equity)
            final_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            daily_equity[final_date] = cash

        eq_series = pd.Series(daily_equity).sort_index()
        metrics   = _compute_metrics(
            all_trades, p.initial_equity, eq_series,
            days_with_positions, len(calendar)
        )
        return BacktestResults(p, all_trades, eq_series, metrics)


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(
    trades: list[Trade],
    initial_equity: float,
    equity: pd.Series,
    days_with_positions: int,
    total_calendar_days: int,
) -> dict:
    closed = [t for t in trades if t.is_closed]
    if not closed:
        return {"total_trades": 0}

    r_mults  = [t.r_multiple for t in closed]
    pct_gains = [t.pct_gain  for t in closed]
    pnls     = [t.total_pnl  for t in closed]

    winners      = [g for g in pct_gains if g > 0]
    losers       = [g for g in pct_gains if g <= 0]
    gross_win    = sum(d for d in pnls if d > 0)
    gross_loss   = abs(sum(d for d in pnls if d < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    win_rate   = len(winners) / len(closed)
    avg_win_r  = float(np.mean([r for r in r_mults if r > 0])) if any(r > 0 for r in r_mults) else 0.0
    avg_loss_r = float(np.mean([r for r in r_mults if r <= 0])) if any(r <= 0 for r in r_mults) else 0.0
    # expectancy = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
    expectancy = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r

    final_equity = float(equity.iloc[-1]) if len(equity) > 0 else initial_equity
    total_return = final_equity / initial_equity - 1

    dates = pd.to_datetime(equity.index, errors="coerce").dropna()
    years = (dates[-1] - dates[0]).days / 365.25 if len(dates) >= 2 else 1.0
    cagr  = (final_equity / initial_equity) ** (1 / max(years, 1 / 365)) - 1

    daily_rets = equity.pct_change().dropna()
    std = float(daily_rets.std())
    sharpe = float(daily_rets.mean() / std * np.sqrt(_TRADING_DAYS)) if std > 0 else 0.0
    down_std = float(daily_rets[daily_rets < 0].std())
    sortino  = float(daily_rets.mean() / down_std * np.sqrt(_TRADING_DAYS)) if down_std > 0 else 0.0

    rolling_max = equity.cummax()
    dd          = (equity - rolling_max) / rolling_max
    max_dd      = float(dd.min())
    calmar      = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    # max drawdown duration in calendar days
    in_dd = dd < 0
    max_dur = cur_dur = 0
    for i, flag in enumerate(in_dd):
        if flag:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    trim1_count = sum(1 for t in closed if t.trim1_date)

    exit_reasons: dict[str, int] = {}
    for t in closed:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    time_in_market = (
        days_with_positions / total_calendar_days if total_calendar_days > 0 else 0.0
    )

    return {
        "total_trades":   len(closed),
        "winners":        len(winners),
        "losers":         len(losers),
        "win_rate":       win_rate,
        "avg_win":        float(np.mean(winners)) if winners else 0.0,
        "avg_loss":       float(np.mean(losers))  if losers  else 0.0,
        "avg_r":          float(np.mean(r_mults)),
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
        "avg_hold":       float(np.mean([t.hold_days for t in closed])),
        "avg_mfe":        float(np.mean([t.mfe for t in closed])),
        "avg_mae":        float(np.mean([t.mae for t in closed])),
        "trim1_count":    trim1_count,
        "trim1_rate":     trim1_count / len(closed),
        "total_return":   total_return,
        "cagr":           cagr,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "max_dd_duration": max_dur,
        "time_in_market": time_in_market,
        "final_equity":   final_equity,
        "exit_reasons":   exit_reasons,
    }


# ── signal validator ─────────────────────────────────────────────────────────
#
# answers the core question independently of execution mechanics:
#   "when the engine scores a stock high, does the stock go up?"
#
# entry = open of next trading day after scan date (same as next_open mode)
# forward returns measured at fixed horizons in TRADING days (not calendar)

_FWD_HORIZONS = [5, 10, 20, 60]


class SignalValidator:
    """
    Pure forward-return analysis on scan signals. No stops, no sizing.

    The trading simulation (Backtester) tests whether you make money with a
    specific execution strategy. This tests whether the underlying signals are
    predictive — the simpler and more direct question.
    """

    def __init__(self, params: BacktestParams):
        self.params = params
        self._price_cache: dict[str, pd.DataFrame] = {}

    def _load_scans(self) -> pd.DataFrame:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"database not found: {DB_PATH}")
        p = self.params
        score_expr = (
            "COALESCE(s.raw_score, s.score / NULLIF(mc.regime_multiplier, 0))"
            if p.score_col == "raw_score" else "s.score"
        )
        regime_clause, regime_params = "", []
        if p.min_regime and p.min_regime in _REGIME_ORDER:
            allowed = _REGIME_ORDER[_REGIME_ORDER.index(p.min_regime):]
            placeholders = ",".join("?" * len(allowed))
            regime_clause = f" AND (mc.regime IN ({placeholders}) OR mc.regime IS NULL)"
            regime_params = allowed
        max_score_clause, max_score_params = "", []
        if p.max_score is not None:
            max_score_clause = f" AND {score_expr} <= ?"
            max_score_params = [p.max_score]
        # signal validator uses signal_min_score (lower threshold for full distribution)
        query = f"""
            SELECT s.*, mc.regime,
                   {score_expr} AS _filter_score
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE s.passes_filters = 1
              AND s.scan_date BETWEEN ? AND ?
              AND {score_expr} >= ?
              {max_score_clause}
              {regime_clause}
            ORDER BY s.scan_date ASC, {score_expr} DESC
        """
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                query, conn,
                params=(p.start_date, p.end_date, p.signal_min_score,
                        *max_score_params, *regime_params),
            )

    def _load_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        files = list(DATA_DIR.glob(f"*/{symbol}-*.pkl"))
        if not files:
            return None
        pkl = max(files, key=lambda f: f.stem[-10:])
        try:
            df = pd.read_pickle(pkl)
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            col = "datetime" if "datetime" in df.columns else "date"
            df = df.rename(columns={col: "date"})
            if df["date"].dtype in (np.int64, np.float64, "int64", "float64"):
                df["date"] = pd.to_datetime(df["date"], unit="ms")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.normalize()
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df["ds"] = df["date"].dt.strftime("%Y-%m-%d")
            self._price_cache[symbol] = df
            return df
        except Exception:
            return None

    def _print_db_summary(self) -> None:
        """show available signal counts at common thresholds so the user can
        see how much historical data exists and pick an appropriate signal_min_score."""
        p = self.params
        score_expr = (
            "COALESCE(s.raw_score, s.score / NULLIF(mc.regime_multiplier, 0))"
            if p.score_col == "raw_score" else "s.score"
        )
        with sqlite3.connect(DB_PATH) as conn:
            thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90]
            print("  available signals by score threshold:")
            print(f"  {'threshold':>10}  {'signals':>8}  {'dates':>6}  {'symbols':>8}")
            for t in thresholds:
                row = conn.execute(f"""
                    SELECT COUNT(*), COUNT(DISTINCT s.scan_date), COUNT(DISTINCT s.symbol)
                    FROM scans s
                    LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
                    WHERE s.passes_filters = 1
                      AND s.scan_date BETWEEN ? AND ?
                      AND {score_expr} >= ?
                """, (p.start_date, p.end_date, t)).fetchone()
                marker = " <-- signal_min_score" if t == p.signal_min_score else (
                         " <-- trade min_score" if t == p.min_score else "")
                print(f"  {'>= ' + str(t):>10}  {row[0]:>8,}  {row[1]:>6}  {row[2]:>8}{marker}")
        print()

    def run(self) -> pd.DataFrame:
        """
        Returns one row per signal with raw forward returns at each horizon.
        No execution mechanics — pure price action from signal to N days out.
        """
        self._print_db_summary()
        scans = self._load_scans()
        if scans.empty:
            print("no signals found")
            return pd.DataFrame()
        print(f"validating {len(scans)} signals across "
              f"{scans['scan_date'].nunique()} scan dates "
              f"(signal_min_score={self.params.signal_min_score})...")

        records = []
        for _, row in scans.iterrows():
            sym = str(row["symbol"])
            prices = self._load_prices(sym)
            if prices is None:
                continue

            scan_date = str(row["scan_date"])
            after = prices[prices["ds"] > scan_date]
            if after.empty:
                continue

            # RangeIndex: index value = positional iloc
            entry_iloc  = int(after.index[0])
            entry_price = float(prices.iloc[entry_iloc]["open"])
            if entry_price <= 0:
                continue

            fwd: dict[str, float] = {}
            for h in _FWD_HORIZONS:
                tgt = entry_iloc + h
                fwd[f"fwd_{h}d"] = (
                    (float(prices.iloc[tgt]["close"]) - entry_price) / entry_price
                    if tgt < len(prices) else np.nan
                )

            records.append({
                "symbol":            sym,
                "scan_date":         scan_date,
                "entry_date":        str(prices.iloc[entry_iloc]["ds"]),
                "entry_price":       round(entry_price, 2),
                "score":             float(row.get("_filter_score") or 0),
                "raw_score":         float(row.get("raw_score") or 0),
                "grade":             str(row.get("grade") or ""),
                "market_regime":     str(row.get("regime") or ""),
                "base_quality":      float(row.get("base_quality") or 0),
                "trend_strength":    float(row.get("trend_strength") or 0),
                "relative_strength": float(row.get("relative_strength_score") or 0),
                "volume_profile":    float(row.get("volume_score") or 0),
                "risk_reward_score": float(row.get("rr_score") or 0),
                **fwd,
            })

        return pd.DataFrame(records)

    def print_report(self, df: pd.DataFrame) -> None:
        if df.empty:
            print("no signal data to report")
            return

        p   = self.params
        w   = 70
        sc  = p.score_col if p.score_col in df.columns else "score"

        print("\n" + "=" * w)
        print("SIGNAL VALIDITY ANALYSIS")
        print("  question: does a high score predict the stock going up?")
        print("=" * w)
        print(f"  {'period':<26} {p.start_date} to {p.end_date}")
        print(f"  {'signal score filter':<26} >= {p.signal_min_score}  ({sc})")
        print(f"  {'trade score filter':<26} >= {p.min_score}  (used by backtester)")
        print(f"  {'signals':<26} {len(df)}")
        print(f"  {'unique symbols':<26} {df['symbol'].nunique()}")
        if p.min_regime:
            print(f"  {'min regime':<26} {p.min_regime}+")
        print("-" * w)

        # ── forward return summary ────────────────────────────────────────────
        print("\nforward returns (entry open -> N-day close, no stops):")
        print(f"  {'horizon':<8} {'n':<5} {'hit %':>7} {'avg':>8} {'median':>8}"
              f" {'avg win':>8} {'avg loss':>9}")
        for h in _FWD_HORIZONS:
            col = f"fwd_{h}d"
            if col not in df.columns:
                continue
            v      = df[col].dropna()
            wins   = v[v > 0]
            losses = v[v <= 0]
            print(
                f"  {h}d{'':<5} {len(v):<5} {(v > 0).mean():>7.1%}"
                f" {v.mean():>+8.2%} {v.median():>+8.2%}"
                f" {wins.mean() if len(wins) else 0.0:>+8.2%}"
                f" {losses.mean() if len(losses) else 0.0:>+9.2%}"
            )

        # a hit rate materially above 50% confirms signals have directional edge.
        # avg > 0 confirms the edge is positive in magnitude.
        # check 20d as the primary anchor — it's long enough to matter but short
        # enough that most signals still have data.
        if "fwd_20d" in df.columns:
            v20 = df["fwd_20d"].dropna()
            hit = (v20 > 0).mean()
            note = ("EDGE CONFIRMED" if hit >= 0.55
                    else "WEAK EDGE" if hit >= 0.50
                    else "NO DIRECTIONAL EDGE")
            print(f"\n  20d hit rate {hit:.1%} -> {note}")

        # ── score correlation ─────────────────────────────────────────────────
        print(f"\n{sc} correlation with forward returns")
        print("  (positive = higher score predicts stronger gain):")
        any_signal = False
        for h in _FWD_HORIZONS:
            col   = f"fwd_{h}d"
            valid = df[[sc, col]].dropna()
            if len(valid) < 5:
                continue
            r   = valid[sc].corr(valid[col])
            bar = ("+" * max(0, int(r * 30)) if r >= 0
                   else "-" * max(0, int(abs(r) * 30)))
            note = " <-- scoring is working" if r >= 0.10 else ""
            print(f"  {h}d  {r:+.3f}  |{bar}{note}")
            if abs(r) >= 0.10:
                any_signal = True
        if not any_signal:
            print("  WARNING: score shows no meaningful correlation (<0.10) with"
                  " forward returns at any horizon.")
            print("  The scoring weights may need recalibration.")

        # ── component correlations at 20d ─────────────────────────────────────
        components = ["base_quality", "trend_strength", "relative_strength",
                      "volume_profile", "risk_reward_score"]
        avail_comp = [c for c in components if c in df.columns]
        if "fwd_20d" in df.columns and avail_comp:
            print("\ncomponent scores vs 20d forward return:")
            print("  (positive = component predicts gains, negative = inverse signal):")
            v20 = df[avail_comp + ["fwd_20d"]].dropna()
            for c in avail_comp:
                r   = v20[c].corr(v20["fwd_20d"])
                bar = ("+" * max(0, int(r * 30)) if r >= 0
                       else "-" * max(0, int(abs(r) * 30)))
                flag = " <-- useful predictor" if r >= 0.10 else (
                       " <-- INVERSE SIGNAL — consider penalizing" if r <= -0.10 else "")
                print(f"  {c:<26} {r:+.3f}  |{bar}{flag}")

        # ── score bucket breakdown ────────────────────────────────────────────
        if sc in df.columns and "fwd_20d" in df.columns:
            print("\n20d return by score bucket")
            print("  (monotonic increase = scoring is correctly ordered):")
            d   = df.copy()
            d["bucket"] = (d[sc] // 5 * 5).astype(int)
            bkt = d.groupby("bucket")["fwd_20d"].agg(
                n="count",
                hit_rate=lambda x: (x > 0).mean(),
                avg_ret="mean",
                med_ret="median",
            ).dropna()
            best_hr = bkt["hit_rate"].max()
            print(f"  {'score':<8} {'n':<5} {'hit %':>7} {'avg':>8} {'median':>8}")
            for b, r in bkt.iterrows():
                marker = " <-- best hit rate" if r["hit_rate"] == best_hr else ""
                print(
                    f"  {b:<8} {int(r['n']):<5} {r['hit_rate']:>7.1%}"
                    f" {r['avg_ret']:>+8.2%} {r['med_ret']:>+8.2%}{marker}"
                )

        # ── regime breakdown ──────────────────────────────────────────────────
        if "market_regime" in df.columns and "fwd_20d" in df.columns:
            print("\n20d return by market regime:")
            reg = df.groupby("market_regime")["fwd_20d"].agg(
                n="count",
                hit_rate=lambda x: (x > 0).mean(),
                avg_ret="mean",
                med_ret="median",
            ).dropna()
            print(f"  {'regime':<16} {'n':<5} {'hit %':>7} {'avg':>8} {'median':>8}")
            for regime, r in reg.iterrows():
                print(
                    f"  {regime:<16} {int(r['n']):<5} {r['hit_rate']:>7.1%}"
                    f" {r['avg_ret']:>+8.2%} {r['med_ret']:>+8.2%}"
                )

        # ── top / bottom ──────────────────────────────────────────────────────
        if "fwd_20d" in df.columns:
            display_cols = ["symbol", "entry_date", sc,
                            "fwd_5d", "fwd_10d", "fwd_20d", "market_regime"]
            avail_dc = [c for c in display_cols if c in df.columns]
            valid = df[avail_dc].dropna(subset=["fwd_20d"])
            if len(valid) >= 5:
                top10 = valid.nlargest(10, "fwd_20d").copy()
                bot10 = valid.nsmallest(10, "fwd_20d").copy()
                for tbl in (top10, bot10):
                    for c in ["fwd_5d", "fwd_10d", "fwd_20d"]:
                        if c in tbl.columns:
                            tbl[c] = tbl[c].map(lambda x: f"{x:+.1%}")
                print("\ntop 10 signals by 20d forward return:")
                print(top10.to_string(index=False))
                print("\nbottom 10 signals by 20d forward return:")
                print(bot10.to_string(index=False))

        print("=" * w)


# ── cli ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="qullamaggie-style breakout backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",    default="2019-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--end",      default="2099-12-31", metavar="YYYY-MM-DD")
    parser.add_argument("--min-score", type=float, default=70.0,
        help="minimum score to trade (default: 70). used by backtester.")
    parser.add_argument("--signal-min-score", type=float, default=60.0,
        help=(
            "minimum score for signal validity analysis (default: 60). "
            "lower than --min-score so the validator sees a full score distribution "
            "across historical data. historical_batch records average ~37 raw_score "
            "since they cover all nasdaq stocks, not curated watchlists."
        ))
    parser.add_argument("--max-score", type=float, default=None)
    parser.add_argument(
        "--score-col", choices=["score", "raw_score"], default="raw_score",
        help="raw_score = pre-regime setup quality (recommended); score = regime-adjusted",
    )
    parser.add_argument(
        "--entry", choices=["next_open", "breakout"], default="next_open",
        dest="entry_type",
    )
    parser.add_argument("--breakout-window", type=int, default=15,
        help="days to wait for breakout confirmation (entry=breakout only)")
    parser.add_argument("--trim1-r",    type=float, default=3.0,
        help="R-multiple at which to sell 1/3 and move stop to breakeven")
    parser.add_argument("--trim2-r",    type=float, default=6.0)
    parser.add_argument("--trim2",      action="store_true", dest="trim2_enabled",
        help="enable second trim at trim2-r")
    parser.add_argument("--sma-trail",  type=int,   default=10,
        help="SMA period for trailing stop (after breakeven)")
    parser.add_argument("--max-hold",   type=int,   default=60,
        help="safety valve: force exit after N days")
    parser.add_argument("--positions",  type=int,   default=10)
    parser.add_argument("--risk",       type=float, default=0.005,
        help="fraction of equity risked per trade (Qullamaggie uses 0.003-0.005)")
    parser.add_argument("--equity",     type=float, default=100_000.0)
    parser.add_argument(
        "--min-regime", default="",
        choices=["", "DOWNTREND", "CAUTION", "MIXED", "UPTREND", "BULL"],
    )
    parser.add_argument("--csv", metavar="FILE",
        help="save output to CSV (trade log for 'trade' mode, signal data for 'signal' mode)")
    parser.add_argument(
        "--mode", choices=["trade", "signal", "both"], default="trade",
        help=(
            "trade  = full trading simulation with stops/sizing (default); "
            "signal = pure forward-return validity check (did scored stocks go up?); "
            "both   = run signal analysis first, then trading simulation"
        ),
    )
    args = parser.parse_args()

    params = BacktestParams(
        start_date        = args.start,
        end_date          = args.end,
        min_score         = args.min_score,
        signal_min_score  = args.signal_min_score,
        max_score         = args.max_score,
        score_col        = args.score_col,
        entry_type       = args.entry_type,
        breakout_window  = args.breakout_window,
        trim1_r          = args.trim1_r,
        trim2_r          = args.trim2_r,
        trim2_enabled    = args.trim2_enabled,
        sma_trail_period = args.sma_trail,
        max_hold_days    = args.max_hold,
        max_positions    = args.positions,
        risk_per_trade   = args.risk,
        initial_equity   = args.equity,
        min_regime       = args.min_regime,
    )

    if args.mode in ("signal", "both"):
        validator = SignalValidator(params)
        sig_df    = validator.run()
        validator.print_report(sig_df)
        if args.csv and args.mode == "signal":
            sig_df.to_csv(args.csv, index=False)
            print(f"\nsignal data saved to {args.csv}")

    if args.mode in ("trade", "both"):
        results = Backtester(params).run()
        results.print_report()
        if args.csv and args.mode in ("trade", "both"):
            results.to_dataframe().to_csv(args.csv, index=False)
            print(f"\ntrade log saved to {args.csv}")
