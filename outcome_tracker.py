"""
Outcome Tracker — measures what happened to flagged setups after the scan date.

For every scan that:
  - passed hard filters
  - has no outcomes recorded yet
  - is at least min_days_old calendar days old

...the tracker loads the latest available price data for that stock, slices to
the post-scan window, and records:
  - Whether the breakout level was crossed (close >= breakout_level)
  - Whether the stop was hit      (low  <= stop_level)
  - Whether the target was reached (high >= target_level)
  - Number of trading days to each event
  - Max gain and max drawdown at 10 / 20 / 60 day windows

Price data comes from the pickle cache in data/.  Each pickle covers ~365 days
of daily OHLCV, so the latest pickle for any stock contains all post-scan data
from recent scans without requiring any additional API calls.

Run directly:
    python outcome_tracker.py
    python outcome_tracker.py --min-days 5
    python outcome_tracker.py --db results/breakout.db
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from persistence import ScanPersistence


class OutcomeTracker:
    def __init__(self, db: ScanPersistence, data_dir: str = "data"):
        self.db       = db
        self.data_dir = Path(data_dir)

    # ── Public ───────────────────────────────────────────────────────────────

    def update(self, min_days_old: int = 10) -> int:
        """
        Evaluate all pending scans and write outcomes to the database.

        Returns: number of outcome records saved.
        """
        pending = self.db.get_pending_outcomes(min_days_old)

        if pending.empty:
            print("No pending outcomes to evaluate.")
            return 0

        print(f"Evaluating outcomes for {len(pending)} pending scan(s)...\n")

        # Cache loaded DataFrames so each symbol's pickle is read at most once.
        price_cache: dict[str, Optional[pd.DataFrame]] = {}
        outcomes = []

        for _, scan in pending.iterrows():
            symbol    = scan["symbol"]
            scan_date = scan["scan_date"]

            if symbol not in price_cache:
                price_cache[symbol] = self._load_latest_prices(symbol)

            df = price_cache[symbol]
            if df is None or df.empty:
                print(f"  ⚠  {symbol}: no price data found, skipping")
                continue

            # Slice to trading days strictly after the scan date
            try:
                future_df = df[df.index > pd.Timestamp(scan_date)]
            except Exception:
                continue

            if future_df.empty:
                print(f"  ─  {symbol} ({scan_date}): no post-scan data yet")
                continue

            outcome = self._compute_outcome(scan, future_df)
            if outcome is None:
                continue

            outcomes.append(outcome)

            # One-line summary per evaluated scan
            bo   = "✓ BREAKOUT" if outcome["breakout_triggered"] else "─"
            st   = "✗ STOP"     if outcome["stop_triggered"]     else "─"
            tgt  = "◎ TARGET"   if outcome["target_reached"]     else "─"
            g20  = outcome["max_gain_20d"]
            g20s = f"{g20:+.1%}" if g20 is not None else "n/a"
            print(
                f"  {symbol:6s} ({scan_date})  {bo:10s}  {st:7s}  "
                f"{tgt:9s}  max20d={g20s}"
            )

        saved = self.db.save_outcomes(outcomes)
        print(f"\n✓ Saved {saved} outcome record(s)")
        return saved

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_latest_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Find and load the most recent pickle for a symbol across all
        data/<date>/ subdirectories.  Returns a DatetimeIndex DataFrame
        or None if no pickle is found.
        """
        candidates = sorted(
            self.data_dir.glob(f"*/{symbol}-*.pkl"),
            key=lambda p: p.parent.name,   # dir names are YYYY-MM-DD → sorts correctly
            reverse=True,
        )
        if not candidates:
            return None

        with open(candidates[0], "rb") as f:
            df = pickle.load(f)

        # Normalise to DatetimeIndex — mirrors the logic in engine.py
        if "datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            df = df.set_index("datetime")
            df.index = df.index.normalize()

        return df

    def _compute_outcome(
        self, scan: pd.Series, future_df: pd.DataFrame
    ) -> Optional[dict]:
        """
        Compute all outcome metrics from post-scan OHLCV data.

        Breakout / target checks use the daily high (intraday touch counts).
        Stop checks use the daily low.
        Gain / drawdown calculations use close prices.
        """
        entry_price    = scan.get("price")
        breakout_level = scan.get("breakout_level")
        stop_level     = scan.get("stop_level")
        target_level   = scan.get("target_level")

        if not entry_price or entry_price <= 0:
            return None

        close  = future_df["close"]
        high   = future_df["high"]
        low    = future_df["low"]
        n_days = len(future_df)

        volume = future_df["volume"] if "volume" in future_df.columns else None

        # ── Breakout: close crosses above breakout level ──────────────────────
        breakout_triggered = False
        days_to_breakout: Optional[int] = None
        breakout_day_rel_volume: Optional[float] = None
        if breakout_level and not pd.isna(breakout_level):
            mask = close >= breakout_level
            if mask.any():
                breakout_triggered = True
                bo_idx = int(mask.to_numpy().argmax())
                days_to_breakout = bo_idx + 1
                # record volume expansion on breakout day vs 20-day rolling average
                # high-conviction breakouts show 150-200%+ of average volume
                if volume is not None:
                    avg_vol_20 = float(volume.rolling(20).mean().iloc[bo_idx])
                    bo_vol = float(volume.iloc[bo_idx])
                    if avg_vol_20 > 0:
                        breakout_day_rel_volume = bo_vol / avg_vol_20

        # ── Stop: intraday low touches or undercuts stop level ────────────────
        stop_triggered = False
        days_to_stop: Optional[int] = None
        if stop_level and not pd.isna(stop_level):
            mask = low <= stop_level
            if mask.any():
                stop_triggered = True
                days_to_stop   = int(mask.to_numpy().argmax()) + 1

        # ── Target: intraday high reaches target level ────────────────────────
        target_reached = False
        if target_level and not pd.isna(target_level):
            target_reached = bool((high >= target_level).any())

        # ── Gain / drawdown at fixed windows ─────────────────────────────────
        # Use intraday high/low, not close — a stock that spikes 20% intraday
        # then closes +5% should show +20% gain; a stop-out at the low of day
        # should show the actual drawdown, not the gentler close-to-close figure.
        def max_gain(n: int) -> Optional[float]:
            w = high.iloc[:n]
            return float((w.max() - entry_price) / entry_price) if not w.empty else None

        def max_drawdown(n: int) -> Optional[float]:
            w = low.iloc[:n]
            return float((w.min() - entry_price) / entry_price) if not w.empty else None

        current_price = float(close.iloc[-1])

        return {
            "scan_date":          scan["scan_date"],
            "symbol":             scan["symbol"],
            "outcome_date":       future_df.index[-1].strftime("%Y-%m-%d"),
            "days_elapsed":       n_days,
            "entry_price":        entry_price,
            "current_price":      current_price,
            "pct_change":         (current_price - entry_price) / entry_price,
            "breakout_triggered": int(breakout_triggered),
            "stop_triggered":     int(stop_triggered),
            "target_reached":     int(target_reached),
            "days_to_breakout":   days_to_breakout,
            "days_to_stop":       days_to_stop,
            "max_gain_10d":            max_gain(10),
            "max_gain_20d":            max_gain(20),
            "max_gain_60d":            max_gain(60),
            "max_drawdown_20d":        max_drawdown(20),
            "breakout_day_rel_volume": breakout_day_rel_volume,
        }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update scan outcomes from cached price data"
    )
    parser.add_argument(
        "--min-days", type=int, default=10,
        help="Minimum calendar days since scan before evaluating (default: 10)",
    )
    parser.add_argument(
        "--db", default="results/breakout.db",
        help="Path to SQLite database (default: results/breakout.db)",
    )
    args = parser.parse_args()

    db      = ScanPersistence(db_path=args.db)
    tracker = OutcomeTracker(db)
    tracker.update(min_days_old=args.min_days)
    db.summary()
