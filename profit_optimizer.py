"""
profit_optimizer.py -signal filtering, ATR position sizing, exit strategy comparison.

builds on top of the scan signals in results/breakout.db to answer:
  "which exit strategy maximizes expectancy on the quality-filtered signal set?"

phases:
  1. signal filtering  -composite QUALITY_SCORE (0-100); reject below MIN_QUALITY_SCORE
  2. position sizing   -ATR-based with portfolio risk cap and max position size cap
  3. exit optimization -compare three strategies: fixed-R, ATR-trail, time-stop
  4. regime filter     -optional SPY-above-50SMA gate (--spy-filter)

usage:
    python profit_optimizer.py
    python profit_optimizer.py --min-quality 65 --spy-filter
    python profit_optimizer.py --start 2024-01-01 --r-targets 1.5 2.0 3.0
    python profit_optimizer.py --csv results/trades_opt.csv
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DB_PATH  = Path("results/breakout.db")
DATA_DIR = Path("data")

# sub-component maxes from CLAUDE.md (denominators for normalization)
_SUB_MAX = {
    "volume_score":            30.0,
    "relative_strength_score": 30.0,
    "base_quality":            20.0,
}

_REGIME_ORDER = ["DOWNTREND", "CAUTION", "MIXED", "UPTREND", "BULL"]


# ── quality score config ─────────────────────────────────────────────────────

@dataclass
class QualityConfig:
    # component weights -must sum to 1.0
    # ordered by empirical predictive power from DB analysis (CLAUDE.md)
    weight_volume:      float = 0.35   # volume_score: dominant predictor, config weight=50
    weight_rs:          float = 0.25   # rs_score: r=+0.076, config weight=25
    weight_prior_move:  float = 0.25   # strongest single feature r=+0.287
    weight_tightness:   float = 0.10   # base_quality: r=-0.023 (weak, still useful as gate)
    weight_consol:      float = 0.05   # consol_days structure
    # threshold -signals below this are logged and rejected
    min_quality_score:  float = 50.0
    # prior_move scaling: 75% filter floor = 0 pts; 300%+ = 100 pts
    prior_move_floor:   float = 0.75
    prior_move_ceil:    float = 3.00
    # consol_days: 35-60d is empirically optimal per DB analysis (CLAUDE.md)
    consol_optimal_low:  int  = 35
    consol_optimal_high: int  = 60
    consol_max:          int  = 90    # stale base beyond this


# ── simulation params ────────────────────────────────────────────────────────

@dataclass
class SimParams:
    start_date:         str   = "2019-01-01"
    end_date:           str   = "2099-12-31"
    min_engine_score:   float = 70.0     # engine raw_score gate (applied before quality)
    min_engine_regime:  str   = "CAUTION"
    min_consol_days:    int   = 5
    # position sizing (phase 2)
    initial_equity:     float = 100_000.0
    risk_pct:           float = 0.01     # fraction of portfolio risked per trade (default 1%)
    max_position_pct:   float = 0.05     # max single position as fraction of equity (default 5%)
    # ATR params (used for trailing stop and sizing reference)
    atr_period:         int   = 14
    # strategy 1: fixed-R exits (each r_target is a separate sub-strategy)
    r_targets:          list  = field(default_factory=lambda: [1.5, 2.0, 3.0])
    # strategy 2: ATR trailing stop
    atr_trail_mult:     float = 2.0     # trail = high_watermark - N * ATR
    atr_trail_activate: float = 1.0     # R profit required before trail activates
    # strategy 3: time-based exit
    time_stop_days:     int   = 15      # exit after N days (overrides trail/fixed if sooner)
    # universal safety valve for all strategies
    max_hold_days:      int   = 30
    # optional SPY 50-SMA regime filter (phase 4)
    spy_filter:         bool  = False
    # quality config (phase 1)
    quality:            QualityConfig = field(default_factory=QualityConfig)


# ── per-trade exit result ────────────────────────────────────────────────────

@dataclass
class ExitResult:
    ticker:        str
    scan_date:     str
    entry_date:    str
    quality_score: float
    raw_score:     float
    entry_price:   float
    stop_price:    float
    exit_price:    float
    exit_reason:   str
    r_multiple:    float
    pct_gain:      float
    duration_days: int
    atr:           float
    shares:        float
    position_value: float
    strategy:      str


# ── quality scorer (phase 1) ──────────────────────────────────────────────────

class QualityScorer:
    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg

    def score(self, row: pd.Series) -> tuple[float, str]:
        """
        returns (quality_score 0-100, filter_reason).
        filter_reason is empty string when signal passes.
        """
        cfg = self.cfg

        # volume component -normalize raw score to 0-1
        vol_pct = min(1.0, float(row.get("volume_score") or 0) / _SUB_MAX["volume_score"])

        # RS component
        rs_pct = min(1.0, float(row.get("relative_strength_score") or 0) / _SUB_MAX["relative_strength_score"])

        # prior move -rescale between floor and ceil
        pm = float(row.get("prior_move_pct") or 0)
        span = cfg.prior_move_ceil - cfg.prior_move_floor
        pm_pct = min(1.0, max(0.0, (pm - cfg.prior_move_floor) / span)) if span > 0 else 0.0

        # base tightness -base_quality normalized
        tightness_pct = min(1.0, float(row.get("base_quality") or 0) / _SUB_MAX["base_quality"])

        # consolidation days -35-60d is optimal, linear ramp outside that range
        cd = int(row.get("consol_days") or 0)
        if cfg.consol_optimal_low <= cd <= cfg.consol_optimal_high:
            cd_pct = 1.0
        elif cd < cfg.consol_optimal_low:
            cd_pct = max(0.0, (cd - 5) / max(1, cfg.consol_optimal_low - 5))
        else:
            cd_pct = max(0.0, 1.0 - (cd - cfg.consol_optimal_high) / max(1, cfg.consol_max - cfg.consol_optimal_high))

        quality = 100.0 * (
            cfg.weight_volume     * vol_pct +
            cfg.weight_rs         * rs_pct +
            cfg.weight_prior_move * pm_pct +
            cfg.weight_tightness  * tightness_pct +
            cfg.weight_consol     * cd_pct
        )
        quality = round(quality, 1)

        reason = ""
        if quality < cfg.min_quality_score:
            reason = (
                f"quality={quality:.1f} < {cfg.min_quality_score}  "
                f"[vol={vol_pct:.0%} rs={rs_pct:.0%} pm={pm_pct:.0%} "
                f"tight={tightness_pct:.0%} cd={cd_pct:.0%}]"
            )
        return quality, reason


# ── profit optimizer ─────────────────────────────────────────────────────────

class ProfitOptimizer:
    def __init__(self, params: SimParams | None = None):
        self.params = params or SimParams()
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._scorer = QualityScorer(self.params.quality)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_scans(self) -> pd.DataFrame:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"db not found: {DB_PATH}")
        p = self.params
        regime_clause = ""
        regime_params: list = []
        if p.min_engine_regime and p.min_engine_regime in _REGIME_ORDER:
            allowed = _REGIME_ORDER[_REGIME_ORDER.index(p.min_engine_regime):]
            ph = ",".join("?" * len(allowed))
            regime_clause = f" AND (mc.regime IN ({ph}) OR mc.regime IS NULL)"
            regime_params = allowed
        consol_clause = (
            f" AND COALESCE(s.consol_days, 0) >= {p.min_consol_days}"
            if p.min_consol_days else ""
        )
        score_expr = "COALESCE(s.raw_score, s.score / NULLIF(mc.regime_multiplier, 0))"
        query = f"""
            SELECT s.*, mc.regime, mc.regime_multiplier,
                   {score_expr} AS _raw_score
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE s.passes_filters = 1
              AND s.scan_date BETWEEN ? AND ?
              AND {score_expr} >= ?
              {regime_clause}
              {consol_clause}
            ORDER BY s.scan_date ASC, {score_expr} DESC
        """
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                query, conn,
                params=(p.start_date, p.end_date, p.min_engine_score, *regime_params),
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

    def _load_spy(self) -> Optional[pd.DataFrame]:
        """load SPY price data and compute 50-SMA for regime filter"""
        for sym in ("SPY", "$SPX", "COMPX", "QQQ"):
            prices = self._load_prices(sym)
            if prices is not None:
                df = prices.copy()
                df["sma50"] = df["close"].rolling(50, min_periods=1).mean()
                return df
        return None

    # ── ATR + entry finder ────────────────────────────────────────────────────

    def _compute_atr(self, prices: pd.DataFrame, entry_date: str, period: int) -> float:
        sub = prices[prices["ds"] <= entry_date].tail(period + 1)
        if len(sub) < 2:
            return 0.0
        hi = sub["high"].values.astype(float)
        lo = sub["low"].values.astype(float)
        cl = sub["close"].values.astype(float)
        tr = np.maximum(
            hi[1:] - lo[1:],
            np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1]))
        )
        tail = tr[-period:] if len(tr) >= period else tr
        return float(np.mean(tail)) if len(tail) > 0 else 0.0

    def _find_entry(
        self, scan_date: str, prices: pd.DataFrame
    ) -> Optional[tuple[str, float, float]]:
        """returns (entry_date, entry_price, initial_stop) or None"""
        after = prices[prices["ds"] > scan_date]
        if after.empty:
            return None
        r = after.iloc[0]
        entry = float(r["open"])
        stop  = float(r["low"])
        # enforce minimum stop distance -near-doji entry days inflate share count
        min_dist = entry * 0.02
        if entry - stop < min_dist:
            stop = entry - min_dist
        return str(r["ds"]), entry, stop

    # ── position sizing (phase 2) ─────────────────────────────────────────────

    def _size_position(self, entry: float, stop: float, equity: float) -> float:
        """
        shares = (equity * risk_pct) / (entry - stop)
        capped at max_position_pct * equity
        returns share count (float, caller rounds down)
        """
        p = self.params
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0.0
        shares_by_risk = equity * p.risk_pct / risk_per_share
        shares_by_size = equity * p.max_position_pct / entry
        return min(shares_by_risk, shares_by_size)

    # ── exit strategy 1: fixed R-multiple ─────────────────────────────────────

    def _sim_fixed_r(
        self,
        entry_date: str,
        entry: float,
        stop: float,
        r_target: float,
        prices: pd.DataFrame,
    ) -> tuple[float, float, int, str]:
        """returns (exit_price, r_multiple, bars_held, exit_reason)"""
        p    = self.params
        risk = entry - stop
        if risk <= 0:
            return entry, 0.0, 0, "invalid_stop"
        target = entry + r_target * risk
        bars   = prices[prices["ds"] > entry_date].reset_index(drop=True)
        for i, (_, bar) in enumerate(bars.iterrows()):
            op = float(bar["open"])
            hi = float(bar["high"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            if lo <= stop:
                fill = stop if op >= stop else op
                return fill, (fill - entry) / risk, i + 1, "stop"
            if hi >= target:
                return target, r_target, i + 1, f"target_{r_target}R"
            if i + 1 >= p.max_hold_days:
                return cl, (cl - entry) / risk, i + 1, "time"
        if not bars.empty:
            cl = float(bars.iloc[-1]["close"])
            return cl, (cl - entry) / risk, len(bars), "eod"
        return entry, 0.0, 0, "no_data"

    # ── exit strategy 2: ATR trailing stop ────────────────────────────────────

    def _sim_atr_trail(
        self,
        entry_date: str,
        entry: float,
        stop: float,
        atr: float,
        prices: pd.DataFrame,
    ) -> tuple[float, float, int, str]:
        p          = self.params
        risk       = entry - stop
        if risk <= 0 or atr <= 0:
            return entry, 0.0, 0, "invalid"
        trail_dist    = p.atr_trail_mult * atr
        activate_at   = entry + p.atr_trail_activate * risk
        trail_active  = False
        trail_stop    = stop
        high_water    = entry
        bars          = prices[prices["ds"] > entry_date].reset_index(drop=True)
        for i, (_, bar) in enumerate(bars.iterrows()):
            op = float(bar["open"])
            hi = float(bar["high"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            high_water = max(high_water, hi)
            if not trail_active and hi >= activate_at:
                trail_active = True
            if trail_active:
                trail_stop = max(trail_stop, high_water - trail_dist)
            current_stop = trail_stop if trail_active else stop
            if lo <= current_stop:
                fill = current_stop if op >= current_stop else op
                reason = "trail_stop" if trail_active else "stop"
                return fill, (fill - entry) / risk, i + 1, reason
            if i + 1 >= p.max_hold_days:
                return cl, (cl - entry) / risk, i + 1, "time"
        if not bars.empty:
            cl = float(bars.iloc[-1]["close"])
            return cl, (cl - entry) / risk, len(bars), "eod"
        return entry, 0.0, 0, "no_data"

    # ── exit strategy 3: time-based stop ──────────────────────────────────────

    def _sim_time_stop(
        self,
        entry_date: str,
        entry: float,
        stop: float,
        prices: pd.DataFrame,
    ) -> tuple[float, float, int, str]:
        p    = self.params
        risk = entry - stop
        if risk <= 0:
            return entry, 0.0, 0, "invalid_stop"
        bars = prices[prices["ds"] > entry_date].reset_index(drop=True)
        for i, (_, bar) in enumerate(bars.iterrows()):
            op = float(bar["open"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            if lo <= stop:
                fill = stop if op >= stop else op
                return fill, (fill - entry) / risk, i + 1, "stop"
            if i + 1 >= p.time_stop_days:
                return cl, (cl - entry) / risk, i + 1, f"time_{p.time_stop_days}d"
            if i + 1 >= p.max_hold_days:
                return cl, (cl - entry) / risk, i + 1, "time"
        if not bars.empty:
            cl = float(bars.iloc[-1]["close"])
            return cl, (cl - entry) / risk, len(bars), "eod"
        return entry, 0.0, 0, "no_data"

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades: list[ExitResult]) -> dict:
        if not trades:
            return {}
        rm   = [t.r_multiple for t in trades]
        pcts = [t.pct_gain   for t in trades]
        wins  = [r for r in rm if r > 0]
        losses = [r for r in rm if r <= 0]
        wr  = len(wins) / len(rm)
        aw  = float(np.mean(wins))   if wins   else 0.0
        al  = float(np.mean(losses)) if losses else 0.0
        exp = wr * aw + (1 - wr) * al
        gw  = sum(r for r in rm if r > 0)
        gl  = abs(sum(r for r in rm if r < 0))
        pf  = gw / gl if gl > 0 else float("inf")
        # simplified equity curve: 1 unit risked per trade at risk_pct of equity
        rp = self.params.risk_pct
        equity = np.ones(len(rm) + 1)
        for i, r in enumerate(rm):
            equity[i + 1] = equity[i] * (1 + r * rp)
        rolling_max = np.maximum.accumulate(equity)
        dd = (equity - rolling_max) / rolling_max
        max_dd = float(dd.min())
        exit_reasons: dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        return {
            "n":             len(trades),
            "win_rate":      wr,
            "avg_win_r":     aw,
            "avg_loss_r":    al,
            "expectancy":    round(exp, 3),
            "profit_factor": round(pf, 2),
            "avg_r":         round(float(np.mean(rm)), 3),
            "avg_gain_pct":  round(float(np.mean(pcts)), 4),
            "avg_hold_days": round(float(np.mean([t.duration_days for t in trades])), 1),
            "max_drawdown":  round(max_dd, 4),
            "exit_reasons":  exit_reasons,
        }

    # ── SPY regime check (phase 4) ────────────────────────────────────────────

    def _spy_above_50sma(self, spy: Optional[pd.DataFrame], date: str) -> bool:
        if spy is None:
            return True
        row = spy[spy["ds"] <= date]
        if row.empty:
            return True
        last = row.iloc[-1]
        return float(last["close"]) >= float(last["sma50"])

    # ── reporting ─────────────────────────────────────────────────────────────

    def _print_summary(
        self,
        total_signals: int,
        spy_filtered: int,
        quality_filtered: int,
        passed: int,
        by_strategy: dict[str, dict],
        quality_scores: list[float],
    ) -> None:
        p = self.params
        w = 74
        print("\n" + "=" * w)
        print("PROFIT OPTIMIZER - SIGNAL FILTERING & EXIT STRATEGY COMPARISON")
        print("=" * w)
        print(f"  period:              {p.start_date} to {p.end_date}")
        print(f"  engine score filter: >= {p.min_engine_score} (raw_score)")
        print(f"  engine regime gate:  >= {p.min_engine_regime}")
        print(f"  min consol days:     {p.min_consol_days}")
        print(f"  spy 50-SMA filter:   {'on' if p.spy_filter else 'off'}")
        print(f"  min quality score:   {p.quality.min_quality_score}")
        print(f"  risk per trade:      {p.risk_pct:.1%}")
        print(f"  max position:        {p.max_position_pct:.1%}")
        print()

        # phase 1 filter stats
        print("phase 1 - signal filtering:")
        print(f"  signals from DB:     {total_signals}")
        if p.spy_filter:
            print(f"  spy-filtered out:    {spy_filtered}")
        print(f"  quality-filtered out: {quality_filtered}")
        print(f"  signals passed:      {passed}")
        if quality_scores:
            print(f"  quality dist:        min={min(quality_scores):.1f}  "
                  f"mean={np.mean(quality_scores):.1f}  max={max(quality_scores):.1f}")

        # phase 2 sizing note
        print()
        print("phase 2 - position sizing formula:")
        print(f"  shares = (equity * {p.risk_pct:.1%}) / (entry - stop)")
        print(f"  capped at {p.max_position_pct:.1%} of equity per position")
        print(f"  (see 'position_value' column in trade log CSV)")

        # phase 3 comparison table
        print()
        print("phase 3 - exit strategy comparison:")
        print(f"  {'strategy':<22} {'n':>5} {'wr%':>6} {'avgR':>7} {'exp':>7} "
              f"{'PF':>6} {'maxDD':>7} {'hold':>6}")
        print("  " + "-" * 68)

        best_exp  = max((m.get("expectancy", -99) for m in by_strategy.values()), default=0)
        order     = sorted(by_strategy, key=lambda s: -by_strategy[s].get("expectancy", -99))
        for s in order:
            m = by_strategy[s]
            if not m:
                continue
            marker = " *" if abs(m["expectancy"] - best_exp) < 0.001 else "  "
            dd_str = f"{m['max_drawdown']:.1%}"
            print(
                f"  {s:<22} {m['n']:>5} {m['win_rate']:>6.1%} {m['avg_r']:>+7.3f}"
                f" {m['expectancy']:>+7.3f} {m['profit_factor']:>6.2f} {dd_str:>7}"
                f" {m['avg_hold_days']:>5.1f}d{marker}"
            )
        print("  (* = best expectancy)")

        # best strategy deep-dive
        if order:
            best = order[0]
            m    = by_strategy[best]
            print()
            print(f"best strategy: {best}")
            print(f"  expectancy:    {m['expectancy']:+.3f}R/trade")
            print(f"  win rate:      {m['win_rate']:.1%}  |  avg win: {m['avg_win_r']:+.2f}R  "
                  f"|  avg loss: {m['avg_loss_r']:+.2f}R")
            print(f"  profit factor: {m['profit_factor']:.2f}")
            print(f"  max drawdown:  {m['max_drawdown']:.2%}")
            print("  exit reasons:")
            for reason, cnt in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason:<28} {cnt:4d}  ({cnt / m['n']:.0%})")

        print("=" * w)

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self, csv_path: Optional[str] = None) -> list[ExitResult]:
        p = self.params
        scans = self._load_scans()
        if scans.empty:
            print("no signals matched engine score/regime filters -check DB and thresholds")
            return []

        print(f"loaded {len(scans)} signals across {scans['scan_date'].nunique()} scan dates")

        # preload all price data
        symbols = scans["symbol"].unique()
        print(f"loading price data for {len(symbols)} symbols...")
        for sym in symbols:
            self._load_prices(sym)

        spy = self._load_spy() if p.spy_filter else None
        if p.spy_filter:
            status = "loaded" if spy is not None else "not found -filter disabled"
            print(f"spy 50-SMA filter: {status}")

        all_results: list[ExitResult]   = []
        quality_passed_scores: list[float] = []
        spy_filtered     = 0
        quality_filtered = 0
        no_price_data    = 0

        for _, row in scans.iterrows():
            sym    = str(row["symbol"])
            prices = self._price_cache.get(sym)
            if prices is None:
                no_price_data += 1
                continue

            scan_date = str(row["scan_date"])

            # phase 4: SPY regime gate
            if p.spy_filter and not self._spy_above_50sma(spy, scan_date):
                spy_filtered += 1
                continue

            # phase 1: quality filter
            quality, reason = self._scorer.score(row)
            if reason:
                quality_filtered += 1
                continue
            quality_passed_scores.append(quality)

            # find entry
            entry_result = self._find_entry(scan_date, prices)
            if entry_result is None:
                continue
            entry_date, entry_price, stop_price = entry_result
            risk = entry_price - stop_price
            if risk <= 0:
                continue

            # phase 2: compute position size
            raw_score = float(row.get("_raw_score") or row.get("raw_score") or 0)
            shares    = self._size_position(entry_price, stop_price, p.initial_equity)
            pos_value = math.floor(shares) * entry_price

            # ATR (used for strategy 2 and logged for all)
            atr = self._compute_atr(prices, entry_date, p.atr_period)
            if atr <= 0:
                atr = risk   # fallback: use stop distance as proxy

            def _make(strategy: str, exit_p: float, r_mult: float, dur: int, reason: str) -> ExitResult:
                return ExitResult(
                    ticker         = sym,
                    scan_date      = scan_date,
                    entry_date     = entry_date,
                    quality_score  = quality,
                    raw_score      = raw_score,
                    entry_price    = round(entry_price, 2),
                    stop_price     = round(stop_price, 2),
                    exit_price     = round(exit_p, 2),
                    exit_reason    = reason,
                    r_multiple     = round(r_mult, 3),
                    pct_gain       = round((exit_p - entry_price) / entry_price, 4),
                    duration_days  = dur,
                    atr            = round(atr, 4),
                    shares         = math.floor(shares),
                    position_value = round(pos_value, 2),
                    strategy       = strategy,
                )

            # strategy 1: fixed-R targets
            for rt in p.r_targets:
                ep, rm, dur, rsn = self._sim_fixed_r(entry_date, entry_price, stop_price, rt, prices)
                all_results.append(_make(f"fixed_{rt}R", ep, rm, dur, rsn))

            # strategy 2: ATR trailing stop
            ep, rm, dur, rsn = self._sim_atr_trail(entry_date, entry_price, stop_price, atr, prices)
            all_results.append(_make("atr_trail", ep, rm, dur, rsn))

            # strategy 3: time-based stop
            ep, rm, dur, rsn = self._sim_time_stop(entry_date, entry_price, stop_price, prices)
            all_results.append(_make(f"time_{p.time_stop_days}d", ep, rm, dur, rsn))

        if no_price_data > 0:
            print(f"  skipped {no_price_data} signals with no pickle data")

        if not all_results:
            print("no trades generated after filtering -try lowering min_quality_score")
            return []

        # group and compute metrics per strategy
        by_strategy: dict[str, list[ExitResult]] = {}
        for r in all_results:
            by_strategy.setdefault(r.strategy, []).append(r)
        metrics = {s: self._compute_metrics(trades) for s, trades in by_strategy.items()}

        passed = len(quality_passed_scores)
        self._print_summary(
            total_signals    = len(scans),
            spy_filtered     = spy_filtered,
            quality_filtered = quality_filtered,
            passed           = passed,
            by_strategy      = metrics,
            quality_scores   = quality_passed_scores,
        )

        if csv_path:
            df = pd.DataFrame([
                {
                    "ticker":         r.ticker,
                    "quality_score":  r.quality_score,
                    "raw_score":      r.raw_score,
                    "strategy":       r.strategy,
                    "entry_date":     r.entry_date,
                    "entry_price":    r.entry_price,
                    "stop_price":     r.stop_price,
                    "exit_price":     r.exit_price,
                    "exit_reason":    r.exit_reason,
                    "r_multiple":     r.r_multiple,
                    "pct_gain":       f"{r.pct_gain:.2%}",
                    "duration_days":  r.duration_days,
                    "atr":            r.atr,
                    "shares":         int(r.shares),
                    "position_value": r.position_value,
                }
                for r in all_results
            ])
            df.to_csv(csv_path, index=False)
            n_strat = len(by_strategy)
            n_signals = passed
            print(f"\ntrade log saved to {csv_path}")
            print(f"  {len(df)} rows ({n_signals} signals x {n_strat} strategies)")

        return all_results


# ── cli ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="profit optimizer -signal quality filter + exit strategy comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",        default="2019-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--end",          default="2099-12-31", metavar="YYYY-MM-DD")
    parser.add_argument("--min-score",    type=float, default=70.0,
        help="minimum engine raw_score (first gate before quality filter)")
    parser.add_argument("--min-regime",   default="CAUTION",
        choices=["", "DOWNTREND", "CAUTION", "MIXED", "UPTREND", "BULL"])
    parser.add_argument("--min-consol",   type=int, default=5,
        help="minimum consol_days required")
    parser.add_argument("--min-quality",  type=float, default=50.0,
        help="minimum QUALITY_SCORE (0-100) to allow entry")
    # quality weights
    parser.add_argument("--w-volume",     type=float, default=0.35,
        help="quality weight: volume_score component")
    parser.add_argument("--w-rs",         type=float, default=0.25,
        help="quality weight: relative_strength_score component")
    parser.add_argument("--w-prior-move", type=float, default=0.25,
        help="quality weight: prior_move_pct component")
    parser.add_argument("--w-tightness",  type=float, default=0.10,
        help="quality weight: base_quality (tightness) component")
    parser.add_argument("--w-consol",     type=float, default=0.05,
        help="quality weight: consol_days component")
    # sizing
    parser.add_argument("--equity",       type=float, default=100_000.0)
    parser.add_argument("--risk-pct",     type=float, default=0.01,
        help="fraction of portfolio risked per trade (default 1%%)")
    parser.add_argument("--max-pos-pct",  type=float, default=0.05,
        help="max single position as fraction of equity (default 5%%)")
    # strategies
    parser.add_argument("--r-targets",    type=float, nargs="+", default=[1.5, 2.0, 3.0],
        help="R-multiple targets for strategy 1 (space-separated)")
    parser.add_argument("--atr-period",   type=int,   default=14)
    parser.add_argument("--atr-trail-mult", type=float, default=2.0,
        help="ATR multiplier for trailing stop: trail = high_watermark - N * ATR")
    parser.add_argument("--atr-activate", type=float, default=1.0,
        help="R profit required before ATR trail activates")
    parser.add_argument("--time-stop",    type=int,   default=15,
        help="days before time-based exit fires (strategy 3)")
    parser.add_argument("--max-hold",     type=int,   default=30,
        help="universal safety valve for all strategies (bars)")
    # regime filter
    parser.add_argument("--spy-filter",   action="store_true",
        help="only take trades when SPY is above its 50-day SMA (phase 4)")
    # output
    parser.add_argument("--csv",          metavar="FILE",
        help="save trade-by-trade log to CSV")
    args = parser.parse_args()

    q = QualityConfig(
        weight_volume      = args.w_volume,
        weight_rs          = args.w_rs,
        weight_prior_move  = args.w_prior_move,
        weight_tightness   = args.w_tightness,
        weight_consol      = args.w_consol,
        min_quality_score  = args.min_quality,
    )
    params = SimParams(
        start_date        = args.start,
        end_date          = args.end,
        min_engine_score  = args.min_score,
        min_engine_regime = args.min_regime,
        min_consol_days   = args.min_consol,
        initial_equity    = args.equity,
        risk_pct          = args.risk_pct,
        max_position_pct  = args.max_pos_pct,
        r_targets         = args.r_targets,
        atr_period        = args.atr_period,
        atr_trail_mult    = args.atr_trail_mult,
        atr_trail_activate = args.atr_activate,
        time_stop_days    = args.time_stop,
        max_hold_days     = args.max_hold,
        spy_filter        = args.spy_filter,
        quality           = q,
    )

    ProfitOptimizer(params).run(csv_path=args.csv)
