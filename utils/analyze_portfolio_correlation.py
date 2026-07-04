"""
phase 3 — portfolio-level correlation check (follow-up #5 in experiments.md).

directive's concern: momentum setups cluster — when one fires, several often fire
together, correlated. the backtester's fixed --positions=10 cap doesn't model
correlation between concurrently-held positions; it just caps count. if realized
concurrent-position correlation is high, realized portfolio drawdown runs worse than
per-trade stats (which treat each trade in isolation) suggest.

four checks, run against a full-period backtest at current default params:
  1. signal clustering — daily count of qualifying signals (why this matters at all)
  2. concurrent-position daily-return correlation vs a non-concurrent same-symbol
     baseline and vs each stock's own market-beta (COMPX) correlation, to separate
     "everything correlates with the market" from excess momentum-cluster correlation
  3. effective-N: how many independent bets --positions=10 actually buys, given
     realized correlation (standard 1/(1+(N-1)*rho) diversification-ratio formula)
  4. realized vs naive-independence daily portfolio volatility, using the actual
     backtest equity curve vs an equal-weight/no-correlation reconstruction

dev/holdout split (same boundaries as elsewhere in this pass) shown for consistency,
though this is descriptive, not a tuned parameter — no selection-on-holdout risk here.

usage: uv run python analyze_portfolio_correlation.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import Backtester, BacktestParams
from validation import final_holdout_split

HISTORY_DIR = Path("data") / "history"
OUT_FILE = Path("data") / "validation_cache" / "portfolio_correlation_results.json"
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10


def check1_signal_clustering():
    with sqlite3.connect("results/breakout.db") as conn:
        df = pd.read_sql(
            "SELECT scan_date, COUNT(*) as n FROM scans WHERE passes_filters=1 GROUP BY scan_date",
            conn,
        )
    print("\n--- Check 1: daily signal clustering (passes_filters=1) ---")
    print(df["n"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    zero_days_note = f"scan dates with 0 signals aren't in this table at all"
    print(f"({zero_days_note})")
    return {"describe": df["n"].describe().to_dict()}


def _load_price(symbol: str) -> pd.DataFrame | None:
    path = HISTORY_DIR / f"{symbol}-full.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        df = pickle.load(f)
    df = df.reset_index().rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    df["ret"] = df["close"].pct_change()
    return df


def _held_mask(trades, symbol: str, calendar: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=calendar)
    for t in trades:
        if t.symbol != symbol or not t.is_closed:
            continue
        entry, exit_ = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        mask.loc[(calendar >= entry) & (calendar <= exit_)] = True
    return mask


def checks_2_3_4(trades, daily_equity: pd.Series, label: str, results_params: BacktestParams) -> dict:
    print(f"\n{'='*70}\n{label}  (n_trades={len(trades)})\n{'='*70}")
    closed = [t for t in trades if t.is_closed]
    symbols = sorted({t.symbol for t in closed})
    calendar = pd.date_range(
        min(pd.Timestamp(t.entry_date) for t in closed),
        max(pd.Timestamp(t.exit_date) for t in closed),
        freq="D",
    )

    prices = {}
    for s in symbols:
        p = _load_price(s)
        if p is not None:
            prices[s] = p

    # wide daily-return matrix, NaN except when that symbol was actually held
    ret_matrix = pd.DataFrame(index=calendar, columns=list(prices.keys()), dtype=float)
    for s, p in prices.items():
        mask = _held_mask(closed, s, calendar)
        aligned = p["ret"].reindex(calendar)
        ret_matrix[s] = aligned.where(mask)

    held_counts = (~ret_matrix.isna()).sum(axis=1)
    corr = ret_matrix.corr(min_periods=10)
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    concurrent_corr = float(np.mean(off_diag)) if len(off_diag) else float("nan")

    # baseline: correlation of the SAME symbol pairs computed over ALL days both
    # had *any* price history (not just concurrently-held days) — isolates the
    # "held together" effect from each pair's general/baseline co-movement.
    full_ret_matrix = pd.DataFrame(index=calendar, columns=list(prices.keys()), dtype=float)
    for s, p in prices.items():
        full_ret_matrix[s] = p["ret"].reindex(calendar)
    full_corr = full_ret_matrix.corr(min_periods=60)
    full_off_diag = full_corr.values[np.triu_indices_from(full_corr.values, k=1)]
    full_off_diag = full_off_diag[~np.isnan(full_off_diag)]
    baseline_corr = float(np.mean(full_off_diag)) if len(full_off_diag) else float("nan")

    # market-beta baseline: each traded symbol's correlation to COMPX during its
    # OWN held periods (not pairwise — this is "how correlated is a typical held
    # position with the market", the floor you'd expect from beta alone)
    compx = _load_price("COMPX")
    beta_corrs = []
    if compx is not None:
        for s, p in prices.items():
            mask = _held_mask(closed, s, calendar)
            held_days = calendar[mask.reindex(calendar, fill_value=False)]
            if len(held_days) < 15:
                continue
            sr = p["ret"].reindex(held_days)
            mr = compx["ret"].reindex(held_days)
            c = sr.corr(mr)
            if not np.isnan(c):
                beta_corrs.append(c)
    market_beta_corr = float(np.mean(beta_corrs)) if beta_corrs else float("nan")

    print(f"concurrent-position pairwise correlation:      {concurrent_corr:+.4f}  (n pairs={len(off_diag)})")
    print(f"same-pair baseline correlation (all days):     {baseline_corr:+.4f}  (n pairs={len(full_off_diag)})")
    print(f"avg correlation of held position vs COMPX:      {market_beta_corr:+.4f}  (n={len(beta_corrs)})")
    print(f"excess correlation (concurrent - baseline):     {concurrent_corr - baseline_corr:+.4f}")

    # check 3: effective N
    avg_n_held = float(held_counts[held_counts > 0].mean()) if (held_counts > 0).any() else 0.0
    max_n_held = int(held_counts.max())
    rho = max(concurrent_corr, 0.0) if not np.isnan(concurrent_corr) else 0.0
    eff_n = avg_n_held / (1 + (avg_n_held - 1) * rho) if avg_n_held > 0 else 0.0
    print(f"\navg concurrent positions held (days with >=1): {avg_n_held:.2f}  (max={max_n_held}, cap=10)")
    print(f"effective N (diversification-adjusted):        {eff_n:.2f}  at rho={rho:+.4f}")
    print(f"diversification ratio realized/nominal:         {eff_n/avg_n_held:.2%}" if avg_n_held > 0 else "n/a")

    # check 4: realized vs naive-independence daily portfolio vol, using the SAME
    # risk-based sizing the backtester actually uses (not equal-weight, and NOT
    # renormalized to 100% invested — both would confound the correlation effect
    # with other effects: equal-weight ignores risk-based sizing discipline,
    # renormalizing ignores cash drag from unfilled position slots, which matters
    # here since avg positions held is well below the 10-slot cap).
    # raw_weight_i = min(risk_per_trade / stop_distance_pct_i, max_position_pct) —
    # the actual fraction of equity a position this size would represent, per
    # BacktestParams' own sizing formula. stop_distance_pct held constant per trade
    # episode (ignores the breakeven-stop adjustment after trim1, documented
    # simplification).
    p = results_params
    weight_matrix = pd.DataFrame(index=calendar, columns=list(prices.keys()), dtype=float)
    for t in closed:
        if t.symbol not in prices or t.entry_price <= 0 or t.initial_stop >= t.entry_price:
            continue
        stop_dist = (t.entry_price - t.initial_stop) / t.entry_price
        raw_w = min(p.risk_per_trade / max(stop_dist, 0.005), p.max_position_pct)
        entry, exit_ = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        span = (calendar >= entry) & (calendar <= exit_)
        weight_matrix.loc[span, t.symbol] = raw_w

    per_symbol_var = (ret_matrix.std(skipna=True) ** 2)
    naive_daily_var = (weight_matrix.pow(2) * per_symbol_var).sum(axis=1, skipna=True)
    held_any = weight_matrix.notna().sum(axis=1) > 0
    naive_daily_var = naive_daily_var[held_any]

    daily_rets = daily_equity.pct_change().dropna()
    realized_vol = float(daily_rets.std())
    naive_vol = float(np.sqrt(naive_daily_var.mean())) if len(naive_daily_var) else float("nan")
    print(f"\nrealized daily portfolio return std:  {realized_vol:.4%}")
    print(f"naive independence-assumed std:       {naive_vol:.4%}  (risk-based weights, zero correlation assumed)")
    print(f"realized/naive ratio:                 {realized_vol/naive_vol:.2f}x" if naive_vol > 0 else "n/a")

    return {
        "n_trades": len(closed), "n_symbols": len(symbols),
        "concurrent_corr": concurrent_corr, "baseline_corr": baseline_corr,
        "market_beta_corr": market_beta_corr,
        "avg_n_held": avg_n_held, "max_n_held": max_n_held,
        "effective_n": eff_n, "realized_vol": realized_vol, "naive_vol": naive_vol,
    }


def main():
    params = BacktestParams(start_date="2019-06-10", end_date="2026-06-30")
    print(f"running full-period backtest: {params.start_date} .. {params.end_date}, positions={params.max_positions}")
    results = Backtester(params).run()
    print(f"total closed trades: {len([t for t in results.trades if t.is_closed])}")

    signal_stats = check1_signal_clustering()

    full_stats = checks_2_3_4(results.trades, results.daily_equity, "FULL PERIOD", params)

    with sqlite3.connect("results/breakout.db") as conn:
        dates_df = pd.read_sql("SELECT DISTINCT scan_date FROM scans WHERE passes_filters=1", conn)
    dev_dates, holdout_dates, holdout_start, holdout_end = final_holdout_split(
        dates_df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"\nholdout block for consistency check: {holdout_start} .. {holdout_end}")

    dev_trades = [t for t in results.trades if t.is_closed and t.entry_date < holdout_start]
    holdout_trades = [t for t in results.trades if t.is_closed and t.entry_date >= holdout_start]
    dev_equity = results.daily_equity[results.daily_equity.index < holdout_start]
    holdout_equity = results.daily_equity[results.daily_equity.index >= holdout_start]

    dev_stats = checks_2_3_4(dev_trades, dev_equity, "DEV PERIOD ONLY", params) if len(dev_trades) > 20 else None
    holdout_stats = checks_2_3_4(holdout_trades, holdout_equity, "HOLDOUT PERIOD ONLY", params) if len(holdout_trades) > 20 else None

    out = {
        "signal_clustering": signal_stats, "full_period": full_stats,
        "dev_period": dev_stats, "holdout_period": holdout_stats,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
