"""
phase 4 — regime multiplier calibration.

directive's question: the daily/macro bucket boundaries and multiplier values
(1.00/0.95/0.85/0.70/0.50 daily; 0.60-1.00 macro; panic floor 0.40) are mostly
hand-set round numbers. condition realized expectancy on the regime label at scan
time and check whether the multiplier curve actually tracks it.

three checks:
  1. daily regime layer (market_conditions.regime, fully persisted) vs realized EV
  2. macro regime layer — NOT persisted anywhere, so recomputed historically via
     MacroRegimeAnalyzer against cached COMPX/SPY/IWM full history (fetched fresh
     for SPY/IWM this session — same read-only Schwab call historical_batch.py
     already makes routinely, just not previously cached to data/history/*.pkl)
  3. the blended final multiplier (0.55*daily + 0.45*macro, floored 0.50) vs
     realized EV — does the whole two-layer system, as currently weighted, track
     realized outcomes

all three use passes_filters=1 rows only (the population that actually generates
trade signals) joined to outcomes, block-bootstrapped by scan_date, with dev/holdout
consistency shown (same split boundaries as elsewhere in this pass).

usage: uv run python analyze_regime_calibration.py
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

from config import PARAMETERS
from macro_regime import MacroRegimeAnalyzer
from validation import final_holdout_split

HISTORY_DIR = Path("data") / "history"
CACHE_PATH = Path("data") / "validation_cache" / "macro_regime_history.pkl"
OUT_FILE = Path("data") / "validation_cache" / "regime_calibration_results.json"
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
N_BOOT = 2000

DAILY_MULT = {"BULL": 1.00, "UPTREND": 0.95, "MIXED": 0.85, "CAUTION": 0.70, "DOWNTREND": 0.50}
MACRO_MULT_ORDER = [
    "BULL_RUN", "BULL_TRANSITION", "BULL_CHOP", "INFLECTION",
    "CONSOLIDATION", "BEAR_TRANSITION", "DOWNTREND", "BEAR_CHOP",
]


def block_bootstrap_mean(df: pd.DataFrame, value_col: str, date_col: str = "scan_date",
                          n_boot: int = N_BOOT, seed: int = 42):
    if df.empty:
        return np.nan, np.nan, np.nan
    dates = df[date_col].unique()
    date_groups = {d: g[value_col].to_numpy() for d, g in df.groupby(date_col)}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        vals = np.concatenate([date_groups[d] for d in sampled])
        boot_means[i] = vals.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(df[value_col].mean()), float(lo), float(hi)


def recompute_macro_history() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"using cached macro regime history: {CACHE_PATH}")
        return pd.read_pickle(CACHE_PATH)

    with open(HISTORY_DIR / "COMPX-full.pkl", "rb") as f:
        compx = pickle.load(f)
    with open(HISTORY_DIR / "SPY-full.pkl", "rb") as f:
        spy = pickle.load(f)
    with open(HISTORY_DIR / "IWM-full.pkl", "rb") as f:
        iwm = pickle.load(f)

    with sqlite3.connect("results/breakout.db") as conn:
        dates = pd.read_sql(
            "SELECT DISTINCT scan_date FROM market_conditions ORDER BY scan_date", conn
        )["scan_date"].tolist()

    analyzer = MacroRegimeAnalyzer(PARAMETERS)
    rows = []
    print(f"recomputing macro regime for {len(dates)} historical dates...")
    for i, d in enumerate(dates, 1):
        ts = pd.Timestamp(d)
        c_slice = compx.loc[:ts]
        s_slice = spy.loc[:ts]
        i_slice = iwm.loc[:ts]
        if len(c_slice) < 60:
            continue
        result = analyzer.analyze(compx_df=c_slice, spy_df=s_slice, iwm_df=i_slice)
        rows.append({
            "scan_date": d, "macro_regime": result.regime_label,
            "macro_multiplier": result.macro_multiplier, "panic_state": result.panic_state,
        })
        if i % 300 == 0:
            print(f"  {i}/{len(dates)}")

    out = pd.DataFrame(rows)
    out.to_pickle(CACHE_PATH)
    print(f"wrote {len(out)} rows to {CACHE_PATH}")
    return out


def _load_base() -> pd.DataFrame:
    with sqlite3.connect("results/breakout.db") as conn:
        df = pd.read_sql(
            """
            SELECT s.scan_date, s.symbol, s.score, s.raw_score, o.max_gain_20d
            FROM scans s
            JOIN outcomes o ON s.scan_date = o.scan_date AND s.symbol = o.symbol
            WHERE s.passes_filters = 1
            """,
            conn,
        )
    with sqlite3.connect("results/breakout.db") as conn:
        mc = pd.read_sql(
            "SELECT scan_date, regime AS daily_regime, regime_multiplier AS daily_multiplier "
            "FROM market_conditions", conn,
        )
    return df.merge(mc, on="scan_date", how="left")


def bucket_report(df: pd.DataFrame, bucket_col: str, order: list, label: str):
    rows = []
    for bucket in order:
        sub = df[df[bucket_col] == bucket]
        n = len(sub)
        if n == 0:
            rows.append({"bucket": bucket, "n": 0})
            continue
        ev, lo, hi = block_bootstrap_mean(sub, "max_gain_20d")
        win_rate = float((sub["max_gain_20d"] > 0).mean())
        rows.append({"bucket": bucket, "n": n, "ev_mean_gain_20d": round(ev, 4),
                     "ev_ci95_lo": round(lo, 4), "ev_ci95_hi": round(hi, 4), "win_rate": round(win_rate, 3)})
    out = pd.DataFrame(rows)
    print(f"\n--- {label} ---")
    print(out.to_string(index=False))
    return out


def main():
    macro_hist = recompute_macro_history()
    base = _load_base()
    base = base.merge(macro_hist, on="scan_date", how="left")
    print(f"\nloaded {len(base)} passes_filters=1 rows with outcomes, "
          f"{base['macro_regime'].notna().sum()} matched to a recomputed macro regime")

    # implied blend = actually-applied multiplier, derived from persisted score/raw_score —
    # a direct check that recomputed macro regime roughly reproduces what was live-applied
    base["implied_multiplier"] = base["score"] / base["raw_score"].replace(0, np.nan)
    base["computed_blend"] = (
        0.55 * base["daily_multiplier"] + 0.45 * base["macro_multiplier"]
    ).clip(lower=0.50)
    matched = base.dropna(subset=["implied_multiplier", "computed_blend"])
    matched = matched[(matched["implied_multiplier"] > 0) & (matched["implied_multiplier"] <= 1.5)]
    corr = matched["implied_multiplier"].corr(matched["computed_blend"])
    print(f"sanity check — corr(implied live multiplier, recomputed blend): {corr:.3f} (n={len(matched)})")

    dev, holdout, holdout_start, holdout_end = final_holdout_split(
        base, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"holdout: {holdout_start} .. {holdout_end}")

    results = {}
    for label, d in [("full", base), ("dev", dev), ("holdout", holdout)]:
        print(f"\n{'='*70}\n{label.upper()}\n{'='*70}")
        r1 = bucket_report(d, "daily_regime", list(DAILY_MULT.keys()), f"Check 1: daily regime ({label})")
        r2 = bucket_report(d, "macro_regime", MACRO_MULT_ORDER, f"Check 2: macro regime ({label})")
        d = d.copy()
        d["blend_bucket"] = pd.cut(
            d["computed_blend"], bins=[0, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01],
            labels=["<=0.55", "0.55-0.65", "0.65-0.75", "0.75-0.85", "0.85-0.95", "0.95-1.00"],
        )
        r3 = bucket_report(d, "blend_bucket",
                            ["<=0.55", "0.55-0.65", "0.65-0.75", "0.75-0.85", "0.85-0.95", "0.95-1.00"],
                            f"Check 3: blended multiplier ({label})")
        results[label] = {"daily": r1.to_dict("records"), "macro": r2.to_dict("records"),
                           "blend": r3.to_dict("records")}

    results["sanity_corr_implied_vs_computed"] = corr
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
