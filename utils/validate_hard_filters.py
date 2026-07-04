"""
phase 2 — hard filter re-validation (H5-H9 in experiments.md).

re-checks the 5 hard filters the directive names (ADR>=7%, prior_move>=75%,
consol_days>=5, stop<=5xADR, rs_comp_252>=0) against the FULL scans+outcomes
table (n in the thousands-to-tens-of-thousands per filter), not the 91-example
hand-labeled set the original thresholds leaned on.

methodology, per filter:
  1. restrict to rows passing every OTHER hard filter (isolates the marginal
     effect of the filter under test; mirrors the "n=5,214 stocks failing
     ONLY on consol_days" approach already used once in this repo, per
     CLAUDE.md's 2026-06-12 note).
  2. bucket by the filter's own metric, report n / mean(max_gain_20d) as EV /
     win rate per bucket, with 95% CIs from a block bootstrap resampled by
     scan_date (not by row — rows on the same date and the same symbol across
     nearby dates aren't independent).
  3. re-run the same bucketing on the dev period and the final holdout block
     separately (same split as sweep_trade_mgmt.py) to check the relationship
     replicates rather than being an artifact of the window it was fit on.

ADR (H5) / prior_move (H6) / consol_days (H7) come straight from persisted
`scans` columns. stop_adr_multiple (H8) and rs_comp_252 (H9) require
recompute_filter_features.py's output, since neither is persisted:
  - scans.stop_distance_pct is the 60-day version, not the
    stop_distance_20d_pct the filter actually checks.
  - rs_comp_252 isn't in the scans table at all, despite being enforced live.

usage: uv run python recompute_filter_features.py   (once, ~15min)
       uv run python validate_hard_filters.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from validation import final_holdout_split

DB_PATH = "results/breakout.db"
RECOMPUTED_PATH = Path("data") / "validation_cache" / "recomputed_filter_features.pkl"
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
N_BOOT = 2000


def _load_base() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT s.scan_date, s.symbol, s.price, s.dollar_volume, s.adr_pct,
                   s.stop_distance_pct, s.pct_from_52wk_high, s.prior_move_pct,
                   s.consol_days, o.max_gain_20d, o.breakout_triggered
            FROM scans s
            JOIN outcomes o ON s.scan_date = o.scan_date AND s.symbol = o.symbol
            """,
            conn,
        )
    return df


def _other_filters_mask(df: pd.DataFrame, exclude: str) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if exclude != "price":
        m &= df["price"] >= 5.0
    if exclude != "dollar_volume":
        m &= df["dollar_volume"] >= 10_000_000
    if exclude != "adr":
        m &= df["adr_pct"] >= 0.07
    if exclude != "stop":
        # persisted 60d value used as an approximate gate when NOT testing the
        # stop filter itself — the exact 20d value isn't in this base frame.
        m &= df["stop_distance_pct"] <= 5.0 * df["adr_pct"].clip(lower=0.01)
    if exclude != "pct_from_high":
        m &= df["pct_from_52wk_high"] >= -0.30
    if exclude != "prior_move":
        m &= df["prior_move_pct"] >= 0.75
    if exclude != "consol_days":
        m &= df["consol_days"] >= 5
    return m


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


def bucket_report(df: pd.DataFrame, metric_col: str, bins: list, labels: list, label_name: str):
    d = df.copy()
    d["_bucket"] = pd.cut(d[metric_col], bins=bins, labels=labels, right=False)
    rows = []
    for bucket in labels:
        sub = d[d["_bucket"] == bucket]
        n = len(sub)
        if n == 0:
            rows.append({"bucket": bucket, "n": 0})
            continue
        ev, lo, hi = block_bootstrap_mean(sub, "max_gain_20d")
        win_rate = float((sub["max_gain_20d"] > 0).mean())
        bo_rate = float(sub["breakout_triggered"].fillna(0).astype(float).mean())
        rows.append({
            "bucket": bucket, "n": n, "ev_mean_gain_20d": round(ev, 4),
            "ev_ci95_lo": round(lo, 4), "ev_ci95_hi": round(hi, 4),
            "win_rate": round(win_rate, 3), "breakout_rate": round(bo_rate, 3),
        })
    out = pd.DataFrame(rows)
    print(f"\n--- {label_name} (bucketed on {metric_col}, isolating from other filters) ---")
    print(out.to_string(index=False))
    return out


def run_all(df: pd.DataFrame, period_label: str):
    print(f"\n{'='*78}\n{period_label}\n{'='*78}")

    # H5 — ADR
    sub = df[_other_filters_mask(df, exclude="adr")]
    bucket_report(
        sub, "adr_pct",
        bins=[0, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, np.inf],
        labels=["<5%", "5-7%", "7-10%", "10-12%", "12-15%", "15-20%", "20-25%", "25%+"],
        label_name=f"H5: ADR floor (n={len(sub)}, current threshold = 7%)",
    )

    # H6 — prior move
    sub = df[_other_filters_mask(df, exclude="prior_move")]
    bucket_report(
        sub, "prior_move_pct",
        bins=[0, 0.25, 0.50, 0.75, 1.00, 2.00, np.inf],
        labels=["<25%", "25-50%", "50-75%", "75-100%", "100-200%", "200%+"],
        label_name=f"H6: prior move floor (n={len(sub)}, current threshold = 75%)",
    )

    # H7 — consol_days
    sub = df[_other_filters_mask(df, exclude="consol_days")]
    bucket_report(
        sub, "consol_days",
        bins=[0, 1, 5, 10, 20, 35, 60, np.inf],
        labels=["0", "1-4", "5-9", "10-19", "20-34", "35-59", "60+"],
        label_name=f"H7: consol_days floor (n={len(sub)}, current threshold = 5)",
    )


def run_pickle_dependent(df: pd.DataFrame, period_label: str):
    print(f"\n{'='*78}\n{period_label} (stop_adr_multiple / rs_comp_252)\n{'='*78}")

    # H8 — stop distance relative to ADR, using the REAL 20d value
    df = df.copy()
    df["stop_adr_ratio"] = df["stop_distance_20d_pct"] / df["adr_pct"].clip(lower=0.01)
    sub = df[_other_filters_mask(df, exclude="stop")]
    bucket_report(
        sub, "stop_adr_ratio",
        bins=[0, 2, 3, 4, 5, 6, 8, np.inf],
        labels=["<=2x", "2-3x", "3-4x", "4-5x", "5-6x", "6-8x", "8x+"],
        label_name=f"H8: stop distance vs ADR (n={len(sub)}, current threshold = 5x)",
    )

    # H9 — 12-month RS vs NASDAQ
    sub = df[_other_filters_mask(df, exclude="prior_move")]  # base mask; rs_252 not in _other_filters_mask
    bucket_report(
        sub, "rs_comp_252",
        bins=[-np.inf, -0.20, -0.10, 0.0, 0.10, 0.25, np.inf],
        labels=["<-20%", "-20..-10%", "-10..0%", "0..10%", "10..25%", "25%+"],
        label_name=f"H9: 12-month RS vs NASDAQ (n={len(sub)}, current threshold = 0)",
    )


def main():
    base = _load_base()
    print(f"loaded {len(base)} scan+outcome rows ({base['scan_date'].min()}..{base['scan_date'].max()})")

    dev, holdout, holdout_start, holdout_end = final_holdout_split(
        base, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"dev: {len(dev)} rows | holdout ({holdout_start}..{holdout_end}): {len(holdout)} rows")

    run_all(base, "FULL TABLE (H5-H7)")
    run_all(dev, "DEV PERIOD ONLY (H5-H7)")
    run_all(holdout, "HOLDOUT PERIOD ONLY (H5-H7, single look)")

    if not RECOMPUTED_PATH.exists():
        print(f"\n{RECOMPUTED_PATH} not found — run recompute_filter_features.py first for H8/H9")
        return

    recomputed = pd.read_pickle(RECOMPUTED_PATH)
    merged = base.drop(columns=["adr_pct"]).merge(
        recomputed, on=["scan_date", "symbol"], how="inner"
    )
    print(f"\nmerged with recomputed features: {len(merged)} rows (of {len(base)} base rows)")

    dev_m, holdout_m, _, _ = final_holdout_split(
        merged, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    run_pickle_dependent(merged, "FULL TABLE (H8-H9)")
    run_pickle_dependent(dev_m, "DEV PERIOD ONLY (H8-H9)")
    run_pickle_dependent(holdout_m, "HOLDOUT PERIOD ONLY (H8-H9, single look)")


if __name__ == "__main__":
    main()
