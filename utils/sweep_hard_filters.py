"""
follow-up #1 from experiments.md: trade-simulation re-test of min_consol_days and
stop_adr_multiple, since Phase 2's bucketed-EV findings for both (H7, H8) showed real
headroom but EV-per-bucket doesn't capture entry mechanics or trade count/opportunity
cost — a consol_days=0 stock is already mid-breakout, so a wider stop or looser
consol_days floor changes which trades actually get taken and at what price, not just
which forward-return bucket a row falls into.

same purged-walk-forward + holdout methodology as sweep_trade_mgmt.py, but here the
parameter under test changes which rows enter the simulation at all (a hard filter),
not how an already-selected trade is managed. that means bypassing the live
`passes_filters` flag (baked in at scan time with the CURRENT thresholds) and
reconstructing "would this row pass at threshold X" from raw + recomputed columns,
then feeding the alternate universe into Backtester via its scans_override hook.

two independent sweeps (not combined into a joint config): each parameter gets its own
"pool" of rows passing every OTHER filter, since testing consol_days needs stop/rs_252
already gated and vice versa.

usage: uv run python recompute_filter_features.py   (if not already run)
       uv run python sweep_hard_filters.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import Backtester, BacktestParams
from validation import purged_walk_forward_folds, final_holdout_split

DB_PATH = "results/breakout.db"
RECOMPUTED_PATH = Path("data") / "validation_cache" / "recomputed_filter_features.pkl"
OUT_FILE = Path("data") / "validation_cache" / "sweep_hard_filters_results.json"

TRAIN_MONTHS = 36
TEST_MONTHS = 6
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
MIN_TRADES = 15

CONSOL_CANDIDATES = [0, 2, 3, 4, 5, 7, 10, 15]
STOP_CANDIDATES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
CURRENT_CONSOL = 5
CURRENT_STOP = 5.0

_SCANS_COLS = [
    "scan_date", "symbol", "adr_pct", "score", "raw_score", "grade",
    "base_quality", "trend_strength", "relative_strength_score",
    "volume_score", "rr_score", "breakout_level", "consol_days",
    "prior_move_pct", "price", "dollar_volume", "pct_from_52wk_high",
]


def _load_pool_base() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT s.{', s.'.join(_SCANS_COLS)}, mc.regime, mc.regime_multiplier AS mc_rm
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE s.price >= 5.0
              AND s.dollar_volume >= 10000000
              AND s.adr_pct >= 0.07
              AND s.prior_move_pct >= 0.75
              AND s.pct_from_52wk_high >= -0.30
            """,
            conn,
        )
    df["_filter_score"] = df["raw_score"].fillna(df["score"] / df["mc_rm"].replace(0, np.nan))
    recomputed = pd.read_pickle(RECOMPUTED_PATH)
    merged = df.merge(recomputed, on=["scan_date", "symbol"], how="inner", suffixes=("", "_recalc"))
    merged["stop_adr_ratio"] = merged["stop_distance_20d_pct"] / merged["adr_pct"].clip(lower=0.01)
    return merged


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["scan_date", "_filter_score"], ascending=[True, False]).reset_index(drop=True)


def run_bt(scans_df: pd.DataFrame, start: str, end: str, **overrides) -> dict | None:
    params = BacktestParams(start_date=start, end_date=end, min_consol_days=None, **overrides)
    m = Backtester(params, scans_override=scans_df).run().metrics
    if not m or m.get("total_trades", 0) < MIN_TRADES:
        return None
    return m


def sortino_key(m: dict | None) -> float:
    if m is None:
        return float("-inf")
    s = m.get("sortino", 0.0)
    return s if s == s else float("-inf")


def sweep_one(label: str, pool: pd.DataFrame, metric_col: str, candidates: list, current_val,
              condition_fn, date_df: pd.DataFrame) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    dev_dates, holdout_dates, holdout_start, holdout_end = final_holdout_split(
        date_df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    folds = purged_walk_forward_folds(
        dev_dates, date_col="scan_date", train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"{len(folds)} folds, holdout {holdout_start}..{holdout_end}")

    candidate_pools = {c: _prep(pool[condition_fn(pool[metric_col], c)]) for c in candidates}

    fold_results, winners = [], []
    for fi, fold in enumerate(folds, 1):
        tr_s, tr_e, te_s, te_e = fold["train_start"], fold["train_end"], fold["test_start"], fold["test_end"]
        best_val, best_score = None, float("-inf")
        for c in candidates:
            m = run_bt(candidate_pools[c], tr_s, tr_e)
            score = sortino_key(m)
            if score > best_score:
                best_val, best_score = c, score
        test_m = run_bt(candidate_pools[best_val], te_s, te_e) if best_val is not None else None
        base_test_m = run_bt(candidate_pools[current_val], te_s, te_e)
        print(f"fold {fi}: train-winner={best_val} (train sortino={best_score:+.3f})  "
              f"winner test sortino={sortino_key(test_m):+.3f}  current({current_val}) test sortino={sortino_key(base_test_m):+.3f}")
        winners.append(best_val)
        fold_results.append({
            "fold": fi, "winning_value": best_val, "train_sortino": best_score,
            "winner_test_sortino": sortino_key(test_m), "winner_test_metrics": test_m,
            "current_test_sortino": sortino_key(base_test_m), "current_test_metrics": base_test_m,
        })

    final_choice = Counter(winners).most_common(1)[0][0] if winners else current_val
    print(f"most frequent winner: {final_choice}  (current = {current_val})")

    holdout_current = run_bt(candidate_pools[current_val], holdout_start, holdout_end)
    holdout_final = run_bt(candidate_pools[final_choice], holdout_start, holdout_end)
    print(f"holdout current({current_val}): {holdout_current}")
    print(f"holdout final_choice({final_choice}): {holdout_final}")

    slippage_results = {}
    for lbl, val in [("current", current_val), ("final_choice", final_choice)]:
        slippage_results[lbl] = {}
        for bps in [0, 10, 25, 50]:
            m = run_bt(candidate_pools[val], holdout_start, holdout_end, slippage_bps=bps)
            slippage_results[lbl][bps] = m
            if m:
                print(f"  {lbl:14s} ({val}) slippage={bps:>3}bps  sortino={m['sortino']:+.3f}  "
                      f"calmar={m['calmar']:+.3f}  expectancy={m['expectancy']:+.3f}  n={m['total_trades']}")

    return {
        "label": label, "n_folds": len(folds), "holdout_start": holdout_start, "holdout_end": holdout_end,
        "fold_results": fold_results, "final_choice": final_choice, "current_val": current_val,
        "holdout_current": holdout_current, "holdout_final_choice": holdout_final,
        "slippage_results": slippage_results,
    }


def main():
    pool = _load_pool_base()
    print(f"loaded pool: {len(pool)} rows (all filters except consol_days/stop/rs_252)")
    date_df = pool[["scan_date"]].drop_duplicates()

    # each pool holds every filter except the one under test, at the CURRENT threshold
    consol_pool = pool[pool["stop_adr_ratio"] <= CURRENT_STOP]
    stop_pool = pool[pool["consol_days"] >= 5]

    results = {}
    results["consol_days"] = sweep_one(
        "min_consol_days sweep", consol_pool, "consol_days", CONSOL_CANDIDATES, CURRENT_CONSOL,
        lambda col, c: col >= c, date_df,
    )
    results["stop_adr_multiple"] = sweep_one(
        "stop_adr_multiple sweep", stop_pool, "stop_adr_ratio", STOP_CANDIDATES, CURRENT_STOP,
        lambda col, c: col <= c, date_df,
    )

    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
