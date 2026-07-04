"""
phase 1 — trade management sweep (H1-H4 in experiments.md).

tests trim1_r, sma_trail_period, max_hold_days against purged walk-forward
folds, then confirms the winner once on an untouched final holdout block,
then checks slippage sensitivity on both current defaults and the winner.

coordinate-wise search, not a full grid: sweeps each parameter independently
holding the other two at current defaults (trim1_r=3.0, sma_trail=10,
max_hold=60), picks the per-fold winner by TRAIN sortino, evaluates that
winner OOS on the paired TEST fold, then builds one joint config from each
param's most frequent per-fold winner and checks it too. a full 6x5x4=120
grid would be ~10x the backtester runs for a modest gain in resolution on
three parameters that interact weakly (trim timing, trailing-exit speed,
and hold-time cap operate through different mechanisms).

usage: uv run python sweep_trade_mgmt.py
output: data/validation_cache/sweep_trade_mgmt_results.json
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import json
import sqlite3
from pathlib import Path

import pandas as pd

from backtester import Backtester, BacktestParams
from validation import purged_walk_forward_folds, final_holdout_split

DB_PATH = "results/breakout.db"
OUT_DIR = Path("data") / "validation_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "sweep_trade_mgmt_results.json"

CURRENT = {"trim1_r": 3.0, "sma_trail_period": 10, "max_hold_days": 60}

# candidate values per parameter, swept independently around the current default
CANDIDATES = {
    "trim1_r": [2.0, 2.5, 3.0, 4.0, 5.0],
    "sma_trail_period": [5, 8, 10, 15, 20],
    "max_hold_days": [30, 45, 60, 90],
}

TRAIN_MONTHS = 36
TEST_MONTHS = 6
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
MIN_TRADES = 15  # folds/configs with fewer closed trades are too noisy to rank on


def _scan_date_frame() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT DISTINCT scan_date FROM scans WHERE passes_filters = 1", conn
        )
    return df


def run_bt(start: str, end: str, **overrides) -> dict | None:
    params = BacktestParams(start_date=start, end_date=end, **overrides)
    m = Backtester(params).run().metrics
    if not m or m.get("total_trades", 0) < MIN_TRADES:
        return None
    return m


def sortino_key(m: dict | None) -> float:
    if m is None:
        return float("-inf")
    s = m.get("sortino", 0.0)
    return s if s == s else float("-inf")  # nan-safe


def main():
    date_df = _scan_date_frame()
    dev_dates, holdout_dates, holdout_start, holdout_end = final_holdout_split(
        date_df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"holdout block: {holdout_start} .. {holdout_end} (touched once, at the end)")

    folds = purged_walk_forward_folds(
        dev_dates, date_col="scan_date",
        train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"{len(folds)} purged walk-forward folds in dev period\n")

    per_param_results: dict[str, list[dict]] = {p: [] for p in CANDIDATES}
    joint_results: list[dict] = []
    baseline_results: list[dict] = []

    for fi, fold in enumerate(folds, 1):
        tr_s, tr_e, te_s, te_e = fold["train_start"], fold["train_end"], fold["test_start"], fold["test_end"]
        print(f"fold {fi}/{len(folds)}: train {tr_s}..{tr_e}  test {te_s}..{te_e}")

        # baseline (current defaults) — both train and test, for the overfit-gap reference
        base_train = run_bt(tr_s, tr_e, **CURRENT)
        base_test = run_bt(te_s, te_e, **CURRENT)
        baseline_results.append(
            {"fold": fi, "train_sortino": sortino_key(base_train), "test_sortino": sortino_key(base_test),
             "train_metrics": base_train, "test_metrics": base_test}
        )

        fold_winners = {}
        for param, values in CANDIDATES.items():
            best_val, best_train_m, best_score = None, None, float("-inf")
            for v in values:
                overrides = dict(CURRENT)
                overrides[param] = v
                m = run_bt(tr_s, tr_e, **overrides)
                score = sortino_key(m)
                if score > best_score:
                    best_val, best_train_m, best_score = v, m, score
            if best_val is None:
                continue
            overrides = dict(CURRENT)
            overrides[param] = best_val
            test_m = run_bt(te_s, te_e, **overrides)
            fold_winners[param] = best_val
            per_param_results[param].append(
                {"fold": fi, "winning_value": best_val,
                 "train_sortino": best_score, "test_sortino": sortino_key(test_m),
                 "train_metrics": best_train_m, "test_metrics": test_m}
            )
            print(f"  {param}: train-winner={best_val} (train sortino={best_score:+.3f}, test sortino={sortino_key(test_m):+.3f})")

        joint_overrides = dict(CURRENT)
        joint_overrides.update(fold_winners)
        joint_train = run_bt(tr_s, tr_e, **joint_overrides)
        joint_test = run_bt(te_s, te_e, **joint_overrides)
        joint_results.append(
            {"fold": fi, "config": joint_overrides,
             "train_sortino": sortino_key(joint_train), "test_sortino": sortino_key(joint_test),
             "train_metrics": joint_train, "test_metrics": joint_test}
        )
        print(f"  joint config {joint_overrides}: test sortino={sortino_key(joint_test):+.3f}  "
              f"(baseline test sortino={sortino_key(base_test):+.3f})\n")

    # aggregate: for each param, the value that won the most folds
    from collections import Counter
    final_choice = {}
    for param, results in per_param_results.items():
        counts = Counter(r["winning_value"] for r in results)
        final_choice[param] = counts.most_common(1)[0][0] if counts else CURRENT[param]

    print("=" * 70)
    print("coordinate-wise winners (most frequent across folds):", final_choice)
    mean_base_test = pd.Series([r["test_sortino"] for r in baseline_results if r["test_sortino"] != float("-inf")]).mean()
    mean_joint_test = pd.Series([r["test_sortino"] for r in joint_results if r["test_sortino"] != float("-inf")]).mean()
    print(f"mean OOS sortino — current defaults: {mean_base_test:+.3f}   joint winner config: {mean_joint_test:+.3f}")

    # ── final holdout check (touched exactly once) ──────────────────────────
    print("\nfinal holdout check (single look):")
    holdout_current = run_bt(holdout_start, holdout_end, **CURRENT)
    holdout_final = run_bt(holdout_start, holdout_end, **final_choice)
    print(f"  current defaults  -> {holdout_current}")
    print(f"  final_choice cfg  -> {holdout_final}")

    # ── slippage sensitivity on holdout for both configs ─────────────────────
    print("\nslippage sensitivity (holdout block):")
    slippage_results = {}
    for label, cfg in [("current", CURRENT), ("final_choice", final_choice)]:
        slippage_results[label] = {}
        for bps in [0, 10, 25, 50]:
            m = run_bt(holdout_start, holdout_end, slippage_bps=bps, **cfg)
            slippage_results[label][bps] = m
            if m:
                print(f"  {label:14s} slippage={bps:>3}bps  sortino={m['sortino']:+.3f}  "
                      f"calmar={m['calmar']:+.3f}  expectancy={m['expectancy']:+.3f}  n={m['total_trades']}")
            else:
                print(f"  {label:14s} slippage={bps:>3}bps  insufficient trades")

    out = {
        "holdout_start": holdout_start, "holdout_end": holdout_end,
        "n_folds": len(folds),
        "per_param_results": per_param_results,
        "joint_results": joint_results,
        "baseline_results": baseline_results,
        "final_choice": final_choice,
        "mean_baseline_test_sortino": None if pd.isna(mean_base_test) else mean_base_test,
        "mean_joint_test_sortino": None if pd.isna(mean_joint_test) else mean_joint_test,
        "holdout_current": holdout_current,
        "holdout_final_choice": holdout_final,
        "slippage_results": slippage_results,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
