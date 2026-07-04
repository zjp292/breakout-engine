"""
part D of the scoring deep dive: does the choice of ranking/gating score translate
into realized P&L differences at all?

parts A-C found no component of the hand score (nor the full score, nor any subset)
has STABLE rank-predictive power within the filter-passing universe — everything
oscillates around spearman 0.02-0.07 with sign flips across fold windows. the
elasticnet is the only candidate that was never meaningfully negative in any fold,
but its holdout edge over raw_score is not significant (CI [-0.023,+0.067]).

so the deployment question: at MATCHED selectivity (each candidate admits the same
fraction of signals the live raw_score>=70 gate admits), do realized sortino/calmar
differ? four pre-registered candidates:

  raw_score    — status quo (stored column, vintage-mixture caveat noted in writeup)
  elasticnet   — fold-honest: refit on each fold's train window, dev-only for holdout
  prior_move   — strongest single univariate feature per the original validation
  random       — seeded uniform; direct test of "does ranking matter at all"

pre-registered expectation (HD in the writeup): candidates are indistinguishable on
holdout because slots are often uncontested (phase 3: avg 4.5 of 10 filled) and the
underlying population is identical. if even RANDOM matches raw_score, the score's
gate/rank role is doing ~nothing beyond subsampling — the filters are the product.

trend_strength alone (holdout spearman +0.156 in part B) is deliberately EXCLUDED:
it was identified ON the holdout, so testing it here would be selection contamination.
it can be pre-registered for validation on future data only.

usage: uv run python sweep_score_candidates.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import Backtester, BacktestParams
from ml_baseline_comparison import (
    FEATURES, TARGET, _fit_transform, fit_elasticnet, predict_elasticnet,
)
from validation import final_holdout_split, purged_walk_forward_folds

DB_PATH = "results/breakout.db"
OUT_FILE = Path("data") / "validation_cache" / "sweep_score_candidates_results.json"

TRAIN_MONTHS = 36
TEST_MONTHS = 6
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
MIN_TRADES = 15
RANDOM_SEED = 42

_BT_COLS = [
    "scan_date", "symbol", "adr_pct", "score", "raw_score", "grade",
    "base_quality", "trend_strength", "relative_strength_score",
    "volume_score", "rr_score", "breakout_level",
]


def _load_pool() -> pd.DataFrame:
    feature_cols = [f for f in FEATURES if f not in _BT_COLS]
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT s.{', s.'.join(_BT_COLS)}, {', '.join('s.' + f for f in feature_cols)},
                   mc.regime, mc.regime_multiplier AS mc_rm, o.max_gain_20d
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            JOIN outcomes o ON s.scan_date = o.scan_date AND s.symbol = o.symbol
            WHERE s.passes_filters = 1
            """,
            conn,
        )
    rng = np.random.default_rng(RANDOM_SEED)
    df["random_score"] = rng.uniform(size=len(df))
    return df


def _prep(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_filter_score"] = d[score_col]
    return d.sort_values(["scan_date", "_filter_score"], ascending=[True, False]).reset_index(drop=True)


def run_bt(scans_df: pd.DataFrame, start: str, end: str, min_score: float) -> dict | None:
    params = BacktestParams(
        start_date=start, end_date=end, min_score=min_score,
        min_regime="CAUTION", min_consol_days=None,
    )
    m = Backtester(params, scans_override=scans_df).run().metrics
    if not m or m.get("total_trades", 0) < MIN_TRADES:
        return None
    return m


def key_metrics(m: dict | None) -> dict:
    if m is None:
        return {"sortino": None, "calmar": None, "expectancy": None, "n": 0}
    return {
        "sortino": round(m["sortino"], 3), "calmar": round(m["calmar"], 3),
        "expectancy": round(m["expectancy"], 3), "n": m["total_trades"],
        "win_rate": round(m["win_rate"], 3), "max_drawdown": round(m["max_drawdown"], 3),
    }


def en_predict(train: pd.DataFrame, apply_to: pd.DataFrame) -> np.ndarray:
    trX, apX = _fit_transform(train, apply_to, FEATURES)
    scaler, model = fit_elasticnet(trX, train[TARGET])
    return predict_elasticnet(scaler, model, apX)


def evaluate_window(pool, train_mask, window_mask, window_start, window_end):
    """head-to-head on one window: thresholds from train quantiles, matched selectivity"""
    train = pool[train_mask]
    frac = float((train["raw_score"] >= 70).mean())
    results = {}

    window_pool = pool.copy()
    window_pool.loc[window_mask, "_en"] = en_predict(train, pool[window_mask])
    # en threshold must come from train-window predictions, not test predictions
    train_en = en_predict(train, train)

    for cand, col, thr in [
        ("raw_score", "raw_score", float(train["raw_score"].quantile(1 - frac))),
        ("elasticnet", "_en", float(np.quantile(train_en, 1 - frac))),
        ("prior_move", "prior_move_pct", float(train["prior_move_pct"].dropna().quantile(1 - frac))),
        ("random", "random_score", float(train["random_score"].quantile(1 - frac))),
    ]:
        frame = _prep(window_pool.dropna(subset=[col]) if col != "_en" else window_pool[window_mask].dropna(subset=[col]), col)
        m = run_bt(frame, window_start, window_end, min_score=thr)
        results[cand] = key_metrics(m)
    return frac, results


def main():
    pool = _load_pool()
    print(f"loaded pool: {len(pool)} filter-passing rows with outcomes")

    dates = pd.to_datetime(pool["scan_date"])
    date_df = pool[["scan_date"]].drop_duplicates()
    dev_dates, _, holdout_start, holdout_end = final_holdout_split(
        date_df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    folds = purged_walk_forward_folds(
        dev_dates, date_col="scan_date", train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"{len(folds)} folds, holdout {holdout_start}..{holdout_end}\n")

    fold_results = []
    for fi, fold in enumerate(folds, 1):
        tr_s, tr_e, te_s, te_e = fold["train_start"], fold["train_end"], fold["test_start"], fold["test_end"]
        train_mask = (pool["scan_date"] >= tr_s) & (pool["scan_date"] <= tr_e)
        test_mask = (pool["scan_date"] >= te_s) & (pool["scan_date"] <= te_e)
        frac, res = evaluate_window(pool, train_mask, test_mask, te_s, te_e)
        print(f"fold {fi} (test {te_s}..{te_e}, gate admits {frac:.0%} of train):")
        for cand, km in res.items():
            print(f"  {cand:12s} sortino={km['sortino']}  calmar={km['calmar']}  "
                  f"expectancy={km['expectancy']}  n={km['n']}")
        fold_results.append({"fold": fi, "gate_frac": frac, "results": res})

    print("\nfinal holdout (single look):")
    dev_mask = pool["scan_date"].isin(dev_dates["scan_date"])
    holdout_mask = (pool["scan_date"] >= holdout_start) & (pool["scan_date"] <= holdout_end)
    frac, holdout_res = evaluate_window(pool, dev_mask, holdout_mask, holdout_start, holdout_end)
    for cand, km in holdout_res.items():
        print(f"  {cand:12s} sortino={km['sortino']}  calmar={km['calmar']}  "
              f"expectancy={km['expectancy']}  n={km['n']}  win_rate={km.get('win_rate')}")

    # mean OOS sortino per candidate across folds
    print("\nmean OOS sortino across folds:")
    means = {}
    for cand in ["raw_score", "elasticnet", "prior_move", "random"]:
        vals = [f["results"][cand]["sortino"] for f in fold_results if f["results"][cand]["sortino"] is not None]
        means[cand] = float(np.mean(vals)) if vals else None
        print(f"  {cand:12s} {means[cand]}")

    out = {
        "n_folds": len(folds), "holdout_start": holdout_start, "holdout_end": holdout_end,
        "fold_results": fold_results, "holdout_results": holdout_res, "mean_oos_sortino": means,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
