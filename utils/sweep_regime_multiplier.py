"""
follow-up #0 from experiments.md: trade-simulation re-test of the regime multiplier
system, since Phase 4's bucketed EV showed flat-to-inverted calibration (both layers,
worst in the holdout year) but EV-tracking isn't the only justification for regime-based
sizing — it could still help via variance/drawdown reduction even with flat EV, and raw
EV doesn't capture entry mechanics or portfolio effects any more than H7/H8's did.

key methodological wrinkle found in Phase 4: the backtester's default score_col="raw_score"
means every backtest run in this whole pass (Phase 1, sweep_hard_filters, etc.) has been
completely regime-blind for entry decisions — it only ever used the daily regime for the
min_regime hard gate (blocking DOWNTREND entries). The continuous multiplier that live
trading actually applies to produce total_score (gated against min_score_watchlist=70) has
never been exercised in a backtest, and can't be read from the persisted scans.score column
either (degenerate — equals raw_score for 97.8% of history, per Phase 4). So this script
reconstructs it: reconstructed_score = raw_score * blend_multiplier(date), using
market_conditions.regime_multiplier (daily, persisted) blended with the macro multiplier
recomputed in analyze_regime_calibration.py (data/validation_cache/macro_regime_history.pkl).

three candidates, isolating the two separate regime mechanisms:
  A "current backtester convention" — gate on raw_score, block DOWNTREND entries
    (min_regime="CAUTION") — this is what every prior backtest run in this pass actually did
  B "regime-adjusted" — gate on reconstructed_score (mirrors what live trading's
    min_score_watchlist threshold actually filters on), same DOWNTREND block
  C "fully regime-blind" — gate on raw_score, no DOWNTREND block (min_regime="")
    — direct test of whether the hard DOWNTREND gate specifically is worth keeping,
    given Phase 4 found DOWNTREND days had unexpectedly high EV

same purged-walk-forward + holdout + slippage methodology as the other trade-sim follow-ups.

usage: uv run python analyze_regime_calibration.py   (if macro_regime_history.pkl missing)
       uv run python sweep_regime_multiplier.py
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
MACRO_CACHE = Path("data") / "validation_cache" / "macro_regime_history.pkl"
OUT_FILE = Path("data") / "validation_cache" / "sweep_regime_multiplier_results.json"

TRAIN_MONTHS = 36
TEST_MONTHS = 6
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
MIN_TRADES = 15

CANDIDATES = ["A_current", "B_regime_adjusted", "C_no_downtrend_block"]

_SCANS_COLS = [
    "scan_date", "symbol", "adr_pct", "score", "raw_score", "grade",
    "base_quality", "trend_strength", "relative_strength_score",
    "volume_score", "rr_score", "breakout_level",
]


def _load_pool() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT s.{', s.'.join(_SCANS_COLS)},
                   mc.regime, mc.regime_multiplier AS mc_rm
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE s.passes_filters = 1
            """,
            conn,
        )
    macro = pd.read_pickle(MACRO_CACHE)[["scan_date", "macro_multiplier"]]
    df = df.merge(macro, on="scan_date", how="left")
    df["computed_blend"] = (
        0.55 * df["mc_rm"].fillna(1.0) + 0.45 * df["macro_multiplier"].fillna(1.0)
    ).clip(lower=0.50)
    df["reconstructed_score"] = df["raw_score"] * df["computed_blend"]
    return df


def _prep(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_filter_score"] = d[score_col]
    return d.sort_values(["scan_date", "_filter_score"], ascending=[True, False]).reset_index(drop=True)


def run_bt(scans_df: pd.DataFrame, start: str, end: str, min_regime: str, **overrides) -> dict | None:
    params = BacktestParams(start_date=start, end_date=end, min_regime=min_regime,
                             min_consol_days=None, **overrides)
    m = Backtester(params, scans_override=scans_df).run().metrics
    if not m or m.get("total_trades", 0) < MIN_TRADES:
        return None
    return m


def sortino_key(m: dict | None) -> float:
    if m is None:
        return float("-inf")
    s = m.get("sortino", 0.0)
    return s if s == s else float("-inf")


def main():
    pool = _load_pool()
    print(f"loaded pool: {len(pool)} passes_filters=1 rows with regime data")

    pools = {
        "A_current": (_prep(pool, "raw_score"), "CAUTION"),
        "B_regime_adjusted": (_prep(pool, "reconstructed_score"), "CAUTION"),
        "C_no_downtrend_block": (_prep(pool, "raw_score"), ""),
    }

    date_df = pool[["scan_date"]].drop_duplicates()
    dev_dates, holdout_dates, holdout_start, holdout_end = final_holdout_split(
        date_df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    folds = purged_walk_forward_folds(
        dev_dates, date_col="scan_date", train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"{len(folds)} folds, holdout {holdout_start}..{holdout_end}")

    fold_results, winners = [], []
    for fi, fold in enumerate(folds, 1):
        tr_s, tr_e, te_s, te_e = fold["train_start"], fold["train_end"], fold["test_start"], fold["test_end"]
        train_scores = {}
        for cand in CANDIDATES:
            df, min_regime = pools[cand]
            m = run_bt(df, tr_s, tr_e, min_regime)
            train_scores[cand] = sortino_key(m)
        winner = max(train_scores, key=train_scores.get)
        winners.append(winner)

        test_scores = {}
        for cand in CANDIDATES:
            df, min_regime = pools[cand]
            m = run_bt(df, te_s, te_e, min_regime)
            test_scores[cand] = sortino_key(m)

        print(f"fold {fi}: train sortino {train_scores}  -> winner={winner}  test sortino {test_scores}")
        fold_results.append({"fold": fi, "train_scores": train_scores, "test_scores": test_scores, "winner": winner})

    final_choice = Counter(winners).most_common(1)[0][0] if winners else "A_current"
    print(f"\nmost frequent winner: {final_choice}")

    print("\nfinal holdout check (single look, all 3 candidates):")
    holdout_results = {}
    for cand in CANDIDATES:
        df, min_regime = pools[cand]
        m = run_bt(df, holdout_start, holdout_end, min_regime)
        holdout_results[cand] = m
        if m:
            print(f"  {cand:22s} sortino={m['sortino']:+.3f}  calmar={m['calmar']:+.3f}  "
                  f"expectancy={m['expectancy']:+.3f}  n={m['total_trades']}  win_rate={m['win_rate']:.1%}")

    print("\nslippage sensitivity (holdout, A_current vs final_choice):")
    slippage_results = {}
    for cand in {"A_current", final_choice}:
        df, min_regime = pools[cand]
        slippage_results[cand] = {}
        for bps in [0, 10, 25, 50]:
            m = run_bt(df, holdout_start, holdout_end, min_regime, slippage_bps=bps)
            slippage_results[cand][bps] = m
            if m:
                print(f"  {cand:22s} slippage={bps:>3}bps  sortino={m['sortino']:+.3f}  "
                      f"calmar={m['calmar']:+.3f}  expectancy={m['expectancy']:+.3f}  n={m['total_trades']}")

    out = {
        "n_folds": len(folds), "holdout_start": holdout_start, "holdout_end": holdout_end,
        "fold_results": fold_results, "final_choice": final_choice,
        "holdout_results": holdout_results, "slippage_results": slippage_results,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
