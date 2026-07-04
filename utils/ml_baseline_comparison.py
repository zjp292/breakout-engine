"""
phase 5 — ML baseline vs the hand-tuned score.

directive's explicit test, last priority by design: build a regularized baseline
(ElasticNet + a shallow GBM) on raw features, purged-CV'd, predicting forward return.
compare OOS performance against (a) the current hand-weighted raw_score and (b) ranking
by prior_move_pct alone within the filter-passing set. if the hand-tuned score doesn't
clearly beat the dumb baseline OOS, that's the headline finding for this pass — not a
reason to add another sub-component.

this is the most direct test of the concern the whole pass started from: the engine's
own validation section found raw_score correlates *negatively* with returns (r=-0.198,
n=28) and false positives score higher than confirmed breakouts on the 91-example
hand-labeled set. this uses the full outcomes table (n=10,648) instead.

features are the RAW inputs (adr_pct, prior_move_pct, rs_comp_*, etc.), not the
hand-scored sub-components (base_quality, trend_strength, ...) — using the sub-components
would bias the comparison toward reproducing the hand-tuned score's own internal logic
rather than genuinely re-deriving weights from scratch.

usage: uv run python ml_baseline_comparison.py
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import RobustScaler

from validation import purged_walk_forward_folds, final_holdout_split

DB_PATH = "results/breakout.db"
OUT_FILE = Path("data") / "validation_cache" / "ml_baseline_results.json"

FEATURES = [
    "adr_pct", "prior_move_pct", "consol_days", "consol_range_pct",
    "vcp_contraction_ratio", "pct_from_52wk_high", "dollar_volume",
    "volume_dryup_ratio", "stop_distance_pct", "stop_distance_20d_pct",
    "rs_comp_20", "rs_comp_60", "rs_comp_120", "rs_comp_252",
    "stage2", "ma_alignment", "vcp_contracting",
]
TARGET = "max_gain_20d"

TRAIN_MONTHS = 36
TEST_MONTHS = 6
HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
CLIP_PCTL = (1, 99)


def _load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT scan_date, symbol, raw_score, {', '.join(FEATURES)}, {TARGET}
            FROM scans s JOIN outcomes o USING (scan_date, symbol)
            WHERE s.passes_filters = 1
            """,
            conn,
        )
    return df


def _fit_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list):
    """median-impute + winsorize on train stats only, no leakage into test."""
    tr, te = train[cols].copy(), test[cols].copy()
    medians = tr.median()
    tr = tr.fillna(medians)
    te = te.fillna(medians)
    lo = tr.quantile(CLIP_PCTL[0] / 100)
    hi = tr.quantile(CLIP_PCTL[1] / 100)
    tr = tr.clip(lower=lo, upper=hi, axis=1)
    te = te.clip(lower=lo, upper=hi, axis=1)
    return tr, te


def _clip_target(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile(CLIP_PCTL[0] / 100), y.quantile(CLIP_PCTL[1] / 100)
    return y.clip(lower=lo, upper=hi)


def fit_elasticnet(train_X, train_y):
    scaler = RobustScaler()
    Xs = scaler.fit_transform(train_X)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], alphas=np.logspace(-4, 1, 30),
        cv=5, max_iter=5000, random_state=42,
    )
    model.fit(Xs, _clip_target(train_y))
    return scaler, model


def predict_elasticnet(scaler, model, X):
    return model.predict(scaler.transform(X))


def fit_gbm(train_X, train_y):
    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    model.fit(train_X, _clip_target(train_y))
    return model


def spearman(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def main():
    df = _load_data()
    print(f"loaded {len(df)} filter-passing rows with outcomes")

    dev, holdout, holdout_start, holdout_end = final_holdout_split(
        df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    folds = purged_walk_forward_folds(
        dev, date_col="scan_date", train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"{len(folds)} folds, holdout {holdout_start}..{holdout_end} (n={len(holdout)})")

    fold_results = []
    for fi, fold in enumerate(folds, 1):
        tr, te = fold["train_df"], fold["test_df"]
        trX, teX = _fit_transform(tr, te, FEATURES)
        try_y, test_y = tr[TARGET], te[TARGET]

        scaler, en_model = fit_elasticnet(trX, try_y)
        en_pred = predict_elasticnet(scaler, en_model, teX)

        gbm_model = fit_gbm(trX, try_y)
        gbm_pred = gbm_model.predict(teX)

        result = {
            "fold": fi,
            "raw_score":       spearman(te["raw_score"], test_y),
            "prior_move_pct":  spearman(te["prior_move_pct"], test_y),
            "elasticnet":      spearman(en_pred, test_y),
            "gbm":             spearman(gbm_pred, test_y),
            "n_train": len(tr), "n_test": len(te),
        }
        print(f"fold {fi}: n_tr={len(tr)} n_te={len(te)}  "
              f"raw_score={result['raw_score']:+.3f}  prior_move={result['prior_move_pct']:+.3f}  "
              f"elasticnet={result['elasticnet']:+.3f}  gbm={result['gbm']:+.3f}")
        fold_results.append(result)

    means = {
        k: float(np.nanmean([r[k] for r in fold_results]))
        for k in ["raw_score", "prior_move_pct", "elasticnet", "gbm"]
    }
    print(f"\nmean OOS spearman across {len(folds)} folds: {means}")

    winner = max(["elasticnet", "gbm"], key=lambda k: means[k])
    print(f"best ML candidate on dev folds: {winner}")

    print(f"\nfinal holdout check (single look, refit {winner} on full dev period):")
    devX, holdX = _fit_transform(dev, holdout, FEATURES)
    if winner == "elasticnet":
        scaler, model = fit_elasticnet(devX, dev[TARGET])
        holdout_pred = predict_elasticnet(scaler, model, holdX)
    else:
        model = fit_gbm(devX, dev[TARGET])
        holdout_pred = model.predict(holdX)

    holdout_results = {
        "raw_score":      spearman(holdout["raw_score"], holdout[TARGET]),
        "prior_move_pct": spearman(holdout["prior_move_pct"], holdout[TARGET]),
        winner:           spearman(holdout_pred, holdout[TARGET]),
    }
    for k, v in holdout_results.items():
        print(f"  {k:16s} holdout spearman = {v:+.4f}")

    if winner == "elasticnet":
        coefs = dict(zip(FEATURES, model.coef_))
        print("\nElasticNet learned coefficients (standardized features):")
        for f, c in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            print(f"  {f:24s} {c:+.4f}")
    else:
        importances = dict(zip(FEATURES, model.feature_importances_)) if hasattr(model, "feature_importances_") else {}
        print("\n(HistGradientBoostingRegressor has no native feature_importances_; skip)")

    out = {
        "n_folds": len(folds), "fold_results": fold_results, "mean_oos_spearman": means,
        "winner": winner, "holdout_start": holdout_start, "holdout_end": holdout_end,
        "holdout_results": holdout_results,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
