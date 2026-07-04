"""
purged + embargoed chronological cross-validation, per Lopez de Prado.

why this exists: optimizer.py's walk_forward_folds puts test_start immediately
after train_end with no gap. labels here (max_gain_10d/20d/60d) look forward up
to 60 days, so training rows within ~60 days of test_start have outcome windows
that extend into the test period — the model would be partly "trained" on
information that only exists because the test period's price action already
happened. purging removes those rows; embargo adds an extra buffer past the
purge cutoff for residual serial correlation (weekends/holidays stretching the
label horizon, feature autocorrelation near the boundary).

only one-sided (before test_start) because folds are chronological walk-forward,
not scattered k-fold — train is always strictly before test, so there's no
"after test" boundary to purge on the train side.

usage:
    from validation import purged_walk_forward_folds, final_holdout_split

    dev_df, holdout_df, hs, he = final_holdout_split(df, holdout_months=12)
    for fold in purged_walk_forward_folds(dev_df, train_months=6, test_months=3):
        ... fit on fold["train_df"], evaluate on fold["test_df"] ...
    # exactly one look at holdout_df, at the very end
"""
from __future__ import annotations

import pandas as pd


def _gap_cutoff(boundary: pd.Timestamp, label_horizon_days: int, embargo_days: int) -> pd.Timestamp:
    return boundary - pd.Timedelta(days=label_horizon_days + embargo_days)


def purged_walk_forward_folds(
    df: pd.DataFrame,
    date_col: str = "scan_date",
    train_months: int = 6,
    test_months: int = 3,
    label_horizon_days: int = 60,
    embargo_days: int = 10,
    expanding: bool = True,
) -> list[dict]:
    """
    same fold cadence as optimizer.py's walk_forward_folds (test_months-wide
    blocks stepping forward from data_start + train_months), but every train
    fold is truncated at test_start - (label_horizon_days + embargo_days)
    instead of test_start - 1 day.
    """
    dt = pd.to_datetime(df[date_col])
    work = df.copy()
    work["_dt"] = dt

    data_start = dt.min()
    data_end = dt.max()
    folds = []
    test_start = data_start + pd.DateOffset(months=train_months)

    while test_start <= data_end:
        test_end = min(
            test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1),
            data_end,
        )
        purge_cutoff = _gap_cutoff(test_start, label_horizon_days, embargo_days)
        train_end_raw = test_start - pd.Timedelta(days=1)
        train_start = (
            data_start
            if expanding
            else train_end_raw - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)
        )

        train_raw = work[(work["_dt"] >= train_start) & (work["_dt"] <= train_end_raw)]
        train_df = train_raw[train_raw["_dt"] <= purge_cutoff].drop(columns=["_dt"])
        test_df = work[(work["_dt"] >= test_start) & (work["_dt"] <= test_end)].drop(columns=["_dt"])

        if len(train_df) >= 20 and len(test_df) >= 5:
            folds.append(
                {
                    "train_df": train_df.copy(),
                    "test_df": test_df.copy(),
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": min(train_end_raw, purge_cutoff).strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "n_train_raw": len(train_raw),
                    "n_train_purged": len(train_raw) - len(train_df),
                }
            )

        test_start = test_start + pd.DateOffset(months=test_months)

    return folds


def final_holdout_split(
    df: pd.DataFrame,
    date_col: str = "scan_date",
    holdout_months: int = 12,
    label_horizon_days: int = 60,
    embargo_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    carve off the most recent holdout_months as a block touched exactly once.
    dev rows whose label window would bleed into the holdout period are
    purged from dev too — otherwise "pick the best config using dev data"
    would be indirectly informed by holdout-period price action.
    """
    dt = pd.to_datetime(df[date_col])
    work = df.copy()
    work["_dt"] = dt

    data_end = dt.max()
    holdout_start = data_end - pd.DateOffset(months=holdout_months) + pd.Timedelta(days=1)
    purge_cutoff = _gap_cutoff(holdout_start, label_horizon_days, embargo_days)

    dev_df = work[work["_dt"] <= purge_cutoff].drop(columns=["_dt"])
    holdout_df = work[work["_dt"] >= holdout_start].drop(columns=["_dt"])

    return dev_df, holdout_df, holdout_start.strftime("%Y-%m-%d"), data_end.strftime("%Y-%m-%d")


# ── sanity checks (run directly: `uv run python validation.py`) ─────────────

if __name__ == "__main__":
    import numpy as np

    rng = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    synth = pd.DataFrame({"scan_date": rng, "value": np.arange(len(rng))})

    folds = purged_walk_forward_folds(
        synth, train_months=12, test_months=3, label_horizon_days=60, embargo_days=10
    )
    assert folds, "expected at least one fold on 4 years of synthetic daily data"

    for f in folds:
        test_start = pd.Timestamp(f["test_start"])
        train_dates = pd.to_datetime(f["train_df"]["scan_date"])
        assert train_dates.max() < test_start, (
            f"leak: train contains dates >= test_start ({f['train_end']} vs {f['test_start']})"
        )
        # every kept train row's label window (row_date + 60d) must resolve
        # strictly before test_start, with the embargo buffer respected too
        latest_allowed = test_start - pd.Timedelta(days=60 + 10)
        assert train_dates.max() <= latest_allowed, (
            f"embargo violated: train has rows within {60+10}d of test_start"
        )
        assert f["n_train_purged"] > 0, "expected purge to actually remove rows near the boundary"
        print(
            f"fold train={f['train_start']}..{f['train_end']} "
            f"(purged {f['n_train_purged']} of {f['n_train_raw']}) "
            f"test={f['test_start']}..{f['test_end']}"
        )

    dev_df, holdout_df, hs, he = final_holdout_split(
        synth, holdout_months=6, label_horizon_days=60, embargo_days=10
    )
    assert pd.to_datetime(dev_df["scan_date"]).max() < pd.Timestamp(hs)
    assert pd.to_datetime(holdout_df["scan_date"]).min() == pd.Timestamp(hs)
    gap_days = (pd.Timestamp(hs) - pd.to_datetime(dev_df["scan_date"]).max()).days
    assert gap_days >= 60 + 10, f"holdout gap too small: {gap_days}d"
    print(f"holdout {hs}..{he} — dev/holdout gap = {gap_days}d (n_dev={len(dev_df)}, n_holdout={len(holdout_df)})")

    print("all sanity checks passed")
