"""
weight optimizer for the breakout engine scoring system.

finds component weights that maximize predictive power against
actual forward return data from the outcomes table. supports
multiple objective functions and train/test validation to flag
overfitting.

usage:
    python optimizer.py
    python optimizer.py --objective sim_pf
    python optimizer.py --objective composite
    python optimizer.py --objective corr_20d
    python optimizer.py --walk-forward
    python optimizer.py --walk-forward --train-months 9 --test-months 3
    python optimizer.py --walk-forward --rolling --objective composite

objectives:
    sim_pf        — simulated profit factor: stops model is used to assign
                    loss = stop_distance_pct when stopped out before breakout,
                    gain = max_gain_20d otherwise. most aligned with actual
                    trading profit. (default)
    composite     — 0.5 * corr_20d + 0.5 * peak_win_rate. balanced signal
                    that avoids fixed-endpoint bias of corr_realized.
    peak_win_rate — % of top-quartile stocks with max_gain_20d >= 5%.
                    directly rewards explosive early moves.
    corr_20d      — pearson r between score and max_gain_20d (peak, not endpoint)
    corr_10d      — pearson r between score and max_gain_10d
    corr_realized — pearson r between score and pct_change at outcome_date.
                    NOTE: biased toward slow grinders, not breakouts.
    win_rate      — win rate of top-quartile stocks by realized pct_change
    profit_factor — profit factor of top-quartile by realized pct_change
    spearman_20d  — spearman rank corr vs max_gain_20d. robust to outliers;
                    a few 100%+ movers can dominate pearson and cause overfit.
    spearman_10d  — spearman rank corr vs max_gain_10d

regularization:
    --regularize        add L2 penalty to keep weights near current values.
                        prevents extreme swings like trend_strength 15→44.
    --reg-strength N    penalty weight (default: 0.005). try 0.01-0.05 for
                        tighter constraint; 0.0 is identical to no regularize.

walk-forward:
    --walk-forward     rolling optimize/test across time; replaces single 70/30 split
    --train-months N   initial training window length (default: 6)
    --test-months  N   out-of-sample test window per fold (default: 3)
    --rolling          use fixed-size rolling window; default is expanding
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

DB_PATH = Path("results/breakout.db")

CURRENT_WEIGHTS = {
    "base_quality": 20,
    "trend_strength": 20,
    "relative_strength_score": 30,
    "volume_score": 30,
}
COMPONENTS = list(CURRENT_WEIGHTS.keys())
CURRENT_W = np.array(list(CURRENT_WEIGHTS.values()), dtype=float)

_DB_TO_CONFIG = {
    "base_quality": "base_quality",
    "trend_strength": "trend_strength",
    "relative_strength_score": "relative_strength",
    "volume_score": "volume_profile",
}

# each component gets between 10 and 50 points — prevents degenerate all-in weights
BOUNDS = [(10.0, 50.0)] * len(COMPONENTS)

# win threshold for peak_win_rate: breakout must move >= this to count as a win
PEAK_WIN_THRESHOLD = 0.05

# realistic R-multiple cap for sim_pf breakout exits.
# qullamaggie: "cannot make 10x your risk? skip it." his avg winner is 10-20x his
# avg loser at 20-35% win rate. 5R systematically undervalues the rare 300-500%
# winners that drive his actual returns. raised to 10R to reflect realistic exits.
R_CAP = 10.0


# ── data loading ──────────────────────────────────────────────────────────────


def load_data(min_score: float = 0.0, passes_only: bool = True) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"database not found: {DB_PATH}")

    filters = []
    if passes_only:
        filters.append("s.passes_filters = 1")
    if min_score > 0:
        filters.append(f"s.raw_score >= {min_score}")
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        SELECT
            s.scan_date,
            s.base_quality,
            s.trend_strength,
            s.relative_strength_score,
            s.volume_score,
            s.rr_score,
            s.raw_score,
            s.score,
            s.regime_multiplier,
            s.stop_distance_pct,
            o.max_gain_10d,
            o.max_gain_20d,
            o.max_gain_60d,
            o.max_drawdown_20d,
            o.pct_change,
            o.breakout_triggered,
            o.stop_triggered,
            o.days_to_breakout,
            o.days_to_stop
        FROM scans s
        JOIN outcomes o ON s.symbol = o.symbol AND s.scan_date = o.scan_date
        {where}
        ORDER BY s.scan_date
    """
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(query, con)

    df = df.dropna(subset=["max_gain_20d", "max_gain_10d"] + COMPONENTS)
    return df


def normalize_components(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """normalize each component to 0-1 then apply weights to produce a composite score"""
    normed = np.column_stack(
        [df[col].values / CURRENT_WEIGHTS[col] for col in COMPONENTS]
    )
    normed = np.clip(normed, 0.0, 1.0)
    return normed @ weights


# ── objective functions ───────────────────────────────────────────────────────


def obj_correlation(weights: np.ndarray, df: pd.DataFrame, outcome_col: str) -> float:
    scores = normalize_components(df, weights)
    y = df[outcome_col].values
    if np.std(scores) == 0 or np.std(y) == 0:
        return 1.0
    return -np.corrcoef(scores, y)[0, 1]


def obj_spearman(weights: np.ndarray, df: pd.DataFrame, outcome_col: str) -> float:
    """
    spearman rank correlation vs outcome column.
    more robust than pearson: a handful of 100%+ outlier gains can dominate
    pearson and cause the optimizer to overfit to those extreme cases.
    rank correlation rewards correctly ordering stocks, not predicting magnitudes.
    """
    scores = normalize_components(df, weights)
    y = df[outcome_col].values
    if np.std(scores) == 0:
        return 1.0
    corr, _ = spearmanr(scores, y)
    return -corr if not np.isnan(corr) else 1.0


def obj_win_rate(weights: np.ndarray, df: pd.DataFrame) -> float:
    """win rate of top-quartile stocks by realized pct_change at outcome_date"""
    scores = normalize_components(df, weights)
    top = df["pct_change"].values[scores >= np.percentile(scores, 75)]
    return -(top > 0).mean() if len(top) else 1.0


def obj_profit_factor(weights: np.ndarray, df: pd.DataFrame) -> float:
    """profit factor of top-quartile stocks by realized pct_change"""
    scores = normalize_components(df, weights)
    gains = df["pct_change"].values[scores >= np.percentile(scores, 75)]
    if len(gains) == 0:
        return 1.0
    gw = gains[gains > 0].sum()
    gl = abs(gains[gains < 0].sum())
    return -(gw / gl) if gl > 0 else -10.0


def obj_peak_win_rate(weights: np.ndarray, df: pd.DataFrame) -> float:
    """
    % of top-quartile stocks where max_gain_20d >= PEAK_WIN_THRESHOLD.
    directly rewards explosive early moves — the core breakout edge.
    """
    scores = normalize_components(df, weights)
    top = df["max_gain_20d"].values[scores >= np.percentile(scores, 75)]
    return -(top >= PEAK_WIN_THRESHOLD).mean() if len(top) else 1.0


def obj_sim_profit_factor(weights: np.ndarray, df: pd.DataFrame) -> float:
    """
    simulates a realistic breakout trade for each top-quartile stock.

    three mutually exclusive outcomes (checked in priority order):
    - breakout triggered before stop: exit at max_gain_20d (take the move)
    - stop triggered before breakout: exit at -stop_distance_pct (take the loss)
    - neither triggered within 60 bars: hold to outcome_date → use pct_change

    using max_gain_20d for ALL non-stop cases is wrong — it credits stocks that
    fell 20% with their single best intraday tick, inflating win rates to 90%+.
    the three-case model produces realistic profit factors in the 1-5x range.
    """
    scores = normalize_components(df, weights)
    top_mask = scores >= np.percentile(scores, 75)
    top = df[top_mask]
    if len(top) == 0:
        return 1.0

    stop_dist = top["stop_distance_pct"].fillna(0.05).clip(lower=0.001).values
    max_gain = top["max_gain_20d"].values
    realized = top["pct_change"].values
    stop_trig = top["stop_triggered"].fillna(0).values.astype(bool)
    brk_trig = top["breakout_triggered"].fillna(0).values.astype(bool)
    days_stop = top["days_to_stop"].fillna(9999).values
    days_brk = top["days_to_breakout"].fillna(9999).values

    broke_first = brk_trig & (~stop_trig | (days_brk <= days_stop))
    stopped_first = stop_trig & (~brk_trig | (days_stop < days_brk))
    # neither case: no breakout, no stop → use actual realized return (honest)

    # cap breakout gains at R_CAP × stop_dist — a trailing stop can plausibly
    # capture this; crediting the full max_gain_20d peak is unrealistic
    capped_gain = np.minimum(max_gain, stop_dist * R_CAP)
    trade_returns = np.where(
        broke_first, capped_gain, np.where(stopped_first, -stop_dist, realized)
    )

    gw = trade_returns[trade_returns > 0].sum()
    gl = abs(trade_returns[trade_returns < 0].sum())
    return -(gw / gl) if gl > 0 else -10.0


OBJECTIVES = {
    "sim_pf": obj_sim_profit_factor,
    "composite": lambda w, df: (
        0.5 * obj_correlation(w, df, "max_gain_20d") + 0.5 * obj_peak_win_rate(w, df)
    ),
    "peak_win_rate": obj_peak_win_rate,
    "corr_20d": lambda w, df: obj_correlation(w, df, "max_gain_20d"),
    "corr_10d": lambda w, df: obj_correlation(w, df, "max_gain_10d"),
    "corr_realized": lambda w, df: obj_correlation(w, df, "pct_change"),
    "win_rate": obj_win_rate,
    "profit_factor": obj_profit_factor,
    # rank-based: robust to outliers, harder to game than pearson
    "spearman_20d": lambda w, df: obj_spearman(w, df, "max_gain_20d"),
    "spearman_10d": lambda w, df: obj_spearman(w, df, "max_gain_10d"),
}


# ── constraints ───────────────────────────────────────────────────────────────


def make_constraints():
    return [{"type": "eq", "fun": lambda w: w.sum() - 100.0}]


# ── optimization ─────────────────────────────────────────────────────────────


def optimize(
    df: pd.DataFrame,
    objective: str = "sim_pf",
    n_restarts: int = 20,
    regularize: bool = False,
    reg_strength: float = 0.005,
) -> tuple[np.ndarray, float]:
    """
    regularize=True adds an L2 penalty proportional to the squared deviation
    from CURRENT_W. this prevents the optimizer from making extreme weight
    swings (e.g. trend_strength: 15 → 44) that fit training data but collapse
    on the test set. reg_strength=0.005 is a light touch; increase to 0.02+
    for more conservative suggestions.
    """
    obj_fn = OBJECTIVES[objective]

    if regularize:

        def constrained_obj(w):
            w = np.clip(w, BOUNDS[0][0], BOUNDS[0][1])
            w = w / w.sum() * 100.0
            base_loss = obj_fn(w, df)
            # L2 penalty: normalized by 100 so reg_strength is scale-independent
            penalty = reg_strength * np.sum(((w - CURRENT_W) / 100.0) ** 2)
            return base_loss + penalty
    else:

        def constrained_obj(w):
            w = np.clip(w, BOUNDS[0][0], BOUNDS[0][1])
            w = w / w.sum() * 100.0
            return obj_fn(w, df)

    best_result = None
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        x0 = rng.dirichlet(np.ones(len(COMPONENTS))) * 100
        x0 = np.clip(x0, BOUNDS[0][0], BOUNDS[0][1])
        x0 = x0 / x0.sum() * 100.0

        result = minimize(
            constrained_obj,
            x0,
            method="SLSQP",
            bounds=BOUNDS,
            constraints=make_constraints(),
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    w = np.clip(best_result.x, BOUNDS[0][0], BOUNDS[0][1])
    w = w / w.sum() * 100.0
    return w, -best_result.fun


# ── evaluation ────────────────────────────────────────────────────────────────


def evaluate(weights: np.ndarray, df: pd.DataFrame) -> dict:
    scores = normalize_components(df, weights)
    top_mask = scores >= np.percentile(scores, 75)

    r20 = (
        np.corrcoef(scores, df["max_gain_20d"].values)[0, 1]
        if np.std(scores) > 0
        else 0.0
    )
    r10 = (
        np.corrcoef(scores, df["max_gain_10d"].values)[0, 1]
        if np.std(scores) > 0
        else 0.0
    )
    rr = (
        np.corrcoef(scores, df["pct_change"].values)[0, 1]
        if np.std(scores) > 0
        else 0.0
    )

    # realized-return metrics (top quartile)
    top_real = df["pct_change"].values[top_mask]
    real_gw = top_real[top_real > 0].sum()
    real_gl = abs(top_real[top_real < 0].sum())

    # peak-gain metrics: win = max_gain_20d >= threshold (explosive move)
    top_peak = df["max_gain_20d"].values[top_mask]
    peak_wr = (top_peak >= PEAK_WIN_THRESHOLD).mean()

    # simulated profit factor: three-case model (same logic as obj_sim_profit_factor)
    stop_dist = df["stop_distance_pct"].fillna(0.05).clip(lower=0.001).values[top_mask]
    realized = df["pct_change"].values[top_mask]
    stop_trig = df["stop_triggered"].fillna(0).values[top_mask].astype(bool)
    brk_trig = df["breakout_triggered"].fillna(0).values[top_mask].astype(bool)
    days_stop = df["days_to_stop"].fillna(9999).values[top_mask]
    days_brk = df["days_to_breakout"].fillna(9999).values[top_mask]
    broke_first = brk_trig & (~stop_trig | (days_brk <= days_stop))
    stopped_first = stop_trig & (~brk_trig | (days_stop < days_brk))
    capped_peak = np.minimum(top_peak, stop_dist * R_CAP)
    sim_ret = np.where(
        broke_first, capped_peak, np.where(stopped_first, -stop_dist, realized)
    )
    sim_gw = sim_ret[sim_ret > 0].sum()
    sim_gl = abs(sim_ret[sim_ret < 0].sum())

    return {
        "n": len(df),
        # correlation metrics
        "corr_20d": round(float(r20), 4),
        "corr_10d": round(float(r10), 4),
        "corr_realized": round(float(rr), 4),
        # realized-return metrics (top quartile)
        "real_wr": round(float((top_real > 0).mean()), 3),
        "real_pf": round(float(real_gw / real_gl) if real_gl > 0 else 0.0, 3),
        "real_avg": round(float(top_real.mean()), 4),
        # peak-gain metrics (breakout-aligned)
        "peak_wr": round(float(peak_wr), 3),
        "peak_avg": round(float(top_peak.mean()), 4),
        # simulated profit factor (stop-loss model + peak-20d exit)
        "sim_pf": round(float(sim_gw / sim_gl) if sim_gl > 0 else 0.0, 3),
        "sim_wr": round(float((sim_ret > 0).mean()), 3),
        "sim_avg": round(float(sim_ret.mean()), 4),
    }


def print_weights(label: str, weights: np.ndarray) -> None:
    print(f"\n  {label}")
    for name, w in zip(COMPONENTS, weights):
        bar = "#" * int(w / 2)
        print(f"    {name:<28} {w:5.1f}  {bar}")
    print(f"    {'total':<28} {weights.sum():5.1f}")


def print_eval(label: str, m: dict) -> None:
    print(f"\n  [{label}]  n={m['n']}")
    print(f"    corr peak-20d:   {m['corr_20d']:+.4f}")
    print(f"    corr peak-10d:   {m['corr_10d']:+.4f}")
    print(f"    corr realized:   {m['corr_realized']:+.4f}")
    print(f"    ── simulated trade (stop-loss model, {R_CAP:.0f}R-capped exit) ──")
    print(
        f"    sim profit factor: {m['sim_pf']:.3f}   win rate: {m['sim_wr']:.1%}   avg: {m['sim_avg']:+.2%}"
    )
    print(f"    ── top-quartile realized return ──")
    print(
        f"    real profit factor:{m['real_pf']:.3f}   win rate: {m['real_wr']:.1%}   avg: {m['real_avg']:+.2%}"
    )
    print(f"    ── peak gain (max_gain_20d) ──")
    print(
        f"    peak win rate ({PEAK_WIN_THRESHOLD:.0%}+): {m['peak_wr']:.1%}   avg peak: {m['peak_avg']:+.2%}"
    )


# ── train/test split ──────────────────────────────────────────────────────────


def time_split(df: pd.DataFrame, train_frac: float = 0.70):
    df_sorted = df.sort_values("scan_date").reset_index(drop=True)
    split = int(len(df_sorted) * train_frac)
    return df_sorted.iloc[:split].copy(), df_sorted.iloc[split:].copy()


# ── walk-forward validation ───────────────────────────────────────────────────

_METRIC_KEY = {
    "sim_pf": "sim_pf",
    "composite": "sim_pf",
    "peak_win_rate": "peak_wr",
    "corr_20d": "corr_20d",
    "corr_10d": "corr_10d",
    "corr_realized": "corr_realized",
    "win_rate": "real_wr",
    "profit_factor": "real_pf",
    "spearman_20d": "corr_20d",
    "spearman_10d": "corr_10d",
}


def walk_forward_folds(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 3,
    expanding: bool = True,
) -> list[tuple]:
    dt = pd.to_datetime(df["scan_date"])
    df = df.copy()
    df["_dt"] = dt

    data_start = dt.min()
    data_end = dt.max()
    folds = []
    test_start = data_start + pd.DateOffset(months=train_months)

    while test_start <= data_end:
        test_end = min(
            test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1),
            data_end,
        )
        train_end = test_start - pd.Timedelta(days=1)
        train_start = (
            data_start
            if expanding
            else train_end - pd.DateOffset(months=train_months) + pd.Timedelta(days=1)
        )

        train_df = df[(df["_dt"] >= train_start) & (df["_dt"] <= train_end)].drop(
            columns=["_dt"]
        )
        test_df = df[(df["_dt"] >= test_start) & (df["_dt"] <= test_end)].drop(
            columns=["_dt"]
        )

        if len(train_df) >= 20 and len(test_df) >= 5:
            folds.append(
                (
                    train_df.copy(),
                    test_df.copy(),
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )
            )

        test_start = test_start + pd.DateOffset(months=test_months)

    return folds


def run_walk_forward(
    df: pd.DataFrame,
    objective: str = "sim_pf",
    n_restarts: int = 20,
    train_months: int = 6,
    test_months: int = 3,
    expanding: bool = True,
    regularize: bool = False,
    reg_strength: float = 0.005,
) -> list[dict]:
    folds = walk_forward_folds(df, train_months, test_months, expanding)
    if not folds:
        dt = pd.to_datetime(df["scan_date"])
        d0, d1 = dt.min(), dt.max()
        span = (d1.year - d0.year) * 12 + (d1.month - d0.month)
        need = train_months + test_months
        print(f"  data range: {d0.date()} → {d1.date()} (~{span} months)")
        print(
            f"  need at least {need} months for one fold (train={train_months} + test={test_months})"
        )
        if span >= 3:
            sug_tr = max(2, span - 2)
            print(f"  try: --train-months {sug_tr} --test-months {span - sug_tr}")
        print("insufficient data — no walk-forward folds possible")
        return []

    mkey = _METRIC_KEY[objective]
    results = []
    for i, (tr_df, te_df, tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        print(
            f"  fold {i}/{len(folds)}: train={tr_s}–{tr_e} (n={len(tr_df)})"
            f"  test={te_s}–{te_e} (n={len(te_df)})  ...",
            end=" ",
            flush=True,
        )
        opt_w, _ = optimize(
            tr_df,
            objective=objective,
            n_restarts=n_restarts,
            regularize=regularize,
            reg_strength=reg_strength,
        )
        tr_m = evaluate(opt_w, tr_df)
        te_m = evaluate(opt_w, te_df)
        print(f"train {mkey}={tr_m[mkey]:+.4f}  test {mkey}={te_m[mkey]:+.4f}")

        results.append(
            {
                "fold": i,
                "train_start": tr_s,
                "train_end": tr_e,
                "test_start": te_s,
                "test_end": te_e,
                "n_train": len(tr_df),
                "n_test": len(te_df),
                "weights": opt_w,
                "train_metric": tr_m[mkey],
                "test_metric": te_m[mkey],
                "train_metrics": tr_m,
                "test_metrics": te_m,
            }
        )
    return results


def print_walk_forward_report(
    fold_results: list[dict],
    objective: str,
    df_full: pd.DataFrame,
    expanding: bool,
) -> None:
    if not fold_results:
        return

    mkey = _METRIC_KEY[objective]
    W = 76
    mode = "expanding" if expanding else "rolling"
    print("\n" + "=" * W)
    print("WALK-FORWARD VALIDATION REPORT")
    print(f"objective: {objective}  |  folds: {len(fold_results)}  |  window: {mode}")
    print("=" * W)

    hdr = (
        f"  {'fold':<4}  {'train period':<23}  {'test period':<23}"
        f"  {'n_tr':>4}  {'n_te':>4}  {'train':>8}  {'test':>8}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for r in fold_results:
        print(
            f"  {r['fold']:<4}  {r['train_start']}–{r['train_end']}  "
            f"{r['test_start']}–{r['test_end']}  "
            f"{r['n_train']:>4}  {r['n_test']:>4}  "
            f"{r['train_metric']:>+8.4f}  {r['test_metric']:>+8.4f}"
        )

    oos = [r["test_metric"] for r in fold_results]
    ins = [r["train_metric"] for r in fold_results]
    ddof = 1 if len(fold_results) > 1 else 0
    print(sep)
    print(f"  {'mean':<59}  {np.mean(ins):>+8.4f}  {np.mean(oos):>+8.4f}")
    print(
        f"  {'std':<59}  {np.std(ins, ddof=ddof):>8.4f}  {np.std(oos, ddof=ddof):>8.4f}"
    )

    gap = np.mean(ins) - np.mean(oos)
    verdict = (
        "WARNING: significant overfitting"
        if gap > 0.05
        else "MILD overfitting"
        if gap > 0.02
        else "OK — gap is small"
    )
    print(f"\n  overfit gap (in-sample minus OOS): {gap:+.4f}  ({verdict})")

    # weight stability across folds
    all_w = np.vstack([r["weights"] for r in fold_results])
    n_folds = len(fold_results)
    cw = 6
    print(f"\n  weight stability across {n_folds} folds:")
    header = f"  {'component':<28}" + "".join(
        f"  {'f' + str(r['fold']):>{cw}}" for r in fold_results
    )
    header += f"  {'mean':>{cw}}  {'std':>{cw}}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, comp in enumerate(COMPONENTS):
        col = all_w[:, j]
        row = f"  {comp:<28}" + "".join(f"  {v:>{cw}.1f}" for v in col)
        row += f"  {col.mean():>{cw}.1f}  {col.std(ddof=ddof):>{cw}.2f}"
        print(row)

    avg_w = all_w.mean(axis=0)
    avg_w = avg_w / avg_w.sum() * 100.0
    cur_full = evaluate(CURRENT_W, df_full)
    print(f"\n  current weights — full dataset {mkey}: {cur_full[mkey]:+.4f}")
    print(f"  walk-forward OOS mean {mkey}:          {np.mean(oos):+.4f}")
    print_weights("avg walk-forward weights (normalized to 100)", avg_w)
    print("=" * W)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="breakout engine weight optimizer")
    parser.add_argument(
        "--objective",
        choices=list(OBJECTIVES.keys()),
        default="sim_pf",
        help="metric to maximize (default: sim_pf)",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--passes-only",
        action="store_true",
        default=True,
        help="only use stocks that pass all hard filters (default: true)",
    )
    parser.add_argument("--no-passes-only", dest="passes_only", action="store_false")
    parser.add_argument(
        "--restarts",
        type=int,
        default=20,
        help="number of random optimizer restarts (default: 20)",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="use rolling (fixed-size) train window instead of expanding",
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="add L2 penalty to keep weights near current values (reduces overfitting)",
    )
    parser.add_argument(
        "--reg-strength",
        type=float,
        default=0.005,
        help="L2 regularization strength (default: 0.005; try 0.01-0.05 for tighter constraint)",
    )
    args = parser.parse_args()

    print(
        f"\nloading data (passes_only={args.passes_only}, min_score={args.min_score})..."
    )
    df = load_data(min_score=args.min_score, passes_only=args.passes_only)
    print(f"loaded {len(df)} records with outcome data")

    if len(df) < 30:
        print("warning: fewer than 30 records — results will likely overfit")

    if args.walk_forward:
        expanding = not args.rolling
        print(
            f"\nrunning walk-forward validation "
            f"(train={args.train_months}mo, test={args.test_months}mo, "
            f"{'expanding' if expanding else 'rolling'} window)..."
        )
        fold_results = run_walk_forward(
            df,
            objective=args.objective,
            n_restarts=args.restarts,
            train_months=args.train_months,
            test_months=args.test_months,
            expanding=expanding,
            regularize=args.regularize,
            reg_strength=args.reg_strength,
        )
        print_walk_forward_report(fold_results, args.objective, df, expanding)
        return

    train, test = time_split(df, train_frac=0.70)
    print(f"train: {len(train)} rows  |  test: {len(test)} rows")

    W = 62
    print("\n" + "=" * W)
    print("BREAKOUT ENGINE — WEIGHT OPTIMIZER")
    print(f"objective: {args.objective}")
    print("=" * W)

    print_weights("current weights", CURRENT_W)
    print_eval("current — train", evaluate(CURRENT_W, train))
    print_eval("current — test ", evaluate(CURRENT_W, test))
    print_eval("current — full ", evaluate(CURRENT_W, df))

    print(f"\noptimizing on train set ({len(train)} records)...")
    opt_w, train_metric = optimize(
        train,
        objective=args.objective,
        n_restarts=args.restarts,
        regularize=args.regularize,
        reg_strength=args.reg_strength,
    )

    print("-" * W)
    print_weights("optimized weights", opt_w)
    print(f"  train {args.objective}: {train_metric:+.4f}")
    print_eval("optimized — train", evaluate(opt_w, train))
    print_eval("optimized — test ", evaluate(opt_w, test))
    print_eval("optimized — full ", evaluate(opt_w, df))

    print("\n" + "=" * W)
    print("suggested config.py update:")
    print("  'weights': {")
    for comp, val in zip(COMPONENTS, opt_w):
        print(f"    '{_DB_TO_CONFIG[comp]}': {round(val)},")
    print("  }")
    print(f"  (sum: {sum(round(v) for v in opt_w)})")

    print("\ndelta from current weights:")
    for name, cur, opt in zip(COMPONENTS, CURRENT_W, opt_w):
        d = opt - cur
        print(f"  {name:<28} {'+' if d >= 0 else ''}{d:.1f}")
    print("=" * W)

    if len(test) >= 10:
        mkey = _METRIC_KEY[args.objective]
        cur_oos = evaluate(CURRENT_W, test)[mkey]
        opt_oos = evaluate(opt_w, test)[mkey]
        direction = "GOOD" if opt_oos > cur_oos else "OVERFIT WARNING"
        print(f"\nout-of-sample {mkey}: {cur_oos:.4f} → {opt_oos:.4f} ({direction})")


if __name__ == "__main__":
    main()
