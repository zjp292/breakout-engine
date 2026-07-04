"""
deep dive on the phase 5 open thread: is the 100-pt hand-tuned score earning its
complexity, and why do two analyses in this repo disagree about stage2?

part A — stage2 conflict resolution. the 2026-06 univariate finding (stage2=True
  EV=0.195 vs False EV=0.276) got stage2 removed from scoring; phase 5's elasticnet
  learned a POSITIVE stage2 coefficient controlling for everything else. classic
  confounding candidate. checks: confounder balance table, short-history cut
  (stage2 requires ~200 bars so recent IPOs can never be stage2=True — and recent
  listings are momentum monsters), stratified EV, and a block-bootstrapped
  regression coefficient.

part B — component ablation. which of the four components carries the score's
  signal? per-component spearman on each purged-fold test window + holdout, plus
  all 15 equal-weight z-sum subsets and the current-weight formula. nothing is
  fitted, so there's no leakage — the fold structure is for stability-across-time,
  and the dev-fold winner is declared BEFORE looking at holdout.

part C — is the elasticnet edge real? paired block bootstrap (resample scan
  dates, compute both spearmans on the same resample, take the difference) for
  EN vs raw_score, EN vs prior_move, raw_score vs prior_move on the holdout.

usage: uv run python analyze_scoring_deepdive.py
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_baseline_comparison import (
    FEATURES, TARGET, _fit_transform, fit_elasticnet, predict_elasticnet,
)
from validation import final_holdout_split, purged_walk_forward_folds

DB_PATH = "results/breakout.db"
OUT_FILE = Path("data") / "validation_cache" / "scoring_deepdive_results.json"

HOLDOUT_MONTHS = 12
LABEL_HORIZON_DAYS = 60
EMBARGO_DAYS = 10
N_BOOT = 2000

COMPONENTS = {
    "base_quality": (20.0, 10.0),
    "trend_strength": (14.0, 15.0),
    "relative_strength_score": (30.0, 25.0),
    "volume_score": (30.0, 50.0),
}
CONFOUNDERS = [
    "adr_pct", "prior_move_pct", "pct_from_52wk_high", "rs_comp_60",
    "consol_days", "consol_range_pct", "dollar_volume",
]


def _load() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT s.scan_date, s.symbol, s.raw_score, s.stage2,
                   s.base_quality, s.trend_strength, s.relative_strength_score, s.volume_score,
                   s.rs_comp_252, {', '.join('s.' + c for c in CONFOUNDERS)},
                   {', '.join('s.' + f for f in FEATURES if f not in CONFOUNDERS and f not in ('stage2', 'rs_comp_252'))},
                   o.max_gain_20d
            FROM scans s JOIN outcomes o USING (scan_date, symbol)
            WHERE s.passes_filters = 1
            """,
            conn,
        )
    return df


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def block_boot_mean_diff(df, group_col, value_col, n_boot=N_BOOT, seed=42):
    """bootstrap CI for EV(group=True) - EV(group=False), resampled by scan_date"""
    dates = df["scan_date"].unique()
    groups = {d: g for d, g in df.groupby("scan_date")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        sub = pd.concat([groups[d] for d in sampled])
        t = sub.loc[sub[group_col] == 1, value_col]
        f = sub.loc[sub[group_col] == 0, value_col]
        if len(t) < 5 or len(f) < 5:
            continue
        diffs.append(t.mean() - f.mean())
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired_boot_spearman_diff(df, score_a, score_b, target, n_boot=N_BOOT, seed=42):
    """bootstrap CI for spearman(a,target) - spearman(b,target), same resample both sides"""
    dates = df["scan_date"].unique()
    groups = {d: g for d, g in df.groupby("scan_date")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        sub = pd.concat([groups[d] for d in sampled])
        ra = spearman(sub[score_a], sub[target])
        rb = spearman(sub[score_b], sub[target])
        if np.isnan(ra) or np.isnan(rb):
            continue
        diffs.append(ra - rb)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "p_gt_0": float((diffs > 0).mean()),
    }


# ── part A ────────────────────────────────────────────────────────────────────

def part_a(df, dev, holdout):
    print("\n" + "=" * 78)
    print("PART A — stage2 conflict resolution")
    print("=" * 78)
    out = {}

    for label, d in [("full", df), ("dev", dev), ("holdout", holdout)]:
        t = d[d["stage2"] == 1][TARGET]
        f = d[d["stage2"] == 0][TARGET]
        mean_diff, lo, hi = block_boot_mean_diff(d, "stage2", TARGET)
        print(f"\n[{label}] stage2=True n={len(t)} EV={t.mean():.4f} | "
              f"stage2=False n={len(f)} EV={f.mean():.4f} | "
              f"diff={mean_diff:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")
        out[f"univariate_{label}"] = {
            "n_true": int(len(t)), "ev_true": float(t.mean()),
            "n_false": int(len(f)), "ev_false": float(f.mean()),
            "diff": mean_diff, "ci": [lo, hi],
        }

    # confounder balance: standardized mean difference true vs false
    print("\nconfounder balance (standardized mean difference, stage2 True - False):")
    smds = {}
    for c in CONFOUNDERS + ["rs_comp_252"]:
        t = df.loc[df["stage2"] == 1, c].dropna()
        f = df.loc[df["stage2"] == 0, c].dropna()
        pooled_sd = np.sqrt((t.std() ** 2 + f.std() ** 2) / 2)
        smd = (t.mean() - f.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        smds[c] = float(smd)
        print(f"  {c:24s} smd={smd:+.3f}  (T mean={t.mean():.3f}, F mean={f.mean():.3f})")
    out["confounder_smd"] = smds

    # short-history cut: stage2 requires ~200 bars; rs_comp_252 null ~= <252 bars listed.
    # recent listings can never be stage2=True and are the explosive-momentum cohort.
    short = df[df["rs_comp_252"].isna()]
    longh = df[df["rs_comp_252"].notna()]
    print(f"\nshort-history rows (rs_comp_252 null): n={len(short)}, EV={short[TARGET].mean():.4f}, "
          f"stage2=True share={short['stage2'].mean():.1%}")
    print(f"long-history  rows:                    n={len(longh)}, EV={longh[TARGET].mean():.4f}, "
          f"stage2=True share={longh['stage2'].mean():.1%}")
    mean_diff, lo, hi = block_boot_mean_diff(longh, "stage2", TARGET)
    t = longh[longh["stage2"] == 1][TARGET]
    f = longh[longh["stage2"] == 0][TARGET]
    print(f"stage2 effect WITHIN long-history only: True EV={t.mean():.4f} (n={len(t)}) vs "
          f"False EV={f.mean():.4f} (n={len(f)}) | diff={mean_diff:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")
    out["short_history"] = {
        "n_short": int(len(short)), "ev_short": float(short[TARGET].mean()),
        "n_long": int(len(longh)), "ev_long": float(longh[TARGET].mean()),
        "long_only_diff": mean_diff, "long_only_ci": [lo, hi],
        "long_only_ev_true": float(t.mean()), "long_only_ev_false": float(f.mean()),
    }

    # stratified EV by ADR quintile and prior_move quintile
    for strat in ["adr_pct", "prior_move_pct"]:
        d = df.dropna(subset=[strat])
        d = d.assign(_q=pd.qcut(d[strat], 5, labels=False, duplicates="drop"))
        print(f"\nstage2 EV within {strat} quintiles (True vs False):")
        rows = []
        for q, g in d.groupby("_q"):
            t = g[g["stage2"] == 1][TARGET]
            f = g[g["stage2"] == 0][TARGET]
            if len(t) < 20 or len(f) < 20:
                continue
            print(f"  q{q}: True EV={t.mean():.4f} (n={len(t)})  False EV={f.mean():.4f} (n={len(f)})  "
                  f"diff={t.mean()-f.mean():+.4f}")
            rows.append({"q": int(q), "ev_true": float(t.mean()), "n_true": int(len(t)),
                         "ev_false": float(f.mean()), "n_false": int(len(f))})
        out[f"stratified_{strat}"] = rows

    # regression: winsorized target ~ stage2 + z(confounders), block-bootstrapped coef
    d = df.dropna(subset=CONFOUNDERS).copy()
    y = d[TARGET].clip(d[TARGET].quantile(0.01), d[TARGET].quantile(0.99))
    X = pd.DataFrame({"stage2": d["stage2"].astype(float)})
    for c in CONFOUNDERS:
        col = d[c].clip(d[c].quantile(0.01), d[c].quantile(0.99))
        X[c] = (col - col.mean()) / col.std()
    X["_intercept"] = 1.0
    d = d.assign(_y=y, **{f"_x_{c}": X[c] for c in X.columns})

    xcols = [f"_x_{c}" for c in X.columns]
    dates = d["scan_date"].unique()
    groups = {dt: g for dt, g in d.groupby("scan_date")}
    rng = np.random.default_rng(42)
    coefs = []
    for _ in range(N_BOOT):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        sub = pd.concat([groups[dt] for dt in sampled])
        Xb, yb = sub[xcols].to_numpy(), sub["_y"].to_numpy()
        try:
            beta = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            coefs.append(beta[0])  # stage2 is first column
        except np.linalg.LinAlgError:
            continue
    coefs = np.array(coefs)
    print(f"\nOLS stage2 coefficient controlling for {len(CONFOUNDERS)} confounders "
          f"(block-bootstrapped): {np.mean(coefs):+.4f} CI95=[{np.percentile(coefs, 2.5):+.4f},"
          f"{np.percentile(coefs, 97.5):+.4f}]  P(coef>0)={np.mean(coefs > 0):.3f}")
    out["regression"] = {
        "coef_mean": float(np.mean(coefs)),
        "ci": [float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))],
        "p_gt_0": float(np.mean(coefs > 0)),
        "n": int(len(d)),
    }
    return out


# ── part B ────────────────────────────────────────────────────────────────────

def part_b(df, dev, holdout, folds):
    print("\n" + "=" * 78)
    print("PART B — component ablation")
    print("=" * 78)
    out = {}

    # verify reconstruction: raw_score should equal sum((raw/submax)*weight) unless
    # an earnings penalty (-5/-10) was applied at scan time
    recon = sum((df[c] / m) * w for c, (m, w) in COMPONENTS.items())
    resid = (df["raw_score"] - recon).round(4)
    match = (resid.abs() < 0.01).mean()
    penalty_like = resid.isin([-5.0, -10.0]).mean()
    print(f"reconstruction check: {match:.1%} exact, {penalty_like:.1%} off by earnings "
          f"penalty (-5/-10), {1 - match - penalty_like:.1%} other")
    out["reconstruction"] = {"exact": float(match), "earnings_penalty": float(penalty_like)}

    comp_cols = list(COMPONENTS.keys())

    # z-scores computed on dev only, applied everywhere (holdout stays untouched by stats)
    mu, sd = dev[comp_cols].mean(), dev[comp_cols].std()

    def zsum(frame, subset):
        z = (frame[list(subset)] - mu[list(subset)]) / sd[list(subset)]
        return z.mean(axis=1)

    candidates = {"full_score_current_weights": lambda fr: fr["raw_score"]}
    for r in range(1, 5):
        for subset in combinations(comp_cols, r):
            name = "+".join(s.replace("_score", "").replace("relative_strength", "rs") for s in subset)
            candidates[name] = (lambda ss: lambda fr: zsum(fr, ss))(subset)

    fold_rows = []
    for fi, fold in enumerate(folds, 1):
        te = fold["test_df"]
        row = {"fold": fi}
        for name, fn in candidates.items():
            row[name] = spearman(fn(te), te[TARGET])
        fold_rows.append(row)

    means = {name: float(np.nanmean([r[name] for r in fold_rows])) for name in candidates}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    print("\nmean spearman across 6 dev fold windows (nothing fitted, stability check):")
    for name, m in ranked:
        marker = " <-- full score" if name == "full_score_current_weights" else ""
        print(f"  {name:44s} {m:+.4f}{marker}")
    out["dev_fold_means"] = means

    # declare dev winner FIRST, then evaluate holdout for all (winner marked)
    dev_winner = ranked[0][0]
    print(f"\ndev winner (declared before holdout): {dev_winner}")
    holdout_scores = {name: spearman(fn(holdout), holdout[TARGET]) for name, fn in candidates.items()}
    print("holdout spearman:")
    for name, v in sorted(holdout_scores.items(), key=lambda kv: -kv[1]):
        marker = ""
        if name == dev_winner:
            marker = " <-- dev winner"
        if name == "full_score_current_weights":
            marker += " <-- full score"
        print(f"  {name:44s} {v:+.4f}{marker}")
    out["dev_winner"] = dev_winner
    out["holdout"] = holdout_scores

    # per-component fold-by-fold detail for the four singles
    print("\nper-fold detail (single components + full score):")
    singles = [c.replace("_score", "").replace("relative_strength", "rs") for c in comp_cols]
    hdr = "  fold  " + "  ".join(f"{s:>14s}" for s in singles) + f"  {'full':>14s}"
    print(hdr)
    for r in fold_rows:
        vals = "  ".join(f"{r[s]:+14.4f}" for s in singles)
        print(f"  {r['fold']:>4d}  {vals}  {r['full_score_current_weights']:+14.4f}")
    out["fold_detail"] = fold_rows
    return out


# ── part C ────────────────────────────────────────────────────────────────────

def part_c(dev, holdout):
    print("\n" + "=" * 78)
    print("PART C — significance of the elasticnet edge (paired block bootstrap)")
    print("=" * 78)
    out = {}

    devX, holdX = _fit_transform(dev, holdout, FEATURES)
    scaler, model = fit_elasticnet(devX, dev[TARGET])
    holdout = holdout.copy()
    holdout["_en_pred"] = predict_elasticnet(scaler, model, holdX)

    print(f"holdout point estimates: EN={spearman(holdout['_en_pred'], holdout[TARGET]):+.4f}  "
          f"raw_score={spearman(holdout['raw_score'], holdout[TARGET]):+.4f}  "
          f"prior_move={spearman(holdout['prior_move_pct'], holdout[TARGET]):+.4f}")

    for a, b in [("_en_pred", "raw_score"), ("_en_pred", "prior_move_pct"), ("raw_score", "prior_move_pct")]:
        r = paired_boot_spearman_diff(holdout, a, b, TARGET)
        label_a = "elasticnet" if a == "_en_pred" else a
        print(f"  {label_a} - {b}:  diff={r['mean_diff']:+.4f}  "
              f"CI95=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  P(diff>0)={r['p_gt_0']:.3f}")
        out[f"{label_a}_vs_{b}"] = r
    return out


def main():
    df = _load()
    print(f"loaded {len(df)} filter-passing rows with outcomes")

    dev, holdout, hs, he = final_holdout_split(
        df, date_col="scan_date", holdout_months=HOLDOUT_MONTHS,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    folds = purged_walk_forward_folds(
        dev, date_col="scan_date", train_months=36, test_months=6,
        label_horizon_days=LABEL_HORIZON_DAYS, embargo_days=EMBARGO_DAYS,
    )
    print(f"dev n={len(dev)}, holdout {hs}..{he} n={len(holdout)}, {len(folds)} folds")

    results = {
        "part_a": part_a(df, dev, holdout),
        "part_b": part_b(df, dev, holdout, folds),
        "part_c": part_c(dev, holdout),
    }
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
