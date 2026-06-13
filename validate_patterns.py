"""
validate the scoring engine against confirmed qullamaggie breakout examples
from test_training_data.py.

fetches OHLCV via Schwab API, computes all features, scores at the trading
day BEFORE each known breakout, then measures post-breakout profit.
uses forward returns + scipy to suggest optimal component weights.

usage:
    python validate_patterns.py               # first 28 keys (2020 covid-era)
    python validate_patterns.py --all         # all examples including 2025
    python validate_patterns.py --key 1 6 18  # specific keys only
    python validate_patterns.py --no-cache    # force re-fetch from API
    python validate_patterns.py --optimize    # run weight optimizer
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from config import PARAMETERS
from engine import Features, Scoring
from ingestion import SchwabAPIClient
from test_training_data import TRAINING_DATA

CACHE_DIR = Path("data/validation_cache")

# component normalization maxes — must match scoring method caps (trend raised to 23 in 2026-05)
_COMP_MAXES = {
    "base":   20.0,
    "trend":  23.0,
    "rs":     30.0,
    "volume": 30.0,
}


def _ts(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)


def _prev_bday(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str) - pd.offsets.BDay(1)


def fetch_cached(
    api: SchwabAPIClient,
    symbol: str,
    start_ts: int,
    end_ts: int,
    no_cache: bool = False,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{start_ts}_{end_ts}.pkl"
    if not no_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    df = api.get_index_data(symbol, start_ts, end_ts)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def compute_forward_returns(df: pd.DataFrame, breakout_date: str, end_date: str) -> dict:
    """
    compute post-breakout profit metrics starting from the breakout day close.

    kj_return: simplified qullamaggie trailing stop — exit first close below SMA-10
    max_gain: best possible exit before end_date (peak of trade)
    """
    bo_ts  = pd.Timestamp(breakout_date)
    end_ts = pd.Timestamp(end_date)

    future = df.loc[(df.index >= bo_ts) & (df.index <= end_ts)]
    if len(future) == 0:
        return {}

    entry = future.iloc[0]["close"]
    result = {"entry_price": entry}

    for n, key in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
        idx = min(n, len(future) - 1)
        result[key] = future.iloc[idx]["close"] / entry - 1

    result["max_gain"] = future["close"].max() / entry - 1
    result["max_dd"]   = future["close"].min() / entry - 1

    # qullamaggie trailing stop: entry at breakout close, exit first close below SMA-10
    kj_return = None
    kj_days   = None
    for i, (_, r) in enumerate(future.iterrows()):
        sma10 = r.get("sma_10", float("nan"))
        if i > 0 and not pd.isna(sma10) and r["close"] < sma10:
            kj_return = r["close"] / entry - 1
            kj_days   = i
            break

    if kj_return is None:
        kj_return = future.iloc[-1]["close"] / entry - 1
        kj_days   = len(future) - 1

    result["kj_return"] = kj_return
    result["kj_days"]   = kj_days

    return result


def score_example(
    api: SchwabAPIClient,
    features: Features,
    scoring: Scoring,
    key: int,
    ex: dict,
    no_cache: bool = False,
) -> dict | None:
    ticker       = ex["ticker"]
    breakout_date = ex["breakout_date"]
    end_date     = ex["end_date"]

    if not breakout_date:
        print(f"  {ticker:<6} (key {key}): no breakout_date — skipped")
        return None

    score_day = _prev_bday(breakout_date)
    history_start = (pd.Timestamp(breakout_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    start_ts = _ts(history_start)
    end_ts   = _ts(end_date)

    try:
        stock_df = fetch_cached(api, ticker, start_ts, end_ts, no_cache)
        bench_df = fetch_cached(api, "$COMPX", start_ts, end_ts, no_cache)
    except Exception as e:
        print(f"  {ticker:<6} (key {key}): fetch failed — {e}")
        return None

    if stock_df is None or len(stock_df) < 50:
        n = len(stock_df) if stock_df is not None else 0
        print(f"  {ticker:<6} (key {key}): insufficient data ({n} rows)")
        return None

    try:
        df = features.add_all_features(stock_df, bench_df)
    except Exception as e:
        print(f"  {ticker:<6} (key {key}): feature error — {e}")
        return None

    # score at the day BEFORE the breakout
    valid = df.index[df.index <= score_day]
    if len(valid) == 0:
        print(f"  {ticker:<6} (key {key}): no data before {score_day.date()}")
        return None
    actual_day = valid[-1]
    row = df.loc[actual_day]

    passes, failures = scoring.apply_hard_filters(row)
    bd = scoring.calculate_total_score(row, rs_rank=None)

    result = {
        "key":            key,
        "ticker":         ticker,
        "is_breakout":    ex["is_breakout"] == "True",
        "breakout_date":  breakout_date,
        "score_day":      actual_day.date(),
        "passes_filters": passes,
        "filter_failures": failures,
        "raw":            bd.raw_total,
        "base":           bd.base_quality,
        "trend":          bd.trend_strength,
        "rs":             bd.relative_strength,
        "volume":         bd.volume_profile,
        "grade":          scoring.get_grade(bd.raw_total),
        # normalized component fractions (0-1) for weight optimization
        "base_frac":      bd.base_quality   / _COMP_MAXES["base"],
        "trend_frac":     bd.trend_strength / _COMP_MAXES["trend"],
        "rs_frac":        bd.relative_strength / _COMP_MAXES["rs"],
        "volume_frac":    bd.volume_profile  / _COMP_MAXES["volume"],
        # raw feature values for diagnostics
        "close":          row.get("close",           float("nan")),
        "adr_pct":        row.get("adr_pct",         float("nan")),
        "prior_move":     row.get("prior_move_pct",  float("nan")),
        "range_10":       row.get("range_10",        float("nan")),
        "vd_ratio":       row.get("volume_dryup_ratio", float("nan")),
        "rs_comp_60":     row.get("rs_comp_60",      float("nan")),
        "rs_comp_120":    row.get("rs_comp_120",     float("nan")),
    }

    # forward returns — uses data already fetched through end_date
    fwd = compute_forward_returns(df, breakout_date, end_date)
    result.update(fwd)

    return result


def _spearman(x: list[float], y: list[float]) -> float:
    """spearman r or 0 if too few data points"""
    pairs = [(xi, yi) for xi, yi in zip(x, y) if not (xi != xi or yi != yi)]
    if len(pairs) < 4:
        return 0.0
    xs, ys = zip(*pairs)
    r, _ = stats.spearmanr(xs, ys)
    return float(r) if not np.isnan(r) else 0.0


def optimize_weights(results: list[dict]) -> dict | None:
    """
    find component weights (summing to 100) that maximize spearman(score, kj_return)
    over confirmed breakouts that have forward-return data.

    returns {"base": w1, "trend": w2, "rs": w3, "volume": w4} or None
    """
    bo = [r for r in results if r["is_breakout"] and "kj_return" in r]
    if len(bo) < 6:
        print(f"  only {len(bo)} confirmed breakouts with return data — skipping optimizer")
        return None

    fracs  = np.array([[r["base_frac"], r["trend_frac"], r["rs_frac"], r["volume_frac"]] for r in bo])
    target = np.array([r["kj_return"] for r in bo])

    def neg_spearman(w):
        scores = fracs @ w
        r, _ = stats.spearmanr(scores, target)
        return -r if not np.isnan(r) else 0.0

    # optimize over [0,100] weights, sum = 100
    w0 = np.array([20.0, 20.0, 30.0, 30.0])
    bounds = [(0.0, 80.0)] * 4
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 100.0}]

    res = optimize.minimize(
        neg_spearman,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not res.success:
        print(f"  optimizer did not converge: {res.message}")

    w_opt = res.x
    return {
        "base":   round(w_opt[0], 1),
        "trend":  round(w_opt[1], 1),
        "rs":     round(w_opt[2], 1),
        "volume": round(w_opt[3], 1),
        "_spearman": -res.fun,
        "_n": len(bo),
    }


def print_report(results: list[dict], run_optimizer: bool = False) -> None:
    if not results:
        print("no results to report")
        return

    W = 110
    true_bo  = [r for r in results if r["is_breakout"]]
    false_bo = [r for r in results if not r["is_breakout"]]

    # ── score table ──────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    hdr = (
        f"{'#':<4} {'ticker':<7} {'date':<12} {'raw':>4}"
        f" {'base':>5} {'trd':>5} {'rs':>5} {'vol':>5} {'grade':<4}"
        f" {'5d%':>6} {'20d%':>6} {'kj%':>6} {'max%':>6}  status"
    )
    print(hdr)
    print("-" * W)

    for r in sorted(results, key=lambda x: x["raw"], reverse=True):
        if r["passes_filters"]:
            status = "OK"
        else:
            status = "FAIL: " + "; ".join(r["filter_failures"][:2])
        label = "" if r["is_breakout"] else " [NOT BO]"

        def _pct(key):
            v = r.get(key)
            return f"{v*100:+.0f}" if v is not None else "  --"

        print(
            f"{r['key']:<4} {r['ticker']:<7} {str(r['score_day']):<12}"
            f" {r['raw']:>4.0f} {r['base']:>5.1f} {r['trend']:>5.1f}"
            f" {r['rs']:>5.1f} {r['volume']:>5.1f} {r['grade']:<4}"
            f" {_pct('ret_5d'):>6} {_pct('ret_20d'):>6}"
            f" {_pct('kj_return'):>6} {_pct('max_gain'):>6}"
            f"  {status}{label}"
        )

    print("-" * W)

    # ── summary stats ─────────────────────────────────────────────────────────
    def stats_block(subset: list[dict], label: str) -> None:
        if not subset:
            return
        scored  = [r["raw"] for r in subset]
        passing = [r for r in subset if r["passes_filters"]]
        mean_s  = sum(scored) / len(scored)
        above   = {t: sum(1 for s in scored if s >= t) for t in (70, 60, 50)}
        print(f"\n  {label} (n={len(subset)})")
        print(f"    mean raw: {mean_s:.1f}   min: {min(scored):.0f}   max: {max(scored):.0f}")
        print(f"    passes filters: {len(passing)}/{len(subset)} ({len(passing)/len(subset):.0%})")
        for t, n in above.items():
            print(f"    raw >= {t}: {n}/{len(subset)} ({n/len(subset):.0%})")

        with_returns = [r for r in subset if "kj_return" in r]
        if with_returns:
            kj    = [r["kj_return"] * 100 for r in with_returns]
            mx    = [r["max_gain"]  * 100 for r in with_returns]
            ret20 = [r.get("ret_20d", float("nan")) for r in with_returns]
            ret20 = [v * 100 for v in ret20 if not (v != v)]
            print(f"\n    post-breakout returns (n={len(with_returns)}):")
            print(f"      kj_return:   mean={sum(kj)/len(kj):+.1f}%  min={min(kj):+.1f}%  max={max(kj):+.1f}%")
            if ret20:
                print(f"      ret_20d:     mean={sum(ret20)/len(ret20):+.1f}%  min={min(ret20):+.1f}%  max={max(ret20):+.1f}%")
            print(f"      max_gain:    mean={sum(mx)/len(mx):+.1f}%  min={min(mx):+.1f}%  max={max(mx):+.1f}%")

    stats_block(true_bo,  "confirmed breakouts")
    stats_block(false_bo, "non-breakouts / false positives")

    # ── component correlations with forward returns ───────────────────────────
    with_ret = [r for r in true_bo if "kj_return" in r]
    if with_ret:
        print(f"\n  spearman correlations vs kj_return (n={len(with_ret)} confirmed breakouts):")
        for comp, label in [("base", "base_quality"), ("trend", "trend_strength"),
                             ("rs", "rs"), ("volume", "volume_profile"),
                             ("raw", "raw_score (total)")]:
            r_val = _spearman([r[comp] for r in with_ret], [r["kj_return"] for r in with_ret])
            bar   = "#" * int(abs(r_val) * 20)
            sign  = "+" if r_val >= 0 else "-"
            print(f"    {label:<20} {sign}{abs(r_val):.3f}  [{bar}]")

        print(f"\n  spearman correlations vs max_gain:")
        for comp, label in [("base", "base_quality"), ("trend", "trend_strength"),
                             ("rs", "rs"), ("volume", "volume_profile"),
                             ("raw", "raw_score (total)")]:
            r_val = _spearman([r[comp] for r in with_ret], [r["max_gain"] for r in with_ret])
            bar   = "#" * int(abs(r_val) * 20)
            sign  = "+" if r_val >= 0 else "-"
            print(f"    {label:<20} {sign}{abs(r_val):.3f}  [{bar}]")

    # ── per-component averages ─────────────────────────────────────────────────
    print("\n  per-component averages (confirmed breakouts only):")
    for comp, label, max_pts in [
        ("base",   "base_quality  (max 20)", 20.0),
        ("trend",  "trend_strength(max 23)", 23.0),
        ("rs",     "rs            (max 30)", 30.0),
        ("volume", "volume_profile(max 30)", 30.0),
    ]:
        vals = [r[comp] for r in true_bo]
        if vals:
            mean_v = sum(vals) / len(vals)
            pct    = mean_v / max_pts * 100
            print(f"    {label}  avg={mean_v:.1f}  ({pct:.0f}%)  "
                  f"min={min(vals):.1f}  max={max(vals):.1f}")

    # ── diagnostic averages ────────────────────────────────────────────────────
    print("\n  diagnostic feature averages (confirmed breakouts only):")
    for col, label, fmt in [
        ("prior_move",  "prior_move_pct",  ".1%"),
        ("range_10",    "range_10",        ".1%"),
        ("vd_ratio",    "vd_ratio",        ".2f"),
        ("rs_comp_60",  "rs_comp_60",      ".1%"),
        ("rs_comp_120", "rs_comp_120",     ".1%"),
    ]:
        vals = [r[col] for r in true_bo if not (r[col] != r[col])]
        if vals:
            print(f"    {label:<16} avg={sum(vals)/len(vals):{fmt}}"
                  f"  min={min(vals):{fmt}}  max={max(vals):{fmt}}")

    # ── filter failure analysis ────────────────────────────────────────────────
    blocked = [r for r in true_bo if not r["passes_filters"]]
    if blocked:
        print(f"\n  filter failures on CONFIRMED breakouts ({len(blocked)} stocks):")
        all_failures: dict[str, int] = {}
        for r in blocked:
            for f in r["filter_failures"]:
                all_failures[f] = all_failures.get(f, 0) + 1
        for failure, count in sorted(all_failures.items(), key=lambda x: -x[1]):
            # show avg kj_return for stocks blocked by this filter
            affected = [r for r in blocked if failure in r["filter_failures"]]
            rets = [r["kj_return"] * 100 for r in affected if "kj_return" in r]
            avg_ret = f"{sum(rets)/len(rets):+.1f}%" if rets else "n/a"
            print(f"    {count:2d}x  {failure}  (avg kj={avg_ret})")

    # ── weight optimizer ──────────────────────────────────────────────────────
    if run_optimizer:
        print(f"\n{'-' * W}")
        print("  weight optimizer (maximizing spearman correlation with kj_return)")
        suggested = optimize_weights(results)
        if suggested:
            n   = suggested.pop("_n")
            rho = suggested.pop("_spearman")
            print(f"  optimized on {n} confirmed breakouts  =>  spearman={rho:+.3f}")
            print(f"\n  current weights:  base=20  trend=20  rs=30  volume=30")
            print(f"  suggested weights: base={suggested['base']:.0f}  "
                  f"trend={suggested['trend']:.0f}  "
                  f"rs={suggested['rs']:.0f}  "
                  f"volume={suggested['volume']:.0f}")
            print()
            print("  to apply: update config.py weights dict AND matching sub-component maxes")
            print("  in engine.py score_* methods, then re-run tests.")

    print("=" * W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",      action="store_true", help="include all keys")
    parser.add_argument("--key",      type=int, nargs="+", help="specific keys only")
    parser.add_argument("--no-cache", action="store_true", help="force re-fetch")
    parser.add_argument("--optimize", action="store_true", help="run weight optimizer")
    args = parser.parse_args()

    if args.key:
        subset = {k: TRAINING_DATA[k] for k in args.key if k in TRAINING_DATA}
    elif args.all:
        subset = TRAINING_DATA
    else:
        # first 28 dict entries
        keys   = list(TRAINING_DATA.keys())[:28]
        subset = {k: TRAINING_DATA[k] for k in keys}

    api      = SchwabAPIClient()
    features = Features(PARAMETERS)
    scoring  = Scoring(PARAMETERS)

    print(f"\nvalidating {len(subset)} examples via Schwab API...")
    print("(cached in data/validation_cache/ — use --no-cache to refresh)\n")

    results = []
    for key, ex in sorted(subset.items()):
        ticker = ex["ticker"]
        print(f"  {ticker:<6} (key {key:2d}) ...", end=" ", flush=True)
        r = score_example(api, features, scoring, key, ex, no_cache=args.no_cache)
        if r:
            kj = f"{r['kj_return']*100:+.0f}%" if "kj_return" in r else "n/a"
            mx = f"{r['max_gain']*100:+.0f}%"  if "max_gain"  in r else "n/a"
            status = "OK" if r["passes_filters"] else "FAIL"
            print(f"raw={r['raw']:.0f} [{r['grade']}]  kj={kj}  max={mx}  {status}")
            results.append(r)
        else:
            print("skipped")

    print_report(results, run_optimizer=args.optimize)


if __name__ == "__main__":
    main()
