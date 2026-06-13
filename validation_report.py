"""
before/after comparison: tests the impact of each feature addition
by temporarily disabling specific scoring bonuses and measuring
spearman correlation changes on the filter-passing confirmed set.
"""
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PARAMETERS
from engine import Features, Scoring
from test_training_data import TRAINING_DATA

CACHE_DIR = Path("data/validation_cache")


def _ts(d):
    return int(datetime.strptime(d, "%Y-%m-%d").timestamp() * 1000)


def _prev_bday(d):
    return pd.Timestamp(d) - pd.offsets.BDay(1)


def fetch_cached(api, symbol, start_ts, end_ts):
    p = CACHE_DIR / f"{symbol}_{start_ts}_{end_ts}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return api.get_index_data(symbol, start_ts, end_ts)


def collect_results(api, features, scoring, subset=None):
    td = subset or TRAINING_DATA
    results = []
    for key, ex in sorted(td.items()):
        ticker = ex["ticker"]
        bo_date = ex.get("breakout_date")
        if not bo_date:
            continue
        score_day = _prev_bday(bo_date)
        hist_start = (pd.Timestamp(bo_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        start_ts = _ts(hist_start)
        end_ts = _ts(ex["end_date"])
        try:
            stock_df = fetch_cached(api, ticker, start_ts, end_ts)
            bench_df = fetch_cached(api, "$COMPX", start_ts, end_ts)
        except Exception:
            continue
        if stock_df is None or len(stock_df) < 50:
            continue
        try:
            df = features.add_all_features(stock_df, bench_df)
        except Exception:
            continue
        valid = df.index[df.index <= score_day]
        if len(valid) == 0:
            continue
        row = df.loc[valid[-1]]
        passes, failures = scoring.apply_hard_filters(row)
        bd = scoring.calculate_total_score(row, rs_rank=None)

        future = df.loc[df.index >= pd.Timestamp(bo_date)]
        if len(future) == 0:
            continue
        entry = future.iloc[0]["close"]
        kj_return = None
        for i, (_, r) in enumerate(future.iterrows()):
            sma10 = r.get("sma_10", float("nan"))
            if i > 0 and not pd.isna(sma10) and r["close"] < sma10:
                kj_return = r["close"] / entry - 1
                break
        if kj_return is None:
            kj_return = future.iloc[-1]["close"] / entry - 1
        max_gain = future["close"].max() / entry - 1

        results.append({
            "key": key, "ticker": ticker,
            "is_breakout": ex["is_breakout"] == "True",
            "passes": passes, "raw": bd.raw_total,
            "base": bd.base_quality, "trend": bd.trend_strength,
            "rs": bd.relative_strength, "volume": bd.volume_profile,
            "kj_return": kj_return, "max_gain": max_gain,
            # new feature flags from the row
            "obv_trend": bool(row.get("obv_trend", False)),
            "is_trigger_bar": bool(row.get("is_trigger_bar", False)),
            "weekly_aligned": bool(row.get("weekly_aligned", True)),
            "vcp_count": int(row.get("vcp_contraction_count", -1)),
            "prior_move": float(row.get("prior_move_pct", 0) or 0),
            "base_depth": float(row.get("base_depth", 0) or 0),
        })
    return results


def sp(x, y):
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi == xi and yi == yi]
    if len(pairs) < 4:
        return 0.0, 1.0
    xs, ys = zip(*pairs)
    r, p = spearmanr(xs, ys)
    return (float(r) if not np.isnan(r) else 0.0), float(p)


def feature_activation_report(results):
    """how often do new features fire on filter-passing confirmed breakouts?"""
    pc = [r for r in results if r["is_breakout"] and r["passes"]]
    pf = [r for r in results if not r["is_breakout"] and r["passes"]]
    print(f"\n{'─'*65}")
    print(f"NEW FEATURE ACTIVATION — passing confirmed (n={len(pc)}) vs fp (n={len(pf)})")
    print(f"{'─'*65}")
    print(f"   {'Feature':<28} {'confirmed':>12} {'false_pos':>12} {'lift':>8}")
    features_to_check = [
        ("obv_trend",    "OBV accumulation (True)"),
        ("is_trigger_bar","Trigger bar (True)"),
        ("weekly_aligned","Weekly aligned (True)"),
    ]
    for key, label in features_to_check:
        c_rate = sum(1 for r in pc if r[key]) / len(pc) if pc else 0
        f_rate = sum(1 for r in pf if r[key]) / len(pf) if pf else 0
        lift   = c_rate - f_rate
        print(f"   {label:<28} {c_rate:>12.0%} {f_rate:>12.0%} {lift:>+8.0%}")

    # vcp contraction count distribution
    print(f"\n   VCP contraction count distribution:")
    for count in range(0, 4):
        c_n = sum(1 for r in pc if r["vcp_count"] == count)
        f_n = sum(1 for r in pf if r["vcp_count"] == count)
        print(f"   count={count}:  confirmed={c_n}/{len(pc)} ({c_n/len(pc):.0%})  fp={f_n}/{len(pf)} ({f_n/len(pf):.0%})")

    # base_depth distribution
    print(f"\n   Base depth (penalty zone >0.25):")
    c_deep = sum(1 for r in pc if r["base_depth"] > 0.25)
    f_deep = sum(1 for r in pf if r["base_depth"] > 0.25)
    print(f"   depth>0.25:  confirmed={c_deep}/{len(pc)} ({c_deep/len(pc):.0%})  fp={f_deep}/{len(pf)} ({f_deep/len(pf):.0%})")


def spearman_of_new_features(results):
    """spearman correlation of each new feature flag vs kj_return."""
    pc = [r for r in results if r["is_breakout"] and r["passes"]]
    if not pc:
        return
    print(f"\n{'─'*65}")
    print(f"SPEARMAN: NEW FEATURES vs kj_return (filter-passing confirmed, n={len(pc)})")
    print(f"{'─'*65}")
    print(f"   {'Feature':<28} {'r(kj)':>8} {'p':>7} {'sig':>4}  note")
    kj = [r["kj_return"] for r in pc]
    for key, label in [
        ("obv_trend", "OBV trend"),
        ("is_trigger_bar", "Trigger bar"),
        ("weekly_aligned", "Weekly aligned"),
        ("vcp_count", "VCP contraction count"),
        ("base_depth", "Base depth"),
        ("prior_move", "Prior move pct"),
    ]:
        vals = [float(r[key]) for r in pc]
        r_val, p_val = sp(vals, kj)
        sig = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "")
        note = "low power (n=28)" if p_val >= 0.10 else ""
        print(f"   {label:<28} {r_val:>+8.3f} {p_val:>7.3f} {sig:>4}  {note}")


def prior_move_filter_impact(results):
    """measure the precision/recall impact of the 25% prior move filter."""
    confirmed = [r for r in results if r["is_breakout"]]
    false_pos  = [r for r in results if not r["is_breakout"]]

    # without prior_move filter: would these stocks pass?
    # blocked_by_prior_move are stocks that fail ONLY because of prior_move (already captured in failures)
    # but since the analysis script doesn't store per-filter info, use the prior_move value directly
    MIN_PM = PARAMETERS.get("min_prior_move_pct", 0.25)
    blocked_conf = [r for r in confirmed if r["prior_move"] < MIN_PM]
    blocked_fp   = [r for r in false_pos  if r["prior_move"] < MIN_PM]
    print(f"\n{'─'*65}")
    print(f"PRIOR MOVE FILTER IMPACT (min={MIN_PM:.0%})")
    print(f"{'─'*65}")
    print(f"   Confirmed blocked by prior_move: {len(blocked_conf)}/{len(confirmed)}")
    if blocked_conf:
        for r in blocked_conf:
            kj = r['kj_return'] * 100
            print(f"     {r['ticker']:<8} prior={r['prior_move']:.1%}  kj={kj:+.0f}%  max={r['max_gain']*100:+.0f}%")
    print(f"   False pos blocked by prior_move: {len(blocked_fp)}/{len(false_pos)}")
    if blocked_fp:
        for r in blocked_fp:
            kj = r['kj_return'] * 100
            print(f"     {r['ticker']:<8} prior={r['prior_move']:.1%}  kj={kj:+.0f}%")


def main():
    from ingestion import SchwabAPIClient
    api = SchwabAPIClient()
    features = Features(PARAMETERS)
    scoring = Scoring(PARAMETERS)

    print("collecting results (from cache)...")
    results = collect_results(api, features, scoring)
    print(f"scored {len(results)} examples\n")

    sep = "=" * 65
    print(sep)
    print("FEATURE IMPACT ANALYSIS — 2026-05-16 IMPROVEMENTS")
    print(sep)

    feature_activation_report(results)
    spearman_of_new_features(results)
    prior_move_filter_impact(results)

    # summary calibration
    pc = [r for r in results if r["is_breakout"] and r["passes"]]
    pf = [r for r in results if not r["is_breakout"] and r["passes"]]
    print(f"\n{'─'*65}")
    print(f"SUMMARY — filter-passing set (n={len(pc)} confirmed, n={len(pf)} fp)")
    print(f"{'─'*65}")
    kj = [r["kj_return"] * 100 for r in pc]
    print(f"   Confirmed avg kj_return  : {np.mean(kj):+.1f}%  (median: {np.median(kj):+.1f}%)")
    kj_fp = [r["kj_return"] * 100 for r in pf]
    print(f"   False pos avg kj_return  : {np.mean(kj_fp):+.1f}%  (median: {np.median(kj_fp):+.1f}%)")
    # win rate
    wins = sum(1 for r in pc if r["kj_return"] > 0)
    print(f"   Confirmed win rate       : {wins}/{len(pc)} = {wins/len(pc):.0%}")
    fp_wins = sum(1 for r in pf if r["kj_return"] > 0)
    print(f"   False pos win rate       : {fp_wins}/{len(pf)} = {fp_wins/len(pf):.0%}")
    # precision
    prec = len(pc) / (len(pc) + len(pf))
    print(f"   Filter precision         : {len(pc)}/{len(pc)+len(pf)} = {prec:.0%}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
