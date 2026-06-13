"""
precision/recall and score calibration analysis on filter-passing stocks only.
isolates the tradeable subset and measures score discrimination.
"""
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from config import PARAMETERS
from engine import Features, Scoring
from test_training_data import TRAINING_DATA

CACHE_DIR = Path("data/validation_cache")
_COMP_MAXES = {"base": 20.0, "trend": 23.0, "rs": 30.0, "volume": 30.0}


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


def score_example(api, features, scoring, key, ex):
    ticker = ex["ticker"]
    bo_date = ex.get("breakout_date")
    if not bo_date:
        return None
    score_day = _prev_bday(bo_date)
    hist_start = (pd.Timestamp(bo_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    start_ts = _ts(hist_start)
    end_ts = _ts(ex["end_date"])
    try:
        stock_df = fetch_cached(api, ticker, start_ts, end_ts)
        bench_df = fetch_cached(api, "$COMPX", start_ts, end_ts)
    except Exception:
        return None
    if stock_df is None or len(stock_df) < 50:
        return None
    try:
        df = features.add_all_features(stock_df, bench_df)
    except Exception:
        return None
    valid = df.index[df.index <= score_day]
    if len(valid) == 0:
        return None
    row = df.loc[valid[-1]]
    passes, failures = scoring.apply_hard_filters(row)
    bd = scoring.calculate_total_score(row, rs_rank=None)

    future = df.loc[df.index >= pd.Timestamp(bo_date)]
    if len(future) == 0:
        return None
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

    return {
        "key": key,
        "ticker": ticker,
        "is_breakout": ex["is_breakout"] == "True",
        "passes": passes,
        "failures": failures,
        "raw": bd.raw_total,
        "base": bd.base_quality,
        "trend": bd.trend_strength,
        "rs": bd.relative_strength,
        "volume": bd.volume_profile,
        "kj_return": kj_return,
        "max_gain": max_gain,
        "base_frac": bd.base_quality / _COMP_MAXES["base"],
        "trend_frac": bd.trend_strength / _COMP_MAXES["trend"],
        "rs_frac": bd.relative_strength / _COMP_MAXES["rs"],
        "volume_frac": bd.volume_profile / _COMP_MAXES["volume"],
    }


def sp(x, y):
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi == xi and yi == yi]
    if len(pairs) < 4:
        return 0.0, 1.0
    xs, ys = zip(*pairs)
    r, p = spearmanr(xs, ys)
    return (float(r) if not np.isnan(r) else 0.0), float(p)


def main():
    from ingestion import SchwabAPIClient
    api = SchwabAPIClient()
    features = Features(PARAMETERS)
    scoring = Scoring(PARAMETERS)

    results = []
    for key, ex in sorted(TRAINING_DATA.items()):
        r = score_example(api, features, scoring, key, ex)
        if r:
            results.append(r)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"BREAKOUT ENGINE — VALIDATION ANALYSIS  (n={len(results)} examples)")
    print(sep)

    confirmed = [r for r in results if r["is_breakout"]]
    false_pos  = [r for r in results if not r["is_breakout"]]
    pass_conf  = [r for r in confirmed if r["passes"]]
    pass_fp    = [r for r in false_pos if r["passes"]]

    # 1. Recall / Precision
    print(f"\n{'─'*72}")
    print("1. RECALL & PRECISION")
    print(f"{'─'*72}")
    recall = len(pass_conf) / len(confirmed) if confirmed else 0
    fp_rate = len(pass_fp) / len(false_pos) if false_pos else 0
    precision = len(pass_conf) / (len(pass_conf) + len(pass_fp)) if (pass_conf or pass_fp) else 0
    print(f"   Recall  : {len(pass_conf):>3}/{len(confirmed):>3} confirmed breakouts pass filters = {recall:.1%}")
    print(f"   FP Rate : {len(pass_fp):>3}/{len(false_pos):>3} false positives   pass filters = {fp_rate:.1%}")
    print(f"   Precision (passing set): {len(pass_conf)}/{len(pass_conf)+len(pass_fp)} = {precision:.1%}")

    # 2. Score separation (all)
    print(f"\n{'─'*72}")
    print("2. SCORE SEPARATION — ALL EXAMPLES")
    print(f"{'─'*72}")
    c_scores  = [r["raw"] for r in confirmed]
    fp_scores = [r["raw"] for r in false_pos]
    print(f"   Confirmed (n={len(c_scores)}):  mean={np.mean(c_scores):.1f}  med={np.median(c_scores):.0f}  std={np.std(c_scores):.1f}")
    print(f"   False pos (n={len(fp_scores)}):  mean={np.mean(fp_scores):.1f}  med={np.median(fp_scores):.0f}  std={np.std(fp_scores):.1f}")
    _, p_mw = stats.mannwhitneyu(c_scores, fp_scores, alternative="greater")
    print(f"   Mann-Whitney U p={p_mw:.3f}  (p<0.05 = confirmed score > false-pos is significant)")
    print(f"   NOTE: false positives score HIGHER avg — dominated by low-price/micro-cap")
    print(f"         confirmed breakouts that have huge returns but fail price/vol filters")

    # 3. Score separation (filter-passing only)
    print(f"\n{'─'*72}")
    print("3. SCORE SEPARATION — FILTER-PASSING STOCKS ONLY")
    print(f"{'─'*72}")
    pc_s  = [r["raw"] for r in pass_conf]
    pfp_s = [r["raw"] for r in pass_fp]
    if pc_s:
        print(f"   Passing confirmed (n={len(pc_s)}):  mean={np.mean(pc_s):.1f}  med={np.median(pc_s):.0f}  std={np.std(pc_s):.1f}")
    if pfp_s:
        print(f"   Passing false pos (n={len(pfp_s)}):  mean={np.mean(pfp_s):.1f}  med={np.median(pfp_s):.0f}  std={np.std(pfp_s):.1f}")
    if len(pc_s) >= 3 and len(pfp_s) >= 3:
        _, p2 = stats.mannwhitneyu(pc_s, pfp_s, alternative="greater")
        print(f"   Mann-Whitney U p={p2:.3f}")

    # 4. Score calibration (filter-passing)
    print(f"\n{'─'*72}")
    print("4. SCORE CALIBRATION — FILTER-PASSING STOCKS")
    print(f"{'─'*72}")
    all_passing = pass_conf + pass_fp
    print(f"   {'Bucket':<10} {'n':>4} {'TP':>4} {'FP':>4} {'Precision':>10} {'avg_kj':>8} {'med_kj':>8} {'win_rt':>8}")
    for lo, hi in [(80, 101), (70, 80), (60, 70), (50, 60), (0, 50)]:
        bucket = [r for r in all_passing if lo <= r["raw"] < hi]
        if not bucket:
            continue
        tp   = [r for r in bucket if r["is_breakout"]]
        fp_b = [r for r in bucket if not r["is_breakout"]]
        prec = len(tp) / len(bucket)
        kj_v = [r["kj_return"] * 100 for r in tp]
        avg_kj = f"{np.mean(kj_v):+.0f}%" if kj_v else "n/a"
        med_kj = f"{np.median(kj_v):+.0f}%" if kj_v else "n/a"
        win_rt = f"{sum(1 for k in kj_v if k > 0)/len(kj_v):.0%}" if kj_v else "n/a"
        print(f"   {lo:>2}-{hi:<5}    {len(bucket):>4} {len(tp):>4} {len(fp_b):>4} {prec:>10.0%} {avg_kj:>8} {med_kj:>8} {win_rt:>8}")

    # 5. Spearman correlations on filter-passing confirmed
    print(f"\n{'─'*72}")
    print(f"5. SPEARMAN CORRELATIONS — FILTER-PASSING CONFIRMED BREAKOUTS (n={len(pass_conf)})")
    print(f"{'─'*72}")
    kj_v = [r["kj_return"] for r in pass_conf]
    mx_v = [r["max_gain"] for r in pass_conf]
    print(f"   {'Component':<22} {'r(kj)':>8} {'p':>7} {'sig':>4}  {'r(max)':>8} {'p':>7} {'sig':>4}")
    for k, label in [("base","base_quality"),("trend","trend_strength"),("rs","relative_strength"),
                     ("volume","volume_profile"),("raw","raw_score")]:
        comp_v = [r[k] for r in pass_conf]
        r_kj, p_kj = sp(comp_v, kj_v)
        r_mx, p_mx = sp(comp_v, mx_v)
        sig_kj = "**" if p_kj < 0.05 else ("*" if p_kj < 0.10 else "")
        sig_mx = "**" if p_mx < 0.05 else ("*" if p_mx < 0.10 else "")
        print(f"   {label:<22} {r_kj:>+8.3f} {p_kj:>7.3f} {sig_kj:>4}  {r_mx:>+8.3f} {p_mx:>7.3f} {sig_mx:>4}")

    # 6. Component averages
    print(f"\n{'─'*72}")
    print("6. COMPONENT AVERAGES — PASSING CONFIRMED vs PASSING FALSE POSITIVES")
    print(f"{'─'*72}")
    for k, label, mx in [("base","base_quality",20),("trend","trend_strength",23),
                          ("rs","relative_strength",30),("volume","volume_profile",30)]:
        c_v  = [r[k] for r in pass_conf]
        fp_v = [r[k] for r in pass_fp]
        c_str  = f"{np.mean(c_v):.1f}/{mx} ({np.mean(c_v)/mx:.0%})" if c_v else "n/a"
        fp_str = f"{np.mean(fp_v):.1f}/{mx} ({np.mean(fp_v)/mx:.0%})" if fp_v else "n/a"
        diff_str = ""
        if c_v and fp_v:
            diff = np.mean(c_v) - np.mean(fp_v)
            diff_str = f"  diff={diff:+.1f}"
        print(f"   {label:<22}  confirmed={c_str}  fp={fp_str}{diff_str}")

    # 7. Filter failure summary
    print(f"\n{'─'*72}")
    print("7. TOP FILTER FAILURES — CONFIRMED BREAKOUTS BLOCKED")
    print(f"{'─'*72}")
    blocked = [r for r in confirmed if not r["passes"]]
    all_f: dict = {}
    for r in blocked:
        for f in r["failures"]:
            key2 = f.split(" ")[0] + " " + f.split(" ")[1] if len(f.split(" ")) > 1 else f
            all_f[key2] = all_f.get(key2, 0) + 1
    top5 = sorted(all_f.items(), key=lambda x: -x[1])[:8]
    for name, count in top5:
        print(f"   {count:>3}x  {name}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
