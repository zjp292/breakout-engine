import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("results/breakout.db")

# all filter-passing, consol_days>=5, with outcome data
all_df = pd.read_sql_query("""
    SELECT s.raw_score, s.base_quality, s.trend_strength, s.relative_strength_score,
           s.volume_score, s.adr_pct, s.prior_move_pct, s.consol_days,
           s.rs_comp_120, s.vcp_contracting, s.vcp_contraction_ratio,
           s.volume_dryup_ratio, s.pct_from_52wk_high,
           o.max_gain_20d, o.max_gain_60d, o.stop_triggered, o.breakout_triggered
    FROM scans s
    JOIN outcomes o ON s.scan_date=o.scan_date AND s.symbol=o.symbol
    WHERE s.passes_filters=1
      AND COALESCE(s.consol_days,0) >= 5
      AND o.max_gain_20d IS NOT NULL
""", conn)

bo = all_df[all_df["breakout_triggered"] == 1].copy()

print(f"all records:        n={len(all_df)}")
print(f"breakout_triggered: n={len(bo)} ({len(bo)/len(all_df):.1%})")
print(f"avg 20d gain (all):       {all_df['max_gain_20d'].mean():.1%}")
print(f"avg 20d gain (breakouts): {bo['max_gain_20d'].mean():.1%}")
print()

features = [
    "raw_score", "base_quality", "trend_strength", "relative_strength_score",
    "volume_score", "adr_pct", "prior_move_pct", "consol_days",
    "rs_comp_120", "vcp_contracting", "vcp_contraction_ratio",
    "volume_dryup_ratio", "pct_from_52wk_high",
]

print("=" * 65)
print("Spearman r vs max_gain_20d — ALL filter-passing records:")
print("=" * 65)
results_all = []
for f in features:
    joined = all_df[["max_gain_20d", f]].dropna()
    if len(joined) < 10:
        continue
    r = joined[f].corr(joined["max_gain_20d"], method="spearman")
    results_all.append((r, f, len(joined)))
results_all.sort(key=lambda x: -abs(x[0]))
for r, f, n in results_all:
    bar = "+" * max(0, int(r * 50)) if r >= 0 else "-" * max(0, int(abs(r) * 50))
    print(f"  {f:<32} {r:+.3f}  n={n:<5} |{bar}")

print()
print("=" * 65)
print("Spearman r vs max_gain_20d — BREAKOUTS ONLY:")
print("=" * 65)
results_bo = []
for f in features:
    joined = bo[["max_gain_20d", f]].dropna()
    if len(joined) < 10:
        continue
    r = joined[f].corr(joined["max_gain_20d"], method="spearman")
    results_bo.append((r, f, len(joined)))
results_bo.sort(key=lambda x: -abs(x[0]))
for r, f, n in results_bo:
    bar = "+" * max(0, int(r * 50)) if r >= 0 else "-" * max(0, int(abs(r) * 50))
    print(f"  {f:<32} {r:+.3f}  n={n:<5} |{bar}")

print()
print("=" * 65)
print("What predicts a breakout occurring? (mean by triggered=0/1)")
print("=" * 65)
print(all_df.groupby("breakout_triggered")[
    ["adr_pct", "prior_move_pct", "raw_score", "vcp_contracting", "volume_dryup_ratio", "consol_days"]
].mean().round(3).to_string())

print()
print("=" * 65)
print("Score bucket vs 20d gain (all records):")
print("=" * 65)
all_df["bucket"] = (all_df["raw_score"] // 5 * 5).astype(int)
print(all_df.groupby("bucket")["max_gain_20d"].agg(
    n="count", mean="mean", median="median", pct_pos=lambda x: (x > 0).mean()
).round(3).to_string())

print()
print("=" * 65)
print("ADR bucket vs 20d gain:")
print("=" * 65)
bins = [0, 0.07, 0.10, 0.15, 0.20, 0.30, 1.0]
labels = ["<7%", "7-10%", "10-15%", "15-20%", "20-30%", "30%+"]
all_df["adr_bucket"] = pd.cut(all_df["adr_pct"], bins=bins, labels=labels)
print(all_df.groupby("adr_bucket", observed=True)["max_gain_20d"].agg(
    n="count", mean="mean", median="median"
).round(3).to_string())

print()
print("=" * 65)
print("Prior move bucket vs 20d gain:")
print("=" * 65)
bins2 = [0, 0.25, 0.50, 1.0, 2.0, 10.0]
labels2 = ["25-50%", "50-100%", "100-200%", "200%+", "500%+"]
pm = all_df[all_df["prior_move_pct"] > 0].copy()
pm["pm_bucket"] = pd.cut(pm["prior_move_pct"], bins=bins2, labels=labels2)
print(pm.groupby("pm_bucket", observed=True)["max_gain_20d"].agg(
    n="count", mean="mean", median="median"
).round(3).to_string())
