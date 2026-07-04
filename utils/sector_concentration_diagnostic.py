"""
cheap diagnostic for item #5 part B (sector concentration). before any full build:
1) fetch sector/industry for filter-passing symbols (yfinance, cached)
2) prevalence: how sector-concentrated is each day's candidate book (top-10 by raw_score)?
3) does concentration predict WORSE outcomes / deeper drawdowns (the tail-risk hypothesis)?
"""
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path
import sys, os, pickle
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
def p(*a): print(*a); sys.stdout.flush()

CACHE = Path("data/validation_cache/sector_map.pkl")

def fetch_sectors(symbols):
    import yfinance as yf
    def one(s):
        try:
            info = yf.Ticker(s).info
            return s, info.get("sector"), info.get("industry")
        except Exception:
            return s, None, None
    out = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        for i,(s,sec,ind) in enumerate(ex.map(one, symbols),1):
            out[s] = (sec, ind)
            if i % 100 == 0: p(f"  fetched {i}/{len(symbols)}")
    return out

def main():
    c = sqlite3.connect('results/breakout.db')
    syms = [r[0] for r in c.execute("SELECT DISTINCT symbol FROM scans WHERE passes_filters=1")]
    p(f"filter-passing symbols: {len(syms)}")
    if CACHE.exists():
        smap = pickle.load(open(CACHE,"rb")); p(f"loaded cached sector map: {len(smap)}")
        missing = [s for s in syms if s not in smap]
        if missing:
            p(f"fetching {len(missing)} new..."); smap.update(fetch_sectors(missing)); pickle.dump(smap, open(CACHE,"wb"))
    else:
        smap = fetch_sectors(syms); pickle.dump(smap, open(CACHE,"wb"))
    cov = sum(1 for s in syms if smap.get(s,(None,))[0])
    p(f"sector coverage: {cov}/{len(syms)} ({100*cov/len(syms):.0f}%)")

    # pool: filter-passing, raw_score>=70 (traded gate), with outcomes
    df = pd.read_sql("""SELECT s.scan_date, s.symbol, s.raw_score, o.max_gain_20d, o.max_drawdown_20d
                        FROM scans s JOIN outcomes o ON s.scan_date=o.scan_date AND s.symbol=o.symbol
                        WHERE s.passes_filters=1 AND s.raw_score>=70 AND o.max_gain_20d IS NOT NULL""", c)
    df["sector"] = df["symbol"].map(lambda s: (smap.get(s) or (None,None))[0])
    df = df.dropna(subset=["sector"])
    p(f"\npool (raw_score>=70, has sector+outcome): {len(df)} rows, {df.scan_date.nunique()} dates")
    p("top sectors:"); p(df["sector"].value_counts().head(10).to_string())

    # per-date candidate book = top-10 by raw_score
    recs = []
    for dt, g in df.groupby("scan_date"):
        g10 = g.sort_values("raw_score", ascending=False).head(10)
        if len(g10) < 3:  # need a real book to talk about concentration
            continue
        shares = g10["sector"].value_counts(normalize=True)
        maxshare = shares.iloc[0]
        hhi = (shares**2).sum()
        recs.append(dict(scan_date=dt, n=len(g10), maxshare=maxshare, hhi=hhi,
                         mean_gain=g10.max_gain_20d.mean(), median_gain=g10.max_gain_20d.median(),
                         min_gain=g10.max_gain_20d.min(), mean_dd=g10.max_drawdown_20d.mean(),
                         worst_dd=g10.max_drawdown_20d.min()))
    d = pd.DataFrame(recs)
    p(f"\ndays with a book (>=3 signals): {len(d)}")

    # A. prevalence
    p("\n=== A. sector concentration of the daily top-10 book ===")
    p(f"max single-sector share: mean={d.maxshare.mean():.2f} median={d.maxshare.median():.2f} "
      f"p90={d.maxshare.quantile(.9):.2f} p99={d.maxshare.quantile(.99):.2f}")
    for thr in [0.4,0.5,0.6,0.7]:
        p(f"  days with >{int(thr*100)}% of book in ONE sector: {(d.maxshare>thr).mean()*100:.1f}%")

    # B. does concentration predict worse outcomes / deeper drawdowns?
    p("\n=== B. outcomes by book concentration (tail-risk test) ===")
    d["bucket"] = pd.cut(d.maxshare, [0,0.3,0.5,0.7,1.01], labels=["<30%","30-50%","50-70%",">70%"])
    agg = d.groupby("bucket", observed=True).agg(
        n=("scan_date","size"), mean_gain=("mean_gain","mean"), median_gain=("median_gain","mean"),
        worst_pos_gain=("min_gain","mean"), mean_drawdown=("mean_dd","mean"), worst_drawdown=("worst_dd","mean"))
    p(agg.round(3).to_string())
    p("\n(tail-risk hypothesis predicts: higher concentration -> lower mean_gain and/or deeper mean_drawdown/worst_drawdown)")

    # correlation
    from scipy import stats
    for col in ["mean_gain","mean_dd","worst_dd","min_gain"]:
        r,pv = stats.spearmanr(d.maxshare, d[col])
        p(f"  spearman(maxshare, {col}) = {r:+.3f}  p={pv:.3f}")

if __name__ == "__main__":
    main()
