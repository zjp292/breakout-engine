"""
A/B: current scoring formula vs two reweighted variants, through the actual backtester
at matched selectivity (Part-D methodology). scores recomputed fresh from data/history
(no vintage-soup stored raw_score). candidates: current, M1 (dryup->binary), M2
(M1 + base weight cut), random (noise floor).
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path
import sqlite3, sys, pickle
from pathlib import Path
import numpy as np, pandas as pd

from backtester import Backtester, BacktestParams
from engine import Features, Scoring
from config import PARAMETERS
from validation import final_holdout_split, purged_walk_forward_folds

def p(*a): print(*a); sys.stdout.flush()

DB="results/breakout.db"
CACHE=Path("data/validation_cache/reweight_scores.pkl")
TRAIN_MONTHS, TEST_MONTHS, HOLDOUT_MONTHS = 36, 6, 12
LABEL_HORIZON, EMBARGO = 60, 10
MIN_TRADES=15
RANDOM_SEED=42

def recompute():
    conn=sqlite3.connect(DB)
    keys=pd.read_sql("SELECT symbol, scan_date FROM scans WHERE passes_filters=1", conn)
    conn.close()
    with open("data/history/COMPX-full.pkl","rb") as f: compx=pickle.load(f)
    feats=Features(PARAMETERS); sc=Scoring(PARAMETERS); sc.regime_multiplier=1.0
    by_sym=keys.groupby("symbol")["scan_date"].apply(set).to_dict()
    recs=[]
    for i,(sym,dates) in enumerate(by_sym.items(),1):
        pth=Path(f"data/history/{sym}-full.pkl")
        if not pth.exists(): continue
        try:
            with open(pth,"rb") as f: df=pickle.load(f)
            if len(df)<100: continue
            fdf=feats.add_all_features(df, compx)
        except Exception as e:
            p(f"  skip {sym}: {e}"); continue
        dstr=fdf.index.strftime("%Y-%m-%d")
        want=fdf[dstr.isin(dates)]
        for date,row in want.iterrows():
            bd=sc.calculate_total_score(row, rs_rank=None)
            d=bd.details
            dv=d.get("volume_dollar_volume",0.0); vd=d.get("volume_volume_contraction",0.0); adr=d.get("volume_adr",0.0)
            base,trend,rs=bd.base_quality,bd.trend_strength,bd.relative_strength
            dryup_bin = 3.0 if vd>=10.5 else 0.0
            vol_m = dv+dryup_bin+adr           # modified volume, submax 19
            cur = bd.raw_total
            m1 = (base/20)*10 + (trend/14)*15 + (rs/30)*25 + (vol_m/19)*50
            m2 = (base/20)*5  + (trend/14)*15 + (rs/30)*27 + (vol_m/19)*53
            recs.append((sym,date.strftime("%Y-%m-%d"),cur,m1,m2))
        if i%200==0: p(f"  recompute {i}/{len(by_sym)}")
    out=pd.DataFrame(recs,columns=["symbol","scan_date","score_current","score_m1","score_m2"])
    out.to_pickle(CACHE)
    p(f"recomputed {len(out)} rows -> {CACHE}")
    return out

def load_pool(scores):
    _BT=["scan_date","symbol","adr_pct","score","raw_score","grade","base_quality",
         "trend_strength","relative_strength_score","volume_score","rr_score","breakout_level"]
    with sqlite3.connect(DB) as conn:
        pool=pd.read_sql(f"""SELECT s.{', s.'.join(_BT)}, mc.regime, mc.regime_multiplier AS mc_rm,
                             o.max_gain_20d
                             FROM scans s LEFT JOIN market_conditions mc ON s.scan_date=mc.scan_date
                             JOIN outcomes o ON s.scan_date=o.scan_date AND s.symbol=o.symbol
                             WHERE s.passes_filters=1""",conn)
    pool=pool.merge(scores,on=["symbol","scan_date"],how="inner")
    rng=np.random.default_rng(RANDOM_SEED)
    pool["score_random"]=rng.uniform(size=len(pool))
    return pool

def prep(df,col):
    d=df.dropna(subset=[col]).copy(); d["_filter_score"]=d[col]
    return d.sort_values(["scan_date","_filter_score"],ascending=[True,False]).reset_index(drop=True)

def run_bt(frame,start,end,thr):
    params=BacktestParams(start_date=start,end_date=end,min_score=thr,
                          min_regime="CAUTION",min_consol_days=None)
    m=Backtester(params,scans_override=frame).run().metrics
    if not m or m.get("total_trades",0)<MIN_TRADES: return None
    return m

def km(m):
    if m is None: return dict(sortino=None,calmar=None,expectancy=None,n=0,win_rate=None)
    return dict(sortino=round(m["sortino"],3),calmar=round(m["calmar"],3),
                expectancy=round(m["expectancy"],3),n=m["total_trades"],win_rate=round(m["win_rate"],3))

CANDS=["score_current","score_m1","score_m2","score_random"]

def evaluate(pool,train_mask,win_mask,ws,we):
    train=pool[train_mask]
    # matched selectivity: each candidate admits the same train fraction that raw_score>=70 does
    base_frac=float((train["raw_score"]>=70).mean())
    wp=pool[win_mask]; res={}
    for c in CANDS:
        thr=float(train[c].quantile(1-base_frac))
        res[c]=km(run_bt(prep(wp,c),ws,we,thr))
    return base_frac,res

def main():
    if CACHE.exists():
        scores=pd.read_pickle(CACHE); p(f"loaded cached scores: {len(scores)} rows")
    else:
        scores=recompute()
    pool=load_pool(scores)
    p(f"pool with recomputed scores + outcomes: {len(pool)} rows")
    dd=pool[["scan_date"]].drop_duplicates()
    dev_dates,_,hs,he=final_holdout_split(dd,date_col="scan_date",holdout_months=HOLDOUT_MONTHS,
                                          label_horizon_days=LABEL_HORIZON,embargo_days=EMBARGO)
    folds=purged_walk_forward_folds(dev_dates,date_col="scan_date",train_months=TRAIN_MONTHS,
                                    test_months=TEST_MONTHS,label_horizon_days=LABEL_HORIZON,embargo_days=EMBARGO)
    p(f"{len(folds)} folds, holdout {hs}..{he}\n")
    fold_res=[]
    for fi,fold in enumerate(folds,1):
        tr_s,tr_e,te_s,te_e=fold["train_start"],fold["train_end"],fold["test_start"],fold["test_end"]
        trm=(pool["scan_date"]>=tr_s)&(pool["scan_date"]<=tr_e)
        tem=(pool["scan_date"]>=te_s)&(pool["scan_date"]<=te_e)
        frac,res=evaluate(pool,trm,tem,te_s,te_e)
        p(f"fold {fi} (test {te_s}..{te_e}, gate admits {frac:.0%}):")
        for c in CANDS:
            k=res[c]; p(f"  {c:14s} sortino={k['sortino']} calmar={k['calmar']} exp={k['expectancy']} n={k['n']}")
        fold_res.append(res)
    p("\nmean OOS sortino across folds:")
    for c in CANDS:
        vals=[f[c]["sortino"] for f in fold_res if f[c]["sortino"] is not None]
        p(f"  {c:14s} {np.mean(vals):.3f}" if vals else f"  {c:14s} n/a")
    p("\nFINAL HOLDOUT (single look):")
    devm=pool["scan_date"].isin(dev_dates["scan_date"]); hm=(pool["scan_date"]>=hs)&(pool["scan_date"]<=he)
    frac,hr=evaluate(pool,devm,hm,hs,he)
    for c in CANDS:
        k=hr[c]; p(f"  {c:14s} sortino={k['sortino']} calmar={k['calmar']} exp={k['expectancy']} n={k['n']} win={k['win_rate']}")

if __name__=="__main__":
    main()
