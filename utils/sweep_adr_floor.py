"""
ADR floor trade-sim (experiments.md follow-up #3 / item #4). sweep the ADR hard-filter
floor 0.07(current)->0.12 through the actual backtester over purged folds + holdout.
gating uses FRESH recomputed current-formula score (reweight_scores.pkl), not vintage
stored raw_score. each candidate restricts the pool to adr_pct>=floor, then gates
score_current>=70 as usual. answers: does the clean per-trade EV headroom (H5) survive
the shrinking-universe / fill-rate tradeoff at the portfolio level?
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path
import sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
from backtester import Backtester, BacktestParams
from validation import final_holdout_split, purged_walk_forward_folds

def p(*a): print(*a); sys.stdout.flush()
DB="results/breakout.db"; CACHE=Path("data/validation_cache/reweight_scores.pkl")
TRAIN_MONTHS,TEST_MONTHS,HOLDOUT_MONTHS=36,6,12
LABEL_HORIZON,EMBARGO=60,10; MIN_TRADES=15
FLOORS=[0.07,0.08,0.09,0.10,0.11,0.12]
GATE=None  # set at runtime: matched selectivity to live raw_score>=70

def load_pool():
    scores=pd.read_pickle(CACHE)
    _BT=["scan_date","symbol","adr_pct","score","raw_score","grade","base_quality",
         "trend_strength","relative_strength_score","volume_score","rr_score","breakout_level"]
    with sqlite3.connect(DB) as conn:
        pool=pd.read_sql(f"""SELECT s.{', s.'.join(_BT)}, mc.regime, mc.regime_multiplier AS mc_rm
                             FROM scans s LEFT JOIN market_conditions mc ON s.scan_date=mc.scan_date
                             WHERE s.passes_filters=1""",conn)
    return pool.merge(scores[["symbol","scan_date","score_current"]],on=["symbol","scan_date"],how="inner")

def prep(df):
    d=df.copy(); d["_filter_score"]=d["score_current"]
    return d.sort_values(["scan_date","_filter_score"],ascending=[True,False]).reset_index(drop=True)

def run_bt(frame,start,end):
    m=Backtester(BacktestParams(start_date=start,end_date=end,min_score=GATE,
                 min_regime="CAUTION",min_consol_days=None),scans_override=frame).run().metrics
    if not m or m.get("total_trades",0)<MIN_TRADES: return None
    return m

def km(m):
    if m is None: return dict(sortino=None,calmar=None,expectancy=None,n=0)
    return dict(sortino=round(m["sortino"],3),calmar=round(m["calmar"],3),
                expectancy=round(m["expectancy"],3),n=m["total_trades"])

def main():
    global GATE
    pool=load_pool(); p(f"pool: {len(pool)} rows")
    frac_live=float((pool["raw_score"]>=70).mean())
    GATE=float(pool["score_current"].quantile(1-frac_live))
    p(f"live selectivity: {frac_live:.1%} of pool -> gate score_current>={GATE:.2f}")
    dd=pool[["scan_date"]].drop_duplicates()
    dev_dates,_,hs,he=final_holdout_split(dd,date_col="scan_date",holdout_months=HOLDOUT_MONTHS,
                                          label_horizon_days=LABEL_HORIZON,embargo_days=EMBARGO)
    folds=purged_walk_forward_folds(dev_dates,date_col="scan_date",train_months=TRAIN_MONTHS,
                                    test_months=TEST_MONTHS,label_horizon_days=LABEL_HORIZON,embargo_days=EMBARGO)
    p(f"{len(folds)} folds, holdout {hs}..{he}\n")
    # dev-fold mean sortino per floor
    fold_metrics={f:[] for f in FLOORS}
    for fi,fold in enumerate(folds,1):
        te_s,te_e=fold["test_start"],fold["test_end"]
        tem=(pool["scan_date"]>=te_s)&(pool["scan_date"]<=te_e)
        line=f"fold {fi} ({te_s}..{te_e}): "
        for fl in FLOORS:
            sub=pool[tem & (pool["adr_pct"]>=fl)]
            k=km(run_bt(prep(sub),te_s,te_e))
            fold_metrics[fl].append(k["sortino"])
            line+=f" {fl:.2f}:s={k['sortino']}/n={k['n']}"
        p(line)
    p("\nmean dev-fold sortino by ADR floor:")
    for fl in FLOORS:
        vals=[v for v in fold_metrics[fl] if v is not None]
        p(f"  floor {fl:.2f}: {np.mean(vals):.3f} (folds with trades: {len(vals)}/{len(FLOORS)})" if vals else f"  floor {fl:.2f}: n/a")
    p("\nFINAL HOLDOUT by ADR floor:")
    hm=(pool["scan_date"]>=hs)&(pool["scan_date"]<=he)
    for fl in FLOORS:
        sub=pool[hm & (pool["adr_pct"]>=fl)]
        k=km(run_bt(prep(sub),hs,he))
        tag=" <- current" if abs(fl-0.07)<1e-9 else ""
        p(f"  floor {fl:.2f}: sortino={k['sortino']} calmar={k['calmar']} exp={k['expectancy']} n={k['n']}{tag}")

if __name__=="__main__":
    main()
