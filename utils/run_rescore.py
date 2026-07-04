"""
full historical re-score (experiments.md R2-C: fix raw_score vintage mixture — stored
scores predate current scoring code). reuses the tested historical_batch phase2_score
(force_rescore) + phase3_rs_calibration, driven from the local full-history pickles
(no nasdaq-list fetch); skips phase1 download and phase4 market-conditions.
INSERT OR REPLACE upsert on (scan_date, symbol) -> non-destructive to un-rescorable rows.

WARNING: rewrites all scan/outcome rows for symbols with a data/history pickle (~16h,
~4.5M rows). BACK UP results/breakout.db first. needs a valid Schwab token (one $COMPX
call) and, on Windows, PYTHONUTF8=1 (historical_batch prints non-cp1252 chars).

usage (from repo root):  PYTHONUTF8=1 python utils/run_rescore.py
"""
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_here)
_sys.path[:0] = [_here, _ROOT]
_os.chdir(_ROOT)  # historical_batch uses repo-relative DB/data paths

import sqlite3, time
import pandas as pd
from config import PARAMETERS
import historical_batch as hb
from persistence import ScanPersistence


def main():
    symbols = sorted({q.stem.replace("-full", "") for q in hb.HISTORY_DIR.glob("*-full.pkl")}
                     - {"COMPX", "$COMPX"})
    print(f"symbols from pickles: {len(symbols)}"); sys.stdout.flush()

    ScanPersistence(hb.DB_PATH)  # ensure schema
    with sqlite3.connect(hb.DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

    t0 = time.time()
    hb.phase2_score(symbols, pd.Timestamp("2019-06-01"), pd.Timestamp.today().normalize(),
                    hb.DB_PATH, PARAMETERS, force_rescore=True, workers=4)
    print(f"phase2 done in {(time.time()-t0)/60:.1f} min")
    hb.phase3_rs_calibration(hb.DB_PATH, PARAMETERS.get("min_score_alert", 80),
                             PARAMETERS.get("min_score_watchlist", 70))
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    import sys
    main()
