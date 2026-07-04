"""
recomputes the two hard-filter inputs that were never persisted to `scans`:
  - rs_comp_252   (require_positive_rs_252 filter — H9)
  - stop_distance_20d_pct  (the actual value stop_adr_multiple gates on — H8;
    the persisted scans.stop_distance_pct column is the 60-day version, used
    for display/backtester, NOT what apply_hard_filters checks)

both are cheap to recompute from the full-history pickles already on disk
(data/history/{symbol}-full.pkl, data/history/COMPX-full.pkl) via the same
Features pipeline historical_batch.py uses — no new API calls needed.

usage: uv run python recompute_filter_features.py
output: data/validation_cache/recomputed_filter_features.pkl
        columns: symbol, scan_date, rs_comp_252, stop_distance_20d_pct, adr_pct
"""
from __future__ import annotations
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path

import pickle
import time
from pathlib import Path

import pandas as pd

from config import PARAMETERS
from engine import Features

HISTORY_DIR = Path("data") / "history"
OUT_DIR = Path("data") / "validation_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "recomputed_filter_features.pkl"


def main():
    with open(HISTORY_DIR / "COMPX-full.pkl", "rb") as f:
        compx = pickle.load(f)

    features = Features(PARAMETERS)
    symbol_files = sorted(HISTORY_DIR.glob("*-full.pkl"))
    symbol_files = [p for p in symbol_files if p.stem.replace("-full", "") != "COMPX"]
    print(f"recomputing features for {len(symbol_files)} symbols...")

    frames = []
    t0 = time.time()
    for i, path in enumerate(symbol_files, 1):
        symbol = path.stem.replace("-full", "")
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if len(df) < 100:
                continue
            fdf = features.add_all_features(df, compx)
            out = fdf[["rs_comp_252", "stop_distance_20d_pct", "adr_pct"]].copy()
            out["symbol"] = symbol
            out["scan_date"] = fdf.index.strftime("%Y-%m-%d")
            frames.append(out.reset_index(drop=True))
        except Exception as e:
            print(f"  skip {symbol}: {e}")
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(symbol_files)} ({elapsed:.0f}s elapsed, ~{elapsed/i*len(symbol_files):.0f}s total est.)")

    result = pd.concat(frames, ignore_index=True)
    result.to_pickle(OUT_FILE)
    print(f"wrote {len(result)} rows to {OUT_FILE} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
