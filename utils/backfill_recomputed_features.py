"""
one-time backfill: populates the newly-added scans.rs_comp_252 and
scans.stop_distance_20d_pct columns for existing rows, using the cache already
built by recompute_filter_features.py (5,054,546 symbol-days recomputed from
data/history/*.pkl). New rows going forward get these columns natively from
historical_batch.py / persistence.py — this script is a one-time catch-up for
rows written before those columns existed.

loads the cache into a temp SQLite table (indexed on scan_date, symbol) and does
a single set-based UPDATE rather than millions of individual parameterized
UPDATEs — the correlated-subquery form is a few orders of magnitude faster in
SQLite than executemany row-by-row for a table this size.

usage: uv run python backfill_recomputed_features.py
"""
import sys as _sys, os as _os  # utils/ path bootstrap
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]  # utils/ and repo root on path
import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = "results/breakout.db"
CACHE_PATH = Path("data") / "validation_cache" / "recomputed_filter_features.pkl"


def main():
    cache = pd.read_pickle(CACHE_PATH)[["scan_date", "symbol", "rs_comp_252", "stop_distance_20d_pct"]]
    print(f"loaded {len(cache)} rows from recompute cache")

    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()

    conn.execute("DROP TABLE IF EXISTS _recompute_cache")
    cache.to_sql("_recompute_cache", conn, index=False)
    conn.execute("CREATE INDEX idx_recompute_cache_key ON _recompute_cache(scan_date, symbol)")
    print(f"staged temp table ({time.time()-t0:.0f}s)")

    before = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE rs_comp_252 IS NOT NULL"
    ).fetchone()[0]

    conn.execute("""
        UPDATE scans
        SET rs_comp_252 = (
                SELECT c.rs_comp_252 FROM _recompute_cache c
                WHERE c.scan_date = scans.scan_date AND c.symbol = scans.symbol
            ),
            stop_distance_20d_pct = (
                SELECT c.stop_distance_20d_pct FROM _recompute_cache c
                WHERE c.scan_date = scans.scan_date AND c.symbol = scans.symbol
            )
        WHERE EXISTS (
            SELECT 1 FROM _recompute_cache c
            WHERE c.scan_date = scans.scan_date AND c.symbol = scans.symbol
        )
    """)
    conn.commit()
    print(f"backfill UPDATE complete ({time.time()-t0:.0f}s total)")

    conn.execute("DROP TABLE _recompute_cache")
    conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) FROM scans WHERE rs_comp_252 IS NOT NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    print(f"rs_comp_252 populated: {before} -> {after} of {total} total rows")

    conn.close()
    print(f"done ({time.time()-t0:.0f}s total)")
    print("(skipped VACUUM to avoid ~2x temp disk usage on a 2GB+ db — "
          "run manually later if you want to reclaim space from the ALTER TABLE churn)")


if __name__ == "__main__":
    main()
