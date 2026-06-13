"""
three-phase historical data pipeline for the breakout engine.

phase 1 — download:  fetches 7yr + 400d buffer per nasdaq symbol
                     → data/history/{SYMBOL}-full.pkl
phase 2 — score:     features + hard-filter + scoring + forward outcomes
                     → sqlite scans + outcomes tables (rs rank = 0 placeholder)
phase 3 — rs rank:   cross-sectional percentile of rs_comp_60 per date
                     → updates relative_strength_score, raw_score, score, grade, signal

the outcome tracker and backtester both automatically pick up the full-history
pkls: outcome tracker sorts by parent dir name ('history' > '20XX'), backtester
compares stem[-10:] ('SYM-full' > '2025-04-30' lexicographically).

usage:
    uv run python historical_batch.py
    uv run python historical_batch.py --start 2022-01-01 --end 2023-12-31
    uv run python historical_batch.py --skip-download
    uv run python historical_batch.py --skip-scoring
    uv run python historical_batch.py --skip-rs-calibration
    uv run python historical_batch.py --symbols AAPL MSFT NVDA
    uv run python historical_batch.py --workers 4 --delay 0.55 --force-download
"""

import argparse
import math
import pickle
import sqlite3
import threading
import time
import sys
import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import PARAMETERS
from engine import Features, Scoring
from persistence import ScanPersistence

HISTORY_DIR = Path("data") / "history"
PROGRESS_FILE = HISTORY_DIR / ".progress"
DB_PATH = "results/breakout.db"


# ── symbol list ───────────────────────────────────────────────────────────────

def get_nasdaq_symbols() -> list[str]:
    for source_name, fetch_fn in [
        ("nasdaq screener api", _symbols_from_screener),
        ("nasdaq ftp directory", _symbols_from_ftp),
    ]:
        try:
            symbols = fetch_fn()
            if symbols:
                print(f"fetched {len(symbols)} symbols (via {source_name})")
                return symbols
        except Exception as e:
            print(f"  {source_name} unavailable: {e}")

    print("\nerror: could not fetch nasdaq symbol list.")
    print("  pass symbols manually:  --symbols AAPL MSFT NVDA ...")
    sys.exit(1)


def _symbols_from_screener() -> list[str]:
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {"tableonly": "true", "exchange": "nasdaq", "download": "true"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["data"]["rows"]
    return sorted(
        r["symbol"]
        for r in rows
        if r.get("symbol", "").strip().isalpha() and len(r["symbol"]) <= 5
    )


def _symbols_from_ftp() -> list[str]:
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep="|")
    df = df[df["Symbol"].notna()]
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$")]
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    return sorted(df["Symbol"].unique().tolist())


# ── phase 1: download ─────────────────────────────────────────────────────────

def _fetch_ohlcv(client, symbol: str, start_ts: int, end_ts: int, max_retries: int = 3) -> pd.DataFrame | None:
    url = f"{client.base_url_market_data}/pricehistory"
    params = {
        "symbol": symbol,
        "periodType": "year",
        "frequencyType": "daily",
        "frequency": 1,
        "needExtendedHoursData": "false",
        "needPreviousClose": "false",
        "startDate": start_ts,
        "endDate": end_ts,
    }

    for attempt in range(max_retries):
        try:
            token = client.get_access_token()
            headers = {"Authorization": f"Bearer {token}"}
            res = requests.get(url, headers=headers, params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
        except requests.HTTPError:
            if res.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            continue
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            continue

        if "candles" not in data or not data["candles"]:
            return None

        df = pd.DataFrame(data["candles"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")
        df.index = df.index.normalize()
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 100 else None

    return None


def phase1_download(
    symbols: list[str],
    start_ts: int,
    end_ts: int,
    workers: int,
    delay: float,
    force: bool,
):
    from ingestion import SchwabAPIClient

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if not force and PROGRESS_FILE.exists():
        done = set(PROGRESS_FILE.read_text().splitlines())

    to_process = [s for s in symbols if s not in done]
    total = len(to_process)

    print(f"\nphase 1 — download")
    print(f"  {len(done)} already done, {total} remaining")

    if not to_process:
        print("  nothing to do — use --force-download to reprocess")
        return

    client = SchwabAPIClient()
    _lock = threading.Lock()
    _last_call = [0.0]
    _progress_f = open(PROGRESS_FILE, "a")
    saved_count = [0]
    fail_count = [0]

    def download_symbol(symbol: str) -> bool:
        with _lock:
            elapsed = time.monotonic() - _last_call[0]
            if elapsed < delay:
                time.sleep(delay - elapsed)
            _last_call[0] = time.monotonic()

        df = _fetch_ohlcv(client, symbol, start_ts, end_ts)
        if df is None:
            return False

        out_path = HISTORY_DIR / f"{symbol}-full.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(df, f)
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_symbol, sym): sym for sym in to_process}

        for future in as_completed(futures):
            sym = futures[future]
            try:
                ok = future.result()
                with _lock:
                    if ok:
                        saved_count[0] += 1
                    else:
                        fail_count[0] += 1
                    _progress_f.write(f"{sym}\n")
                    _progress_f.flush()
            except Exception:
                with _lock:
                    fail_count[0] += 1

            finished = saved_count[0] + fail_count[0]
            if finished % 25 == 0 or finished == total:
                print(
                    f"  [{finished}/{total}] saved={saved_count[0]} failed={fail_count[0]}",
                    end="\r", flush=True,
                )

    _progress_f.close()
    print(f"\n  done: {saved_count[0]} saved, {fail_count[0]} failed")


# ── phase 2 helpers ───────────────────────────────────────────────────────────

def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


def _safe_bool(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else int(bool(f))
    except (TypeError, ValueError):
        return None


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "C+"
    if score >= 60: return "C"
    return "D"


def _signal(score: float, min_alert: float = 80, min_watch: float = 70) -> str:
    if score >= min_alert: return "STRONG BUY - Alert"
    if score >= min_watch: return "BUY - Watch Closely"
    if score >= 60: return "HOLD - Monitor"
    return "PASS"


def _build_scan_record(date_str: str, symbol: str, row: pd.Series, breakdown, config: dict, passes: bool) -> dict:
    consol_range = _safe_float(row.get("consol_range_60"))
    if consol_range is None:
        consol_range = _safe_float(row.get("consol_range_15"))

    return {
        "scan_date":               date_str,
        "symbol":                  symbol,
        "passes_filters":          int(bool(passes)),
        "score":                   float(breakdown.total),
        "raw_score":               float(breakdown.raw_total),
        "grade":                   _grade(breakdown.raw_total),
        "signal":                  _signal(breakdown.total,
                                            config.get("min_score_alert", 80),
                                            config.get("min_score_watchlist", 70)),
        "base_quality":            float(breakdown.base_quality),
        "trend_strength":          float(breakdown.trend_strength),
        "relative_strength_score": float(breakdown.relative_strength),
        "volume_score":            float(breakdown.volume_profile),
        "rr_score":                float(breakdown.risk_reward),
        "price":                   _safe_float(row.get("close")),
        "breakout_level":          _safe_float(row.get("breakout_level")),
        "stop_level":              _safe_float(row.get("stop_level")),
        "target_level":            _safe_float(row.get("target_level")),
        "stop_distance_pct":       _safe_float(row.get("stop_distance_pct")),
        "potential_r":             _safe_float(row.get("potential_r")),
        "consol_days":             _safe_int(row.get("consol_days")),
        "consol_range_pct":        consol_range,
        "vcp_contracting":         _safe_bool(row.get("vcp_contracting")),
        "vcp_contraction_ratio":   _safe_float(row.get("vcp_contraction_ratio")),
        "stage2":                  _safe_bool(row.get("stage2")),
        "pct_from_52wk_high":      _safe_float(row.get("pct_from_52wk_high")),
        "ma_alignment":            _safe_bool(row.get("ma_alignment")),
        "prior_move_pct":          _safe_float(row.get("prior_move_pct")),
        "rs_comp_20":              _safe_float(row.get("rs_comp_20")),
        "rs_comp_60":              _safe_float(row.get("rs_comp_60")),
        "rs_comp_120":             _safe_float(row.get("rs_comp_120")),
        "dollar_volume":           _safe_float(row.get("dollar_volume")),
        "adr_pct":                 _safe_float(row.get("adr_pct")),
        "volume_dryup_ratio":      _safe_float(row.get("volume_dryup_ratio")),
        "market_regime":           None,
        "market_score":            None,
        "regime_multiplier":       1.0,
    }


def _compute_outcome(
    date_str: str,
    symbol: str,
    entry_price: float,
    row: pd.Series,
    future_df: pd.DataFrame,
) -> dict | None:
    if future_df.empty or entry_price <= 0:
        return None

    f10 = future_df["high"].iloc[:10]
    f20h = future_df["high"].iloc[:20]
    f20l = future_df["low"].iloc[:20]
    f60 = future_df["high"].iloc[:60]
    f60c = future_df["close"].iloc[:60]

    max_gain_10d = _safe_float((f10.max() - entry_price) / entry_price) if len(f10) else None
    max_gain_20d = _safe_float((f20h.max() - entry_price) / entry_price) if len(f20h) else None
    max_gain_60d = _safe_float((f60.max() - entry_price) / entry_price) if len(f60) else None
    max_drawdown_20d = _safe_float((f20l.min() - entry_price) / entry_price) if len(f20l) else None

    breakout_level = _safe_float(row.get("breakout_level"))
    stop_level = _safe_float(row.get("stop_level"))
    target_level = _safe_float(row.get("target_level"))

    breakout_triggered, days_to_breakout = 0, None
    if breakout_level is not None and len(f60):
        above = f60 > breakout_level
        if above.any():
            breakout_triggered = 1
            days_to_breakout = int(np.argmax(above.values)) + 1

    stop_triggered, days_to_stop = 0, None
    if stop_level is not None and len(f60c):
        below = f60c < stop_level
        if below.any():
            stop_triggered = 1
            days_to_stop = int(np.argmax(below.values)) + 1

    target_reached = 0
    if target_level is not None and len(f60):
        if (f60 > target_level).any():
            target_reached = 1

    outcome_idx = min(59, len(future_df) - 1)
    outcome_date = future_df.index[outcome_idx].strftime("%Y-%m-%d")
    current_price = float(future_df["close"].iloc[outcome_idx])
    pct_change = _safe_float((current_price - entry_price) / entry_price)

    return {
        "scan_date":          date_str,
        "symbol":             symbol,
        "outcome_date":       outcome_date,
        "days_elapsed":       outcome_idx + 1,
        "entry_price":        entry_price,
        "current_price":      current_price,
        "pct_change":         pct_change,
        "breakout_triggered": breakout_triggered,
        "stop_triggered":     stop_triggered,
        "target_reached":     target_reached,
        "days_to_breakout":   days_to_breakout,
        "days_to_stop":       days_to_stop,
        "max_gain_10d":       max_gain_10d,
        "max_gain_20d":       max_gain_20d,
        "max_gain_60d":       max_gain_60d,
        "max_drawdown_20d":   max_drawdown_20d,
    }


def _existing_scan_dates(db_path: str, symbol: str) -> set[str]:
    # check ALL records (passing + failing) so re-runs don't re-score already-processed dates
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT scan_date FROM scans WHERE symbol = ?",
            (symbol,),
        ).fetchall()
    return {r[0] for r in rows}


def _write_records(db_path: str, scan_records: list, outcome_records: list):
    with sqlite3.connect(db_path) as conn:
        if scan_records:
            conn.executemany("""
                INSERT OR REPLACE INTO scans (
                    scan_date, symbol, passes_filters,
                    score, raw_score, grade, signal,
                    base_quality, trend_strength, relative_strength_score,
                    volume_score, rr_score,
                    price, breakout_level, stop_level, target_level,
                    stop_distance_pct, potential_r,
                    consol_days, consol_range_pct,
                    vcp_contracting, vcp_contraction_ratio,
                    stage2, pct_from_52wk_high, ma_alignment, prior_move_pct,
                    rs_comp_20, rs_comp_60, rs_comp_120,
                    dollar_volume, adr_pct, volume_dryup_ratio,
                    market_regime, market_score, regime_multiplier
                ) VALUES (
                    :scan_date, :symbol, :passes_filters,
                    :score, :raw_score, :grade, :signal,
                    :base_quality, :trend_strength, :relative_strength_score,
                    :volume_score, :rr_score,
                    :price, :breakout_level, :stop_level, :target_level,
                    :stop_distance_pct, :potential_r,
                    :consol_days, :consol_range_pct,
                    :vcp_contracting, :vcp_contraction_ratio,
                    :stage2, :pct_from_52wk_high, :ma_alignment, :prior_move_pct,
                    :rs_comp_20, :rs_comp_60, :rs_comp_120,
                    :dollar_volume, :adr_pct, :volume_dryup_ratio,
                    :market_regime, :market_score, :regime_multiplier
                )
            """, scan_records)

        if outcome_records:
            conn.executemany("""
                INSERT OR REPLACE INTO outcomes (
                    scan_date, symbol, outcome_date, days_elapsed,
                    entry_price, current_price, pct_change,
                    breakout_triggered, stop_triggered, target_reached,
                    days_to_breakout, days_to_stop,
                    max_gain_10d, max_gain_20d, max_gain_60d, max_drawdown_20d
                ) VALUES (
                    :scan_date, :symbol, :outcome_date, :days_elapsed,
                    :entry_price, :current_price, :pct_change,
                    :breakout_triggered, :stop_triggered, :target_reached,
                    :days_to_breakout, :days_to_stop,
                    :max_gain_10d, :max_gain_20d, :max_gain_60d, :max_drawdown_20d
                )
            """, outcome_records)


def _score_symbol(
    symbol: str,
    compx_df: pd.DataFrame,
    scan_start: pd.Timestamp,
    scan_end: pd.Timestamp,
    features: Features,
    scoring: Scoring,
    db_path: str,
    config: dict,
    force_rescore: bool = False,
) -> tuple[int, int]:
    pkl_path = HISTORY_DIR / f"{symbol}-full.pkl"
    if not pkl_path.exists():
        return 0, 0

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    if len(df) < 100:
        return 0, 0

    # force_rescore ignores existing records so failing rows get backfilled on re-runs
    existing = set() if force_rescore else _existing_scan_dates(db_path, symbol)
    feature_df = features.add_all_features(df, compx_df)

    scan_window = feature_df.loc[
        (feature_df.index >= scan_start) & (feature_df.index <= scan_end)
    ]

    scan_records = []
    outcome_records = []

    for date, row in scan_window.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        if date_str in existing:
            continue

        passes, _ = scoring.apply_hard_filters(row)

        # score every stock regardless of filter result — failing records are saved
        # with passes_filters=0 so the optimizer can analyze filter boundary effects
        breakdown = scoring.calculate_total_score(row, rs_rank=None)
        scan_records.append(_build_scan_record(date_str, symbol, row, breakdown, config, passes))

        # outcomes use next-day open as entry price, not scan-day close.
        # scan-day close is unknowable at signal time (scan runs after market close).
        idx = feature_df.index.get_loc(date)
        future_rows = df.iloc[idx + 1:]
        if not future_rows.empty:
            entry_price = _safe_float(future_rows.iloc[0]["open"])
            if entry_price:
                outcome = _compute_outcome(date_str, symbol, entry_price, row, future_rows)
                if outcome:
                    outcome_records.append(outcome)

    if scan_records or outcome_records:
        _write_records(db_path, scan_records, outcome_records)

    return len(scan_records), len(outcome_records)


# ── phase 2: score ────────────────────────────────────────────────────────────

def phase2_score(
    symbols: list[str],
    scan_start: pd.Timestamp,
    scan_end: pd.Timestamp,
    db_path: str,
    config: dict,
    force_rescore: bool = False,
):
    from ingestion import SchwabAPIClient

    print(f"\nphase 2 — score + outcomes")

    # fetch compx covering full scan range + warmup buffer
    buffer_start = scan_start - timedelta(days=400)
    start_ts = int(buffer_start.timestamp() * 1000)
    end_ts = int(scan_end.timestamp() * 1000)

    print("  fetching $compx benchmark...")
    compx_df = SchwabAPIClient().get_index_data("$COMPX", start_ts, end_ts)
    print(f"  compx: {len(compx_df)} bars ({compx_df.index[0].date()} → {compx_df.index[-1].date()})")

    features = Features(config)
    scoring = Scoring(config)
    scoring.regime_multiplier = 1.0  # no regime gating for historical data

    available = {p.stem.split("-")[0] for p in HISTORY_DIR.glob("*-full.pkl")}
    to_score = [s for s in symbols if s in available]

    trading_days = max(1, (scan_end - scan_start).days * 5 // 7)
    estimated_rows = len(to_score) * trading_days
    print(f"  {len(to_score)} symbols to score")
    print(f"  estimated records: ~{estimated_rows:,} (all stocks, passing + failing)")
    print(f"  NOTE: failing stocks are now saved (passes_filters=0) for filter analysis.")
    print(f"        DB may grow significantly. use --skip-scoring to skip if not needed.")
    if force_rescore:
        print(f"  force-rescore: ignoring existing records, re-processing all dates")

    total_scans    = 0
    total_outcomes = 0
    total_passing  = 0

    for i, symbol in enumerate(to_score, 1):
        try:
            n_scans, n_outcomes = _score_symbol(
                symbol, compx_df, scan_start, scan_end, features, scoring, db_path, config,
                force_rescore=force_rescore,
            )
            total_scans    += n_scans
            total_outcomes += n_outcomes
        except Exception:
            pass

        if i % 50 == 0 or i == len(to_score):
            print(
                f"  [{i}/{len(to_score)}] scans={total_scans:,} outcomes={total_outcomes:,}",
                end="\r", flush=True,
            )

    print(f"\n  done: {total_scans:,} scan records, {total_outcomes:,} outcome records")


# ── phase 3: rs rank calibration ─────────────────────────────────────────────

def phase3_rs_calibration(db_path: str, min_alert: float = 80, min_watch: float = 70):
    """
    Compute cross-sectional RS percentile rank across the FULL downloaded universe
    (passing + failing stocks) so the rank reflects genuine market leadership, not
    just rank-within-the-already-filtered-set.

    Thresholds match the live engine (engine.py score_relative_strength):
      95th+  -> 8 pts  (very top is often distribution, not accumulation)
      85-95th -> 10 pts (sweet spot: leaders before the crowd finds them)
      75-85th ->  7 pts
      65-75th ->  4 pts
      <65th   ->  0 pts

    Both passing and failing records get their relative_strength_score and
    raw_score updated. The passes_filters column remains the source of truth
    for which records the backtester and optimizer should trade.
    """
    print(f"\nphase 3 — rs rank calibration (full universe)")

    with sqlite3.connect(db_path) as conn:
        dates = [
            r[0]
            for r in conn.execute(
                # calibrate on any date that has at least one filter-passing record
                # so we don't waste time on dates with only non-tradeable stocks
                "SELECT DISTINCT scan_date FROM scans WHERE passes_filters = 1 ORDER BY scan_date"
            ).fetchall()
        ]

    print(f"  calibrating {len(dates)} scan dates across full downloaded universe...")
    updated = 0

    for date_str in dates:
        with sqlite3.connect(db_path) as conn:
            # query ALL records for this date — passing + failing — to build the universe
            rows = conn.execute(
                """SELECT id, rs_comp_60, relative_strength_score, raw_score, passes_filters
                   FROM scans
                   WHERE scan_date = ? AND rs_comp_60 IS NOT NULL""",
                (date_str,),
            ).fetchall()

        if len(rows) < 2:
            continue

        ids           = [r[0] for r in rows]
        rs_values     = pd.Series([r[1] for r in rows], dtype=float)
        rs_scores     = np.array([r[2] for r in rows], dtype=float)
        raw_scores    = np.array([r[3] for r in rows], dtype=float)
        passes_flags  = [bool(r[4]) for r in rows]

        # rank against the full universe — a stock at the 90th percentile of
        # 4,000 NASDAQ stocks means far more than 90th of 2 filter-passing stocks
        percentiles = rs_values.rank(pct=True, method="average") * 100

        updates = []
        for i, (row_id, pct) in enumerate(zip(ids, percentiles)):
            # mirror engine.py score_relative_strength exactly
            if pct >= 95:
                rank_pts = 8.0    # top bucket penalized: distribution risk at the pivot
            elif pct >= 85:
                rank_pts = 10.0   # sweet spot: leaders before the crowd finds them
            elif pct >= 75:
                rank_pts = 7.0
            elif pct >= 65:
                rank_pts = 4.0
            else:
                rank_pts = 0.0

            new_rs    = rs_scores[i] + rank_pts
            new_raw   = raw_scores[i] + rank_pts
            new_score = new_raw   # regime_multiplier = 1.0 for all historical records

            updates.append((
                new_rs, new_raw, new_score,
                _grade(new_raw),
                _signal(new_score, min_alert, min_watch),
                row_id,
            ))

        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                """UPDATE scans
                   SET relative_strength_score = ?,
                       raw_score = ?,
                       score = ?,
                       grade = ?,
                       signal = ?
                   WHERE id = ?""",
                updates,
            )

        updated += len(updates)

    print(f"  updated {updated:,} records with full-universe cross-sectional rs rank")


# ── phase 4: backfill market conditions ──────────────────────────────────────

def phase4_backfill_market_conditions(db_path: str, config: dict):
    from ingestion import SchwabAPIClient
    from market_condition import MarketConditionAnalyzer

    print("\nphase 4 — backfill market conditions")

    with sqlite3.connect(db_path) as conn:
        dates = [r[0].strip().rstrip("\\") for r in conn.execute("""
            SELECT DISTINCT s.scan_date
            FROM scans s
            LEFT JOIN market_conditions mc ON s.scan_date = mc.scan_date
            WHERE mc.scan_date IS NULL AND s.passes_filters = 1
            ORDER BY s.scan_date
        """).fetchall() if r[0]]

    # determine the full range needed: either missing dates or just today for refresh
    if dates:
        fetch_start = pd.Timestamp(dates[0]) - timedelta(days=400)
        fetch_end   = pd.Timestamp(dates[-1])
    else:
        fetch_end   = pd.Timestamp("today")
        fetch_start = fetch_end - timedelta(days=365 * 8)

    start_ts = int(fetch_start.timestamp() * 1000)
    end_ts   = int(fetch_end.timestamp() * 1000)

    client = SchwabAPIClient()
    print("  fetching index data...")

    compx_df = client.get_index_data("$COMPX", start_ts, end_ts)
    print(f"  compx: {len(compx_df)} bars ({compx_df.index[0].date()} → {compx_df.index[-1].date()})")

    # always save/refresh compx; backtester's momentum gate reads this pkl
    compx_pkl = HISTORY_DIR / "COMPX-full.pkl"
    compx_df.to_pickle(compx_pkl)
    print(f"  saved compx history → {compx_pkl}")

    if not dates:
        print("  all scan dates already have market conditions")
        return

    print(f"  {len(dates)} dates need market conditions")

    spy_df = None
    try:
        spy_df = client.get_index_data("SPY", start_ts, end_ts)
        print(f"  spy:   {len(spy_df)} bars")
    except Exception as e:
        print(f"  spy unavailable: {e}")

    iwm_df = None
    try:
        iwm_df = client.get_index_data("IWM", start_ts, end_ts)
        print(f"  iwm:   {len(iwm_df)} bars")
    except Exception as e:
        print(f"  iwm unavailable: {e}")

    analyzer = MarketConditionAnalyzer(config)
    db = ScanPersistence(db_path)

    saved = skipped = 0
    for i, date_str in enumerate(dates, 1):
        date_ts = pd.Timestamp(date_str)

        compx_slice = compx_df[compx_df.index <= date_ts]
        if len(compx_slice) < 50:
            skipped += 1
            continue

        spy_slice = spy_df[spy_df.index <= date_ts] if spy_df is not None else None
        iwm_slice = iwm_df[iwm_df.index <= date_ts] if iwm_df is not None else None

        with sqlite3.connect(db_path) as conn:
            stocks = conn.execute("""
                SELECT symbol, price, stage2, pct_from_52wk_high
                FROM scans
                WHERE scan_date = ? AND passes_filters = 1
            """, (date_str,)).fetchall()

        # build minimal 1-row DataFrames per stock for the breadth scorer;
        # all records here passed hard filters so close > sma_50 by definition
        stock_dfs = {}
        for sym, price, stage2, pct_high in stocks:
            if price is None:
                continue
            stock_dfs[sym] = pd.DataFrame([{
                "close":               price,
                "sma_50":              price * 0.99,
                "sma_200":             0,
                "stage2":              bool(stage2),
                "pct_from_52wk_high":  pct_high if pct_high is not None else -1.0,
            }])

        try:
            result = analyzer.analyze(
                compx_df=compx_slice,
                spy_df=spy_slice,
                iwm_df=iwm_slice,
                stock_feature_dfs=stock_dfs or None,
            )
            db._save_market_condition(date_str, result)
            saved += 1
        except Exception:
            skipped += 1
            continue

        if i % 100 == 0 or i == len(dates):
            print(f"  [{i}/{len(dates)}] saved={saved} skipped={skipped}", end="\r", flush=True)

    print(f"\n  done: {saved} market condition records saved, {skipped} skipped")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="historical batch pipeline")
    parser.add_argument(
        "--start",
        default=(datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d"),
        help="scan start date yyyy-mm-dd (default: 7 years ago)",
    )
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="scan end date yyyy-mm-dd (default: today)",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel download workers (default: 4)")
    parser.add_argument("--delay", type=float, default=0.55, help="min seconds between api calls (default: 0.55)")
    parser.add_argument("--symbols", nargs="+", metavar="TICKER", help="specific symbols instead of full nasdaq list")
    parser.add_argument("--skip-download", action="store_true", help="skip phase 1")
    parser.add_argument("--skip-scoring", action="store_true", help="skip phase 2")
    parser.add_argument("--skip-rs-calibration", action="store_true", help="skip phase 3")
    parser.add_argument("--skip-market-conditions", action="store_true", help="skip phase 4 (market conditions backfill)")
    parser.add_argument("--force-download", action="store_true", help="ignore .progress and re-download all")
    parser.add_argument(
        "--force-rescore", action="store_true",
        help=(
            "re-score all dates even if already in DB. use this when upgrading from a DB "
            "that only has passes_filters=1 records and you want to backfill the failing "
            "records for filter analysis. slower — clears and rewrites all scan/outcome rows."
        ),
    )
    args = parser.parse_args()

    config = PARAMETERS.copy()

    print("historical batch pipeline")
    print(f"  range:   {args.start} → {args.end}")

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        print(f"  symbols: {len(symbols)} user-specified")
    else:
        symbols = get_nasdaq_symbols()

    scan_start = pd.Timestamp(args.start)
    scan_end = pd.Timestamp(args.end)

    # 400-day buffer before scan start covers sma_200 warmup
    dl_start = datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=400)
    dl_end = datetime.strptime(args.end, "%Y-%m-%d")
    start_ts = int(dl_start.timestamp() * 1000)
    end_ts = int(dl_end.timestamp() * 1000)

    # ensure db schema is initialized
    ScanPersistence(DB_PATH)

    if not args.skip_download:
        phase1_download(symbols, start_ts, end_ts, args.workers, args.delay, args.force_download)

    if not args.skip_scoring:
        phase2_score(symbols, scan_start, scan_end, DB_PATH, config,
                     force_rescore=args.force_rescore)

    if not args.skip_rs_calibration:
        phase3_rs_calibration(
            DB_PATH,
            config.get("min_score_alert", 80),
            config.get("min_score_watchlist", 70),
        )

    if not args.skip_market_conditions:
        phase4_backfill_market_conditions(DB_PATH, config)

    ScanPersistence(DB_PATH).summary()
    print("done.")


if __name__ == "__main__":
    main()
