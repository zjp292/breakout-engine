"""
Scan result persistence for backtesting and continuous improvement.

Three SQLite tables:
  scans             — snapshot of every processed stock at scan time
                      (ALL stocks, not just those that passed filters)
  outcomes          — what happened after each scan; populated by the
                      outcome tracker once enough time has elapsed
  market_conditions — daily market regime snapshot from MarketConditionResult

Usage:
    from persistence import ScanPersistence

    db = ScanPersistence()
    db.save_scan(date_str, scored_dfs, engine.market_condition)
    db.summary()
"""

import math
import sqlite3
from pathlib import Path

import pandas as pd


class ScanPersistence:
    def __init__(self, db_path: str = "results/breakout.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date               TEXT    NOT NULL,
                    symbol                  TEXT    NOT NULL,
                    passes_filters          INTEGER NOT NULL DEFAULT 0,

                    -- Scoring (NULL for stocks that failed hard filters)
                    score                   REAL,
                    raw_score               REAL,
                    grade                   TEXT,
                    signal                  TEXT,
                    base_quality            REAL,
                    trend_strength          REAL,
                    relative_strength_score REAL,
                    volume_score            REAL,
                    rr_score                REAL,

                    -- Price levels — snapshot at scan time
                    price                   REAL,
                    breakout_level          REAL,
                    stop_level              REAL,
                    target_level            REAL,
                    stop_distance_pct       REAL,
                    potential_r             REAL,

                    -- Base structure
                    consol_days             INTEGER,
                    consol_range_pct        REAL,
                    vcp_contracting         INTEGER,
                    vcp_contraction_ratio   REAL,

                    -- Trend & RS
                    stage2                  INTEGER,
                    pct_from_52wk_high      REAL,
                    ma_alignment            INTEGER,
                    prior_move_pct          REAL,
                    rs_comp_20              REAL,
                    rs_comp_60              REAL,
                    rs_comp_120             REAL,

                    -- Volume / liquidity
                    dollar_volume           REAL,
                    adr_pct                 REAL,
                    volume_dryup_ratio      REAL,

                    -- Market context embedded per row
                    market_regime           TEXT,
                    market_score            REAL,
                    regime_multiplier       REAL,

                    UNIQUE(scan_date, symbol)
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date           TEXT    NOT NULL,
                    symbol              TEXT    NOT NULL,
                    outcome_date        TEXT    NOT NULL,
                    days_elapsed        INTEGER,
                    entry_price         REAL,
                    current_price       REAL,
                    pct_change          REAL,
                    breakout_triggered  INTEGER DEFAULT 0,
                    stop_triggered      INTEGER DEFAULT 0,
                    target_reached      INTEGER DEFAULT 0,
                    days_to_breakout    INTEGER,
                    days_to_stop        INTEGER,
                    max_gain_10d        REAL,
                    max_gain_20d        REAL,
                    max_gain_60d        REAL,
                    max_drawdown_20d    REAL,
                    UNIQUE(scan_date, symbol, outcome_date)
                );

                CREATE TABLE IF NOT EXISTS market_conditions (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date               TEXT    NOT NULL UNIQUE,
                    regime                  TEXT,
                    score                   REAL,
                    regime_multiplier       REAL,
                    index_trend_score       REAL,
                    distribution_score      REAL,
                    follow_through_score    REAL,
                    breadth_score           REAL,
                    momentum_score          REAL,
                    distribution_day_count  INTEGER,
                    stalling_day_count      INTEGER,
                    pct_above_50sma         REAL,
                    pct_in_stage2           REAL,
                    pct_near_52wk_high      REAL,
                    ftd_found               INTEGER,
                    ftd_valid               INTEGER,
                    ftd_date                TEXT,
                    ftd_days_ago            INTEGER,
                    spy_above_200           INTEGER,
                    iwm_above_200           INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_scans_date     ON scans(scan_date);
                CREATE INDEX IF NOT EXISTS idx_scans_symbol   ON scans(symbol);
                CREATE INDEX IF NOT EXISTS idx_scans_score    ON scans(score DESC);
                CREATE INDEX IF NOT EXISTS idx_scans_filtered ON scans(passes_filters);
                CREATE INDEX IF NOT EXISTS idx_outcomes_scan  ON outcomes(scan_date, symbol);
            """)

        # Schema migration: add raw_score to existing databases that predate this column
        with self._connect() as conn:
            try:
                conn.execute("ALTER TABLE scans ADD COLUMN raw_score REAL")
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── Write ────────────────────────────────────────────────────────────────

    def save_scan(
        self,
        date_str: str,
        scored_dfs: dict,
        market_condition=None,
    ) -> int:
        """
        Persist every processed stock for a scan date, plus the market regime.

        Saves ALL stocks regardless of whether they passed hard filters.
        The passes_filters column lets you analyse filter effectiveness later.

        Args:
            date_str:         Scan date as 'YYYY-MM-DD'.
            scored_dfs:       Dict of {symbol: scored_dataframe} from Engine.
            market_condition: MarketConditionResult from Engine (optional).

        Returns:
            Number of stock records upserted.
        """
        if market_condition is not None:
            self._save_market_condition(date_str, market_condition)

        mc_regime     = getattr(market_condition, "regime",            None)
        mc_score      = getattr(market_condition, "score",             None)
        mc_multiplier = getattr(market_condition, "regime_multiplier", None)

        records = []
        for symbol, df in scored_dfs.items():
            if df.empty:
                continue

            row    = df.iloc[-1]
            passes = bool(row.get("passes_filters", False))

            # Score columns are 0 for filtered-out stocks; store as NULL instead.
            def s(col):
                return self._safe_float(row, col) if passes else None

            # consol_range column name varies with config-driven lookback window.
            # Use explicit None check — 0.0 (extremely tight base) must not be treated as falsy.
            consol_range = self._safe_float(row, "consol_range_60")
            if consol_range is None:
                consol_range = self._safe_float(row, "consol_range_15")

            records.append({
                "scan_date":               date_str,
                "symbol":                  symbol,
                "passes_filters":          int(passes),
                "score":                   s("total_score"),
                "raw_score":               s("raw_score"),
                "grade":                   row.get("grade")  if passes else None,
                "signal":                  row.get("signal") if passes else None,
                "base_quality":            s("score_base_quality"),
                "trend_strength":          s("score_trend_strength"),
                "relative_strength_score": s("score_relative_strength"),
                "volume_score":            s("score_volume_profile"),
                "rr_score":                s("score_risk_reward"),
                # Price levels — always stored regardless of filter outcome
                "price":                   self._safe_float(row, "close"),
                "breakout_level":          self._safe_float(row, "breakout_level"),
                "stop_level":              self._safe_float(row, "stop_level"),
                "target_level":            self._safe_float(row, "target_level"),
                "stop_distance_pct":       self._safe_float(row, "stop_distance_pct"),
                "potential_r":             self._safe_float(row, "potential_r"),
                # Base structure
                "consol_days":             self._safe_int(row,  "consol_days"),
                "consol_range_pct":        consol_range,
                "vcp_contracting":         self._safe_bool(row, "vcp_contracting"),
                "vcp_contraction_ratio":   self._safe_float(row, "vcp_contraction_ratio"),
                # Trend & RS
                "stage2":                  self._safe_bool(row,  "stage2"),
                "pct_from_52wk_high":      self._safe_float(row, "pct_from_52wk_high"),
                "ma_alignment":            self._safe_bool(row,  "ma_alignment"),
                "prior_move_pct":          self._safe_float(row, "prior_move_pct"),
                "rs_comp_20":              self._safe_float(row, "rs_comp_20"),
                "rs_comp_60":              self._safe_float(row, "rs_comp_60"),
                "rs_comp_120":             self._safe_float(row, "rs_comp_120"),
                # Volume
                "dollar_volume":           self._safe_float(row, "dollar_volume"),
                "adr_pct":                 self._safe_float(row, "adr_pct"),
                "volume_dryup_ratio":      self._safe_float(row, "volume_dryup_ratio"),
                # Market context
                "market_regime":           mc_regime,
                "market_score":            mc_score,
                "regime_multiplier":       mc_multiplier,
            })

        if not records:
            return 0

        with self._connect() as conn:
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
            """, records)

        return len(records)

    def _save_market_condition(self, date_str: str, mc) -> None:
        """Upsert a MarketConditionResult into the market_conditions table."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO market_conditions (
                    scan_date, regime, score, regime_multiplier,
                    index_trend_score, distribution_score, follow_through_score,
                    breadth_score, momentum_score,
                    distribution_day_count, stalling_day_count,
                    pct_above_50sma, pct_in_stage2, pct_near_52wk_high,
                    ftd_found, ftd_valid, ftd_date, ftd_days_ago,
                    spy_above_200, iwm_above_200
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                mc.regime,
                mc.score,
                mc.regime_multiplier,
                mc.index_trend_score,
                mc.distribution_score,
                mc.follow_through_score,
                mc.breadth_score,
                mc.momentum_score,
                mc.distribution_day_count,
                mc.stalling_day_count,
                mc.pct_above_50sma,
                mc.pct_in_stage2,
                mc.pct_near_52wk_high,
                int(mc.ftd_found),
                int(mc.ftd_valid) if mc.ftd_valid is not None else None,
                mc.ftd_date,
                mc.ftd_days_ago,
                int(mc.spy_above_200) if mc.spy_above_200 is not None else None,
                int(mc.iwm_above_200) if mc.iwm_above_200 is not None else None,
            ))

    def save_outcomes(self, outcomes: list) -> int:
        """
        Upsert outcome records produced by the outcome tracker.

        Each dict must contain at minimum:
            scan_date, symbol, outcome_date, days_elapsed,
            entry_price, current_price

        Returns: number of records upserted.
        """
        if not outcomes:
            return 0
        with self._connect() as conn:
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
            """, outcomes)
        return len(outcomes)

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_scans(
        self,
        from_date: str = None,
        to_date: str = None,
        passed_only: bool = False,
        min_score: float = None,
    ) -> pd.DataFrame:
        """
        Load historical scan records as a DataFrame.

        Args:
            from_date:   Inclusive start date 'YYYY-MM-DD'.
            to_date:     Inclusive end date 'YYYY-MM-DD'.
            passed_only: If True, only return stocks that passed hard filters.
            min_score:   If set, only return stocks with score >= min_score.
        """
        clauses, params = [], []
        if from_date:
            clauses.append("scan_date >= ?"); params.append(from_date)
        if to_date:
            clauses.append("scan_date <= ?"); params.append(to_date)
        if passed_only:
            clauses.append("passes_filters = 1")
        if min_score is not None:
            clauses.append("score >= ?"); params.append(min_score)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM scans {where} ORDER BY scan_date DESC, score DESC",
                conn, params=params,
            )

    def load_market_conditions(
        self, from_date: str = None, to_date: str = None
    ) -> pd.DataFrame:
        """Load historical market condition records as a DataFrame."""
        clauses, params = [], []
        if from_date:
            clauses.append("scan_date >= ?"); params.append(from_date)
        if to_date:
            clauses.append("scan_date <= ?"); params.append(to_date)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM market_conditions {where} ORDER BY scan_date DESC",
                conn, params=params,
            )

    def load_outcomes(
        self, from_date: str = None, to_date: str = None
    ) -> pd.DataFrame:
        """Load outcome records as a DataFrame."""
        clauses, params = [], []
        if from_date:
            clauses.append("scan_date >= ?"); params.append(from_date)
        if to_date:
            clauses.append("scan_date <= ?"); params.append(to_date)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM outcomes {where} ORDER BY scan_date DESC",
                conn, params=params,
            )

    def get_pending_outcomes(self, min_days_old: int = 10) -> pd.DataFrame:
        """
        Returns passed-filter scans that have no outcomes recorded yet and are
        at least min_days_old calendar days old.

        Used by the outcome tracker to determine what to evaluate next.
        """
        with self._connect() as conn:
            return pd.read_sql_query("""
                SELECT s.*
                FROM scans s
                LEFT JOIN outcomes o
                    ON s.scan_date = o.scan_date AND s.symbol = o.symbol
                WHERE s.passes_filters = 1
                  AND o.id IS NULL
                  AND julianday('now') - julianday(s.scan_date) >= ?
                ORDER BY s.scan_date ASC, s.score DESC
            """, conn, params=[min_days_old])

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a formatted database summary to stdout."""
        with self._connect() as conn:
            dates    = conn.execute(
                "SELECT DISTINCT scan_date FROM scans ORDER BY scan_date"
            ).fetchall()
            total    = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            passed   = conn.execute(
                "SELECT COUNT(*) FROM scans WHERE passes_filters = 1"
            ).fetchone()[0]
            outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            mc_rows  = conn.execute(
                "SELECT COUNT(*) FROM market_conditions"
            ).fetchone()[0]

        W = 54
        print(f"\n{'═' * W}")
        print(f"  SCAN DATABASE  {self.db_path}")
        print(f"{'═' * W}")
        print(f"  Scan dates       {len(dates)}")
        print(f"  Stocks scanned   {total}")
        print(f"  Passed filters   {passed}")
        print(f"  Outcomes logged  {outcomes}")
        print(f"  Market records   {mc_rows}")
        if dates:
            print(f"  Date range       {dates[0][0]}  →  {dates[-1][0]}")
        print(f"{'═' * W}\n")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(row, col: str):
        val = row.get(col)
        if val is None:
            return None
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(row, col: str):
        val = row.get(col)
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else int(f)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_bool(row, col: str):
        val = row.get(col)
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else int(bool(f))
        except (TypeError, ValueError):
            return None
