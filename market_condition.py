"""
Market Condition Analyzer for Swing Trading

Quantifies whether the broad market environment is suitable for swing trading.

Based on methods from:

  - Kristjan Qullamaggie  (qullamaggie.com / SMB Summit talks)
      "In a bear market or choppy market I will raise cash and not fight the tape.
       After a correction, I don't start buying aggressively until there is a
       confirmed follow-through day.  In the best markets you press hard; in bad
       markets you stay out."

  - William O'Neil / IBD CANSLIM
      The 'M' (Market Direction) in CANSLIM is the #1 factor.  O'Neil introduced
      the Follow-Through Day (FTD) and Distribution Day counting methodology.
      Reference: "How to Make Money in Stocks", O'Neil (4th ed.)

  - Trader Lion / Richard Moglen
      Internal breadth — % of leading stocks in Stage 2, above key MAs, near
      52-week highs — is a powerful leading indicator of market health.

  - Academic backing
      · Breadth-return predictability: Zweig (1986), Pring (2002)
      · Realized volatility forecasting: Andersen et al. (2003)
      · Momentum in equity indexes: Jegadeesh & Titman (1993)
      · High-vol environments and trend-following: Ang et al. (2006)

Score (0–100) → Regime:
  70–100  BULL       — Optimal.  Trade aggressively, full sizing.
  55– 69  UPTREND    — Good.     Trade normally.
  40– 54  MIXED      — Selective. Smaller positions, tighter criteria.
  25– 39  CAUTION    — Very selective.  Reduce exposure significantly.
   0– 24  DOWNTREND  — Avoid new longs.  Cash is a position.

Component weights (total = 100 pts):
  Index Trend       25   SMA alignment + 50d slope + SPY/IWM confirmation
  Distribution Days 20   D-day count over rolling 25-session window
  Follow-Through    20   Recency and validity of most recent FTD
  Internal Breadth  20   Watchlist stocks above 50 SMA / Stage 2 / near highs
  Momentum / Vol    15   21-day ROC + realized volatility (VIX proxy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MarketConditionResult:
    """Full result from the market condition analyzer."""

    # ── Overall ───────────────────────────────────────────────────────────────
    score: float
    regime: str               # BULL / UPTREND / MIXED / CAUTION / DOWNTREND
    regime_multiplier: float  # Applied as multiplier to individual stock scores

    # ── Component scores (sum to 100) ─────────────────────────────────────────
    index_trend_score: float      # 0–25
    distribution_score: float     # 0–20
    follow_through_score: float   # 0–20
    breadth_score: float          # 0–20
    momentum_score: float         # 0–15

    # ── Distribution day details ──────────────────────────────────────────────
    distribution_day_count: int
    stalling_day_count: int

    # ── Follow-through day details ────────────────────────────────────────────
    ftd_found: bool
    ftd_date: Optional[str]
    ftd_days_ago: Optional[int]
    ftd_valid: bool

    # ── Breadth stats (fractions 0–1) ─────────────────────────────────────────
    pct_above_50sma: float
    pct_above_200sma: float
    pct_in_stage2: float
    pct_near_52wk_high: float

    # ── Momentum & volatility ─────────────────────────────────────────────────
    compx_roc_21d: float             # 21-day rate of change of COMPX close
    realized_vol_annualized: float   # 20-day realized vol, annualized (VIX proxy)

    # ── Multi-index state ─────────────────────────────────────────────────────
    compx_sma_alignment: float   # fraction of SMA conditions that are met (0–1)
    spy_above_200: bool
    iwm_above_200: bool

    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────


class MarketConditionAnalyzer:
    """
    Multi-factor market condition scoring for swing trading.

    Score = Index Trend (25)
          + Distribution Days (20)
          + Follow-Through Day (20)
          + Internal Breadth (20)
          + Momentum / Volatility (15)
    """

    # (name, min_score, max_score_inclusive, stock-score multiplier)
    # expanded BULL zone to 70+ (was 80+): the 65-79 band was too wide and contained
    # mixed-outcome sessions that are genuinely BULL-like. shrinking UPTREND to 55-69
    # makes regime boundaries more discriminating (section 6.3, 2026-05).
    REGIMES = [
        ("BULL",      70, 100, 1.00),
        ("UPTREND",   55,  69, 0.95),
        ("MIXED",     40,  54, 0.85),
        ("CAUTION",   25,  39, 0.70),
        ("DOWNTREND",  0,  24, 0.50),
    ]

    _SMA_PERIODS = [10, 20, 50, 150, 200]

    def __init__(self, config: dict):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        compx_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
        iwm_df: Optional[pd.DataFrame] = None,
        stock_feature_dfs: Optional[dict] = None,
    ) -> MarketConditionResult:
        """
        Run the full market condition analysis.

        Args:
            compx_df:          NASDAQ Composite daily OHLCV (primary benchmark).
                               Accepts either a DatetimeIndex-based DF or one
                               with a 'datetime' ms-epoch column.
            spy_df:            S&P 500 ETF daily OHLCV (multi-index confirmation).
            iwm_df:            Russell 2000 ETF daily OHLCV (small-cap breadth).
            stock_feature_dfs: {symbol: feature_df} from the watchlist — used for
                               internal breadth calculation.
        """
        compx = self._prep_df(compx_df)
        spy   = self._prep_df(spy_df)   if spy_df   is not None else None
        iwm   = self._prep_df(iwm_df)   if iwm_df   is not None else None

        compx = self._add_smas(compx)
        if spy is not None:
            spy = self._add_smas(spy)
        if iwm is not None:
            iwm = self._add_smas(iwm)

        idx_score,  idx_det  = self._score_index_trend(compx, spy, iwm)
        dist_score, dist_det = self._score_distribution_days(compx)
        ftd_score,  ftd_det  = self._score_follow_through(compx)
        brd_score,  brd_det  = self._score_internal_breadth(stock_feature_dfs)
        mom_score,  mom_det  = self._score_momentum_and_volatility(compx)

        total = idx_score + dist_score + ftd_score + brd_score + mom_score
        regime, multiplier = self._classify_regime(total)

        spy_above_200 = False
        iwm_above_200 = False
        if spy is not None and len(spy) >= 200:
            last = spy.iloc[-1]
            spy_above_200 = bool(last["close"] > last.get("SMA_200", 0))
        if iwm is not None and len(iwm) >= 200:
            last = iwm.iloc[-1]
            iwm_above_200 = bool(last["close"] > last.get("SMA_200", 0))

        return MarketConditionResult(
            score=round(total, 1),
            regime=regime,
            regime_multiplier=multiplier,
            index_trend_score=round(idx_score, 1),
            distribution_score=round(dist_score, 1),
            follow_through_score=round(ftd_score, 1),
            breadth_score=round(brd_score, 1),
            momentum_score=round(mom_score, 1),
            distribution_day_count=dist_det.get("dist_days", 0),
            stalling_day_count=dist_det.get("stalling_days", 0),
            ftd_found=ftd_det.get("ftd_found", False),
            ftd_date=(
                str(ftd_det["ftd_date"]) if ftd_det.get("ftd_date") else None
            ),
            ftd_days_ago=ftd_det.get("ftd_days_ago"),
            ftd_valid=ftd_det.get("ftd_valid", False),
            pct_above_50sma=brd_det.get("pct_above_50sma", 0.0),
            pct_above_200sma=brd_det.get("pct_above_200sma", 0.0),
            pct_in_stage2=brd_det.get("pct_in_stage2", 0.0),
            pct_near_52wk_high=brd_det.get("pct_near_52wk_high", 0.0),
            compx_roc_21d=mom_det.get("roc_21d", 0.0),
            realized_vol_annualized=mom_det.get("realized_vol", 0.0),
            compx_sma_alignment=idx_det.get("compx_alignment", 0.0),
            spy_above_200=spy_above_200,
            iwm_above_200=iwm_above_200,
            details={
                "index":          idx_det,
                "distribution":   dist_det,
                "follow_through": ftd_det,
                "breadth":        brd_det,
                "momentum":       mom_det,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Component 1: Index Trend Quality  (0–25 pts)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_index_trend(self, compx, spy, iwm) -> tuple[float, dict]:
        """
        Quality of the primary index's trend structure.

        Points breakdown (25 total):
          COMPX SMA alignment   0–15  (6 conditions × 2.5 pts)
          50-SMA slope          0– 4  (is the 50d SMA rising?)
          Multi-index confirm   0– 6  (SPY above 200 SMA +3, IWM above 200 SMA +3)

        The six SMA conditions for COMPX:
          price > SMA_10, price > SMA_20, price > SMA_50,
          price > SMA_150, price > SMA_200, SMA_200 is rising (vs 20 days ago)

        If price is below the 200 SMA, the alignment score is hard-capped at 5 pts
        because no legitimate uptrend exists below the long-term MA.

        Qullamaggie: "If the general market is in a downtrend, 95% of stocks will
        follow.  I want to see indices making new highs, not new lows."
        """
        details = {}
        score   = 0.0

        last  = compx.iloc[-1]
        close = last["close"]

        # ── SMA alignment (0–15) ───────────────────────────────────────────────
        conditions = [
            ("price_above_10",  close > last.get("SMA_10",  0)),
            ("price_above_20",  close > last.get("SMA_20",  0)),
            ("price_above_50",  close > last.get("SMA_50",  0)),
            ("price_above_150", close > last.get("SMA_150", 0)),
            ("price_above_200", close > last.get("SMA_200", 0)),
            ("sma200_rising",   self._is_sma_rising(compx, "SMA_200", 20)),
        ]

        hits            = sum(1 for _, v in conditions if v)
        alignment_score = (hits / len(conditions)) * 15.0
        above_200       = conditions[4][1]   # price_above_200

        # Hard cap when below the 200 SMA — the market is in a downtrend
        if not above_200:
            alignment_score = min(alignment_score, 5.0)

        details["compx_alignment"]  = hits / len(conditions)
        details["sma_conditions"]   = {n: v for n, v in conditions}
        score += alignment_score

        # ── 50-SMA slope (0–4) ────────────────────────────────────────────────
        slope = self._sma_slope_pct(compx, "SMA_50", 10)
        if   slope >  0.015: slope_score = 4.0
        elif slope >  0.005: slope_score = 2.5
        elif slope >  0.000: slope_score = 1.0
        else:                slope_score = 0.0

        details["sma50_slope_pct"] = round(slope, 4)
        score += slope_score

        # ── Multi-index confirmation (0–6) ────────────────────────────────────
        # SPY (large caps, +3) and IWM (small caps, +3).
        # If data is unavailable, award 1.5 partial credit so the system degrades
        # gracefully rather than artificially penalising the COMPX-only case.
        multi = 0.0

        if spy is not None and len(spy) >= 200:
            spy_last = spy.iloc[-1]
            spy_ok   = spy_last["close"] > spy_last.get("SMA_200", 0)
            multi   += 3.0 if spy_ok else 0.0
            details["spy_above_200"] = spy_ok
        else:
            multi += 1.5
            details["spy_above_200"] = None

        if iwm is not None and len(iwm) >= 200:
            iwm_last = iwm.iloc[-1]
            iwm_ok   = iwm_last["close"] > iwm_last.get("SMA_200", 0)
            multi   += 3.0 if iwm_ok else 0.0
            details["iwm_above_200"] = iwm_ok
        else:
            multi += 1.5
            details["iwm_above_200"] = None

        score += multi
        details["multi_index_score"] = multi

        return min(score, 25.0), details

    # ─────────────────────────────────────────────────────────────────────────
    # Component 2: Distribution Day Count  (0–20 pts)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_distribution_days(
        self, compx: pd.DataFrame, lookback: int = 25
    ) -> tuple[float, dict]:
        """
        Count distribution and stalling days in the last 25 sessions.

        Distribution day (O'Neil / IBD):
          - Index declines ≥0.2% on higher volume than the previous session.
          - Reflects institutional ("smart money") selling into the market.
          - Counted over a rolling 25-session window; old days age off naturally.

        Stalling day (O'Neil variant):
          - Index closes modestly higher (<0.2%) on meaningfully higher volume.
          - Signals buying exhaustion — institutions are no longer absorbing supply.

        Scoring: Start at 20 pts.
          -4 per distribution day, -2 per stalling day.
          5+ distribution days → 0 pts (market under heavy distribution — go flat).

        Qullamaggie: "When you see 5–6 distribution days over 3–4 weeks it's time
        to stop buying and start reducing positions."
        """
        if len(compx) < lookback + 1:
            return 15.0, {"note": "insufficient_data"}

        recent               = compx.tail(lookback + 1).copy()
        recent["pct_chg"]    = recent["close"].pct_change()
        recent["vol_pct_chg"] = recent["volume"].pct_change()

        dist_days     = 0
        stalling_days = 0

        for i in range(1, len(recent)):
            pct      = recent["pct_chg"].iloc[i]
            vol      = recent["volume"].iloc[i]
            prev_vol = recent["volume"].iloc[i - 1]

            if pd.isna(pct) or pd.isna(vol) or pd.isna(prev_vol):
                continue

            # Distribution: down ≥0.2% on higher volume
            if pct <= -0.002 and vol > prev_vol:
                dist_days += 1
            # Stalling: up marginally (<0.2%) on ≥5% higher volume
            elif 0 < pct < 0.002 and vol > prev_vol * 1.05:
                stalling_days += 1

        raw   = 20.0 - (dist_days * 4.0) - (stalling_days * 2.0)
        score = max(0.0, raw)
        if dist_days >= 5:
            score = 0.0   # Hard zero: market is under active distribution

        return score, {
            "dist_days":     dist_days,
            "stalling_days": stalling_days,
            "lookback":      lookback,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Component 3: Follow-Through Day  (0–20 pts)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_follow_through(self, compx: pd.DataFrame) -> tuple[float, dict]:
        """
        Detect and score the most recent Follow-Through Day (FTD).

        O'Neil / Qullamaggie FTD methodology:
          1. Market pulls back from a high (correction begins).
          2. Any day that closes higher begins a "rally attempt" (Day 1).
          3. Starting on Day 4 from the rally attempt:
               if index gains ≥1.25% on higher volume → Follow-Through Day.
          4. FTD is INVALIDATED if the index subsequently makes a new low
             below the correction low that triggered the rally attempt.

        No FTD found — market near all-time high (within 5%): 12 pts
          → Ongoing uptrend; no meaningful correction occurred.
        No FTD found — market below highs:                      2 pts
          → Correction underway; waiting for confirmation.
        FTD found, valid, ≤10 sessions ago:                    20 pts
        FTD found, valid, 11–25 sessions ago:               10–18 pts (linear decay)
        FTD found, valid, 26–40 sessions ago:                5–10 pts (aging)
        FTD found, valid, >40 sessions ago:                     5 pts
        FTD found but INVALIDATED:                              0 pts

        Qullamaggie: "After a market correction I don't start buying aggressively
        until there is a confirmed follow-through day."
        """
        ftd = self._detect_follow_through(compx)

        details = {
            "ftd_found":    ftd["ftd_found"],
            "ftd_date":     ftd.get("ftd_date"),
            "ftd_days_ago": ftd.get("ftd_days_ago"),
            "ftd_valid":    ftd.get("ftd_valid", False),
            "ftd_gain":     ftd.get("ftd_gain"),
        }

        if not ftd["ftd_found"]:
            close    = compx["close"].iloc[-1]
            high_252 = compx["high"].rolling(252, min_periods=50).max().iloc[-1]
            pct_from_high = (close - high_252) / high_252 if high_252 and high_252 > 0 else -1.0
            details["pct_from_high"] = round(pct_from_high, 4)

            if   pct_from_high >= -0.05: score = 12.0  # near all-time high — healthy
            elif pct_from_high >= -0.10: score = 8.0
            else:                        score = 2.0   # off highs, no FTD yet

        elif not ftd.get("ftd_valid", False):
            score = 0.0  # FTD was invalidated — danger sign

        else:
            days_ago = ftd.get("ftd_days_ago", 100)
            if   days_ago <= 10:  score = 20.0
            elif days_ago <= 25:  score = 20.0 - ((days_ago - 10) / 15.0) * 10.0
            elif days_ago <= 40:  score = 10.0 - ((days_ago - 25) / 15.0) * 5.0
            else:                 score = 5.0

        return min(score, 20.0), details

    # ─────────────────────────────────────────────────────────────────────────
    # Component 4: Internal Breadth  (0–20 pts)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_internal_breadth(
        self, stock_dfs: Optional[dict]
    ) -> tuple[float, dict]:
        """
        Health of the market measured through watchlist stock behavior.

        The watchlist is pre-screened for momentum, trend, and fundamentals.
        If even these leading stocks can't hold their key moving averages, the
        market is not healthy enough to support swing trading entries.

        Trader Lion / Richard Moglen focuses on:
          - % of leading stocks above 50-day MA (institutional support line)
          - % of stocks in Stage 2 (Minervini's primary uptrend template)
          - % of stocks within 15% of their 52-week high (near breakout territory)

        Points:
          % above 50 SMA        0–7
          % in Stage 2          0–7
          % near 52-week high   0–6
        """
        if not stock_dfs or len(stock_dfs) < 5:
            return 10.0, {"note": "insufficient_data", "n_stocks": len(stock_dfs or {})}

        n = n_above_50 = n_above_200 = n_stage2 = n_near_high = 0

        for symbol, df in stock_dfs.items():
            if df is None or len(df) == 0:
                continue
            last  = df.iloc[-1]
            close = last.get("close", 0)
            n += 1

            # engine.py uses lowercase column names (sma_50, sma_200, ...)
            sma50  = last.get("sma_50",  last.get("SMA_50",  0))
            sma200 = last.get("sma_200", last.get("SMA_200", 0))

            if sma50  and close > sma50:  n_above_50  += 1
            if sma200 and close > sma200: n_above_200 += 1
            if last.get("stage2", False): n_stage2    += 1

            pct = last.get("pct_from_52wk_high", -1.0)
            if pct is not None and not pd.isna(pct) and pct >= -0.15:
                n_near_high += 1

        if n == 0:
            return 10.0, {"note": "no_valid_stocks"}

        pct_50  = n_above_50  / n
        pct_200 = n_above_200 / n
        pct_s2  = n_stage2    / n
        pct_hi  = n_near_high / n

        # % above 50 SMA (0–7)
        if   pct_50 >= 0.70: above50_score = 7.0
        elif pct_50 >= 0.55: above50_score = 5.5
        elif pct_50 >= 0.40: above50_score = 3.5
        elif pct_50 >= 0.25: above50_score = 1.5
        else:                above50_score = 0.0

        # % in Stage 2 (0–7)
        if   pct_s2 >= 0.60: stage2_score = 7.0
        elif pct_s2 >= 0.45: stage2_score = 5.0
        elif pct_s2 >= 0.30: stage2_score = 3.0
        elif pct_s2 >= 0.15: stage2_score = 1.0
        else:                stage2_score = 0.0

        # % near 52-week high (0–6)
        if   pct_hi >= 0.50: near_high_score = 6.0
        elif pct_hi >= 0.35: near_high_score = 4.0
        elif pct_hi >= 0.20: near_high_score = 2.0
        elif pct_hi >= 0.10: near_high_score = 1.0
        else:                near_high_score = 0.0

        score = above50_score + stage2_score + near_high_score

        return score, {
            "n_stocks":            n,
            "pct_above_50sma":     round(pct_50,  3),
            "pct_above_200sma":    round(pct_200, 3),
            "pct_in_stage2":       round(pct_s2,  3),
            "pct_near_52wk_high":  round(pct_hi,  3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Component 5: Momentum & Volatility  (0–15 pts)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_momentum_and_volatility(
        self, compx: pd.DataFrame
    ) -> tuple[float, dict]:
        """
        Rate market momentum and the volatility environment.

        High volatility is the enemy of swing trading:
          - Wide intraday swings make stop placement unreliable.
          - Institutions reduce activity in high-vol environments.
          - Breakouts frequently fail and whipsaw in volatile markets.

        Points:
          21-day ROC           0–8   (is the market going up?)
          Realized volatility  0–7   (is the environment calm?)

        The 20-day realized historical volatility (annualized) serves as a VIX
        proxy when VIX data is unavailable.  Typical VIX ranges:
          <15 → calm/complacent  (~12–15% realized)
          15–20 → normal         (~15–20% realized)
          20–30 → elevated fear  (~20–30% realized)
          >30 → crisis           (>30% realized)
        """
        details = {}

        # 21-day Rate of Change (0–8)
        roc21 = self._rate_of_change(compx, 21)
        details["roc_21d"] = round(roc21, 4)

        if   roc21 >= 0.08:  roc_score = 8.0
        elif roc21 >= 0.04:  roc_score = 6.0
        elif roc21 >= 0.01:  roc_score = 4.0
        elif roc21 >= -0.01: roc_score = 2.0
        elif roc21 >= -0.05: roc_score = 1.0
        else:                roc_score = 0.0

        # 20-day Realized Volatility, annualized (0–7)
        rv = self._realized_volatility(compx, 20)
        details["realized_vol"] = round(rv, 4)

        if   rv < 0.12:  vol_score = 7.0   # <12%  — extremely calm
        elif rv < 0.15:  vol_score = 6.0   # 12–15%
        elif rv < 0.20:  vol_score = 4.5   # 15–20% — mildly elevated
        elif rv < 0.25:  vol_score = 2.5   # 20–25% — elevated
        elif rv < 0.35:  vol_score = 1.0   # 25–35% — high fear
        else:            vol_score = 0.0   # >35%  — extreme fear / crisis

        score = roc_score + vol_score
        details["roc_score"] = roc_score
        details["vol_score"]  = vol_score

        return min(score, 15.0), details

    # ─────────────────────────────────────────────────────────────────────────
    # Regime classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_regime(self, score: float) -> tuple[str, float]:
        # REGIMES ordered high→low; first matching lower-bound wins
        for name, lo, _hi, mult in self.REGIMES:
            if score >= lo:
                return name, mult
        return "DOWNTREND", 0.50

    # ─────────────────────────────────────────────────────────────────────────
    # Follow-Through Day detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_follow_through(
        self, df: pd.DataFrame, lookback: int = 80
    ) -> dict:
        """
        Scan the last N sessions for the most recent valid Follow-Through Day.

        Algorithm (forward scan, then backwards validity check):

        Phase 1 — forward scan, collect FTD candidates:
          Walk forward through the data tracking market state:
          'uptrend'       → watch for 3+ consecutive down days or a -3% single drop
          'correction'    → track the running correction low; any up-close begins a
                           rally attempt
          'rally_attempt' → check for FTD on day 4+ (gain ≥1.25% on higher volume);
                           if market undercuts correction low, back to 'correction'

        Phase 2 — backwards validity check:
          For each candidate FTD (most recent first), check whether the index
          made a new correction low AFTER the FTD.  Return the most recent valid one.

        Returns dict with keys:
          ftd_found, ftd_date, ftd_days_ago, ftd_valid, ftd_gain, correction_low
        """
        work = df.tail(lookback).copy().reset_index()

        # After reset_index() the old DatetimeIndex becomes a column.
        # The column name is whatever the index.name was — typically 'datetime'.
        date_col = "datetime" if "datetime" in work.columns else work.columns[0]

        n = len(work)
        if n < 10:
            return {"ftd_found": False}

        work["pct_chg"] = work["close"].pct_change()

        # Phase 1: forward scan
        ftd_candidates    = []   # list of dicts
        state             = "uptrend"
        correction_low    = float("inf")
        correction_low_idx = 0
        rally_attempt_idx = None
        consecutive_down  = 0

        for i in range(1, n):
            row      = work.iloc[i]
            prev_row = work.iloc[i - 1]
            pct      = row["pct_chg"]

            if pd.isna(pct):
                continue

            if state == "uptrend":
                # Enter correction on a big single-day drop …
                if pct < -0.03:
                    state              = "correction"
                    correction_low     = row["low"]
                    correction_low_idx = i
                    consecutive_down   = 1
                elif pct < 0:
                    consecutive_down += 1
                    # … or after 3+ consecutive down days
                    if consecutive_down >= 3:
                        state  = "correction"
                        window = work.iloc[max(0, i - 4): i + 1]
                        correction_low     = window["low"].min()
                        correction_low_idx = int(window["low"].idxmin())
                else:
                    consecutive_down = 0

            elif state == "correction":
                # Track the running correction low
                if row["low"] < correction_low:
                    correction_low     = row["low"]
                    correction_low_idx = i
                # First close-higher day starts the rally attempt (Day 1)
                if pct > 0:
                    state             = "rally_attempt"
                    rally_attempt_idx = i

            elif state == "rally_attempt":
                # If the market undercuts the correction low, the rally failed
                if row["low"] < correction_low:
                    state              = "correction"
                    correction_low     = row["low"]
                    correction_low_idx = i
                    rally_attempt_idx  = None
                    continue

                # FTD check: Day 4+ from rally attempt, gain ≥1.25%, higher volume
                days_from_rally = i - rally_attempt_idx
                if (
                    days_from_rally >= 3
                    and pct >= 0.0125
                    and row["volume"] > prev_row["volume"]
                ):
                    ftd_candidates.append({
                        "idx":            i,
                        "date":           row[date_col],
                        "gain":           float(pct),
                        "correction_low": correction_low,
                    })
                    # FTD resets the market to an uptrend watch
                    state            = "uptrend"
                    consecutive_down = 0

        if not ftd_candidates:
            return {"ftd_found": False}

        # Phase 2: validate candidates from most recent → oldest
        for candidate in reversed(ftd_candidates):
            ftd_idx  = candidate["idx"]
            corr_low = candidate["correction_low"]
            post_ftd = work.iloc[ftd_idx + 1:]

            # Valid if no subsequent bar's low broke below the correction low
            valid = len(post_ftd) == 0 or post_ftd["low"].min() >= corr_low

            return {
                "ftd_found":      True,
                "ftd_date":       candidate["date"],
                "ftd_days_ago":   n - 1 - ftd_idx,
                "ftd_valid":      valid,
                "ftd_gain":       round(candidate["gain"], 4),
                "correction_low": corr_low,
            }

        return {"ftd_found": False}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a price DataFrame to a sorted DatetimeIndex.
        Handles both DatetimeIndex-based and ms-epoch 'datetime'-column DataFrames.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                ts = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(df["datetime"], errors="coerce")
                df.index = ts
                df = df.drop(columns=["datetime"])
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")
        return df.sort_index()

    def _add_smas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and attach SMA_10/20/50/150/200 if not already present."""
        for p in self._SMA_PERIODS:
            col = f"SMA_{p}"
            if col not in df.columns:
                df[col] = df["close"].rolling(p, min_periods=p).mean()
        return df

    def _is_sma_rising(
        self, df: pd.DataFrame, col: str, lookback: int = 20
    ) -> bool:
        """True if the SMA is higher today than N sessions ago."""
        if col not in df.columns or len(df) < lookback + 1:
            return False
        return bool(df[col].iloc[-1] > df[col].iloc[-lookback])

    def _sma_slope_pct(
        self, df: pd.DataFrame, col: str, lookback: int = 10
    ) -> float:
        """Percentage change in an SMA over the last N sessions."""
        if col not in df.columns or len(df) < lookback + 1:
            return 0.0
        start = df[col].iloc[-lookback]
        end   = df[col].iloc[-1]
        if not start or pd.isna(start):
            return 0.0
        return float((end - start) / start)

    def _rate_of_change(self, df: pd.DataFrame, periods: int) -> float:
        """N-period rate of change of closing price."""
        if len(df) < periods + 1:
            return 0.0
        start = df["close"].iloc[-periods]
        end   = df["close"].iloc[-1]
        if not start or pd.isna(start):
            return 0.0
        return float((end - start) / start)

    def _realized_volatility(self, df: pd.DataFrame, periods: int = 20) -> float:
        """
        Annualized realized volatility from daily log returns.
        Used as a VIX proxy when VIX data is unavailable.
        A normal calm market is ~12–18%; crisis is >30%.
        """
        if len(df) < periods + 1:
            return 0.20   # default to moderate vol
        series   = df["close"].iloc[-(periods + 1):]
        log_rets = np.log(series / series.shift(1)).dropna()
        if len(log_rets) < max(periods // 2, 5):
            return 0.20
        return float(log_rets.std() * np.sqrt(252))
