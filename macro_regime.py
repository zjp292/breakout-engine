"""
macro_regime.py — Institutional-Grade Macro Market Regime Classifier

Answers three fundamental questions about the market environment:

  1. DIRECTION  — Is the market trending up, down, or sideways?
  2. QUALITY    — Is the trend clean and sustained, or choppy and erratic?
  3. VOLATILITY — Is the environment calm or fear-driven?

These combine into a named regime label with a macro_multiplier applied on top of
the daily MarketConditionAnalyzer multiplier. Together they capture sustained
multi-month environments — e.g., "choppy Nasdaq since October" registers as
NEUTRAL × CHOPPY → CONSOLIDATION, reducing exposure systematically.

Methods used:

  1. Choppiness Index (Dreiss, 1993)
     CI = 100 × log₁₀(Σ ATR₁ / (Highest_High − Lowest_Low)) / log₁₀(N)
     < 38.2 → strongly trending  |  > 61.8 → choppy / consolidating
     Advantage: pure price-structure metric, no directional bias.

  2. ADX — Average Directional Index (Wilder, 1978)
     Measures trend STRENGTH regardless of direction (< 20 = no trend, > 40 = strong).
     +DI / −DI divergence gives the directional component.

  3. Linear Regression Slope + R² (Frisch-Waugh-Lovell via numpy)
     Fit to log-price over 21-day and 63-day windows.
     High R² = clean, linear trend. Low R² = noisy / choppy.
     Slope gives the annualized expected return if the trend persists.

  4. Hurst Exponent (variance scaling approximation)
     Estimated via log-variance of lag differences across multiple time scales.
     H > 0.55 → persistent / trending (breakout-friendly)
     H ≈ 0.50 → random walk (choppy, unpredictable)
     H < 0.45 → anti-persistent / mean-reverting

  5. Multi-Timeframe Momentum Confluence (21 / 63 / 126 / 252-day)
     Fraction of horizons that agree in direction. All aligned = high conviction.
     Mixed alignment signals transitioning or consolidating environments.

  6. Volatility Regime (GARCH-inspired short/long vol ratio)
     Short-term (10d) vs long-term (60d) realized vol.
     vol_ratio > 1.30 → vol expansion spike (fear).
     Trailing vol trend detects expanding vs compressing vol regimes.

  7. Price Structure — 3-Month Swing High / Low Analysis
     How wide is the price range relative to the current level?
     Tight range + no new highs = consolidation. Expanding range = trending.

Regime Matrix (direction × quality → label → base multiplier):

  BULLISH × TRENDING      →  BULL_RUN           1.00
  BULLISH × TRANSITIONING →  BULL_TRANSITION    0.92
  BULLISH × CHOPPY        →  BULL_CHOP          0.82
  NEUTRAL × TRENDING      →  INFLECTION         0.78  (flat but linear, rare)
  NEUTRAL × TRANSITIONING →  INFLECTION         0.78
  NEUTRAL × CHOPPY        →  CONSOLIDATION      0.72  ← "Choppy Nasdaq since Oct"
  BEARISH × TRANSITIONING →  BEAR_TRANSITION    0.68
  BEARISH × TRENDING      →  DOWNTREND          0.62
  BEARISH × CHOPPY        →  BEAR_CHOP          0.60

An ELEVATED or EXTREME volatility regime applies an additional penalty of 4–13%.

References:
  Dreiss (1993) — Choppiness Index
  Wilder (1978) — New Concepts in Technical Trading Systems (ADX)
  Hurst (1951) — Long-term Storage Capacity of Reservoirs
  Peters (1994) — Fractal Market Analysis, ch. 4
  Ang & Timmermann (2012) — Regime Changes and Financial Markets (JEL)
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
class MacroRegimeResult:
    """Complete output from MacroRegimeAnalyzer."""

    # ── Three-dimensional classification ──────────────────────────────────
    trend_direction: str   # BULLISH / NEUTRAL / BEARISH
    trend_quality:   str   # TRENDING / TRANSITIONING / CHOPPY
    vol_regime:      str   # CALM / NORMAL / ELEVATED / EXTREME

    # ── Composite label + scores ───────────────────────────────────────────
    regime_label:     str    # BULL_RUN / CONSOLIDATION / DOWNTREND / …
    direction_score:  float  # −1.0 (bearish) → +1.0 (bullish)
    quality_score:    float  #  0.0 (pure chop) → 1.0 (pure trend)
    macro_multiplier: float  #  0.60–1.00

    # ── Choppiness + ADX ──────────────────────────────────────────────────
    choppiness_14: float   # 14-period CI; <38.2 trending, >61.8 choppy
    choppiness_50: float   # 50-period CI; longer-term regime view
    adx_14:        float   # 14-period ADX; 0–100 trend strength
    plus_di:       float   # +DI (bullish directional movement)
    minus_di:      float   # −DI (bearish directional movement)

    # ── Linear regression trend quality ───────────────────────────────────
    reg_slope_21d: float   # Annualized log-slope over last 21 sessions
    reg_r2_21d:    float   # R² of 21-day fit (0=noise, 1=perfect trend)
    reg_slope_63d: float   # Annualized log-slope over last 63 sessions (3 months)
    reg_r2_63d:    float   # R² of 63-day fit

    # ── Hurst exponent ────────────────────────────────────────────────────
    hurst: float   # H > 0.55 trending / H ≈ 0.5 random / H < 0.45 mean-reverting

    # ── Multi-timeframe momentum ──────────────────────────────────────────
    mom_21d:        float  # 21-day simple return
    mom_63d:        float  # 63-day (≈ 3-month) return
    mom_126d:       float  # 126-day (≈ 6-month) return
    mom_252d:       float  # 252-day (≈ 1-year) return
    mom_confluence: float  # −1.0 (all bearish) → +1.0 (all bullish)

    # ── Volatility regime ─────────────────────────────────────────────────
    vol_10d:    float   # 10-day realized vol, annualized
    vol_60d:    float   # 60-day realized vol, annualized (regime baseline)
    vol_ratio:  float   # vol_10d / vol_60d  (>1.30 = vol expansion spike)
    vol_rising: bool    # True if short-term vol is trending higher

    # ── Price structure (3-month swing analysis) ───────────────────────────
    pct_from_swing_high: float  # % below the 3-month highest high (negative)
    pct_from_swing_low:  float  # % above the 3-month lowest low (positive)
    range_width_pct:     float  # total swing range as % of price

    # ── Multi-index confirmation ───────────────────────────────────────────
    spy_above_200: Optional[bool]
    iwm_above_200: Optional[bool]

    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────


class MacroRegimeAnalyzer:
    """
    Institutional-grade macro market regime classifier.

    Combines seven independent signal families to characterize the sustained
    market environment over weeks to months — not just today's snapshot.

    The "choppy Nasdaq since October" pattern produces:
      - Choppiness Index persistently > 61.8 (no directional commitment)
      - ADX < 20 (no sustained trend strength)
      - Low 63-day R² (price oscillating, not following a line)
      - Mixed multi-TF momentum (21d up, 63d flat, 126d up, etc.)
      - Hurst ≈ 0.5 (random-walk behavior, no persistence)
      → Classifies as NEUTRAL × CHOPPY → CONSOLIDATION → ×0.72

    Integration with MarketConditionAnalyzer:
      The final_multiplier applied to stock scores is a weighted blend:
        final = 0.55 × daily_mc_multiplier + 0.45 × macro_multiplier
      The daily analysis keeps short-term signals relevant; macro prevents
      over-sizing in unfavorable medium-term environments.
    """

    # ── Choppiness Index thresholds (Fibonacci levels) ──────────────────────
    CI_TRENDING = 38.2   # below this → strongly trending
    CI_CHOPPY   = 61.8   # above this → choppy / consolidating

    # ── ADX thresholds (Wilder's original guidelines) ───────────────────────
    ADX_WEAK   = 20    # no meaningful trend
    ADX_TREND  = 25    # trend beginning
    ADX_STRONG = 40    # strong trend

    # ── Hurst exponent thresholds ────────────────────────────────────────────
    HURST_TREND   = 0.55   # persistent / trending
    HURST_MEANREV = 0.45   # anti-persistent / mean-reverting

    # ── R² thresholds for 63-day linear regression ───────────────────────────
    R2_STRONG   = 0.75
    R2_MODERATE = 0.40

    # ── Regime label → base multiplier ──────────────────────────────────────
    REGIME_MULTIPLIERS: dict[str, float] = {
        "BULL_RUN":        1.00,
        "BULL_TRANSITION": 0.92,
        "BULL_CHOP":       0.82,
        "INFLECTION":      0.78,
        "CONSOLIDATION":   0.72,
        "BEAR_TRANSITION": 0.68,
        "DOWNTREND":       0.62,
        "BEAR_CHOP":       0.60,
    }

    # ── Direction × Quality → Regime label matrix ───────────────────────────
    _REGIME_MATRIX: dict[tuple[str, str], str] = {
        ("BULLISH", "TRENDING"):      "BULL_RUN",
        ("BULLISH", "TRANSITIONING"): "BULL_TRANSITION",
        ("BULLISH", "CHOPPY"):        "BULL_CHOP",
        ("NEUTRAL", "TRENDING"):      "INFLECTION",
        ("NEUTRAL", "TRANSITIONING"): "INFLECTION",
        ("NEUTRAL", "CHOPPY"):        "CONSOLIDATION",
        ("BEARISH", "TRANSITIONING"): "BEAR_TRANSITION",
        ("BEARISH", "TRENDING"):      "DOWNTREND",
        ("BEARISH", "CHOPPY"):        "BEAR_CHOP",
    }

    def __init__(self, config: dict):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        compx_df: pd.DataFrame,
        spy_df:   Optional[pd.DataFrame] = None,
        iwm_df:   Optional[pd.DataFrame] = None,
    ) -> MacroRegimeResult:
        """
        Run the full macro regime analysis.

        Args:
            compx_df: NASDAQ Composite daily OHLCV (primary benchmark).
                      Accepts DatetimeIndex or ms-epoch 'datetime' column.
            spy_df:   S&P 500 ETF — multi-index directional confirmation.
            iwm_df:   Russell 2000 ETF — small-cap risk-on confirmation.

        Returns:
            MacroRegimeResult with full classification and all raw signals.
        """
        compx = self._prep_df(compx_df)
        spy   = self._prep_df(spy_df) if spy_df is not None else None
        iwm   = self._prep_df(iwm_df) if iwm_df is not None else None

        if len(compx) < 60:
            return self._insufficient_data_result()

        # Compute all seven signal families
        sig = self._compute_signals(compx)

        # Add multi-index confirmation
        sig["spy_above_200"] = self._above_200(spy)
        sig["iwm_above_200"] = self._above_200(iwm)

        # Compute composite scores
        direction_score = self._score_direction(sig)
        quality_score   = self._score_quality(sig)

        # Three-dimensional classification
        trend_direction = self._classify_direction(direction_score, sig)
        trend_quality   = self._classify_quality(quality_score)
        vol_regime      = self._classify_vol(sig)

        # Map to regime label and apply vol penalty
        regime_label = self._REGIME_MATRIX.get(
            (trend_direction, trend_quality), "CONSOLIDATION"
        )
        base_mult  = self.REGIME_MULTIPLIERS.get(regime_label, 0.72)
        macro_mult = self._vol_adjusted_multiplier(base_mult, sig)

        return MacroRegimeResult(
            trend_direction      = trend_direction,
            trend_quality        = trend_quality,
            vol_regime           = vol_regime,
            regime_label         = regime_label,
            direction_score      = round(direction_score, 3),
            quality_score        = round(quality_score, 3),
            macro_multiplier     = round(macro_mult, 3),
            choppiness_14        = round(sig["ci_14"], 2),
            choppiness_50        = round(sig["ci_50"], 2),
            adx_14               = round(sig["adx"], 2),
            plus_di              = round(sig["plus_di"], 2),
            minus_di             = round(sig["minus_di"], 2),
            reg_slope_21d        = round(sig["slope_21"], 4),
            reg_r2_21d           = round(sig["r2_21"], 4),
            reg_slope_63d        = round(sig["slope_63"], 4),
            reg_r2_63d           = round(sig["r2_63"], 4),
            hurst                = round(sig["hurst"], 3),
            mom_21d              = round(sig["mom_21"],  4),
            mom_63d              = round(sig["mom_63"],  4),
            mom_126d             = round(sig["mom_126"], 4),
            mom_252d             = round(sig["mom_252"], 4),
            mom_confluence       = round(sig["mom_confluence"], 3),
            vol_10d              = round(sig["vol_10"],  4),
            vol_60d              = round(sig["vol_60"],  4),
            vol_ratio            = round(sig["vol_ratio"], 3),
            vol_rising           = sig["vol_rising"],
            pct_from_swing_high  = round(sig["pct_from_swing_high"], 4),
            pct_from_swing_low   = round(sig["pct_from_swing_low"],  4),
            range_width_pct      = round(sig["range_width_pct"], 4),
            spy_above_200        = sig["spy_above_200"],
            iwm_above_200        = sig["iwm_above_200"],
            details              = sig,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Signal computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_signals(self, df: pd.DataFrame) -> dict:
        """Compute all seven signal families; return as a flat dict."""
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── 1. Choppiness Index (14 and 50 period) ────────────────────────
        ci_14_s = self._choppiness_index(df, 14)
        ci_50_s = self._choppiness_index(df, 50)
        ci_14 = float(ci_14_s.iloc[-1]) if pd.notna(ci_14_s.iloc[-1]) else 61.8
        ci_50 = float(ci_50_s.iloc[-1]) if pd.notna(ci_50_s.iloc[-1]) else 61.8

        # ── 2. ADX (14-period) ────────────────────────────────────────────
        adx_s, plus_di_s, minus_di_s = self._adx(df, 14)
        adx      = float(adx_s.iloc[-1])      if pd.notna(adx_s.iloc[-1])      else 20.0
        plus_di  = float(plus_di_s.iloc[-1])  if pd.notna(plus_di_s.iloc[-1])  else 20.0
        minus_di = float(minus_di_s.iloc[-1]) if pd.notna(minus_di_s.iloc[-1]) else 20.0

        # ── 3. Linear regression quality (21d and 63d) ────────────────────
        slope_21, r2_21 = self._regression_quality(close, 21)
        slope_63, r2_63 = self._regression_quality(close, 63)

        # ── 4. Hurst Exponent ─────────────────────────────────────────────
        hurst = self._hurst_exponent(close)

        # ── 5. Multi-timeframe momentum ───────────────────────────────────
        mom_21  = self._momentum(close, 21)
        mom_63  = self._momentum(close, 63)
        mom_126 = self._momentum(close, 126)
        mom_252 = self._momentum(close, 252)

        moms = [m for m in [mom_21, mom_63, mom_126, mom_252] if m is not None]
        pos  = sum(1 for m in moms if m > 0)
        neg  = sum(1 for m in moms if m < 0)
        # confluence: +1 = all bullish, −1 = all bearish, 0 = split
        mom_confluence = (pos - neg) / len(moms) if moms else 0.0

        # ── 6. Volatility regime ──────────────────────────────────────────
        vol_10    = self._realized_vol(close, 10)
        vol_60    = self._realized_vol(close, 60)
        vol_ratio = vol_10 / vol_60 if vol_60 > 0 else 1.0
        vol_rising = self._is_vol_rising(close, period=10, lookback=20)

        # ── 7. Price structure — 3-month swing high/low ───────────────────
        n63         = min(63, len(df))
        swing_high  = float(high.iloc[-n63:].max())
        swing_low   = float(low.iloc[-n63:].min())
        current     = float(close.iloc[-1])
        pct_from_sh = (current - swing_high) / swing_high if swing_high > 0 else 0.0
        pct_from_sl = (current - swing_low)  / swing_low  if swing_low  > 0 else 0.0
        range_width = (swing_high - swing_low) / swing_low if swing_low > 0 else 0.0

        # ── 8. Supplementary: trend consistency (% up-days) ──────────────
        ret_21 = close.iloc[-21:].pct_change().dropna()
        tc_21  = float((ret_21 > 0).mean()) if len(ret_21) >= 10 else 0.5
        ret_63 = close.iloc[-63:].pct_change().dropna()
        tc_63  = float((ret_63 > 0).mean()) if len(ret_63) >= 30 else 0.5

        # ── 9. 200-SMA position ───────────────────────────────────────────
        sma200 = close.rolling(200, min_periods=150).mean()
        last_sma200 = sma200.iloc[-1]
        above_200 = bool(current > last_sma200) if pd.notna(last_sma200) else None

        return {
            # Choppiness
            "ci_14":                ci_14,
            "ci_50":                ci_50,
            # ADX
            "adx":                  adx,
            "plus_di":              plus_di,
            "minus_di":             minus_di,
            # Regression
            "slope_21":             slope_21,
            "r2_21":                r2_21,
            "slope_63":             slope_63,
            "r2_63":                r2_63,
            # Hurst
            "hurst":                hurst,
            # Momentum
            "mom_21":               mom_21  or 0.0,
            "mom_63":               mom_63  or 0.0,
            "mom_126":              mom_126 or 0.0,
            "mom_252":              mom_252 or 0.0,
            "mom_confluence":       mom_confluence,
            # Volatility
            "vol_10":               vol_10,
            "vol_60":               vol_60,
            "vol_ratio":            vol_ratio,
            "vol_rising":           vol_rising,
            # Price structure
            "pct_from_swing_high":  pct_from_sh,
            "pct_from_swing_low":   pct_from_sl,
            "range_width_pct":      range_width,
            # Supplementary
            "trend_consistency_21": tc_21,
            "trend_consistency_63": tc_63,
            "above_200":            above_200,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Direction score  −1.0 → +1.0
    # ─────────────────────────────────────────────────────────────────────────

    def _score_direction(self, sig: dict) -> float:
        """
        Weighted combination of five directional signals.

        Weights reflect signal reliability and independence:
          Multi-TF momentum confluence   35% — clearest bull/bear signal
          ADX directional (+DI vs −DI)   20% — ADX-based direction
          63-day regression slope        25% — sustained price direction
          200-SMA position               15% — institutional trend filter
          Trend consistency tilt          5% — fraction of up-days
        """
        components = []

        # 1. Multi-TF momentum confluence (already in [−1, +1])
        components.append((sig["mom_confluence"], 0.35))

        # 2. ADX direction: (+DI − −DI) / (+DI + −DI) → [−1, +1]
        di_sum = sig["plus_di"] + sig["minus_di"]
        di_dir = (sig["plus_di"] - sig["minus_di"]) / di_sum if di_sum > 0 else 0.0
        components.append((float(di_dir), 0.20))

        # 3. Regression slope — normalize: 30% annualized = full +1 or −1
        slope_norm = float(np.clip(sig["slope_63"] / 0.30, -1.0, 1.0))
        components.append((slope_norm, 0.25))

        # 4. 200-SMA position
        above_200 = sig.get("above_200")
        if   above_200 is True:  sma_sig =  0.50
        elif above_200 is False: sma_sig = -0.50
        else:                    sma_sig =  0.00
        components.append((sma_sig, 0.15))

        # 5. Trend consistency tilt: 0.50 = neutral, 0.65 = bullish → +0.30
        tc_sig = float(np.clip((sig["trend_consistency_63"] - 0.50) * 2.0, -1.0, 1.0))
        components.append((tc_sig, 0.05))

        total_weight = sum(w for _, w in components)
        direction    = sum(s * w for s, w in components) / total_weight
        return float(np.clip(direction, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Quality score  0.0 (pure chop) → 1.0 (pure trend)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_quality(self, sig: dict) -> float:
        """
        Weighted combination of five trend-quality signals.

        Weights reflect what each signal captures:
          ADX                 30% — primary trend-strength indicator
          Choppiness Index    30% — direct range vs trend measurement
          63-day R²           25% — linearity of price path
          Hurst exponent      10% — stochastic persistence of the series
          |Mom confluence|     5% — absolute conviction across timeframes
        """
        components = []

        # 1. ADX: 0 at ADX≤20, 0.5 at ADX=25, 1.0 at ADX≥40
        adx = sig["adx"]
        if   adx >= self.ADX_STRONG: adx_q = 1.0
        elif adx >= self.ADX_TREND:
            adx_q = 0.50 + 0.50 * (adx - self.ADX_TREND) / (self.ADX_STRONG - self.ADX_TREND)
        elif adx >= self.ADX_WEAK:
            adx_q = 0.50 * (adx - self.ADX_WEAK) / (self.ADX_TREND - self.ADX_WEAK)
        else:
            adx_q = 0.0
        components.append((adx_q, 0.30))

        # 2. Choppiness Index: CI<38.2 → 1.0, CI>61.8 → 0.0, linear between
        ci = sig["ci_14"]
        if   ci <= self.CI_TRENDING: ci_q = 1.0
        elif ci >= self.CI_CHOPPY:   ci_q = 0.0
        else:
            ci_q = 1.0 - (ci - self.CI_TRENDING) / (self.CI_CHOPPY - self.CI_TRENDING)
        components.append((ci_q, 0.30))

        # 3. 63-day R²: directly in [0, 1]
        components.append((float(sig["r2_63"]), 0.25))

        # 4. Hurst: H<0.45 → 0.0, H>0.55 → 1.0, linear between
        h = sig["hurst"]
        if   h >= self.HURST_TREND:   h_q = 1.0
        elif h <= self.HURST_MEANREV: h_q = 0.0
        else:
            h_q = (h - self.HURST_MEANREV) / (self.HURST_TREND - self.HURST_MEANREV)
        components.append((h_q, 0.10))

        # 5. |Momentum confluence|: 1.0 = all timeframes agree, 0 = split
        abs_conf = abs(sig["mom_confluence"])
        components.append((abs_conf, 0.05))

        total_weight = sum(w for _, w in components)
        quality      = sum(q * w for q, w in components) / total_weight
        return float(np.clip(quality, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_direction(self, score: float, sig: dict) -> str:
        """Thresholds chosen to require meaningful conviction in either direction."""
        if score >= 0.20:
            # if both short-term signals show current decline, cap at NEUTRAL
            # long-term averages can look bullish while the market is actively falling
            if sig.get("mom_21d", 0) < 0 and sig.get("slope_21", 0) < 0:
                return "NEUTRAL"
            return "BULLISH"
        if score <= -0.20: return "BEARISH"
        return "NEUTRAL"

    def _classify_quality(self, score: float) -> str:
        """Upper/lower thirds define TRENDING/CHOPPY; middle = TRANSITIONING."""
        if score >= 0.60: return "TRENDING"
        if score <= 0.35: return "CHOPPY"
        return "TRANSITIONING"

    def _classify_vol(self, sig: dict) -> str:
        """
        Classify volatility regime using 60-day realized vol as the baseline.
        Typical annualized Nasdaq realized vol ranges:
          <12% → extremely calm / complacent
          12–18% → normal swing-trading environment
          18–28% → elevated (wider stops needed, breakouts fail more)
          >28% → extreme fear / crisis
        """
        vol = sig["vol_60"]
        if   vol < 0.12: return "CALM"
        elif vol < 0.18: return "NORMAL"
        elif vol < 0.28: return "ELEVATED"
        else:            return "EXTREME"

    def _vol_adjusted_multiplier(self, base: float, sig: dict) -> float:
        """
        Apply a volatility penalty on top of the base regime multiplier.

        In high-vol environments, trend-following strategies suffer:
          - Stop placement is unreliable (intraday swings hit stops on noise)
          - Institutions reduce position sizes and activity
          - Breakouts fail more frequently in volatile conditions

        Penalty schedule:
          CALM / NORMAL:  no penalty
          ELEVATED:       −4% (breakouts need more confirmation)
          EXTREME:        −10% (minimize new longs; very selective)
          + if vol is actively RISING: additional −3% (expansion in progress)
        """
        vol_regime = self._classify_vol(sig)
        penalty = {"CALM": 0.00, "NORMAL": 0.00, "ELEVATED": 0.04, "EXTREME": 0.10}.get(
            vol_regime, 0.0
        )
        if sig.get("vol_rising") and vol_regime in ("ELEVATED", "EXTREME"):
            penalty += 0.03

        return max(0.60, round(base - penalty, 2))

    # ─────────────────────────────────────────────────────────────────────────
    # Signal implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _choppiness_index(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Choppiness Index (Dreiss, 1993).

        CI = 100 × log₁₀(Σ ATR₁ / (Highest_High − Lowest_Low)) / log₁₀(N)

        Conceptually: how much of the total bar range was consumed by individual
        bars' true ranges?  High consumption = trending. Low = choppy/oscillating.
        """
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_sum    = tr.rolling(period).sum()
        highest_hi = df["high"].rolling(period).max()
        lowest_lo  = df["low"].rolling(period).min()
        range_hl   = highest_hi - lowest_lo

        ci = pd.Series(np.nan, index=df.index)
        valid = range_hl > 0
        ci[valid] = (
            100.0
            * np.log10(atr_sum[valid] / range_hl[valid])
            / np.log10(period)
        )
        return ci

    def _adx(
        self, df: pd.DataFrame, period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (Wilder, 1978).

        Returns (ADX, +DI, -DI) all in [0, 100].

        ADX measures trend STRENGTH regardless of direction.
        +DI and −DI give the directional component.

        ADX < 20 → no meaningful trend (choppy)
        ADX 20–25 → trend forming
        ADX 25–40 → trending market
        ADX > 40 → strong trend (often leads to exhaustion)
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Directional Movement
        up_move   = high.diff()
        down_move = -(low.diff())

        plus_dm_arr  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
        minus_dm_arr = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)
        plus_dm  = pd.Series(plus_dm_arr,  index=df.index, dtype=float)
        minus_dm = pd.Series(minus_dm_arr, index=df.index, dtype=float)

        # Wilder smoothing (RMA)
        atr      = self._wilder_smooth(tr,       period)
        s_plus   = self._wilder_smooth(plus_dm,  period)
        s_minus  = self._wilder_smooth(minus_dm, period)

        # Directional indicators
        plus_di  = 100.0 * s_plus  / atr.replace(0.0, np.nan)
        minus_di = 100.0 * s_minus / atr.replace(0.0, np.nan)

        # DX and ADX
        di_sum = plus_di + minus_di
        dx     = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
        adx    = self._wilder_smooth(dx.fillna(0.0), period)

        return adx, plus_di.fillna(20.0), minus_di.fillna(20.0)

    def _wilder_smooth(self, series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's RMA (Relative Moving Average).
        RMA[i] = (RMA[i-1] × (period − 1) + val[i]) / period
        Equivalent to EWM with alpha = 1/period, adjust=False.
        """
        return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def _regression_quality(
        self, close: pd.Series, period: int
    ) -> tuple[float, float]:
        """
        Fit a linear regression to log(close) over the last `period` sessions.

        Returns (annualized_slope, R²).

        annualized_slope = daily log-slope × 252. Interpretation:
          +0.30 → market is trending up at ~30% annualized rate
          −0.20 → market is declining at ~20% annualized rate

        R² interpretation:
          0.90+ → very clean, linear trend (strong bull/bear run)
          0.50–0.90 → moderate trend with noise
          <0.40 → choppy / non-linear (oscillating, range-bound)
        """
        n = min(period, len(close))
        if n < 10:
            return 0.0, 0.0

        y = np.log(close.iloc[-n:].values.astype(float))
        x = np.arange(n, dtype=float)

        x_mean, y_mean = x.mean(), y.mean()
        ss_xy = float(np.dot(x - x_mean, y - y_mean))
        ss_xx = float(np.dot(x - x_mean, x - x_mean))

        if ss_xx == 0:
            return 0.0, 0.0

        slope     = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred    = slope * x + intercept

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return float(slope * 252), max(0.0, float(r2))   # annualize slope

    def _hurst_exponent(self, close: pd.Series) -> float:
        """
        Hurst Exponent via variance-scaling approximation.

        Estimates the self-similarity exponent H by measuring how the variance
        of log-price differences scales with the lag length.

        For Geometric Brownian Motion: var(log_p[t+τ] − log_p[t]) ∝ τ^1  → H = 0.50
        For persistent series: var ∝ τ^(2H) with H > 0.5

        Uses log-spaced lags across 2 to half the series length to cover
        multiple time scales robustly.

        Reference: Peters (1994) Fractal Market Analysis, ch. 4.
        """
        prices = close.dropna().values[-min(150, len(close)):]
        if len(prices) < 30:
            return 0.5

        log_p = np.log(prices.astype(float))
        max_lag = max(4, len(prices) // 3)
        lags    = np.unique(
            np.logspace(np.log10(2), np.log10(max_lag), 25).astype(int)
        )
        lags = lags[lags >= 2]

        log_lags = []
        log_vars = []
        for lag in lags:
            diffs = log_p[lag:] - log_p[:-lag]
            if len(diffs) < 4:
                continue
            v = np.var(diffs)
            if v > 0:
                log_lags.append(np.log(lag))
                log_vars.append(np.log(v))

        if len(log_vars) < 4:
            return 0.5

        log_lags_arr = np.array(log_lags)
        log_vars_arr = np.array(log_vars)
        mask = np.isfinite(log_lags_arr) & np.isfinite(log_vars_arr)
        if mask.sum() < 4:
            return 0.5

        slope = float(np.polyfit(log_lags_arr[mask], log_vars_arr[mask], 1)[0])
        # slope of log-log regression ≈ 2H for variance scaling
        return float(np.clip(slope / 2.0, 0.0, 1.0))

    def _momentum(self, close: pd.Series, period: int) -> Optional[float]:
        """Simple N-period price return.  Returns None if insufficient data."""
        if len(close) <= period:
            return None
        start = float(close.iloc[-(period + 1)])
        end   = float(close.iloc[-1])
        if start <= 0:
            return None
        return (end - start) / start

    def _realized_vol(self, close: pd.Series, period: int) -> float:
        """Annualized realized volatility from daily log returns."""
        if len(close) < period + 1:
            return 0.20
        series   = close.iloc[-(period + 1):]
        log_rets = np.log(series / series.shift(1)).dropna()
        if len(log_rets) < max(period // 2, 5):
            return 0.20
        return float(log_rets.std() * np.sqrt(252))

    def _is_vol_rising(
        self, close: pd.Series, period: int = 10, lookback: int = 20
    ) -> bool:
        """
        True if the short-term realized vol is meaningfully higher now
        than it was `lookback` sessions ago.  Detects vol expansion in progress.
        """
        if len(close) < period + lookback + 2:
            return False
        vol_now  = self._realized_vol(close,              period)
        vol_prev = self._realized_vol(close.iloc[:-lookback], period)
        return vol_now > vol_prev * 1.10   # 10% threshold to filter noise

    def _above_200(self, df: Optional[pd.DataFrame]) -> Optional[bool]:
        """Return True/False for close vs 200-SMA; None if unavailable."""
        if df is None or len(df) < 200:
            return None
        close   = df["close"]
        sma_200 = close.rolling(200).mean()
        return bool(close.iloc[-1] > sma_200.iloc[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise to a sorted DatetimeIndex; handles ms-epoch datetime column."""
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

    def _insufficient_data_result(self) -> MacroRegimeResult:
        """Return a conservative neutral result when there isn't enough data."""
        return MacroRegimeResult(
            trend_direction="NEUTRAL",   trend_quality="CHOPPY",
            vol_regime="NORMAL",          regime_label="CONSOLIDATION",
            direction_score=0.0,          quality_score=0.5,
            macro_multiplier=0.85,
            choppiness_14=61.8,           choppiness_50=61.8,
            adx_14=20.0,                  plus_di=20.0,    minus_di=20.0,
            reg_slope_21d=0.0,            reg_r2_21d=0.0,
            reg_slope_63d=0.0,            reg_r2_63d=0.0,
            hurst=0.5,
            mom_21d=0.0,  mom_63d=0.0,   mom_126d=0.0,    mom_252d=0.0,
            mom_confluence=0.0,
            vol_10d=0.20,                 vol_60d=0.20,    vol_ratio=1.0,
            vol_rising=False,
            pct_from_swing_high=0.0,      pct_from_swing_low=0.0,
            range_width_pct=0.0,
            spy_above_200=None,           iwm_above_200=None,
        )
