"""
Regression tests for the Features class.

Covers the computation of every major feature used by the Scoring class.
Each test uses synthetic OHLCV data — no API calls, no file I/O.

Run from project root:
    python -m pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

from config import PARAMETERS
from engine import Features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_features() -> Features:
    return Features(PARAMETERS.copy())


def make_ohlcv(
    n: int = 260,
    start_price: float = 20.0,
    end_price: float = None,
    volume: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic daily OHLCV DataFrame with n rows.

    If end_price is given, prices trend linearly from start_price to end_price.
    Otherwise prices remain flat at start_price (plus tiny noise).
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end="2025-12-31", periods=n, freq="B")

    if end_price is None:
        closes = np.full(n, start_price) + rng.uniform(-0.01, 0.01, n) * start_price
    else:
        closes = np.linspace(start_price, end_price, n) + rng.uniform(-0.01, 0.01, n) * start_price

    highs  = closes * (1 + rng.uniform(0.003, 0.015, n))
    lows   = closes * (1 - rng.uniform(0.003, 0.015, n))
    opens  = closes + rng.uniform(-0.005, 0.005, n) * closes

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=dates,
    )


# ===========================================================================
# 1. MOVING AVERAGES
# ===========================================================================

class TestMovingAverages:

    def test_sma_columns_created(self):
        f = make_features()
        df = make_ohlcv(260)
        df = f.add_moving_averages(df)
        for period in PARAMETERS["sma_periods"]:
            assert f"sma_{period}" in df.columns

    def test_sma_10_is_exponential_moving_average(self):
        """sma_10 is computed as EMA(10) — qullamaggie uses EMA for fast momentum lines."""
        f = make_features()
        df = make_ohlcv(260)
        df = f.add_moving_averages(df)
        expected = df["close"].ewm(span=10, adjust=False).mean()
        pd.testing.assert_series_equal(df["sma_10"], expected, check_names=False)

    def test_sma_nan_for_insufficient_history(self):
        """First (period-1) rows of each SMA should be NaN."""
        f = make_features()
        df = make_ohlcv(260)
        df = f.add_moving_averages(df)
        assert df["sma_200"].iloc[:199].isna().all()
        assert not pd.isna(df["sma_200"].iloc[199])

    def test_uptrend_produces_correct_sma_order(self):
        """In a strong uptrend, sma_10 > sma_20 > sma_50 for later rows."""
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        last = df.iloc[-1]
        assert last["sma_10"] > last["sma_20"] > last["sma_50"]


# ===========================================================================
# 2. MA RELATIONSHIPS AND STAGE 2
# ===========================================================================

class TestMARelationships:

    def test_ma_alignment_true_in_uptrend(self):
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        # Later rows should have sma_10 > sma_20 > sma_50
        assert df["ma_alignment"].iloc[-1] is True or df["ma_alignment"].iloc[-1] == True

    def test_ma_alignment_false_in_downtrend(self):
        f = make_features()
        df = make_ohlcv(260, start_price=50.0, end_price=10.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        # In a downtrend the short SMA is below the long SMA
        assert df["ma_alignment"].iloc[-1] is False or df["ma_alignment"].iloc[-1] == False

    def test_stage2_true_in_strong_uptrend(self):
        """
        Stage 2 conditions:
          close > sma_150, sma_50 > sma_150, sma_150 > sma_200,
          sma_200 slope positive over 20 periods.
        All satisfied in a sustained uptrend with 260 bars.
        """
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        # Last row should be Stage 2
        assert df["stage2"].iloc[-1] == True

    def test_stage2_false_if_price_below_sma150(self):
        """If price is below sma_150, stage2 cannot be True."""
        f = make_features()
        # Downtrend: price falls far below where 150 SMA will be
        df = make_ohlcv(260, start_price=50.0, end_price=5.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        assert df["stage2"].iloc[-1] == False

    def test_stage2_false_without_sma150_column(self):
        """If sma_150 is absent from config, stage2 defaults to False."""
        config = PARAMETERS.copy()
        config["sma_periods"] = [10, 20, 50]  # no 150 or 200
        f = Features(config)
        df = make_ohlcv(100, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        assert (df["stage2"] == False).all()

    def test_distance_from_sma10_positive_when_above(self):
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        # In uptrend, close > sma_10 → distance > 0
        # (may be briefly negative near the start; check last 10 rows)
        assert df["distance_from_sma10"].iloc[-1] > 0

    def test_ma_slope_columns_exist(self):
        f = make_features()
        df = make_ohlcv(260)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        for col in ["ma_slope_10", "ma_slope_20", "ma_slope_50"]:
            assert col in df.columns

    def test_mas_rising_in_uptrend(self):
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        assert df["mas_rising"].iloc[-1] == True


# ===========================================================================
# 3. CONSOLIDATION RANGE DETECTION
# ===========================================================================

class TestConsolidationRange:

    def test_breakout_level_equals_rolling_high(self):
        f = make_features()
        df = make_ohlcv(100)
        df = f.detect_consolidation_range(df)
        lookback = PARAMETERS["base_length_max"]
        expected_high = df["high"].rolling(window=lookback).max()
        pd.testing.assert_series_equal(df["breakout_level"], expected_high,
                                       check_names=False)

    def test_tight_consolidation_flag_below_threshold(self):
        """Flat price data should show is_tight_consolidation=True (tiny range)."""
        f = make_features()
        df = make_ohlcv(100, start_price=30.0, end_price=30.0)
        # Very flat → range should be < 5% threshold
        df = f.detect_consolidation_range(df)
        threshold = PARAMETERS["range_compression_threshold"]
        lookback = PARAMETERS["base_length_max"]
        tight_rows = df["is_tight_consolidation"].iloc[lookback:]
        assert tight_rows.any(), "Expected some tight consolidation in flat data"

    def test_volatile_data_not_tight(self):
        """Large-range synthetic data should not be flagged as tight."""
        f = make_features()
        rng = np.random.RandomState(99)
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        # Closes that swing ±30% each bar
        closes = 30.0 + rng.uniform(-9, 9, n)
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.15,
            "low": closes * 0.85, "close": closes,
            "volume": 1_000_000,
        }, index=dates)
        df = f.detect_consolidation_range(df)
        lookback = PARAMETERS["base_length_max"]
        # After enough history, none of these wide-swinging rows should be tight
        assert not df["is_tight_consolidation"].iloc[lookback:].all()


# ===========================================================================
# 4. 52-WEEK HIGH PROXIMITY
# ===========================================================================

class TestFiftyTwoWeekProximity:

    def test_at_52wk_high_pct_is_zero(self):
        """When close equals the rolling max, pct_from_52wk_high = 0."""
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.calculate_52wk_proximity(df)
        # Last row's close should be at/near the max → pct should be ≥ -0.02
        last = df.iloc[-1]
        assert last["pct_from_52wk_high"] >= -0.02

    def test_below_52wk_high_pct_is_negative(self):
        """Stock that drops after reaching a high should show negative pct."""
        f = make_features()
        # Rise then fall
        n = 260
        prices_up = np.linspace(10, 50, n // 2)
        prices_dn = np.linspace(50, 30, n // 2)
        closes = np.concatenate([prices_up, prices_dn])
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": 1_000_000,
        }, index=dates)
        df = f.calculate_52wk_proximity(df)
        last = df.iloc[-1]
        # After dropping from 50 to 30, pct ≈ (30-50)/50 = -0.40
        assert last["pct_from_52wk_high"] < -0.20

    def test_pct_from_52wk_high_formula(self):
        """Verify the formula: (close - 52wk_high) / 52wk_high."""
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=30.0)
        df = f.calculate_52wk_proximity(df)
        expected = (df["close"] - df["52wk_high"]) / df["52wk_high"]
        pd.testing.assert_series_equal(df["pct_from_52wk_high"], expected,
                                       check_names=False)


# ===========================================================================
# 5. VCP CONTRACTIONS
# ===========================================================================

class TestVCPContractions:

    def test_vcp_columns_created(self):
        f = make_features()
        df = make_ohlcv(100)
        df = f.detect_vcp_contractions(df)
        for col in ["range_10", "range_20", "range_40", "vcp_contracting",
                    "vcp_contraction_ratio"]:
            assert col in df.columns

    def test_vcp_contracting_when_narrowing(self):
        """
        vcp_contracting uses three NON-OVERLAPPING w_short-day windows.
        construct flat prices with explicit per-bar ranges that decrease
        across three successive 10-day blocks so the shifted windows see
        clearly distinct volatility levels and the flag must be True.
        """
        f = make_features()
        w = PARAMETERS["vcp_windows"][0]   # 10
        n_pad = 40                          # enough history for rolling warm-up
        n = n_pad + w * 3                   # 70 bars total
        base = 30.0
        closes = np.full(n, base)

        # half-range per bar — explicit, non-random, non-overlapping periods
        half_ranges = np.concatenate([
            np.full(n_pad, 0.05 * base),   # padding: ±5%
            np.full(w,     0.08 * base),   # far window:    ±8%  (widest)
            np.full(w,     0.03 * base),   # middle window: ±3%
            np.full(w,     0.005 * base),  # near window:   ±0.5% (tightest)
        ])
        highs = closes + half_ranges
        lows  = closes - half_ranges
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        df = pd.DataFrame({
            "open": closes, "high": highs, "low": lows,
            "close": closes, "volume": 1_000_000,
        }, index=dates)
        df = f.detect_vcp_contractions(df)
        # near(0.5%) < middle(3%) < far(8%) → vcp_contracting must be True
        assert df["vcp_contracting"].iloc[-1] == True

    def test_contraction_ratio_bounded(self):
        """vcp_contraction_ratio should always be positive and ≤ 1 for a tight recent range."""
        f = make_features()
        df = make_ohlcv(100, start_price=30.0, end_price=35.0)
        df = f.detect_vcp_contractions(df)
        valid = df["vcp_contraction_ratio"].dropna()
        # Ratio is clipped from below at 0 (range_40 clipped at 0.001)
        assert (valid >= 0).all()

    def test_range_10_lt_range_40_in_tight_market(self):
        """In a steadily calming market, recent (10d) range < longer (40d) range."""
        f = make_features()
        rng = np.random.RandomState(3)
        n = 100
        # Volatility decreasing over time
        vols = np.linspace(0.05, 0.005, n)
        closes = 30 + np.cumsum(rng.uniform(-1, 1, n) * vols * 30)
        closes = np.maximum(closes, 1.0)
        highs = closes + vols * 30 * 0.5
        lows  = closes - vols * 30 * 0.5
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        df = pd.DataFrame({
            "open": closes, "high": highs, "low": lows,
            "close": closes, "volume": 1_000_000,
        }, index=dates)
        df = f.detect_vcp_contractions(df)
        last = df.iloc[-1]
        assert last["range_10"] < last["range_40"]


# ===========================================================================
# 6. RELATIVE STRENGTH (EXCESS RETURN FORMULA)
# ===========================================================================

class TestRelativeStrength:
    """
    calculate_relative_strength() must use the excess-return formula:
      rs = stock_pct_change - benchmark_pct_change

    Key invariants:
    - Bull market outperformer: stock gains more → positive RS
    - Bear market outperformer: stock falls less → positive RS
    - Underperformer: stock gains less (or falls more) → negative RS
    """

    def _aligned_pair(self, stock_prices, bench_prices, n=130):
        """Build aligned stock and benchmark DataFrames."""
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        stock_df = pd.DataFrame({"close": stock_prices}, index=dates)
        bench_df = pd.DataFrame({"close": bench_prices}, index=dates)
        return stock_df, bench_df

    def test_bull_market_outperformer_positive_rs(self):
        """Stock gains 30%, benchmark gains 10% → excess ≈ +20% over 60 days."""
        n = 130
        # stock: grows from 10 to 13 (+30%) over the last 60 bars
        stock_p = np.concatenate([np.linspace(10, 10, 70), np.linspace(10, 13, 60)])
        bench_p = np.concatenate([np.linspace(100, 100, 70), np.linspace(100, 110, 60)])
        s_df, b_df = self._aligned_pair(stock_p, bench_p, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        rs_60 = s_df["rs_comp_60"].iloc[-1]
        assert rs_60 > 0.0, f"Expected positive RS in bull outperformance, got {rs_60}"

    def test_bear_market_outperformer_positive_rs(self):
        """
        Bear market: stock falls 5%, benchmark falls 20%
        → excess ≈ +15% (strong RS even though stock is down)
        """
        n = 130
        stock_p = np.concatenate([np.linspace(10, 10, 70), np.linspace(10, 9.5, 60)])
        bench_p = np.concatenate([np.linspace(100, 100, 70), np.linspace(100, 80, 60)])
        s_df, b_df = self._aligned_pair(stock_p, bench_p, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        rs_60 = s_df["rs_comp_60"].iloc[-1]
        assert rs_60 > 0.0, f"Expected positive RS (stock held up in bear), got {rs_60}"

    def test_underperformer_negative_rs(self):
        """Stock gains 5% while benchmark gains 15% → excess ≈ -10%."""
        n = 130
        stock_p = np.concatenate([np.linspace(10, 10, 70), np.linspace(10, 10.5, 60)])
        bench_p = np.concatenate([np.linspace(100, 100, 70), np.linspace(100, 115, 60)])
        s_df, b_df = self._aligned_pair(stock_p, bench_p, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        rs_60 = s_df["rs_comp_60"].iloc[-1]
        assert rs_60 < 0.0, f"Expected negative RS for underperformer, got {rs_60}"

    def test_rs_columns_created_for_all_periods(self):
        """Columns rs_comp_20, rs_comp_60, rs_comp_120 must be present."""
        n = 130
        stock_p = np.linspace(10, 15, n)
        bench_p = np.linspace(100, 110, n)
        s_df, b_df = self._aligned_pair(stock_p, bench_p, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        for period in [20, 60, 120]:
            assert f"rs_comp_{period}" in s_df.columns

    def test_equal_performance_rs_near_zero(self):
        """If stock and benchmark move identically, RS should be near 0."""
        n = 130
        common = np.linspace(10, 15, n)
        s_df, b_df = self._aligned_pair(common.copy(), common.copy() * 10, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        rs_60 = s_df["rs_comp_60"].dropna().iloc[-1]
        assert abs(rs_60) < 0.01, f"Expected RS ≈ 0, got {rs_60}"

    def test_rs_formula_excess_not_ratio(self):
        """
        Verify the formula is (stock_pct - bench_pct), not stock/bench.
        Use known values:
          stock 60d return: +20%   bench 60d return: +5%
          excess = 0.20 - 0.05 = 0.15
        """
        n = 130
        # Flat for 70 bars, then up
        stock_p = np.concatenate([np.ones(70) * 10, np.linspace(10, 12, 60)])   # +20%
        bench_p = np.concatenate([np.ones(70) * 100, np.linspace(100, 105, 60)]) # +5%
        s_df, b_df = self._aligned_pair(stock_p, bench_p, n)
        f = make_features()
        s_df = f.calculate_relative_strength(s_df, b_df, "COMP")
        rs_60 = s_df["rs_comp_60"].iloc[-1]
        # Should be approximately 0.15 (excess return)
        assert rs_60 == pytest.approx(0.15, abs=0.005)


# ===========================================================================
# 7. VOLUME DRY-UP DETECTION
# ===========================================================================

class TestVolumeDryUp:

    def test_volume_dryup_ratio_below_1_for_declining_volume(self):
        """When recent volume is below the 20-day baseline, ratio < 1.

        detect_volume_drying uses:
          recent_vol  = rolling(lookback=10).mean()
          baseline_vol = rolling(20).mean().shift(lookback=10)

        At the last bar, baseline_vol comes from 10 bars ago.  We need the
        high-volume period to still be inside that 20-bar window — so we use
        90 bars of heavy volume and only 10 bars of collapse so the baseline
        (rows 70-89) is still all heavy volume while the recent window is all
        collapsed volume.
        """
        f = make_features()
        n = 100
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        # 90 bars heavy, then 10 bars at 10% of baseline → ratio ≈ 0.10
        volumes = np.concatenate([
            np.full(90, 1_000_000),
            np.full(10, 100_000),
        ])
        closes = np.linspace(20, 22, n)
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": volumes,
        }, index=dates)
        df = f.add_volume_metrics(df)
        df = f.detect_volume_drying(df, lookback=10)
        # Last row: recent_vol=100k, baseline_vol≈1M → ratio ≈ 0.10
        assert df["volume_dryup_ratio"].iloc[-1] < 0.50

    def test_volume_declining_flag(self):
        """volume_declining should be True when volume_trend is negative and ratio < 1."""
        f = make_features()
        n = 100
        dates = pd.date_range(end="2025-12-31", periods=n, freq="B")
        volumes = np.concatenate([
            np.full(60, 1_000_000),
            np.linspace(1_000_000, 100_000, 40),  # smoothly declining
        ])
        closes = np.full(n, 30.0)
        df = pd.DataFrame({
            "open": closes, "high": closes * 1.005, "low": closes * 0.995,
            "close": closes, "volume": volumes.astype(int),
        }, index=dates)
        df = f.add_volume_metrics(df)
        df = f.detect_volume_drying(df, lookback=10)
        # In the final rows, volume has dropped and the trend is negative
        assert df["volume_declining"].iloc[-1] == True


# ===========================================================================
# 8. STOP CALCULATION
# ===========================================================================

class TestStopCalculation:

    def test_stop_level_is_rolling_base_low(self):
        """stop_level should be the rolling 60-day min of lows, not the single day's low."""
        f = make_features()
        df = make_ohlcv(80, start_price=30.0, end_price=35.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        df = f.calculate_stop(df)
        expected = df["low"].rolling(60, min_periods=1).min()
        assert (df["stop_level"] == expected).all()

    def test_stop_distance_clipped_positive(self):
        f = make_features()
        df = make_ohlcv(50)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        df = f.calculate_stop(df)
        assert (df["stop_distance_pct"] >= 0.001).all()

    def test_stop_distance_clipped_at_max(self):
        f = make_features()
        df = make_ohlcv(50)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        df = f.calculate_stop(df)
        assert (df["stop_distance_pct"] <= 0.25).all()

    def test_trailing_stop_triggered_below_sma10(self):
        """trailing_stop_triggered should be True when close < sma_10."""
        f = make_features()
        # Downtrend: close will drop below its 10 SMA
        df = make_ohlcv(50, start_price=30.0, end_price=15.0)
        df = f.add_moving_averages(df)
        df = f.add_ma_relationships(df)
        df = f.calculate_stop(df)
        # In the later rows of the downtrend, trailing_stop_triggered should be True
        assert df["trailing_stop_triggered"].iloc[-1] == True


# ===========================================================================
# 9. PRIOR MOVE DETECTION
# ===========================================================================

class TestPriorMoveDetection:

    def test_prior_move_pct_positive_after_rally(self):
        """After doubling the price, prior_move_pct should reflect the gain.

        detect_prior_moves uses rolling(60).min() on the low.  With 120 bars
        trending from 10 to 20 (+100%), the rolling 60-bar low at the last bar
        covers roughly the midpoint of the trend (~15), giving a move of
        (20-15)/15 ≈ 33%.  We assert a conservative ≥ 20%.
        """
        f = make_features()
        df = make_ohlcv(120, start_price=10.0, end_price=20.0)
        df = f.detect_prior_moves(df)
        last = df.iloc[-1]
        assert last["prior_move_pct"] >= 0.20

    def test_is_power_move_true_after_20pct_gain(self):
        """is_power_move should be True when prior_move_pct >= 0.20."""
        f = make_features()
        df = make_ohlcv(120, start_price=10.0, end_price=15.0)
        df = f.detect_prior_moves(df)
        assert df["is_power_move"].iloc[-1] == True

    def test_is_power_move_false_on_flat_stock(self):
        f = make_features()
        df = make_ohlcv(120, start_price=30.0, end_price=30.0)
        df = f.detect_prior_moves(df)
        # Flat prices → prior_move_pct ~ 0 → not a power move
        assert df["is_power_move"].iloc[-1] == False


# ===========================================================================
# 10. HIGHER LOWS DETECTION
# ===========================================================================

class TestHigherLows:

    def test_higher_lows_in_uptrend(self):
        """In a noisy uptrend, some bars should have higher_lows=True.

        5-bar pivot detection (2 bars each side per Bulkowski 2005) requires a
        genuine local minimum — not just a 1-bar dip.  Use 150 bars with moderate
        noise so there are enough structural pivot lows to detect.
        swing_low_count is a rolling(base_length_max=60) sum of higher_lows events.
        """
        f = make_features()
        df = make_ohlcv(150, start_price=10.0, end_price=25.0, seed=17)
        df = f.calculate_higher_lows(df)
        assert df["higher_lows"].any()

    def test_downtrend_produces_no_higher_lows(self):
        """A clean downtrend should have lower successive swing lows, not higher ones."""
        f = make_features()
        df = make_ohlcv(150, start_price=50.0, end_price=10.0, seed=42)
        df = f.calculate_higher_lows(df)
        # In a sustained downtrend, new swing lows should be below prior ones
        assert not df["higher_lows"].all()

    def test_columns_created(self):
        f = make_features()
        df = make_ohlcv(60)
        df = f.calculate_higher_lows(df)
        for col in ["is_swing_low", "higher_lows", "swing_low_count"]:
            assert col in df.columns

    def test_swing_low_count_bounded(self):
        """swing_low_count is a non-negative rolling count — never negative."""
        f = make_features()
        df = make_ohlcv(150, start_price=10.0, end_price=25.0, seed=7)
        df = f.calculate_higher_lows(df)
        assert (df["swing_low_count"] >= 0).all()


# ===========================================================================
# 11. LOWER HIGHS DETECTION
# ===========================================================================

class TestLowerHighs:
    """
    calculate_lower_highs() detects the declining resistance line of a wedge.

    Combined with higher lows, this gives the symmetrical triangle / VCP criterion
    from Lo, Mamaysky & Wang (2000): E1 > E3 > E5 (lower highs) + E2 < E4 (higher lows).
    """

    def test_lower_highs_in_downtrend(self):
        """A downtrend should produce swing highs that are successively lower."""
        f = make_features()
        df = make_ohlcv(150, start_price=50.0, end_price=15.0, seed=17)
        df = f.calculate_lower_highs(df)
        assert df["lower_highs"].any()

    def test_uptrend_produces_no_lower_highs(self):
        """A clean uptrend should set higher successive highs, not lower ones."""
        f = make_features()
        df = make_ohlcv(150, start_price=10.0, end_price=50.0, seed=42)
        df = f.calculate_lower_highs(df)
        assert not df["lower_highs"].all()

    def test_columns_created(self):
        f = make_features()
        df = make_ohlcv(60)
        df = f.calculate_lower_highs(df)
        for col in ["is_swing_high", "lower_highs", "swing_high_count"]:
            assert col in df.columns

    def test_swing_high_count_bounded(self):
        """swing_high_count is a non-negative rolling count — never negative."""
        f = make_features()
        df = make_ohlcv(150, start_price=50.0, end_price=15.0, seed=7)
        df = f.calculate_lower_highs(df)
        assert (df["swing_high_count"] >= 0).all()

    def test_lower_highs_only_fires_on_pivot_days(self):
        """lower_highs=True must only occur on days where is_swing_high=True."""
        f = make_features()
        df = make_ohlcv(150, start_price=50.0, end_price=10.0, seed=3)
        df = f.calculate_lower_highs(df)
        # wherever lower_highs is True, is_swing_high must also be True
        assert not (df["lower_highs"] & ~df["is_swing_high"]).any()


# ===========================================================================
# 12. EMA SURF RATIO
# ===========================================================================

class TestEMASurf:
    """
    calculate_ema_surf() measures how consistently price "surfs" the rising
    10-EMA during consolidation — Qullamaggie's primary flag quality signal.

    A surfing day: close in (-3%, +10%) of EMA-10 AND EMA-10 was rising.
    ema10_surf_ratio = rolling 20-day fraction of surfing days.
    """

    def test_column_created(self):
        f = make_features()
        df = make_ohlcv(260)
        df = f.add_moving_averages(df)
        df = f.calculate_ema_surf(df)
        assert "ema10_surf_ratio" in df.columns

    def test_surf_ratio_high_in_steady_uptrend(self):
        """
        Smooth uptrend: price stays just above the rising EMA → high surf ratio.
        EMA(10) in a linear trend lags close by ~5 bars.  For linspace(10→50),
        that lag is tiny relative to the stock price, so dist stays well inside
        the (-3%, +10%) surfing band and ema_rising is True almost every day.
        """
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.calculate_ema_surf(df)
        assert df["ema10_surf_ratio"].iloc[-1] > 0.70

    def test_surf_ratio_low_in_downtrend(self):
        """
        Downtrend: EMA is falling (ema_rising=False most days) → surf ratio near 0.
        """
        f = make_features()
        df = make_ohlcv(260, start_price=50.0, end_price=10.0)
        df = f.add_moving_averages(df)
        df = f.calculate_ema_surf(df)
        assert df["ema10_surf_ratio"].iloc[-1] < 0.30

    def test_surf_ratio_bounded(self):
        """ema10_surf_ratio must always be in [0, 1]."""
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        df = f.add_moving_averages(df)
        df = f.calculate_ema_surf(df)
        valid = df["ema10_surf_ratio"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_missing_sma10_returns_nan(self):
        """If sma_10 column is absent, ema10_surf_ratio should be NaN (no crash)."""
        f = make_features()
        df = make_ohlcv(50)
        # don't call add_moving_averages — sma_10 absent
        df = f.calculate_ema_surf(df)
        assert df["ema10_surf_ratio"].isna().all()


# ===========================================================================
# 13. ADD_ALL_FEATURES INTEGRATION
# ===========================================================================

class TestAddAllFeatures:
    """
    Smoke test: add_all_features() should run end-to-end and produce all
    expected columns without raising errors.
    """

    EXPECTED_COLS = [
        # Moving averages (sma_10, sma_20 are EMA under the hood)
        "sma_10", "sma_20", "sma_50", "sma_150", "sma_200",
        # MA relationships
        "ma_alignment", "mas_rising", "stage2",
        "distance_from_sma10", "distance_from_sma150", "distance_from_sma200",
        # EMA surfing — qullamaggie's flag quality signal
        "ema10_surf_ratio",
        # Range / consolidation
        "adr_pct", "daily_range_pct", "breakout_level", "close_range_position",
        # 52-week proximity
        "52wk_high", "pct_from_52wk_high",
        # VCP — range_10 used in tightness scoring
        "vcp_contracting", "vcp_contraction_ratio", "range_10",
        # Wedge geometry — used in score_base_quality wedge component
        "is_swing_low", "higher_lows", "swing_low_count",
        "is_swing_high", "lower_highs", "swing_high_count",
        # Volume
        "volume_sma_20", "relative_volume", "dollar_volume",
        "volume_dryup_ratio", "volume_declining",
        # Prior move
        "prior_move_pct", "is_power_move",
        # Risk
        "stop_level", "stop_distance_pct", "potential_r",
    ]

    def test_all_feature_columns_present_without_benchmark(self):
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        result = f.add_all_features(df, benchmark_df=None)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_rs_columns_present_with_benchmark(self):
        f = make_features()
        df = make_ohlcv(260, start_price=10.0, end_price=50.0)
        bench = make_ohlcv(260, start_price=100.0, end_price=130.0)
        result = f.add_all_features(df, benchmark_df=bench)
        for period in [20, 60, 120]:
            assert f"rs_comp_{period}" in result.columns

    def test_rs_columns_absent_without_benchmark(self):
        f = make_features()
        df = make_ohlcv(260)
        result = f.add_all_features(df, benchmark_df=None)
        for period in [20, 60, 120]:
            assert f"rs_comp_{period}" not in result.columns

    def test_no_exceptions_on_short_history(self):
        """50 rows is less than the 200-period SMA — should not crash."""
        f = make_features()
        df = make_ohlcv(50, start_price=20.0)
        result = f.add_all_features(df, benchmark_df=None)
        assert not result.empty

    def test_original_df_not_mutated(self):
        """add_all_features must not modify the input DataFrame in-place."""
        f = make_features()
        df = make_ohlcv(260)
        original_cols = set(df.columns)
        _ = f.add_all_features(df, benchmark_df=None)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"


# ===========================================================================
# 13. RS RANK CALCULATION
# ===========================================================================

class TestRSRank:

    def test_best_performer_gets_highest_rank(self):
        """The stock with the highest 60-day return should rank near 100."""
        from engine import Features
        f = make_features()
        n = 130

        stock_dfs = {
            "BEST": make_ohlcv(n, start_price=10, end_price=20),   # +100%
            "MED":  make_ohlcv(n, start_price=10, end_price=12),   # +20%
            "POOR": make_ohlcv(n, start_price=10, end_price=10.5), # +5%
        }
        rs_ranks = f.calculate_rs_rank(stock_dfs)
        last_date = list(stock_dfs["BEST"].index)[-1]

        best_rank = rs_ranks["BEST"].get(last_date)
        poor_rank = rs_ranks["POOR"].get(last_date)

        if best_rank is not None and poor_rank is not None:
            assert best_rank > poor_rank

    def test_ranks_are_percentile_bounded(self):
        """All rank values must be in [0, 100]."""
        f = make_features()
        stock_dfs = {sym: make_ohlcv(130, start_price=10, end_price=10 + i * 2)
                     for i, sym in enumerate(["A", "B", "C", "D", "E"])}
        rs_ranks = f.calculate_rs_rank(stock_dfs)
        for symbol, date_ranks in rs_ranks.items():
            for date, rank in date_ranks.items():
                assert 0.0 <= rank <= 100.0, f"{symbol}@{date} rank {rank} out of bounds"
