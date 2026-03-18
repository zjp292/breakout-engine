"""
Comprehensive regression test suite for the Scoring class.

Covers every scoring branch, every hard-filter condition, every grade and
signal boundary, aggregation, regime-multiplier gating, and batch scoring
via score_dataframe.  Any future refactor that changes scoring behaviour
(intentionally or accidentally) will fail one of these tests.

Run from project root:
    python -m pytest tests/ -v
    python -m pytest tests/test_scoring.py -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd

from config import PARAMETERS
from engine import Scoring
from models import ScoreBreakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scoring(regime_multiplier: float = 1.0) -> Scoring:
    """Return a Scoring instance with optional regime multiplier override."""
    s = Scoring(PARAMETERS.copy())
    s.regime_multiplier = regime_multiplier
    return s


def make_row(**overrides) -> pd.Series:
    """
    Build a pd.Series row with sensible defaults that passes all hard filters
    and achieves a moderate score.  Callers override individual fields to
    isolate the behaviour under test.
    """
    defaults = {
        # --- price & SMAs ---
        "close": 50.0,
        "high": 51.0,
        "low": 49.0,
        "open": 50.0,
        "sma_10": 49.0,
        "sma_20": 47.0,
        "sma_50": 44.0,
        "sma_150": 39.0,
        "sma_200": 36.0,
        # --- MA relationships ---
        "distance_from_sma10":  0.02,   # 2% above 10 SMA
        "distance_from_sma20":  0.06,
        "distance_from_sma50":  0.14,
        "distance_from_sma150": 0.28,
        "distance_from_sma200": 0.39,
        "ma_alignment": True,           # sma10 > sma20 > sma50
        "mas_rising": True,             # all short-term slopes positive
        # --- Stage 2 ---
        "stage2": True,
        # --- 52-week proximity ---
        "pct_from_52wk_high": -0.05,    # 5% below high (near breakout)
        # --- Consolidation ---
        "consol_range_60": 0.04,        # 4% range — tight but not extreme
        "consol_days": 10,              # sweet-spot flag length
        # --- VCP ---
        "vcp_contracting": True,
        "vcp_contraction_ratio": 0.30,
        # --- Prior move ---
        "prior_move_pct": 0.30,         # 30% prior move
        "days_since_power_move": 20,
        # --- Volume ---
        "dollar_volume": 50_000_000,    # $50M
        "volume_declining": True,
        "volume_dryup_ratio": 0.70,
        "adr_pct": 0.08,               # 8% ADR (above minimum)
        "relative_volume": 0.80,
        "volume_sma_20": 500_000,
        # --- Relative strength ---
        "rs_comp_20": 0.06,
        "rs_comp_60": 0.12,
        # --- Risk / reward ---
        "stop_distance_pct": 0.08,     # 8% stop / 8% ADR = 1.0x — ideal
        "stop_level": 46.0,
        "potential_r": 3.5,
        "potential_gain_pct": 0.28,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def make_ideal_row() -> pd.Series:
    """
    Row engineered to achieve the maximum possible raw score of 100.

    Component maxima (rebalanced 2026-03):
      base_quality       6 + 5 + 4  = 15
      trend_strength     5 + 4 + 3 + 3 = 15
      relative_strength  8 + 12 + 10 = 30
      volume_profile     6 + 10 + 9 = 25
      risk_reward       10 + 5      = 15
      TOTAL                         = 100
    """
    return make_row(
        # base_quality -> 15
        consol_range_60=0.01,          # tightness = 6
        consol_days=10,                # length = 5
        vcp_contracting=True,
        vcp_contraction_ratio=0.20,    # vcp = 4
        # trend_strength -> 15
        stage2=True,                   # stage = 5
        pct_from_52wk_high=-0.03,      # proximity = 4
        ma_alignment=True,
        mas_rising=True,
        distance_from_sma10=0.02,      # above_10sma + aligned + rising = 3
        prior_move_pct=0.50,
        days_since_power_move=15,      # power_move = 3
        # relative_strength -> 30 (rs_rank handled via score call)
        rs_comp_20=0.15,               # rs_20 = 8
        rs_comp_60=0.25,               # rs_60 = 12
        # volume_profile -> 25
        dollar_volume=150_000_000,     # dv = 6  (>= 10x min)
        volume_declining=True,
        volume_dryup_ratio=0.50,       # vd = 10 (< 0.60)
        adr_pct=0.12,                  # adr = 9  (>= 10%)
        # risk_reward -> 15
        # stop 0.06 / adr 0.12 = 0.5x -> stop_score = 10
        stop_distance_pct=0.06,
        potential_r=6.0,               # r_score = 5
    )


# ===========================================================================
# 1. SCORE BASE QUALITY
# ===========================================================================

class TestScoreBaseQuality:
    """
    score_base_quality() decomposes into three independent sub-scores:
      - Consolidation tightness  (0-6 pts)
      - Base length              (0-5 pts)
      - VCP contraction series   (0-4 pts)
    Max total = 15.
    """

    scoring = make_scoring()

    # --- 1a. Tightness thresholds ---

    @pytest.mark.parametrize("consol_range, expected", [
        (0.00,  6.0),  # extreme edge
        (0.01,  6.0),  # < 2%
        (0.02,  6.0),  # boundary: exactly 2%
        (0.021, 5.0),  # just over 2%, within 3%
        (0.03,  5.0),  # boundary: exactly 3%
        (0.031, 4.0),  # just over 3%, within 5%
        (0.05,  4.0),  # boundary: exactly 5%
        (0.051, 2.0),  # just over 5%, within 8%
        (0.08,  2.0),  # boundary: exactly 8%
        (0.081, 0.0),  # just over 8% — too loose
        (0.20,  0.0),  # very loose
        (1.00,  0.0),  # default missing value
    ])
    def test_tightness(self, consol_range, expected):
        row = make_row(consol_range_60=consol_range, consol_days=10,
                       vcp_contracting=False, vcp_contraction_ratio=1.0)
        score, details = self.scoring.score_base_quality(row)
        assert details["tightness"] == expected, (
            f"consol_range={consol_range} -> expected tightness {expected}, got {details['tightness']}"
        )

    # --- 1b. Base length thresholds ---

    @pytest.mark.parametrize("days, expected", [
        (0,   0.0),   # < 3 — no base
        (2,   0.0),   # < 3
        (3,   2.5),   # micro-flag lower bound
        (4,   2.5),   # micro-flag
        (5,   5.0),   # sweet-spot lower bound
        (10,  5.0),   # sweet-spot middle
        (15,  5.0),   # sweet-spot upper boundary
        (16,  4.0),   # classic VCP
        (30,  4.0),   # classic VCP upper boundary
        (31,  3.0),   # longer VCP
        (45,  3.0),   # longer VCP upper boundary
        (46,  1.5),   # extended
        (60,  1.5),   # extended upper boundary
        (61,  0.5),   # too long
        (90,  0.5),   # way too long
    ])
    def test_base_length(self, days, expected):
        row = make_row(consol_range_60=0.04, consol_days=days,
                       vcp_contracting=False, vcp_contraction_ratio=1.0)
        score, details = self.scoring.score_base_quality(row)
        assert details["base_length"] == expected, (
            f"consol_days={days} -> expected length {expected}, got {details['base_length']}"
        )

    # --- 1c. VCP contraction thresholds ---

    @pytest.mark.parametrize("contracting, ratio, expected", [
        (True,  0.10, 4.0),   # very strong (ratio <= 0.25)
        (True,  0.25, 4.0),   # boundary
        (True,  0.26, 3.0),   # just over 0.25
        (True,  0.40, 3.0),   # boundary (<= 0.40)
        (True,  0.41, 2.0),   # contracting but modest
        (True,  0.80, 2.0),   # contracting, high ratio
        (True,  0.99, 2.0),   # contracting, ratio close to 1
        (False, 0.50, 0.5),   # not contracting, partial contraction
        (False, 0.60, 0.5),   # boundary (<= 0.60)
        (False, 0.61, 0.0),   # flat or expanding
        (False, 1.00, 0.0),   # flat
    ])
    def test_vcp_contraction(self, contracting, ratio, expected):
        row = make_row(consol_range_60=0.04, consol_days=10,
                       vcp_contracting=contracting, vcp_contraction_ratio=ratio)
        score, details = self.scoring.score_base_quality(row)
        assert details["vcp_contraction"] == expected, (
            f"contracting={contracting}, ratio={ratio} -> expected vcp {expected}, got {details['vcp_contraction']}"
        )

    def test_max_score_is_15(self):
        """Perfect base: tightest range + sweet-spot length + strongest VCP."""
        row = make_row(consol_range_60=0.01, consol_days=10,
                       vcp_contracting=True, vcp_contraction_ratio=0.20)
        score, _ = self.scoring.score_base_quality(row)
        assert score == 15.0

    def test_min_score_is_0(self):
        """Worst possible base: loose range, no days, no VCP."""
        row = make_row(consol_range_60=0.20, consol_days=0,
                       vcp_contracting=False, vcp_contraction_ratio=1.0)
        score, _ = self.scoring.score_base_quality(row)
        assert score == 0.0

    def test_missing_fields_default_gracefully(self):
        """Missing optional fields should not raise — defaults yield a finite score."""
        row = pd.Series({"close": 50.0})  # only close, everything else missing
        score, details = self.scoring.score_base_quality(row)
        assert 0.0 <= score <= 15.0


# ===========================================================================
# 2. SCORE TREND STRENGTH
# ===========================================================================

class TestScoreTrendStrength:
    """
    score_trend_strength() decomposes into four sub-scores:
      - Stage 2 MA structure   (0-5 pts)
      - 52-wk high proximity   (0-4 pts)
      - Short-term MA stack    (0-3 pts)
      - Prior power move       (0-3 pts)
    Max total = 15.
    """

    scoring = make_scoring()

    # --- 2a. Stage 2 ---

    def test_stage2_full(self):
        row = make_row(stage2=True)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 5.0

    def test_stage2_price_above_both_long_smas(self):
        """dist_150 > 0, dist_200 > 0, but stage2=False -> 3 pts."""
        row = make_row(stage2=False, distance_from_sma150=0.05, distance_from_sma200=0.10)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 3.0

    def test_stage2_above_200_only(self):
        """dist_150 <= 0, dist_200 > 0 -> 1.5 pts."""
        row = make_row(stage2=False, distance_from_sma150=-0.02, distance_from_sma200=0.05)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 1.5

    def test_stage2_nan_150_with_dist200_positive(self):
        """dist_150 is NaN but dist_200 > 0 -> 1.5 pts (second elif)."""
        row = make_row(stage2=False, distance_from_sma150=np.nan, distance_from_sma200=0.05)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 1.5

    def test_stage2_nan_150_ma_aligned(self):
        """Insufficient history (dist_150=NaN, dist_200=NaN), but MAs aligned -> 2 pts."""
        row = make_row(stage2=False,
                       distance_from_sma150=np.nan, distance_from_sma200=np.nan,
                       ma_alignment=True)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 2.0

    def test_stage2_nan_150_not_aligned(self):
        """Insufficient history and MAs not aligned -> 0 pts."""
        row = make_row(stage2=False,
                       distance_from_sma150=np.nan, distance_from_sma200=np.nan,
                       ma_alignment=False)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 0.0

    def test_stage2_below_both_long_smas(self):
        """Below 150 and 200 SMA -> 0 pts."""
        row = make_row(stage2=False,
                       distance_from_sma150=-0.05, distance_from_sma200=-0.10)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 0.0

    def test_stage2_above_150_below_200(self):
        """dist_150 > 0 but dist_200 < 0 -> 0 pts."""
        row = make_row(stage2=False,
                       distance_from_sma150=0.05, distance_from_sma200=-0.03)
        _, d = self.scoring.score_trend_strength(row)
        assert d["stage2"] == 0.0

    # --- 2b. 52-week high proximity ---

    @pytest.mark.parametrize("pct_from_high, expected", [
        ( 0.00, 4.0),   # at the 52wk high
        (-0.03, 4.0),   # within 5%
        (-0.05, 4.0),   # boundary: exactly -5%
        (-0.06, 3.5),   # just below -5%, within -10%
        (-0.10, 3.5),   # boundary: exactly -10%
        (-0.11, 2.5),   # within -15%
        (-0.15, 2.5),   # boundary: exactly -15%
        (-0.16, 1.5),   # within -20%
        (-0.20, 1.5),   # boundary: exactly -20%
        (-0.21, 0.5),   # within -25%
        (-0.25, 0.5),   # boundary: exactly -25%
        (-0.26, 0.0),   # >25% below
        (-0.50, 0.0),   # far below
    ])
    def test_proximity_to_52wk_high(self, pct_from_high, expected):
        row = make_row(pct_from_52wk_high=pct_from_high)
        _, d = self.scoring.score_trend_strength(row)
        assert d["proximity_to_high"] == expected, (
            f"pct_from_high={pct_from_high} -> expected {expected}, got {d['proximity_to_high']}"
        )

    # --- 2c. Short-term MA structure ---

    @pytest.mark.parametrize("ma_alignment, mas_rising, dist_10, expected", [
        (True,  True,   0.02,  3.0),   # perfect: aligned + rising + above
        (True,  True,  -0.02,  3.0),   # at -2% = boundary for above_10sma
        (True,  True,  -0.025, 2.0),   # aligned + rising, but below 10 SMA
        (True,  False,  0.02,  1.5),   # aligned + above but not rising
        (True,  False, -0.05,  1.0),   # aligned only
        (False, False,  0.05,  0.0),   # poor structure — need sma_10 > sma_20 check
    ])
    def test_ma_structure(self, ma_alignment, mas_rising, dist_10, expected):
        row = make_row(ma_alignment=ma_alignment, mas_rising=mas_rising,
                       distance_from_sma10=dist_10,
                       # ensure sma_10 <= sma_20 for the False/False case
                       sma_10=48.0, sma_20=49.0)
        _, d = self.scoring.score_trend_strength(row)
        assert d["ma_structure"] == expected

    def test_ma_partial_alignment_10_above_20(self):
        """sma_10 > sma_20 but ma_alignment=False -> 0.5 pt."""
        row = make_row(ma_alignment=False, mas_rising=False,
                       sma_10=50.0, sma_20=48.0)
        _, d = self.scoring.score_trend_strength(row)
        assert d["ma_structure"] == 0.5

    # --- 2d. Prior power move ---

    @pytest.mark.parametrize("prior_move, days_since, expected", [
        (0.40, 30,  3.0),   # 40%+ and within 30 days
        (0.40, 31,  2.5),   # 40% but 31 days -> falls to 30%+ tier (<=45d)
        (0.50, 30,  3.0),   # 50%+ within 30 days
        (0.30, 45,  2.5),   # 30%+ within 45 days
        (0.30, 46,  2.0),   # 30% but 46 days -> falls to 20%+ tier (<=60d)
        (0.20, 60,  2.0),   # 20%+ within 60 days
        (0.20, 61,  1.0),   # 20% but 61 days -> >= 15% tier (no day limit)
        (0.15, 999, 1.0),   # 15%+ modest (no recency requirement)
        (0.14, 999, 0.0),   # below 15% — no meaningful prior move
        (0.00, 999, 0.0),   # no move
    ])
    def test_prior_power_move(self, prior_move, days_since, expected):
        row = make_row(prior_move_pct=prior_move, days_since_power_move=days_since)
        _, d = self.scoring.score_trend_strength(row)
        assert d["prior_power_move"] == expected, (
            f"prior_move={prior_move}, days={days_since} -> expected {expected}, got {d['prior_power_move']}"
        )

    def test_max_score_is_15(self):
        """Ideal trend: Stage 2, within 5% of high, perfect MAs, 50% move."""
        row = make_row(
            stage2=True,
            pct_from_52wk_high=-0.03,
            ma_alignment=True, mas_rising=True, distance_from_sma10=0.02,
            prior_move_pct=0.50, days_since_power_move=15,
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 15.0

    def test_min_score_is_0(self):
        """Worst trend: below all MAs, far from high, no prior move."""
        row = make_row(
            stage2=False,
            distance_from_sma150=-0.10, distance_from_sma200=-0.15,
            pct_from_52wk_high=-0.60,
            ma_alignment=False, mas_rising=False,
            sma_10=45.0, sma_20=46.0,   # sma10 < sma20 -> no partial credit
            prior_move_pct=0.05, days_since_power_move=999,
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 0.0


# ===========================================================================
# 3. SCORE RELATIVE STRENGTH
# ===========================================================================

class TestScoreRelativeStrength:
    """
    score_relative_strength() decomposes into:
      - 20-day RS  (0-8 pts)
      - 60-day RS  (0-12 pts)
      - RS rank    (0-10 pts, only when rs_rank supplied)
    Max total = 30.
    """

    scoring = make_scoring()

    # --- 3a. 20-day RS thresholds ---

    @pytest.mark.parametrize("rs_20, expected", [
        ( 0.10, 8.0),   # +10%+ exceptional
        ( 0.15, 8.0),   # above 10%
        ( 0.05, 6.0),   # +5-10%
        ( 0.10 - 1e-9, 6.0),  # just under 10%
        ( 0.02, 4.0),   # +2-5%
        ( 0.05 - 1e-9, 4.0),  # just under 5%
        ( 0.00, 1.5),   # neutral
        ( 0.02 - 1e-9, 1.5),  # just under 2%
        (-0.01, 0.0),   # underperforming
        (-0.20, 0.0),   # deeply underperforming
    ])
    def test_rs_20(self, rs_20, expected):
        row = make_row(rs_comp_20=rs_20, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_20_day"] == expected

    # --- 3b. 60-day RS thresholds ---

    @pytest.mark.parametrize("rs_60, expected", [
        ( 0.20, 12.0),
        ( 0.30, 12.0),
        ( 0.15, 10.0),
        ( 0.20 - 1e-9, 10.0),
        ( 0.10, 8.0),
        ( 0.15 - 1e-9, 8.0),
        ( 0.05, 5.0),
        ( 0.10 - 1e-9, 5.0),
        ( 0.00, 1.5),
        ( 0.05 - 1e-9, 1.5),
        (-0.01, 0.0),
        (-0.30, 0.0),
    ])
    def test_rs_60(self, rs_60, expected):
        row = make_row(rs_comp_20=0.0, rs_comp_60=rs_60)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_60_day"] == expected

    # --- 3c. RS rank percentile ---

    @pytest.mark.parametrize("rs_rank, expected", [
        (90.0, 10.0),   # top 10%
        (95.0, 10.0),
        (80.0, 8.0),    # top 20%
        (89.9, 8.0),
        (70.0, 6.0),    # top 30%
        (79.9, 6.0),
        (60.0, 3.0),    # top 40%
        (69.9, 3.0),
        (59.9, 0.0),    # below 60th percentile
        ( 0.0, 0.0),
    ])
    def test_rs_rank(self, rs_rank, expected):
        row = make_row(rs_comp_20=0.0, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=rs_rank)
        assert d["rs_rank"] == expected

    def test_rs_rank_none_gives_zero_rank_points(self):
        """When rs_rank is not provided, rank component = 0."""
        row = make_row(rs_comp_20=0.0, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_rank"] == 0.0

    def test_max_score_is_30(self):
        row = make_row(rs_comp_20=0.15, rs_comp_60=0.25)
        score, _ = self.scoring.score_relative_strength(row, rs_rank=95.0)
        assert score == 30.0

    def test_min_score_is_0(self):
        row = make_row(rs_comp_20=-0.20, rs_comp_60=-0.30)
        score, _ = self.scoring.score_relative_strength(row, rs_rank=10.0)
        assert score == 0.0


# ===========================================================================
# 4. SCORE VOLUME PROFILE
# ===========================================================================

class TestScoreVolumeProfile:
    """
    score_volume_profile() decomposes into:
      - Dollar volume      (0-6 pts)
      - Volume dry-up      (0-10 pts) — single source of truth
      - ADR %              (0-9 pts)
    Max total = 25.
    """

    scoring = make_scoring()
    MIN_DV = PARAMETERS["dollar_volume_min"]  # 10_000_000

    # --- 4a. Dollar volume ---

    @pytest.mark.parametrize("dollar_vol, expected", [
        (MIN_DV * 10,     6.0),   # >= $100M
        (MIN_DV * 10 + 1, 6.0),
        (MIN_DV * 5,      5.0),   # >= $50M
        (MIN_DV * 10 - 1, 5.0),
        (MIN_DV * 2,      4.0),   # >= $20M
        (MIN_DV * 5 - 1,  4.0),
        (MIN_DV,          2.5),   # meets minimum ($10M)
        (MIN_DV * 2 - 1,  2.5),
        (MIN_DV - 1,      0.0),   # below minimum
        (0,               0.0),
    ])
    def test_dollar_volume(self, dollar_vol, expected):
        row = make_row(dollar_volume=dollar_vol,
                       volume_declining=False, volume_dryup_ratio=1.0, adr_pct=0.0)
        score, d = self.scoring.score_volume_profile(row)
        assert d["dollar_volume"] == expected

    # --- 4b. Volume dry-up ---

    @pytest.mark.parametrize("declining, ratio, expected", [
        (True,  0.50, 10.0),   # very strong (ratio < 0.60)
        (True,  0.59, 10.0),
        (True,  0.60,  7.5),   # solid (0.60 <= ratio < 0.75)
        (True,  0.74,  7.5),
        (True,  0.75,  5.0),   # moderate (0.75 <= ratio < 0.90)
        (True,  0.89,  5.0),
        # declining=True, ratio=0.90: falls to `elif ratio < 1.0` -> 2.5
        (True,  0.90,  2.5),
        (False, 0.70,  2.5),   # not declining but ratio < 1.0
        (False, 0.99,  2.5),   # not declining but ratio just below 1
        (False, 1.00,  0.0),   # no contraction at all
        (False, 1.20,  0.0),   # expanding volume
    ])
    def test_volume_dryup(self, declining, ratio, expected):
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=declining, volume_dryup_ratio=ratio)
        score, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == expected, (
            f"declining={declining}, ratio={ratio} -> expected {expected}, got {d['volume_contraction']}"
        )

    # --- 4c. ADR % ---

    @pytest.mark.parametrize("adr_pct, expected", [
        (0.10, 9.0),    # 10%+ high-octane
        (0.15, 9.0),
        (0.08, 7.0),    # 8%+
        (0.10 - 1e-9, 7.0),
        (0.06, 5.0),    # 6%+
        (0.08 - 1e-9, 5.0),
        (0.05, 3.0),    # at minimum (5%)
        (0.06 - 1e-9, 3.0),
        (0.049, 0.0),   # below minimum
        (0.00,  0.0),
    ])
    def test_adr(self, adr_pct, expected):
        row = make_row(dollar_volume=0, volume_declining=False, volume_dryup_ratio=1.0,
                       adr_pct=adr_pct)
        score, d = self.scoring.score_volume_profile(row)
        assert d["adr"] == expected

    def test_max_score_is_25(self):
        row = make_row(dollar_volume=150_000_000,
                       volume_declining=True, volume_dryup_ratio=0.50,
                       adr_pct=0.12)
        score, _ = self.scoring.score_volume_profile(row)
        assert score == 25.0

    def test_min_score_is_0(self):
        row = make_row(dollar_volume=0, volume_declining=False, volume_dryup_ratio=1.5,
                       adr_pct=0.0)
        score, _ = self.scoring.score_volume_profile(row)
        assert score == 0.0

    def test_volume_dryup_not_duplicated_in_base_quality(self):
        """
        Volume dry-up must live exclusively in score_volume_profile.
        score_base_quality must NOT include a volume-related detail key.
        """
        row = make_row(volume_declining=True, volume_dryup_ratio=0.50)
        _, base_details = self.scoring.score_base_quality(row)
        assert "volume_contraction" not in base_details
        assert "volume_dryup" not in base_details
        assert "volume" not in " ".join(base_details.keys())


# ===========================================================================
# 5. SCORE RISK/REWARD
# ===========================================================================

class TestScoreRiskReward:
    """
    score_risk_reward() decomposes into:
      - Stop distance vs ADR  (0-10 pts)
      - R-multiple potential  (0-5 pts)
    Max total = 15.
    """

    scoring = make_scoring()

    # --- 5a. Stop vs ADR ratio ---
    # stop_in_adr = stop_distance / max(adr_pct, 0.01)

    @pytest.mark.parametrize("stop_dist, adr, expected_stop_score", [
        # Ideal range [0.5x, 1.0x ADR] -> 10 pts
        (0.05, 0.10, 10.0),   # 0.5x ADR exactly (lower boundary)
        (0.075, 0.10, 10.0),  # 0.75x ADR
        (0.10, 0.10, 10.0),   # 1.0x ADR (upper boundary)
        # Too tight < 0.5x -> 3 pts
        (0.03, 0.10, 3.0),    # 0.3x
        (0.001, 0.10, 3.0),   # 0.01x
        # [1.0x, 1.5x] -> 8 pts
        (0.11, 0.10, 8.0),    # 1.1x
        (0.15, 0.10, 8.0),    # 1.5x (boundary)
        # [1.5x, 2.0x] -> 5 pts
        (0.16, 0.10, 5.0),    # 1.6x
        (0.20, 0.10, 5.0),    # 2.0x (boundary)
        # [2.0x, 2.5x] -> 3 pts
        (0.21, 0.10, 3.0),    # 2.1x
        (0.25, 0.10, 3.0),    # 2.5x (boundary)
        # > 2.5x -> 0 pts
        (0.26, 0.10, 0.0),    # 2.6x
        (0.50, 0.10, 0.0),    # extreme
    ])
    def test_stop_vs_adr(self, stop_dist, adr, expected_stop_score):
        row = make_row(stop_distance_pct=stop_dist, adr_pct=adr, potential_r=0.0)
        score, d = self.scoring.score_risk_reward(row)
        assert d["stop_vs_adr"] == expected_stop_score, (
            f"stop={stop_dist}, adr={adr} -> in_adr={stop_dist/adr:.2f} "
            f"expected {expected_stop_score}, got {d['stop_vs_adr']}"
        )

    def test_adr_zero_uses_floor(self):
        """adr=0 should not divide by zero — uses max(adr, 0.01)."""
        row = make_row(stop_distance_pct=0.05, adr_pct=0.0, potential_r=0.0)
        score, d = self.scoring.score_risk_reward(row)
        # 0.05 / 0.01 = 5.0x -> > 2.5 -> stop_score = 0
        assert d["stop_vs_adr"] == 0.0

    # --- 5b. R-multiple ---

    min_r = PARAMETERS["risk_reward_min"]  # 3.0

    @pytest.mark.parametrize("r_multiple, expected_r_score", [
        (5.0, 5.0),          # 5R+ exceptional
        (6.0, 5.0),
        (4.0, 4.0),          # 4R excellent
        (5.0 - 1e-9, 4.0),
        (min_r, 3.0),        # min_r (3.0) minimum acceptable
        (4.0 - 1e-9, 3.0),
        (2.0, 1.5),          # 2R below target
        (min_r - 1e-9, 1.5),
        (1.9, 0.0),          # < 2R poor
        (0.0, 0.0),
    ])
    def test_r_multiple(self, r_multiple, expected_r_score):
        # Use stop/adr in ideal range to isolate r-multiple scoring
        row = make_row(stop_distance_pct=0.05, adr_pct=0.10, potential_r=r_multiple)
        score, d = self.scoring.score_risk_reward(row)
        assert d["r_multiple"] == expected_r_score

    def test_max_score_is_15(self):
        row = make_row(stop_distance_pct=0.05, adr_pct=0.10, potential_r=6.0)
        score, _ = self.scoring.score_risk_reward(row)
        assert score == 15.0

    def test_min_score_is_0(self):
        row = make_row(stop_distance_pct=0.50, adr_pct=0.10, potential_r=0.0)
        score, _ = self.scoring.score_risk_reward(row)
        assert score == 0.0


# ===========================================================================
# 6. HARD FILTERS
# ===========================================================================

class TestApplyHardFilters:
    """
    apply_hard_filters() must independently catch each of the 6 conditions.
    A row that fails one filter should still report all other failures.
    """

    scoring = make_scoring()

    def _passing_row(self) -> pd.Series:
        """Minimal row that satisfies all 6 hard filters."""
        return make_row(
            close=10.0,
            sma_50=9.0,               # close > sma_50
            dollar_volume=15_000_000, # > $10M
            adr_pct=0.06,             # > 5%
            stop_distance_pct=0.06,   # 6% stop / 6% ADR = 1.0x (< 3x)
            pct_from_52wk_high=-0.10, # within 30%
        )

    def test_passing_row_has_no_failures(self):
        passes, failures = self.scoring.apply_hard_filters(self._passing_row())
        assert passes is True
        assert failures == []

    # --- Filter 1: Minimum price ---

    @pytest.mark.parametrize("price, should_pass", [
        (5.00, True),     # exactly at minimum
        (4.99, False),    # just below minimum
        (100.0, True),
        (0.01, False),
    ])
    def test_filter_price(self, price, should_pass):
        row = self._passing_row()
        row["close"] = price
        row["sma_50"] = price - 1.0   # ensure close > sma_50
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("Price" in f for f in failures)

    # --- Filter 2: Minimum dollar volume ---

    @pytest.mark.parametrize("dv, should_pass", [
        (10_000_000,     True),   # exactly at minimum
        ( 9_999_999,     False),  # one dollar below
        (100_000_000,    True),
        (0,              False),
    ])
    def test_filter_dollar_volume(self, dv, should_pass):
        row = self._passing_row()
        row["dollar_volume"] = dv
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("Dollar volume" in f for f in failures)

    # --- Filter 3: Price above 50 SMA ---

    def test_filter_above_50sma_passes(self):
        row = self._passing_row()
        row["close"] = 50.0
        row["sma_50"] = 49.9
        passes, _ = self.scoring.apply_hard_filters(row)
        assert passes is True

    def test_filter_above_50sma_fails(self):
        row = self._passing_row()
        row["close"] = 49.0
        row["sma_50"] = 50.0
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("50 SMA" in f for f in failures)

    def test_filter_equal_to_50sma_fails(self):
        """close == sma_50 means price < sma_50 is False — should pass."""
        row = self._passing_row()
        row["close"] = 50.0
        row["sma_50"] = 50.0
        passes, failures = self.scoring.apply_hard_filters(row)
        assert not any("50 SMA" in f for f in failures)

    # --- Filter 4: Minimum ADR ---

    @pytest.mark.parametrize("adr, should_pass", [
        (0.050, True),    # exactly at minimum
        (0.049, False),   # just below
        (0.100, True),
    ])
    def test_filter_adr(self, adr, should_pass):
        row = self._passing_row()
        row["adr_pct"] = adr
        row["stop_distance_pct"] = adr * 1.0  # 1x ADR
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("ADR" in f for f in failures)

    # --- Filter 5: Stop distance relative to ADR ---

    def test_filter_stop_exactly_3x_adr_passes(self):
        """stop = 3x ADR exactly: 0.15 > 0.15 is False -> passes."""
        row = self._passing_row()
        row["adr_pct"] = 0.05
        row["stop_distance_pct"] = 0.15  # exactly 3.0x ADR
        passes, failures = self.scoring.apply_hard_filters(row)
        assert not any("Stop distance" in f and "ADR" in f for f in failures)

    def test_filter_stop_over_3x_adr_fails(self):
        """stop = 3.01x ADR -> fails."""
        row = self._passing_row()
        row["adr_pct"] = 0.05
        row["stop_distance_pct"] = 0.151  # > 0.15 = 3x ADR
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("Stop distance" in f for f in failures)

    def test_filter_stop_too_tight_fails(self):
        """stop < 0.001 is rejected as irrationally tight."""
        row = self._passing_row()
        row["stop_distance_pct"] = 0.0005
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("too tight" in f for f in failures)

    # --- Filter 6: Within 30% of 52-week high ---

    @pytest.mark.parametrize("pct_from_high, should_pass", [
        (-0.30, True),    # exactly at -30%: -0.30 < -0.30 is False -> passes
        (-0.301, False),  # just over 30% below
        (-0.10, True),
        (-0.50, False),
        (-1.00, False),
    ])
    def test_filter_52wk_high_proximity(self, pct_from_high, should_pass):
        row = self._passing_row()
        row["pct_from_52wk_high"] = pct_from_high
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("52wk high" in f for f in failures)

    def test_filter_52wk_nan_skipped(self):
        """NaN pct_from_52wk_high should not trigger the filter."""
        row = self._passing_row()
        row["pct_from_52wk_high"] = np.nan
        passes, failures = self.scoring.apply_hard_filters(row)
        assert not any("52wk high" in f for f in failures)

    def test_multiple_failures_all_reported(self):
        """When several filters fail, all failure reasons should be present."""
        row = make_row(
            close=2.0,            # fails price
            dollar_volume=100,    # fails dollar volume
            sma_50=999.0,         # fails above-50-SMA
            adr_pct=0.01,         # fails ADR
            stop_distance_pct=1.0, # fails stop (> 3x ADR)
            pct_from_52wk_high=-0.90,  # fails 52wk high
        )
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert len(failures) >= 4


# ===========================================================================
# 7. GRADE AND SIGNAL THRESHOLDS
# ===========================================================================

class TestGetGrade:
    """get_grade() converts a raw score to a letter grade."""

    scoring = make_scoring()

    @pytest.mark.parametrize("raw_score, expected_grade", [
        (100.0, "A+"),
        ( 90.0, "A+"),  # boundary
        ( 89.9, "A"),
        ( 85.0, "A"),   # boundary
        ( 84.9, "A-"),
        ( 80.0, "A-"),  # boundary
        ( 79.9, "B+"),
        ( 75.0, "B+"),  # boundary
        ( 74.9, "B"),
        ( 70.0, "B"),   # boundary
        ( 69.9, "C+"),
        ( 65.0, "C+"),  # boundary
        ( 64.9, "C"),
        ( 60.0, "C"),   # boundary
        ( 59.9, "D"),
        (  0.0, "D"),
    ])
    def test_grade_boundaries(self, raw_score, expected_grade):
        assert self.scoring.get_grade(raw_score) == expected_grade


class TestGetSignalStrength:
    """get_signal_strength() maps a total (regime-adjusted) score to an action."""

    scoring = make_scoring()

    @pytest.mark.parametrize("total_score, expected_signal", [
        (100.0, "STRONG BUY - Alert"),
        ( 80.0, "STRONG BUY - Alert"),   # boundary
        ( 79.9, "BUY - Watch Closely"),
        ( 70.0, "BUY - Watch Closely"),  # boundary
        ( 69.9, "HOLD - Monitor"),
        ( 60.0, "HOLD - Monitor"),       # boundary
        ( 59.9, "PASS"),
        (  0.0, "PASS"),
    ])
    def test_signal_boundaries(self, total_score, expected_signal):
        assert self.scoring.get_signal_strength(total_score) == expected_signal


# ===========================================================================
# 8. CALCULATE TOTAL SCORE — AGGREGATION & REGIME GATING
# ===========================================================================

class TestCalculateTotalScore:
    """
    calculate_total_score() must:
    - Sum the five component scores correctly
    - Apply the regime_multiplier to compute total (not raw_total)
    - Return a ScoreBreakdown with correct field values
    - Include all expected detail keys
    """

    def test_ideal_row_raw_score_100(self):
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        assert bd.raw_total == 100.0, f"Expected 100, got {bd.raw_total}"

    def test_ideal_row_total_equals_raw_when_multiplier_is_1(self):
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        assert bd.total == bd.raw_total

    def test_regime_multiplier_gates_total(self):
        """With multiplier=0.5, total should be half of raw_total."""
        scoring = make_scoring(regime_multiplier=0.5)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        assert bd.total == pytest.approx(bd.raw_total * 0.5)

    def test_regime_multiplier_075_applies_correctly(self):
        scoring = make_scoring(regime_multiplier=0.75)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        assert bd.total == pytest.approx(bd.raw_total * 0.75, rel=1e-6)

    def test_component_sum_equals_raw_total(self):
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=80.0)
        component_sum = (bd.base_quality + bd.trend_strength +
                         bd.relative_strength + bd.volume_profile + bd.risk_reward)
        assert component_sum == pytest.approx(bd.raw_total, rel=1e-6)

    def test_returns_score_breakdown_dataclass(self):
        scoring = make_scoring()
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=50.0)
        assert isinstance(bd, ScoreBreakdown)

    def test_details_dict_has_expected_keys(self):
        """All five detail namespaces must appear in the combined dict."""
        scoring = make_scoring()
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=50.0)
        keys = bd.details.keys()
        assert any("base_" in k for k in keys)
        assert any("trend_" in k for k in keys)
        assert any("rs_" in k for k in keys)
        assert any("volume_" in k for k in keys)
        assert any("rr_" in k for k in keys)

    def test_to_dict_round_trips(self):
        scoring = make_scoring()
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=50.0)
        d = bd.to_dict()
        assert d["raw_total"] == bd.raw_total
        assert d["total"] == bd.total

    def test_score_bounded_between_0_and_100(self):
        """
        Scores must never exceed 100 on an ideal row (raw) or go below 0
        on a worst-case row.
        """
        scoring = make_scoring(regime_multiplier=1.0)
        ideal = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert 0.0 <= ideal.raw_total <= 100.0
        assert 0.0 <= ideal.total <= 100.0

        worst = make_row(
            consol_range_60=1.0, consol_days=0,
            stage2=False, distance_from_sma150=-0.50, distance_from_sma200=-0.50,
            pct_from_52wk_high=-0.99, ma_alignment=False, mas_rising=False,
            sma_10=40.0, sma_20=41.0, prior_move_pct=0.0, days_since_power_move=999,
            rs_comp_20=-0.50, rs_comp_60=-0.50,
            dollar_volume=0, volume_declining=False, volume_dryup_ratio=2.0, adr_pct=0.0,
            stop_distance_pct=0.50, potential_r=0.0,
            vcp_contracting=False, vcp_contraction_ratio=1.0,
        )
        bad = scoring.calculate_total_score(worst, rs_rank=5.0)
        assert bad.raw_total == 0.0
        assert bad.total == 0.0


# ===========================================================================
# 9. SCORE DATAFRAME — BATCH SCORING INTEGRATION
# ===========================================================================

class TestScoreDataframe:
    """
    score_dataframe() applies hard filters row-by-row, skips failing rows,
    and annotates passing rows with component scores, grade, and signal.
    """

    def _build_feature_df(self) -> pd.DataFrame:
        """
        Construct a small DataFrame (5 rows) with all feature columns.
        Row 0: fails price filter ($1 stock)
        Rows 1-4: pass all filters with varying quality.
        """
        dates = pd.date_range("2025-01-06", periods=5, freq="B")
        rows = []
        for i in range(5):
            r = make_row(
                close=1.0 if i == 0 else 50.0,  # row 0 fails price
                sma_50=0.5 if i == 0 else 44.0,
                dollar_volume=100 if i == 0 else 50_000_000,
            )
            rows.append(r)
        df = pd.DataFrame(rows, index=dates)
        return df

    def test_failing_rows_get_zero_scores(self):
        scoring = make_scoring()
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        # Row 0 fails filters — scores stay at 0
        assert not scored.iloc[0]["passes_filters"]
        assert scored.iloc[0]["total_score"] == 0.0

    def test_passing_rows_get_nonzero_scores(self):
        scoring = make_scoring()
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        for i in range(1, 5):
            assert scored.iloc[i]["passes_filters"]
            assert scored.iloc[i]["total_score"] > 0.0

    def test_score_columns_exist(self):
        scoring = make_scoring()
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        expected_cols = [
            "score_base_quality", "score_trend_strength", "score_relative_strength",
            "score_volume_profile", "score_risk_reward",
            "raw_score", "total_score", "grade", "signal", "passes_filters",
        ]
        for col in expected_cols:
            assert col in scored.columns, f"Missing column: {col}"

    def test_grade_uses_raw_score_not_total(self):
        """
        Grade reflects pure setup quality (raw_score).
        In a downtrend (multiplier < 1), total < raw; grade should still
        reflect the raw score bracket.
        """
        scoring = make_scoring(regime_multiplier=0.7)
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        for i in range(1, 5):
            if scored.iloc[i]["passes_filters"]:
                raw = scored.iloc[i]["raw_score"]
                grade = scored.iloc[i]["grade"]
                assert grade == scoring.get_grade(raw)

    def test_signal_uses_total_score(self):
        """Signal should be based on regime-adjusted total, not raw."""
        scoring = make_scoring(regime_multiplier=0.7)
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        for i in range(1, 5):
            if scored.iloc[i]["passes_filters"]:
                total = scored.iloc[i]["total_score"]
                signal = scored.iloc[i]["signal"]
                assert signal == scoring.get_signal_strength(total)

    def test_rs_ranks_applied_per_date(self):
        """RS ranks dict (date -> percentile) must feed into per-row scoring."""
        scoring = make_scoring()
        df = self._build_feature_df()
        # Assign top percentile (95) to all passing rows
        rs_ranks = {date: 95.0 for date in df.index}
        scored_with_rank = scoring.score_dataframe(df, rs_ranks=rs_ranks)
        scored_no_rank = scoring.score_dataframe(df)

        for i in range(1, 5):
            if scored_with_rank.iloc[i]["passes_filters"]:
                assert (scored_with_rank.iloc[i]["score_relative_strength"] >=
                        scored_no_rank.iloc[i]["score_relative_strength"])

    def test_component_scores_sum_to_raw_score(self):
        scoring = make_scoring()
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        for i in range(1, 5):
            if scored.iloc[i]["passes_filters"]:
                row = scored.iloc[i]
                expected_raw = (row["score_base_quality"] + row["score_trend_strength"] +
                                row["score_relative_strength"] + row["score_volume_profile"] +
                                row["score_risk_reward"])
                assert row["raw_score"] == pytest.approx(expected_raw, rel=1e-6)


# ===========================================================================
# 10. GOLDEN REGRESSION SNAPSHOT
# ===========================================================================

class TestGoldenSnapshot:
    """
    End-to-end score sanity checks.  These lock in expected numeric outputs
    so that any future change that shifts values by even 1 point will fail.
    Update intentionally only when scoring logic is deliberately changed.
    """

    def test_ideal_setup_scores_100_raw(self):
        scoring = make_scoring(regime_multiplier=1.0)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert bd.raw_total == 100.0

    def test_ideal_setup_in_downtrend_scores_50_total(self):
        """Severe downtrend: multiplier = 0.5 -> total halved."""
        scoring = make_scoring(regime_multiplier=0.5)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert bd.total == pytest.approx(50.0)

    def test_ideal_setup_grade_is_A_plus(self):
        scoring = make_scoring(regime_multiplier=1.0)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert scoring.get_grade(bd.raw_total) == "A+"

    def test_ideal_setup_signal_is_strong_buy(self):
        scoring = make_scoring(regime_multiplier=1.0)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert scoring.get_signal_strength(bd.total) == "STRONG BUY - Alert"

    def test_ideal_setup_in_downtrend_signal_is_pass(self):
        """Multiplier=0.5 -> total=50 -> PASS (below 60 threshold)."""
        scoring = make_scoring(regime_multiplier=0.5)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=95.0)
        assert scoring.get_signal_strength(bd.total) == "PASS"

    def test_component_maxima(self):
        """Verify each component achieves its documented maximum."""
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        weights = PARAMETERS["weights"]
        assert bd.base_quality == weights["base_quality"]         # 15
        assert bd.trend_strength == weights["trend_strength"]     # 15
        assert bd.relative_strength == weights["relative_strength"] # 30
        assert bd.volume_profile == weights["volume_profile"]     # 25
        assert bd.risk_reward == weights["risk_reward"]           # 15

    def test_weight_totals_sum_to_100(self):
        """Documented weights must sum to 100 — the denominator of the scale."""
        weights = PARAMETERS["weights"]
        assert sum(weights.values()) == 100

    def test_no_score_exceeds_category_weight(self):
        """No single component can score above its weight maximum."""
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=95.0)
        weights = PARAMETERS["weights"]
        assert bd.base_quality <= weights["base_quality"]
        assert bd.trend_strength <= weights["trend_strength"]
        assert bd.relative_strength <= weights["relative_strength"]
        assert bd.volume_profile <= weights["volume_profile"]
        assert bd.risk_reward <= weights["risk_reward"]
