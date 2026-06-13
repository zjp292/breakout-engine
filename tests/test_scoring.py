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
        "distance_from_sma10":  0.02,   # 2% above 10 EMA
        "distance_from_sma20":  0.06,
        "distance_from_sma50":  0.14,
        "distance_from_sma150": 0.28,
        "distance_from_sma200": 0.39,
        "ma_alignment": True,           # ema10 > ema20 > sma50
        "mas_rising": True,             # all short-term slopes positive
        "ema10_surf_ratio": 0.80,       # 80% of recent days hugging the rising EMA
        # --- Stage 2 ---
        "stage2": True,
        # --- 52-week / 90-day proximity ---
        "pct_from_52wk_high": -0.05,    # 5% below high (near breakout)
        "pct_from_90d_high":  -0.05,    # must match 52wk default so filter tests behave
        # --- Consolidation ---
        "consol_range_60": 0.04,        # 4% range — 60-day box
        "range_10": 0.04,               # 4% range — 10-day recent coil (used for tightness)
        "consol_days": 10,              # sweet-spot flag length
        # --- VCP ---
        "vcp_contracting": True,
        "vcp_contraction_ratio": 0.30,
        # vcp_contraction_count not set — scoring falls back to vcp_contracting flag (2.0 pts)
        # --- Wedge geometry ---
        "swing_low_count": 0,           # higher-lows pivot events in base window
        "swing_high_count": 0,          # lower-highs pivot events in base window
        # --- Prior move ---
        "prior_move_pct": 0.80,         # 80% prior move (passes 75% hard filter)
        "days_since_power_move": 20,
        # --- Consolidation depth / breakout proximity ---
        "base_depth": 0.10,             # 10% depth — below 25% penalty threshold
        # breakout_level not set — pivot proximity defaults to 0 pts
        # --- New signal features (all default to off so existing tests are unaffected) ---
        "is_trigger_bar": False,        # trigger bar bonus: off by default
        "obv_trend": False,             # OBV accumulation bonus: off by default
        "weekly_aligned": True,         # weekly alignment: True = no penalty
        "approaching_annual_high": False,  # GH2004 anchoring alpha: off by default
        # --- Volume ---
        "dollar_volume": 50_000_000,    # $50M
        "volume_declining": True,
        "volume_dryup_ratio": 0.70,
        "adr_pct": 0.08,               # 8% ADR (above minimum)
        "relative_volume": 0.80,
        "volume_sma_20": 500_000,
        "close_range_position": 0.50,   # mid-range close — demand bonus threshold is 0.70
        "volume_vs_6m_avg": 0.50,       # LS2000: 50% of 1-year avg — genuinely quiet base (no penalty)
        # --- Relative strength ---
        "rs_comp_252": 0.10,           # 10% 12-month excess return — passes rs_252 filter
        "rs_comp_120": 0.15,
        "rs_comp_60":  0.12,
        # --- Risk / reward ---
        "stop_distance_pct":     0.08, # 8% stop / 8% ADR = 1.0x — ideal (backtester)
        "stop_distance_20d_pct": 0.08, # 20-day stop (used by hard filter)
        "stop_level": 46.0,
        "potential_r": 3.5,
        "potential_gain_pct": 0.28,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def make_ideal_row() -> pd.Series:
    """
    Row engineered to achieve the maximum possible raw score of 100.

    Component maxima (2026-06: 52wk proximity removed as anti-predictive; base_length peak at 35-60d):
      base_quality       6 + 4 + 4 + 6      = 20   (tightness+length+VCP+wedge; trigger capped at 20)
      trend_strength     4 + 8 + 2          = 14   (MA+prior_move+approaching_annual_high; GH2004)
      relative_strength  12 + 8 + 10        = 30   (60d primary 12pts, 120d secondary 8pts, rank 10pts)
      volume_profile     6 + 14 + 10        = 30   (adr 12-15% required for 10pts; peak EV=0.429)
      risk_reward        0 (excluded from scoring)

    Normalization: (raw/sub_max)*weight
      base:   (20/20)*10 = 10
      trend:  (14/14)*15 = 15
      rs:     (30/30)*25 = 25
      volume: (30/30)*50 = 50
      TOTAL               = 100
    """
    return make_row(
        # base_quality -> 20 (6+4+4+6)
        range_10=0.01,                    # tightness = 6 (ratio=0.01/0.15=0.067 ≤ 0.75)
        consol_range_60=0.01,             # fallback also tight
        consol_days=40,                   # length = 4 (optimal 35-60d window per DB)
        vcp_contraction_count=3,          # vcp = 4 (3 contractions = full VCP)
        vcp_contracting=True,
        vcp_contraction_ratio=0.20,
        swing_low_count=3,                # wedge: hl >= 2
        swing_high_count=3,               # wedge: lh >= 2 → wedge = 6
        is_trigger_bar=False,             # trigger bar: capped at 20, no extra needed
        # trend_strength -> 14 (4+8+2; approaching_annual_high: GH2004)
        ma_alignment=True,
        mas_rising=True,
        distance_from_sma10=0.02,
        ema10_surf_ratio=0.80,            # surf_ratio >= 0.75 + aligned + rising = 4
        prior_move_pct=2.5,
        days_since_power_move=30,         # 200%+ within 60d → power_move = 8
        approaching_annual_high=True,     # -3% to -15% from 52wk high → +2 pts (GH2004)
        weekly_aligned=True,              # no weekly penalty
        # relative_strength -> 30 (rs_rank handled via score call; use rs_rank=90)
        rs_comp_60=0.55,                  # rs_60 = 12 (>=0.50, primary signal 2026-06)
        rs_comp_120=0.25,                 # rs_120 = 8 (>=0.20, secondary)
        # volume_profile -> 30 (6+14+10)
        dollar_volume=150_000_000,        # dv = 6  (>= 10x min)
        volume_declining=True,
        volume_dryup_ratio=0.50,          # vd = 14 (< 0.60; OBV off by default)
        adr_pct=0.13,                     # adr = 10 (12-15% peak, EV=0.429)
        obv_trend=False,                  # OBV bonus off — base vd already 14
        # stop fields kept so the hard-filter still passes (used in filter tests)
        stop_distance_pct=0.06,
        potential_r=6.0,
    )


# ===========================================================================
# 1. SCORE BASE QUALITY
# ===========================================================================

class TestScoreBaseQuality:
    """
    score_base_quality() decomposes into four sub-scores (restructured 2026-05):
      - Recent tightness   (0-6 pts): range_10 — 10-day range, not 60-day box
      - Base length        (0-4 pts)
      - VCP contraction    (0-4 pts)
      - Wedge geometry     (0-6 pts): higher pivot lows + lower pivot highs
    Max total = 20.
    """

    scoring = make_scoring()

    # --- 1a. Tightness thresholds (ADR-relative: ratio = range_10 / adr_pct) ---
    # thresholds: ≤0.75→6  ≤1.25→5  ≤2.0→4  ≤3.5→2  >3.5→0

    @pytest.mark.parametrize("range_val, adr, expected", [
        # ratio ≤ 0.75 → 6 pts
        (0.00,  0.10, 6.0),   # ratio 0.0
        (0.06,  0.10, 6.0),   # ratio 0.6
        (0.075, 0.10, 6.0),   # ratio 0.75 — boundary
        # ratio ≤ 1.25 → 5 pts
        (0.076, 0.10, 5.0),   # ratio 0.76 — just over
        (0.10,  0.10, 5.0),   # ratio 1.0
        (0.125, 0.10, 5.0),   # ratio 1.25 — boundary
        # ratio ≤ 2.0 → 4 pts
        (0.126, 0.10, 4.0),   # ratio 1.26 — just over
        (0.15,  0.10, 4.0),   # ratio 1.5
        (0.20,  0.10, 4.0),   # ratio 2.0 — boundary
        # ratio ≤ 3.5 → 2 pts
        (0.201, 0.10, 2.0),   # ratio 2.01 — just over
        (0.25,  0.10, 2.0),   # ratio 2.5
        (0.35,  0.10, 2.0),   # ratio 3.5 — boundary
        # ratio > 3.5 → 0 pts
        (0.351, 0.10, 0.0),   # ratio 3.51 — just over
        (0.50,  0.10, 0.0),   # ratio 5.0
        (1.00,  0.10, 0.0),   # extreme
        # verify it's truly ADR-relative (same ratio → same score, different absolutes)
        (0.20,  0.04, 0.0),   # range 20%, adr 4%  → ratio 5.0  → 0
        (0.20,  0.15, 4.0),   # range 20%, adr 15% → ratio 1.33 → 4
        (0.20,  0.30, 6.0),   # range 20%, adr 30% → ratio 0.67 → 6
    ])
    def test_tightness(self, range_val, adr, expected):
        row = make_row(range_10=range_val, adr_pct=adr, consol_days=10,
                       vcp_contracting=False, vcp_contraction_ratio=1.0)
        score, details = self.scoring.score_base_quality(row)
        assert details["tightness"] == expected, (
            f"range_10={range_val}, adr={adr} (ratio={range_val/adr:.2f})"
            f" -> expected tightness {expected}, got {details['tightness']}"
        )

    def test_tightness_falls_back_to_consol_range_when_range_10_missing(self):
        """If range_10 is absent, scoring falls back to consol_range_60."""
        row = make_row(consol_days=10, vcp_contracting=False, vcp_contraction_ratio=1.0)
        # Remove range_10 to force fallback
        row = row.drop("range_10")
        row["consol_range_60"] = 0.01   # should get tightness = 6
        score, details = self.scoring.score_base_quality(row)
        assert details["tightness"] == 6.0

    # --- 1b. Base length thresholds (max 4 pts) ---

    @pytest.mark.parametrize("days, expected", [
        # DB-validated: 35-60d best (+30.6% mean), 20-35d very good (+27.5%),
        # short flags underperform longer bases; >60d stalls (+5.6%)
        (0,   0.0),   # no base
        (4,   0.0),   # below hard-filter minimum
        (5,   2.5),   # short flag minimum
        (9,   2.5),   # short flag
        (10,  3.0),   # normal flag lower bound
        (19,  3.0),   # normal flag
        (20,  3.5),   # very good lower bound
        (34,  3.5),   # very good upper bound
        (35,  4.0),   # optimal lower bound (DB best)
        (50,  4.0),   # optimal middle
        (60,  4.0),   # optimal upper bound
        (61,  1.0),   # too long — stalls
        (90,  1.0),   # way too long
    ])
    def test_base_length(self, days, expected):
        row = make_row(consol_days=days,
                       vcp_contracting=False, vcp_contraction_ratio=1.0)
        score, details = self.scoring.score_base_quality(row)
        assert details["base_length"] == expected, (
            f"consol_days={days} -> expected length {expected}, got {details['base_length']}"
        )

    # --- 1c. VCP contraction — count-based scoring (max 4 pts) ---
    # count = consecutive non-overlapping windows each narrower than the one before.
    # 3+ contractions = textbook VCP; 2 = solid; 1 = early; 0 = none detected.
    # fallback when count not provided: vcp_contracting flag (2 pts) or ratio (0.5 pts).

    @pytest.mark.parametrize("count, expected", [
        (3, 4.0),   # 3 contractions: full textbook VCP
        (4, 4.0),   # >3 also capped at 4.0
        (2, 3.0),   # 2 contractions: solid VCP
        (1, 2.0),   # 1 contraction: early VCP
        (0, 0.0),   # count available but 0 contractions
    ])
    def test_vcp_contraction_count(self, count, expected):
        row = make_row(consol_days=10,
                       vcp_contraction_count=count, vcp_contracting=False)
        score, details = self.scoring.score_base_quality(row)
        assert details["vcp_contraction"] == expected, (
            f"count={count} -> expected {expected}, got {details['vcp_contraction']}"
        )

    def test_vcp_contracting_flag_fallback(self):
        """When count not provided (absent from row), fall back to vcp_contracting flag → 2 pts."""
        row = make_row(consol_days=10, vcp_contracting=True)
        # drop count so fallback fires
        row = row.drop("vcp_contraction_count") if "vcp_contraction_count" in row.index else row
        score, details = self.scoring.score_base_quality(row)
        assert details["vcp_contraction"] == 2.0

    def test_vcp_ratio_fallback(self):
        """When count not provided and vcp_contracting=False but ratio <= 0.60 → 0.5 pts."""
        row = make_row(consol_days=10, vcp_contracting=False, vcp_contraction_ratio=0.50)
        row = row.drop("vcp_contraction_count") if "vcp_contraction_count" in row.index else row
        score, details = self.scoring.score_base_quality(row)
        assert details["vcp_contraction"] == 0.5

    def test_vcp_no_signal_fallback(self):
        """count not provided, vcp_contracting=False, ratio=1.0 → 0 pts."""
        row = make_row(consol_days=10, vcp_contracting=False, vcp_contraction_ratio=1.0)
        row = row.drop("vcp_contraction_count") if "vcp_contraction_count" in row.index else row
        score, details = self.scoring.score_base_quality(row)
        assert details["vcp_contraction"] == 0.0

    # --- 1d. Wedge geometry thresholds (max 6 pts) ---

    @pytest.mark.parametrize("hl_count, lh_count, expected", [
        (2, 2, 6.0),   # textbook convergence: multiple events both sides
        (3, 2, 6.0),   # hl >= 2 and lh >= 2 regardless of hl being higher
        (2, 3, 6.0),
        (2, 1, 4.5),   # both >= 1, total = 3 >= 3 → well-confirmed
        (1, 2, 4.5),   # both >= 1, total = 3
        (3, 1, 4.5),   # both >= 1, total = 4
        (1, 1, 3.0),   # one event each side — early-stage wedge
        (2, 0, 2.0),   # ascending base: rising support only, no overhead compression
        (3, 0, 2.0),
        (1, 0, 1.0),   # one higher low — minimal structural evidence
        (0, 1, 0.0),   # lower highs only → descending channel, no credit
        (0, 2, 0.0),   # lower highs only
        (0, 0, 0.0),   # no structure
    ])
    def test_wedge_geometry(self, hl_count, lh_count, expected):
        row = make_row(swing_low_count=hl_count, swing_high_count=lh_count)
        score, details = self.scoring.score_base_quality(row)
        assert details["wedge_geometry"] == expected, (
            f"hl={hl_count}, lh={lh_count} -> expected wedge {expected}, got {details['wedge_geometry']}"
        )

    def test_max_score_is_20(self):
        """Perfect base: tight recent range + optimal-length (35-60d) + 3-contraction VCP + full wedge."""
        row = make_row(
            range_10=0.01, consol_days=40,       # 35-60d window → 4 pts (DB optimal)
            vcp_contraction_count=3,              # count=3 → 4 pts
            vcp_contracting=True, vcp_contraction_ratio=0.20,
            swing_low_count=3, swing_high_count=3,
            is_trigger_bar=False,                 # trigger capped at 20 anyway
        )
        score, _ = self.scoring.score_base_quality(row)
        assert score == 20.0

    def test_trigger_bar_bonus_caps_at_20(self):
        """Trigger bar adds 1.5 pts but total is capped at 20 when base is already maxed."""
        row = make_row(
            range_10=0.01, consol_days=40,       # 35-60d → 4 pts
            vcp_contraction_count=3,
            swing_low_count=3, swing_high_count=3,
            is_trigger_bar=True,
        )
        score, details = self.scoring.score_base_quality(row)
        assert score == 20.0  # cap holds
        assert details["trigger_bar"] == 1.5

    def test_trigger_bar_adds_bonus_on_partial_base(self):
        """Trigger bar adds 1.5 pts when base is not maxed."""
        row = make_row(
            range_10=0.04, consol_days=40,        # tightness=6, length=4 (35-60d optimal)
            vcp_contraction_count=1,               # vcp = 2
            swing_low_count=0, swing_high_count=0, # wedge = 0
            is_trigger_bar=True,
        )
        score, details = self.scoring.score_base_quality(row)
        # 6 + 4 + 2 + 0 + 1.5 = 13.5 (no cap triggered)
        assert score == 13.5
        assert details["trigger_bar"] == 1.5

    def test_min_score_is_0(self):
        """Worst possible base: range ratio > 3.5, no days, no VCP, no wedge."""
        # range_10=0.50 / adr_pct=0.08 → ratio 6.25 > 3.5 → tightness 0
        row = make_row(
            range_10=0.50, consol_range_60=0.50, consol_days=0,
            vcp_contraction_count=0,            # explicit 0 → no VCP score
            vcp_contracting=False, vcp_contraction_ratio=1.0,
            swing_low_count=0, swing_high_count=0,
            is_trigger_bar=False,
        )
        score, _ = self.scoring.score_base_quality(row)
        assert score == 0.0

    def test_missing_fields_default_gracefully(self):
        """Missing optional fields should not raise — all missing → score = 0."""
        row = pd.Series({"close": 50.0})  # only close, everything else missing
        score, details = self.scoring.score_base_quality(row)
        assert score == 0.0


# ===========================================================================
# 2. SCORE TREND STRENGTH
# ===========================================================================

class TestScoreTrendStrength:
    """
    score_trend_strength() decomposes into four sub-scores:
      - Stage 2 MA structure   (0-5 pts)
      - 52-wk high proximity   (0-5 pts)
      - Short-term MA stack    (0-4 pts)
      - Prior power move       (0-6 pts)
    Max total = 20.
    """

    scoring = make_scoring()

    # --- 2a. Stage 2 (removed from scoring 2026-06) ---
    # DB: stage2=True EV=0.195 vs stage2=False EV=0.276 (anti-predictive — full Stage 2
    # stocks are extended; fresh breakout stocks outperform by 40%). Kept in details
    # for persistence compatibility.

    def test_stage2_detail_always_zero(self):
        """Stage 2 is no longer scored; detail key must exist at 0.0."""
        for s2 in [True, False]:
            row = make_row(stage2=s2)
            _, d = self.scoring.score_trend_strength(row)
            assert d["stage2"] == 0.0, f"stage2={s2}: expected 0.0, got {d['stage2']}"

    # --- 2b. Short-term MA structure ---
    # surf_ratio = ema10_surf_ratio: rolling fraction of days price hugged the rising EMA.
    # replaces the old single-day "above_10sma" binary — surf_ratio is the differentiator.

    @pytest.mark.parametrize("ma_alignment, mas_rising, surf_ratio, expected", [
        (True,  True,  0.80, 4.0),   # aligned + rising + clean surfing ≥ 0.75
        (True,  True,  0.55, 3.5),   # aligned + rising + moderate surfing ≥ 0.50
        (True,  True,  0.30, 3.0),   # aligned + rising, sloppy base
        (True,  False, 0.70, 2.0),   # aligned + not rising + good surfing ≥ 0.65
        (True,  False, 0.30, 1.0),   # aligned only, poor surfing
        (False, False, 0.00, 0.0),   # no alignment, sma_10 < sma_20 → 0
    ])
    def test_ma_structure(self, ma_alignment, mas_rising, surf_ratio, expected):
        row = make_row(ma_alignment=ma_alignment, mas_rising=mas_rising,
                       ema10_surf_ratio=surf_ratio,
                       # ensure sma_10 <= sma_20 for the False/False case
                       sma_10=48.0, sma_20=49.0)
        _, d = self.scoring.score_trend_strength(row)
        assert d["ma_structure"] == expected, (
            f"alignment={ma_alignment}, rising={mas_rising}, surf={surf_ratio} "
            f"-> expected {expected}, got {d['ma_structure']}"
        )

    def test_ma_partial_alignment_10_above_20(self):
        """sma_10 > sma_20 but ma_alignment=False -> 0.5 pt."""
        row = make_row(ma_alignment=False, mas_rising=False,
                       sma_10=50.0, sma_20=48.0)
        _, d = self.scoring.score_trend_strength(row)
        assert d["ma_structure"] == 0.5

    # --- 2d. Prior power move ---

    @pytest.mark.parametrize("prior_move, days_since, expected", [
        # 200%+ flagpole within 60d → 8 pts
        (2.5,  30, 8.0),
        (2.0,  60, 8.0),   # exactly 200%
        (2.0,  61, 4.0),   # 200% but >60d → falls to 75%+ within 90d tier
        # 100-200% within 60d → 7 pts; >60d falls to 75%+ tier (4 pts)
        (1.5,  45, 7.0),
        (1.0,  60, 7.0),   # exactly 100%
        (1.0,  61, 4.0),   # 100% but >60d → falls to 75%+ within 90d tier
        # 75-100% within 60d → 5.5 pts
        (0.90, 30, 5.5),
        (0.75, 60, 5.5),   # exactly 75%
        (0.75, 61, 4.0),   # 75% but >60d → falls to 75%+ within 90d tier
        (0.75, 90, 4.0),   # exactly 75% within 90d boundary
        (0.75, 91, 0.0),   # 75% but >90d → 0 pts
        # below 75% → 0 pts (blocked by hard filter; min_prior_move_pct=0.75)
        (0.74, 30, 0.0),
        (0.50, 60, 0.0),
        (0.30, 60, 0.0),
        (0.20, 60, 0.0),
        (0.00, 999, 0.0),
    ])
    def test_prior_power_move(self, prior_move, days_since, expected):
        row = make_row(prior_move_pct=prior_move, days_since_power_move=days_since)
        _, d = self.scoring.score_trend_strength(row)
        assert d["prior_power_move"] == expected, (
            f"prior_move={prior_move}, days={days_since} -> expected {expected}, got {d['prior_power_move']}"
        )

    def test_max_score_is_14(self):
        """Ideal trend: perfect MAs + 200%+ flagpole + approaching annual high. sub-max=14 (4+8+2)."""
        row = make_row(
            ma_alignment=True, mas_rising=True, distance_from_sma10=0.02,
            ema10_surf_ratio=0.80,
            prior_move_pct=2.5, days_since_power_move=30,  # 200%+ → 8 pts
            approaching_annual_high=True, consol_days=10,   # GH2004 → +2 pts
            weekly_aligned=True,
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 14.0

    def test_max_score_without_approaching_is_12(self):
        """Without approaching_annual_high, ideal trend scores 4+8=12."""
        row = make_row(
            ma_alignment=True, mas_rising=True, distance_from_sma10=0.02,
            ema10_surf_ratio=0.80,
            prior_move_pct=2.5, days_since_power_move=30,
            approaching_annual_high=False,
            weekly_aligned=True,
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 12.0

    def test_trend_score_75pct_move(self):
        """75% prior move within 60d → 5.5 pts. MA+prior = 4+5.5 = 9.5."""
        row = make_row(
            ma_alignment=True, mas_rising=True, ema10_surf_ratio=0.80,
            prior_move_pct=0.75, days_since_power_move=15,
            weekly_aligned=True,
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 9.5

    def test_min_score_is_0(self):
        """Worst trend: below all MAs, no prior move."""
        row = make_row(
            stage2=False,
            distance_from_sma150=-0.10, distance_from_sma200=-0.15,
            ma_alignment=False, mas_rising=False,
            sma_10=45.0, sma_20=46.0,     # sma10 < sma20 → no partial credit
            prior_move_pct=0.05, days_since_power_move=999,
            weekly_aligned=True,           # no weekly penalty
        )
        score, _ = self.scoring.score_trend_strength(row)
        assert score == 0.0

    # --- 2d. Weekly alignment soft penalty ---

    def test_weekly_aligned_no_penalty(self):
        row = make_row(weekly_aligned=True, stage2=True,
                       prior_move_pct=0.50, days_since_power_move=15)
        _, d = self.scoring.score_trend_strength(row)
        assert d["weekly_alignment"] == 0.0

    def test_weekly_misaligned_penalty(self):
        row = make_row(weekly_aligned=False, stage2=True,
                       prior_move_pct=0.50, days_since_power_move=15)
        score_aligned, _ = self.scoring.score_trend_strength(
            make_row(weekly_aligned=True, stage2=True,
                     prior_move_pct=0.50, days_since_power_move=15)
        )
        score_misaligned, d = self.scoring.score_trend_strength(row)
        assert d["weekly_alignment"] == -5.0
        assert score_misaligned == max(0.0, score_aligned - 5.0)

    # --- 2e. Approaching annual high (George & Hwang 2004) ---

    def test_approaching_annual_high_adds_2pts(self):
        """approaching_annual_high=True AND consol_days>=10 → +2 pts (GH2004 anchoring alpha)."""
        row = make_row(
            approaching_annual_high=True, consol_days=10,
            ma_alignment=False, mas_rising=False,
            sma_10=45.0, sma_20=46.0,
            prior_move_pct=0.0, days_since_power_move=999,
            weekly_aligned=True,
        )
        _, d = self.scoring.score_trend_strength(row)
        assert d["approaching_annual_high"] == 2.0

    def test_approaching_annual_high_requires_min_consol_days(self):
        """consol_days < 10 suppresses the GH2004 bonus even when flag is True."""
        row_short = make_row(approaching_annual_high=True, consol_days=5, weekly_aligned=True)
        row_long  = make_row(approaching_annual_high=True, consol_days=10, weekly_aligned=True)
        _, d_short = self.scoring.score_trend_strength(row_short)
        _, d_long  = self.scoring.score_trend_strength(row_long)
        assert d_short["approaching_annual_high"] == 0.0
        assert d_long["approaching_annual_high"]  == 2.0

    def test_approaching_annual_high_off_adds_zero(self):
        """approaching_annual_high=False → 0 pts regardless of consol_days."""
        row = make_row(approaching_annual_high=False, consol_days=30, weekly_aligned=True)
        _, d = self.scoring.score_trend_strength(row)
        assert d["approaching_annual_high"] == 0.0


# ===========================================================================
# 3. SCORE RELATIVE STRENGTH
# ===========================================================================

class TestScoreRelativeStrength:
    """
    score_relative_strength() decomposes into:
      - 60-day RS   (0-12 pts)  — PRIMARY: DB EV=0.303 (Q4) vs EV=0.231 (Q1), monotonic
      - 120-day RS  (0-8 pts)   — secondary, non-monotonic (Q3 is lowest EV=0.199)
      - RS rank     (0-10 pts, only when rs_rank supplied)
    Max total = 30.

    2026-06 swap: 60d → 12pts (primary), 120d → 8pts (secondary).
    DB analysis (n=4662, prior>=75%): strong60d+weak120d EV=0.301 (best combo);
    weak60d+strong120d EV=0.186 (worst). Prior weighting had these backwards.
    """

    scoring = make_scoring()

    # --- 3a. 60-day RS thresholds (now PRIMARY, 0-12 pts) ---

    @pytest.mark.parametrize("rs_60, expected", [
        ( 0.50, 12.0),   # top quartile in filter-passing cohort (EV=0.303)
        ( 0.80, 12.0),   # above 50%
        ( 0.25,  9.0),   # 25-50%
        ( 0.50 - 1e-9,  9.0),  # just under 50%
        ( 0.12,  6.0),   # 12-25%
        ( 0.25 - 1e-9,  6.0),
        ( 0.00,  2.0),   # neutral (consolidating, not underperforming)
        ( 0.12 - 1e-9,  2.0),
        (-0.01,  0.0),   # underperforming benchmark
        (-0.30,  0.0),
    ])
    def test_rs_60(self, rs_60, expected):
        row = make_row(rs_comp_120=0.0, rs_comp_60=rs_60)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_60_day"] == expected

    # --- 3b. 120-day RS thresholds (secondary, 0-8 pts) ---

    @pytest.mark.parametrize("rs_120, expected", [
        ( 0.20,  8.0),   # positive trend over 4 months
        ( 0.50,  8.0),   # above 20%
        ( 0.10,  6.0),   # 10-20%
        ( 0.20 - 1e-9,  6.0),
        ( 0.04,  3.0),   # 4-10%
        ( 0.10 - 1e-9,  3.0),
        ( 0.00,  1.0),   # neutral 4-month trend
        ( 0.04 - 1e-9,  1.0),
        (-0.01,  0.0),   # underperforming
        (-0.20,  0.0),
    ])
    def test_rs_120(self, rs_120, expected):
        row = make_row(rs_comp_120=rs_120, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_120_day"] == expected

    # --- 3c. RS rank percentile ---

    @pytest.mark.parametrize("rs_rank, expected", [
        ( 95.0,  8.0),   # top 5%: crowded names, capped below max
        (100.0,  8.0),
        ( 90.0, 10.0),   # 85-95th: leadership sweet spot (max)
        ( 85.0, 10.0),
        ( 84.9,  7.0),   # 75-85th
        ( 75.0,  7.0),
        ( 74.9,  4.0),   # 65-75th
        ( 65.0,  4.0),
        ( 64.9,  0.0),   # below 65th
        (  0.0,  0.0),
    ])
    def test_rs_rank(self, rs_rank, expected):
        row = make_row(rs_comp_120=0.0, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=rs_rank)
        assert d["rs_rank"] == expected

    def test_rs_rank_none_gives_zero_rank_points(self):
        """When rs_rank is not provided, rank component = 0."""
        row = make_row(rs_comp_120=0.0, rs_comp_60=0.0)
        score, d = self.scoring.score_relative_strength(row, rs_rank=None)
        assert d["rs_rank"] == 0.0

    def test_max_score_is_30(self):
        row = make_row(rs_comp_120=0.25, rs_comp_60=0.50)  # 60d>=0.50=12pts, 120d>=0.20=8pts
        score, _ = self.scoring.score_relative_strength(row, rs_rank=90.0)
        assert score == 30.0

    def test_min_score_is_0(self):
        row = make_row(rs_comp_120=-0.20, rs_comp_60=-0.30)
        score, _ = self.scoring.score_relative_strength(row, rs_rank=10.0)
        assert score == 0.0


# ===========================================================================
# 4. SCORE VOLUME PROFILE
# ===========================================================================

class TestScoreVolumeProfile:
    """
    score_volume_profile() decomposes into:
      - Dollar volume      (0-6 pts)
      - Volume dry-up      (0-14 pts) — single source of truth
      - ADR %              (0-10 pts)
    Max total = 30.
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
        (True,  0.50, 14.0),   # very strong (ratio < 0.60)
        (True,  0.59, 14.0),
        (True,  0.60, 10.5),   # solid (0.60 <= ratio < 0.75)
        (True,  0.74, 10.5),
        (True,  0.75,  7.0),   # moderate (0.75 <= ratio < 0.90)
        (True,  0.89,  7.0),
        # declining=True, ratio=0.90: falls to `elif ratio < 1.0` -> 3.5
        (True,  0.90,  3.5),
        (False, 0.70,  3.5),   # not declining but ratio < 1.0
        (False, 0.99,  3.5),   # not declining but ratio just below 1
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
        # peak 12-15%: DB EV=0.429 (prior>=75% cohort, n=257)
        (0.12,  10.0),
        (0.14,  10.0),
        (0.15 - 1e-9, 10.0),  # just below 15% boundary
        # 15-20%: EV=0.378 → 9 pts
        (0.15,   9.0),
        (0.20,   9.0),   # exactly at upper boundary → still 9 pts
        # 20-25%: EV=0.222 → 5.5 pts
        (0.21,   5.5),
        (0.24,   5.5),
        # >=25%: EV=0.055 (terrible) → 2 pts
        (0.25,   2.0),
        (0.30,   2.0),
        # 10-12%: EV=0.331 → 7.5 pts
        (0.10,   7.5),
        (0.12 - 1e-9, 7.5),
        # 8-10% → 5 pts
        (0.08,   5.0),
        (0.10 - 1e-9, 5.0),
        # 7-8% → 2.5 pts (hard-filter minimum)
        (0.07,   2.5),
        (0.08 - 1e-9, 2.5),
        # below 7% → 0 pts
        (0.069,  0.0),
        (0.00,   0.0),
    ])
    def test_adr(self, adr_pct, expected):
        row = make_row(dollar_volume=0, volume_declining=False, volume_dryup_ratio=1.0,
                       adr_pct=adr_pct)
        score, d = self.scoring.score_volume_profile(row)
        assert d["adr"] == expected

    def test_max_score_is_30(self):
        row = make_row(dollar_volume=150_000_000,
                       volume_declining=True, volume_dryup_ratio=0.50,
                       adr_pct=0.13)  # 12-15% = peak (10 pts, EV=0.429)
        score, _ = self.scoring.score_volume_profile(row)
        assert score == 30.0

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

    # --- 4d. Volume dry-up gating on consol_days ---

    def test_volume_dryup_suppressed_when_not_in_base(self):
        """consol_days=0 means not in a base: dry-up signal is suppressed entirely."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.50,
                       consol_days=0)
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 0.0

    def test_volume_dryup_active_when_in_base(self):
        """consol_days=5 means in an active base: dry-up signal fires normally."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.50,
                       consol_days=5)
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 14.0

    # --- 4e. OBV accumulation bonus ---

    def test_obv_trend_adds_bonus_when_vd_score_high(self):
        """obv_trend=True with strong dry-up (vd_score >= 7) -> +2 pts.
        use ratio=0.60 (vd_score=10.5) so the +2 bonus is visible below the 14-pt cap."""
        row_base = make_row(dollar_volume=0, adr_pct=0.0,
                            volume_declining=True, volume_dryup_ratio=0.60,
                            obv_trend=False)
        row_obv  = make_row(dollar_volume=0, adr_pct=0.0,
                            volume_declining=True, volume_dryup_ratio=0.60,
                            obv_trend=True)
        _, d_base = self.scoring.score_volume_profile(row_base)
        _, d_obv  = self.scoring.score_volume_profile(row_obv)
        assert d_obv["volume_contraction"] == d_base["volume_contraction"] + 2.0

    def test_obv_trend_adds_1pt_when_vd_score_low(self):
        """obv_trend=True with weak dry-up (vd_score < 7) -> +1 pt."""
        row_base = make_row(dollar_volume=0, adr_pct=0.0,
                            volume_declining=False, volume_dryup_ratio=0.70,
                            obv_trend=False)
        row_obv  = make_row(dollar_volume=0, adr_pct=0.0,
                            volume_declining=False, volume_dryup_ratio=0.70,
                            obv_trend=True)
        _, d_base = self.scoring.score_volume_profile(row_base)
        _, d_obv  = self.scoring.score_volume_profile(row_obv)
        assert d_obv["volume_contraction"] == d_base["volume_contraction"] + 1.0

    def test_obv_bonus_does_not_exceed_14_cap(self):
        """OBV bonus must not push vd_score above the 14-pt sub-component cap."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.50,
                       obv_trend=True)
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] <= 14.0

    def test_obv_false_does_not_change_score(self):
        """obv_trend=False (default) leaves volume_contraction unchanged."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.60,
                       obv_trend=False)
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 10.5

    # --- 4e. Lee-Swaminathan (2000) historical volume penalty ---

    def test_ls2000_penalty_fires_when_base_vol_elevated_vs_historical(self):
        """volume_vs_6m_avg > 0.90 deducts 2 pts — base not structurally quiet."""
        row_normal = make_row(dollar_volume=0, adr_pct=0.0,
                              volume_declining=True, volume_dryup_ratio=0.60,
                              volume_vs_6m_avg=0.50)  # base at 50% of 1yr avg: genuinely quiet
        row_elevated = make_row(dollar_volume=0, adr_pct=0.0,
                                volume_declining=True, volume_dryup_ratio=0.60,
                                volume_vs_6m_avg=0.95)  # base at 95% of 1yr avg: flagpole lull only
        _, d_normal   = self.scoring.score_volume_profile(row_normal)
        _, d_elevated = self.scoring.score_volume_profile(row_elevated)
        assert d_elevated["volume_contraction"] == d_normal["volume_contraction"] - 2.0

    def test_ls2000_penalty_absent_when_base_vol_quiet(self):
        """volume_vs_6m_avg <= 0.90 leaves vd_score unchanged."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.60,
                       volume_vs_6m_avg=0.90)  # exactly at threshold: no penalty (strict >)
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 10.5

    def test_ls2000_penalty_floored_at_zero(self):
        """penalty cannot push vd_score below 0."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=False, volume_dryup_ratio=1.0,
                       volume_vs_6m_avg=0.95)  # vd_score=0 before penalty → stays at 0
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 0.0

    def test_ls2000_penalty_absent_when_feature_missing(self):
        """missing volume_vs_6m_avg (None) must not raise and must not penalize."""
        row = make_row(dollar_volume=0, adr_pct=0.0,
                       volume_declining=True, volume_dryup_ratio=0.60)
        row = row.drop("volume_vs_6m_avg")  # simulate pre-feature historical data
        _, d = self.scoring.score_volume_profile(row)
        assert d["volume_contraction"] == 10.5  # unpenalized baseline


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
    apply_hard_filters() must independently catch each of the 9 conditions.
    A row that fails one filter should still report all other failures.
    """

    scoring = make_scoring()

    def _passing_row(self) -> pd.Series:
        """Minimal row that satisfies all hard filters."""
        return make_row(
            close=10.0,
            sma_50=9.0,               # close > sma_50
            dollar_volume=15_000_000, # > $10M
            adr_pct=0.08,             # > 7% (min raised from 5% to 7% in 2026-06)
            stop_distance_pct=0.08,   # 8% stop / 8% ADR = 1.0x (< 5x)
            stop_distance_20d_pct=0.08,
            pct_from_52wk_high=-0.10, # within 30%
            consol_days=5,            # meets new min_consol_days filter
            prior_move_pct=0.80,      # > 75% (min raised to 75% in 2026-06)
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
        (0.070, True),    # exactly at new minimum (raised from 5% to 7% in 2026-06)
        (0.069, False),   # just below
        (0.100, True),
        (0.050, False),   # old minimum now fails
    ])
    def test_filter_adr(self, adr, should_pass):
        row = self._passing_row()
        row["adr_pct"] = adr
        row["stop_distance_20d_pct"] = adr * 1.0  # 1x ADR — well within the 5x limit
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("ADR" in f for f in failures)

    # --- Filter 5: Stop distance relative to ADR ---

    def test_filter_stop_exactly_5x_adr_passes(self):
        """stop = 5x ADR exactly: 0.40 > 0.40 is False -> passes."""
        row = self._passing_row()
        row["adr_pct"] = 0.08
        row["stop_distance_20d_pct"] = 0.40  # exactly 5.0x ADR
        passes, failures = self.scoring.apply_hard_filters(row)
        assert not any("Stop distance" in f and "ADR" in f for f in failures)

    def test_filter_stop_over_5x_adr_fails(self):
        """stop = 5.01x ADR -> fails."""
        row = self._passing_row()
        row["adr_pct"] = 0.08
        row["stop_distance_20d_pct"] = 0.401  # > 0.40 = 5x ADR
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("Stop distance" in f for f in failures)

    def test_filter_stop_too_tight_fails(self):
        """stop < 0.001 is rejected as irrationally tight."""
        row = self._passing_row()
        row["stop_distance_20d_pct"] = 0.0005
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
        """both windows set to same value — tests the threshold directly."""
        row = self._passing_row()
        row["pct_from_52wk_high"] = pct_from_high
        row["pct_from_90d_high"]  = pct_from_high
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass
        if not should_pass:
            assert any("high" in f for f in failures)

    def test_filter_52wk_far_but_90d_near_passes(self):
        """post-crash setup: 52wk high is pre-crash; 90d high is recent flagpole top."""
        row = self._passing_row()
        row["pct_from_52wk_high"] = -0.60
        row["pct_from_90d_high"]  = -0.08
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is True
        assert not any("high" in f for f in failures)

    def test_filter_both_windows_far_fails(self):
        """genuine downtrend: both windows far below high -> fails."""
        row = self._passing_row()
        row["pct_from_52wk_high"] = -0.50
        row["pct_from_90d_high"]  = -0.45
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("high" in f for f in failures)

    def test_filter_52wk_nan_skipped(self):
        """NaN pct_from_52wk_high falls back to 90d — filter still evaluates."""
        row = self._passing_row()
        row["pct_from_52wk_high"] = np.nan
        row["pct_from_90d_high"]  = -0.05
        passes, failures = self.scoring.apply_hard_filters(row)
        assert not any("high" in f for f in failures)

    def test_multiple_failures_all_reported(self):
        """When several filters fail, all failure reasons should be present."""
        row = make_row(
            close=2.0,                   # fails price
            dollar_volume=100,           # fails dollar volume
            sma_50=999.0,                # fails above-50-SMA
            adr_pct=0.01,                # fails ADR
            stop_distance_20d_pct=1.0,   # fails stop (> 3x ADR)
            pct_from_52wk_high=-0.90,    # fails 52wk/90d high
            pct_from_90d_high=-0.90,     # both windows far below
        )
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert len(failures) >= 4

    # --- Filter 7: Prior move minimum (flagpole requirement) ---

    @pytest.mark.parametrize("prior_move, should_pass", [
        # raised from 25% → 30% → 50% → 75% (2026-06): DB EV monotonically improves
        # >=50% EV=0.203, >=75% EV=0.236, >=100% EV=0.262
        (0.75, True),   # exactly at new minimum — passes
        (1.00, True),   # solid flagpole
        (2.00, True),   # large flagpole
        (0.74, False),  # just below new minimum
        (0.50, False),  # old threshold — now fails
        (0.30, False),  # older threshold — fails
        (0.10, False),  # no flagpole
        (0.00, False),  # flat stock
    ])
    def test_filter_prior_move(self, prior_move, should_pass):
        row = self._passing_row()
        row["prior_move_pct"] = prior_move
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass, (
            f"prior_move={prior_move:.0%} expected pass={should_pass}, got {passes}"
        )
        if not should_pass:
            assert any("Prior move" in f for f in failures)

    def test_filter_prior_move_missing_treated_as_zero(self):
        """Missing prior_move_pct defaults to 0.0, which fails the filter."""
        row = self._passing_row()
        row["prior_move_pct"] = None
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("Prior move" in f for f in failures)

    # --- Filter 8: Minimum consolidation days ---

    @pytest.mark.parametrize("consol, should_pass", [
        (5,  True),   # exactly at minimum
        (10, True),   # healthy base
        (30, True),   # long base
        (4,  False),  # one day short
        (1,  False),  # too short
        (0,  False),  # not in a base at all — the most common DB state
    ])
    def test_filter_consol_days(self, consol, should_pass):
        row = self._passing_row()
        row["consol_days"] = consol
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes == should_pass, (
            f"consol_days={consol} expected pass={should_pass}, got {passes}"
        )
        if not should_pass:
            assert any("Consolidation" in f for f in failures)

    def test_filter_consol_days_none_treated_as_zero(self):
        """consol_days=None defaults to 0, which fails the filter."""
        row = self._passing_row()
        row["consol_days"] = None
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("Consolidation" in f for f in failures)

    # --- Filter 9: 12-month RS gate (AQR momentum universe, Moskowitz et al. 2012) ---

    def test_filter_rs_252_negative_fails(self):
        """stock that underperformed NASDAQ over 12M is not a momentum leader."""
        row = self._passing_row()
        row["rs_comp_252"] = -0.10
        passes, failures = self.scoring.apply_hard_filters(row)
        assert passes is False
        assert any("12-month RS" in f for f in failures)

    def test_filter_rs_252_positive_passes(self):
        """stock that outperformed NASDAQ over 12M clears the gate."""
        row = self._passing_row()
        row["rs_comp_252"] = 0.05
        passes, failures = self.scoring.apply_hard_filters(row)
        rs_failures = [f for f in failures if "12-month RS" in f]
        assert rs_failures == []

    def test_filter_rs_252_exactly_zero_passes(self):
        """matching NASDAQ exactly (rs_252=0) is not underperformance."""
        row = self._passing_row()
        row["rs_comp_252"] = 0.0
        passes, failures = self.scoring.apply_hard_filters(row)
        rs_failures = [f for f in failures if "12-month RS" in f]
        assert rs_failures == []

    def test_filter_rs_252_missing_skipped(self):
        """no rs_comp_252 key (< 252 bars of history) → filter silently skipped."""
        row = self._passing_row()
        row = row.drop("rs_comp_252", errors="ignore")
        passes, failures = self.scoring.apply_hard_filters(row)
        rs_failures = [f for f in failures if "12-month RS" in f]
        assert rs_failures == []

    def test_filter_rs_252_nan_skipped(self):
        """NaN rs_comp_252 (insufficient history) → filter silently skipped."""
        row = self._passing_row()
        row["rs_comp_252"] = float("nan")
        passes, failures = self.scoring.apply_hard_filters(row)
        rs_failures = [f for f in failures if "12-month RS" in f]
        assert rs_failures == []

    def test_filter_rs_252_disabled_by_config(self):
        """require_positive_rs_252=False disables the gate entirely."""
        import copy
        from config import PARAMETERS
        cfg = copy.deepcopy(PARAMETERS)
        cfg["require_positive_rs_252"] = False
        s = Scoring(cfg)
        row = self._passing_row()
        row["rs_comp_252"] = -0.50
        passes, failures = s.apply_hard_filters(row)
        rs_failures = [f for f in failures if "12-month RS" in f]
        assert rs_failures == []


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
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        assert bd.raw_total == 100.0, f"Expected 100, got {bd.raw_total}"

    def test_ideal_row_total_equals_raw_when_multiplier_is_1(self):
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        assert bd.total == bd.raw_total

    def test_regime_multiplier_gates_total(self):
        """With multiplier=0.5, total should be half of raw_total."""
        scoring = make_scoring(regime_multiplier=0.5)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        assert bd.total == pytest.approx(bd.raw_total * 0.5)

    def test_regime_multiplier_075_applies_correctly(self):
        scoring = make_scoring(regime_multiplier=0.75)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        assert bd.total == pytest.approx(bd.raw_total * 0.75, rel=1e-6)

    def test_component_sum_equals_raw_total(self):
        """
        raw_total = sum of (raw_component / sub_max) * config_weight.
        sub-component maxes (denominators): base=20, trend=17, rs=30, volume=30.
        config weights differ (10/15/25/50), so simple sum != raw_total.
        """
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=80.0)
        weights = PARAMETERS["weights"]
        _sub_maxes = {"base_quality": 20.0, "trend_strength": 14.0,
                      "relative_strength": 30.0, "volume_profile": 30.0}
        expected = (
            bd.base_quality      / _sub_maxes["base_quality"]      * weights["base_quality"]
            + bd.trend_strength  / _sub_maxes["trend_strength"]    * weights["trend_strength"]
            + bd.relative_strength / _sub_maxes["relative_strength"] * weights["relative_strength"]
            + bd.volume_profile  / _sub_maxes["volume_profile"]    * weights["volume_profile"]
        )
        assert bd.raw_total == pytest.approx(expected, rel=1e-6)

    def test_returns_score_breakdown_dataclass(self):
        scoring = make_scoring()
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=50.0)
        assert isinstance(bd, ScoreBreakdown)

    def test_details_dict_has_expected_keys(self):
        """All five detail namespaces must appear in the combined dict.
        rr_ keys are retained for display even though risk_reward=0 in the breakdown."""
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
        ideal = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert 0.0 <= ideal.raw_total <= 100.0
        assert 0.0 <= ideal.total <= 100.0

        worst = make_row(
            range_10=1.0, consol_range_60=1.0, consol_days=0,
            stage2=False, distance_from_sma150=-0.50, distance_from_sma200=-0.50,
            pct_from_52wk_high=-0.99, ma_alignment=False, mas_rising=False,
            sma_10=40.0, sma_20=41.0, prior_move_pct=0.0, days_since_power_move=999,
            rs_comp_120=-0.50, rs_comp_60=-0.50,
            dollar_volume=0, volume_declining=False, volume_dryup_ratio=2.0, adr_pct=0.0,
            stop_distance_pct=0.50, potential_r=0.0,
            vcp_contracting=False, vcp_contraction_ratio=1.0,
            swing_low_count=0, swing_high_count=0,
        )
        bad = scoring.calculate_total_score(worst, rs_rank=5.0)
        assert bad.raw_total == 0.0
        assert bad.total == 0.0


# ===========================================================================
# 9. EARNINGS PROXIMITY PENALTY
# ===========================================================================

class TestEarningsPenalty:
    """
    calculate_total_score() reduces raw_total when earnings are imminent.
    days_to_earnings missing or None → no penalty (backward-compatible with
    historical scoring rows that predate the earnings feature).
    """

    def _score_with_dte(self, days_to_earnings):
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        row = pd.Series({**row.to_dict(), "days_to_earnings": days_to_earnings})
        return scoring.calculate_total_score(row, rs_rank=90.0)

    def test_no_penalty_when_dte_missing(self):
        """rows without days_to_earnings (historical data) get no penalty."""
        scoring = make_scoring(regime_multiplier=1.0)
        bd_no_dte = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert bd_no_dte.raw_total == 100.0

    def test_no_penalty_when_dte_is_nan(self):
        bd = self._score_with_dte(float("nan"))
        assert bd.raw_total == 100.0

    def test_no_penalty_when_dte_is_none(self):
        bd = self._score_with_dte(None)
        assert bd.raw_total == 100.0

    def test_no_penalty_when_dte_15(self):
        """15 days to earnings is beyond the penalty window."""
        bd = self._score_with_dte(15)
        assert bd.raw_total == 100.0

    def test_five_pt_penalty_when_dte_10(self):
        bd = self._score_with_dte(10)
        assert bd.raw_total == pytest.approx(95.0)

    def test_five_pt_penalty_when_dte_7(self):
        bd = self._score_with_dte(7)
        assert bd.raw_total == pytest.approx(95.0)

    def test_ten_pt_penalty_when_dte_5(self):
        bd = self._score_with_dte(5)
        assert bd.raw_total == pytest.approx(90.0)

    def test_ten_pt_penalty_when_dte_0(self):
        """earnings today still triggers the hard penalty."""
        bd = self._score_with_dte(0)
        assert bd.raw_total == pytest.approx(90.0)

    def test_no_penalty_when_dte_negative(self):
        """past earnings (dte < 0) should NOT be penalized."""
        bd = self._score_with_dte(-3)
        assert bd.raw_total == 100.0

    def test_penalty_floored_at_zero(self):
        """a stock that scores near 0 + earnings penalty can't go negative."""
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_row(
            range_10=1.0, consol_days=0, stage2=False,
            distance_from_sma150=-0.50, distance_from_sma200=-0.50,
            pct_from_52wk_high=-0.99, ma_alignment=False, mas_rising=False,
            prior_move_pct=0.0, days_since_power_move=999,
            rs_comp_120=-0.50, rs_comp_60=-0.50,
            dollar_volume=0, volume_declining=False, volume_dryup_ratio=2.0,
            adr_pct=0.0, vcp_contracting=False, days_to_earnings=3,
        )
        bd = scoring.calculate_total_score(row)
        assert bd.raw_total >= 0.0


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
        """
        raw_score = sum of (component_raw / sub_max) * config_weight for each component.

        config weights (20/15/20/45) differ from sub-component maxes (20/20/30/30);
        the normalization step means raw sub-component scores don't simply add to raw_score.
        """
        scoring = make_scoring()
        df = self._build_feature_df()
        scored = scoring.score_dataframe(df)
        weights = PARAMETERS["weights"]
        # actual max output of each scoring method (denominators in normalization)
        _sub_maxes = {
            "base_quality": 20.0, "trend_strength": 14.0,
            "relative_strength": 30.0, "volume_profile": 30.0,
        }
        for i in range(1, 5):
            if scored.iloc[i]["passes_filters"]:
                row = scored.iloc[i]
                expected_raw = (
                    row["score_base_quality"]      / _sub_maxes["base_quality"]      * weights["base_quality"]
                    + row["score_trend_strength"]  / _sub_maxes["trend_strength"]    * weights["trend_strength"]
                    + row["score_relative_strength"] / _sub_maxes["relative_strength"] * weights["relative_strength"]
                    + row["score_volume_profile"]  / _sub_maxes["volume_profile"]    * weights["volume_profile"]
                )
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
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert bd.raw_total == 100.0

    def test_ideal_setup_in_downtrend_scores_50_total(self):
        """Severe downtrend: multiplier = 0.5 -> total halved."""
        scoring = make_scoring(regime_multiplier=0.5)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert bd.total == pytest.approx(50.0)

    def test_ideal_setup_grade_is_A_plus(self):
        scoring = make_scoring(regime_multiplier=1.0)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert scoring.get_grade(bd.raw_total) == "A+"

    def test_ideal_setup_signal_is_strong_buy(self):
        scoring = make_scoring(regime_multiplier=1.0)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert scoring.get_signal_strength(bd.total) == "STRONG BUY - Alert"

    def test_ideal_setup_in_downtrend_signal_is_pass(self):
        """Multiplier=0.5 -> total=50 -> PASS (below 60 threshold)."""
        scoring = make_scoring(regime_multiplier=0.5)
        bd = scoring.calculate_total_score(make_ideal_row(), rs_rank=90.0)
        assert scoring.get_signal_strength(bd.total) == "PASS"

    def test_component_maxima(self):
        """
        Verify each scoring method hits its documented raw sub-component max on an ideal row.

        Sub-component maxes (denominators in calculate_total_score normalization):
          base_quality:      6+4+4+6      = 20   (trigger bar capped at 20)
          trend_strength:    4+8+2        = 14   (approaching_annual_high added 2026-06: GH2004)
          relative_strength: 12+8+10      = 30
          volume_profile:    6+14+10      = 30   (OBV bonus capped within vd 14)

        Config weights (what each component contributes to raw_total):
          base=10, trend=15, rs=25, volume=50
          Normalization (raw/sub_max * weight) yields raw_total = 100 on ideal row.
        """
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        assert bd.base_quality == 20.0
        assert bd.trend_strength == 14.0
        assert bd.relative_strength == 30.0
        assert bd.volume_profile == 30.0
        assert bd.risk_reward == 0.0

    def test_weight_totals_sum_to_100(self):
        """Documented weights must sum to 100 — the denominator of the scale."""
        weights = PARAMETERS["weights"]
        assert sum(weights.values()) == 100

    def test_no_score_exceeds_sub_component_max(self):
        """No scoring method output can exceed its documented sub-component max."""
        scoring = make_scoring(regime_multiplier=1.0)
        row = make_ideal_row()
        bd = scoring.calculate_total_score(row, rs_rank=90.0)
        # sub-component maxes — the actual upper bound of each scoring method's output
        assert bd.base_quality <= 20.0
        assert bd.trend_strength <= 14.0
        assert bd.relative_strength <= 30.0
        assert bd.volume_profile <= 30.0
        assert bd.risk_reward == 0.0


# ===========================================================================
# 11. D&M PANIC STATE — MacroRegimeAnalyzer (Daniel & Moskowitz 2016)
# ===========================================================================

class TestDMPanicState:
    """
    Daniel & Moskowitz (2016): momentum crashes are forecastable when the market
    has fallen >20% over 24 months AND current vol is ELEVATED/EXTREME.
    In that panic state the macro multiplier floor drops 0.60 → 0.40.

    Tests cover:
      - _vol_adjusted_multiplier floor logic directly (unit tests, no data needed)
      - full analyze() end-to-end: panic vs non-panic synthetic market data
    """

    from config import PARAMETERS as _params

    def _make_analyzer(self):
        from macro_regime import MacroRegimeAnalyzer
        return MacroRegimeAnalyzer(self.config)

    @property
    def config(self):
        from config import PARAMETERS
        return PARAMETERS.copy()

    def _make_ohlcv(self, n: int, drift: float, sigma: float, seed: int = 42) -> pd.DataFrame:
        """synthetic daily OHLCV with deterministic drift and vol"""
        rng = np.random.default_rng(seed)
        returns = rng.normal(drift, sigma, n)
        prices = 100.0 * np.cumprod(1 + returns)
        highs = prices * (1 + np.abs(rng.normal(0, sigma / 2, n)))
        lows = prices * (1 - np.abs(rng.normal(0, sigma / 2, n)))
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "open":   prices * 0.999,
            "high":   np.maximum(prices, highs),
            "low":    np.minimum(prices, lows),
            "close":  prices,
            "volume": 1_000_000,
        }, index=dates)
        return df

    # --- unit: multiplier floor ---

    def test_panic_state_lowers_floor_to_0_40(self):
        """panic_state=True + EXTREME vol → floor is 0.40, not 0.60"""
        from macro_regime import MacroRegimeAnalyzer
        analyzer = MacroRegimeAnalyzer(self.config)
        # EXTREME vol: vol_60 > 0.28 — pass a sig dict that triggers it
        sig = {"vol_60": 0.35, "vol_ratio": 1.5, "vol_rising": True}
        # base_mult 0.60 - 0.10 (extreme) - 0.03 (rising) = 0.47; without panic floor=0.60 → clamps to 0.60
        # with panic floor=0.40 → result is 0.47
        result_no_panic = analyzer._vol_adjusted_multiplier(0.60, sig, panic_state=False)
        result_panic    = analyzer._vol_adjusted_multiplier(0.60, sig, panic_state=True)
        assert result_no_panic == 0.60, f"non-panic floor should be 0.60, got {result_no_panic}"
        assert result_panic    == 0.47, f"panic floor 0.40 should allow 0.47, got {result_panic}"

    def test_non_panic_elevated_vol_still_floors_at_0_60(self):
        """elevated vol without panic state: floor remains 0.60"""
        from macro_regime import MacroRegimeAnalyzer
        analyzer = MacroRegimeAnalyzer(self.config)
        sig = {"vol_60": 0.22, "vol_ratio": 1.4, "vol_rising": False}
        # base 0.62 (BEAR_CHOP) - 0.04 (elevated) = 0.58; floor=0.60 → result 0.60
        result = analyzer._vol_adjusted_multiplier(0.62, sig, panic_state=False)
        assert result == 0.60, f"non-panic floor should clamp to 0.60, got {result}"

    # --- integration: full analyze() ---

    def test_panic_state_true_when_market_down_with_high_vol(self):
        """declining market (-40% over 2y) + extreme vol → panic_state=True"""
        from macro_regime import MacroRegimeAnalyzer
        # drift=-0.001/day ≈ -22% annual; sigma=0.02/day ≈ 32% annual (EXTREME)
        # over 510 days: expected drawdown ≈ exp(-0.001*504) - 1 ≈ -40%
        df = self._make_ohlcv(n=510, drift=-0.001, sigma=0.02, seed=7)
        analyzer = MacroRegimeAnalyzer(self.config)
        result = analyzer.analyze(df)
        # verify the conditions are actually met before checking the flag
        assert result.details.get("market_drawdown_24m", 0) < -0.20, (
            f"expected drawdown < -20%, got {result.details.get('market_drawdown_24m')}"
        )
        assert result.vol_regime in ("ELEVATED", "EXTREME"), (
            f"expected elevated vol, got {result.vol_regime}"
        )
        assert result.panic_state is True

    def test_panic_state_false_when_market_rising_despite_high_vol(self):
        """rising market (strong drift) + extreme vol → panic_state=False (no drawdown)"""
        from macro_regime import MacroRegimeAnalyzer
        # drift=+0.005/day ≈ 120% annual; at 504 bars expected log-return=2.52
        # sigma=0.02 gives std=0.449 in log space — P(drawdown>20%) ≈ 0 with this drift
        df = self._make_ohlcv(n=510, drift=+0.005, sigma=0.02, seed=7)
        analyzer = MacroRegimeAnalyzer(self.config)
        result = analyzer.analyze(df)
        assert result.details.get("market_drawdown_24m", 0) > -0.20, (
            f"rising market should not have >20% drawdown, got {result.details.get('market_drawdown_24m')}"
        )
        assert result.panic_state is False

    def test_panic_state_false_when_market_down_but_vol_calm(self):
        """market down >20% but vol CALM → panic_state=False (D&M requires both conditions)"""
        from macro_regime import MacroRegimeAnalyzer
        # drift=-0.001/day; sigma=0.005/day ≈ 7.9% annual (CALM: <12%)
        df = self._make_ohlcv(n=510, drift=-0.001, sigma=0.005, seed=7)
        analyzer = MacroRegimeAnalyzer(self.config)
        result = analyzer.analyze(df)
        assert result.details.get("market_drawdown_24m", 0) < -0.20, (
            f"expected drawdown < -20%, got {result.details.get('market_drawdown_24m')}"
        )
        assert result.vol_regime in ("CALM", "NORMAL"), (
            f"expected calm vol, got {result.vol_regime}"
        )
        assert result.panic_state is False

    def test_panic_state_floor_respects_config_override(self):
        """panic_state_floor config key overrides the 0.40 default"""
        from macro_regime import MacroRegimeAnalyzer
        cfg = self.config.copy()
        cfg["panic_state_floor"] = 0.35
        analyzer = MacroRegimeAnalyzer(cfg)
        sig = {"vol_60": 0.35, "vol_ratio": 1.5, "vol_rising": True}
        # base 0.60 - 0.10 - 0.03 = 0.47; custom floor 0.35 → result is 0.47
        result = analyzer._vol_adjusted_multiplier(0.60, sig, panic_state=True)
        assert result == 0.47, f"custom floor 0.35 should not clamp 0.47, got {result}"
