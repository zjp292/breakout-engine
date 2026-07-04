"""
Tests for MarketConditionAnalyzer — specifically the momentum universe vol
penalty introduced in P8 (Barroso & Santa-Clara 2015).

Run from project root:
    python -m pytest tests/test_market_condition.py -v
"""

import pytest
import numpy as np
import pandas as pd

from config import PARAMETERS
from market_condition import MarketConditionAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_analyzer(score_momentum_universe_vol: bool = True) -> MarketConditionAnalyzer:
    cfg = PARAMETERS.copy()
    cfg["score_momentum_universe_vol"] = score_momentum_universe_vol
    return MarketConditionAnalyzer(cfg)


def make_compx(n: int = 300, trend: float = 0.0003) -> pd.DataFrame:
    """Synthetic COMPX DataFrame with a gentle uptrend."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 16000 * np.cumprod(1 + trend + np.random.normal(0, 0.005, n))
    df = pd.DataFrame({
        "open":   close * 0.998,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float) * 1000,
    }, index=dates)
    return df


def make_stock_dfs(
    n_stocks: int = 10,
    n_bars: int = 300,
    daily_vol: float = 0.01,
    recent_vol_multiplier: float = 1.0,
) -> dict:
    """
    Build a dict of synthetic stock feature DataFrames.
    recent_vol_multiplier > 1 inflates the last 21 bars' vol to simulate
    a momentum-universe vol spike.
    """
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    result = {}
    rng = np.random.default_rng(42)

    for i in range(n_stocks):
        returns = rng.normal(0, daily_vol, n_bars)
        # spike recent vol if requested
        if recent_vol_multiplier != 1.0:
            returns[-21:] = rng.normal(0, daily_vol * recent_vol_multiplier, 21)
        close = 50.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({"close": close}, index=dates)
        result[f"STOCK{i}"] = df

    return result


# ---------------------------------------------------------------------------
# Tests: _score_momentum_universe_vol
# ---------------------------------------------------------------------------

class TestMomentumUniverseVol:
    """
    Tests for the Barroso & Santa-Clara (2015) momentum universe vol penalty
    applied in MarketConditionAnalyzer._score_momentum_universe_vol().
    """

    def test_no_penalty_in_calm_market(self):
        """When momentum-stock vol is at or below baseline → no penalty (0 pts)."""
        analyzer = make_analyzer()
        stock_dfs = make_stock_dfs(n_stocks=10, daily_vol=0.01, recent_vol_multiplier=1.0)
        penalty, details = analyzer._score_momentum_universe_vol(stock_dfs)
        assert penalty == 0.0

    def test_penalty_fires_when_vol_elevated(self):
        """When recent 21d vol is 50%+ above 63d baseline → -5 pts."""
        analyzer = make_analyzer()
        # 4x vol spike in the last 21 bars — should easily exceed 1.5x ratio
        stock_dfs = make_stock_dfs(n_stocks=10, daily_vol=0.005, recent_vol_multiplier=5.0)
        penalty, details = analyzer._score_momentum_universe_vol(stock_dfs)
        assert penalty == -5.0
        assert details["momentum_universe_vol_ratio"] > 1.5
        assert details["momentum_universe_vol_penalty"] == -5.0

    def test_returns_zero_when_insufficient_stocks(self):
        """Fewer than 5 stocks → skip and return 0 (graceful degradation)."""
        analyzer = make_analyzer()
        stock_dfs = make_stock_dfs(n_stocks=3)
        penalty, details = analyzer._score_momentum_universe_vol(stock_dfs)
        assert penalty == 0.0
        assert details["momentum_universe_vol_ratio"] is None

    def test_returns_zero_when_no_stock_dfs(self):
        """None stock_dfs → skip and return 0."""
        analyzer = make_analyzer()
        penalty, details = analyzer._score_momentum_universe_vol(None)
        assert penalty == 0.0

    def test_returns_zero_for_short_history(self):
        """Fewer than 63+21 bars of history → insufficient data → 0."""
        analyzer = make_analyzer()
        # only 50 bars — below the 84-bar minimum
        stock_dfs = make_stock_dfs(n_stocks=10, n_bars=50)
        penalty, details = analyzer._score_momentum_universe_vol(stock_dfs)
        assert penalty == 0.0

    def test_ratio_stored_in_details(self):
        """details dict always carries momentum_universe_vol_ratio key."""
        analyzer = make_analyzer()
        stock_dfs = make_stock_dfs(n_stocks=10)
        _, details = analyzer._score_momentum_universe_vol(stock_dfs)
        assert "momentum_universe_vol_ratio" in details
        assert "momentum_universe_vol_penalty" in details

    def test_flag_off_skips_penalty(self):
        """When score_momentum_universe_vol=False, even a spike gives 0 penalty."""
        analyzer = make_analyzer(score_momentum_universe_vol=False)
        compx    = make_compx()
        # high-vol stock universe that would normally trigger the penalty
        stock_dfs = make_stock_dfs(n_stocks=10, daily_vol=0.005, recent_vol_multiplier=5.0)
        result = analyzer.analyze(compx, stock_feature_dfs=stock_dfs)
        # penalty suppressed → details should not contain the penalty key
        mom_details = result.details.get("momentum", {})
        assert mom_details.get("momentum_universe_vol_penalty", 0.0) == 0.0

    def test_penalty_reduces_total_score(self):
        """A vol-spiked universe should score lower than a calm one, all else equal."""
        rng = np.random.default_rng(0)
        compx = make_compx()

        calm_dfs   = make_stock_dfs(n_stocks=10, daily_vol=0.01, recent_vol_multiplier=1.0)
        spiked_dfs = make_stock_dfs(n_stocks=10, daily_vol=0.005, recent_vol_multiplier=6.0)

        analyzer = make_analyzer()
        calm_result   = analyzer.analyze(compx, stock_feature_dfs=calm_dfs)
        spiked_result = analyzer.analyze(compx, stock_feature_dfs=spiked_dfs)

        # vol spike should penalise the score
        assert spiked_result.score <= calm_result.score

    def test_penalty_stored_in_mom_details(self):
        """The momentum_universe_vol details appear under result.details['momentum']."""
        analyzer  = make_analyzer()
        compx     = make_compx()
        stock_dfs = make_stock_dfs(n_stocks=10, daily_vol=0.005, recent_vol_multiplier=5.0)
        result    = analyzer.analyze(compx, stock_feature_dfs=stock_dfs)
        mom_det   = result.details.get("momentum", {})
        assert "momentum_universe_vol_ratio"  in mom_det
        assert "momentum_universe_vol_penalty" in mom_det
