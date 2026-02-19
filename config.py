PARAMETERS = {
    # base
    "base_length_min": 3,
    "base_length_max": 60,
    "base_length_optimal": (5, 15),
    "range_compression_threshold": 0.05,
    # moving averages — includes 150/200 for Minervini Stage 2 verification
    "sma_periods": [10, 20, 50, 150, 200],
    "ma_distance_optimal": 0.03,
    # volatility
    "atr_period": 20,
    "adr_period": 20,
    # volume
    "volume_avg_period": 20,
    "volume_surge_multiplier": 1.5,
    "dollar_volume_min": 10_000_000,
    # relative strength
    "rs_lookback_periods": [20, 60, 120],
    # risk
    "stop_loss_max_pct": 0.08,
    "risk_reward_min": 3.0,
    # filtering
    "min_price": 5.0,
    "min_adr_pct": 0.05,
    "pct_from_52wk_high_max": 0.30,  # Must be within 30% of 52-week high
    # scoring — weights = max points per category (total = 100)
    "weights": {
        "base_quality": 25,      # Tight VCP base structure
        "trend_strength": 30,    # Stage 2 + 52wk proximity + MA alignment + prior move
        "relative_strength": 25, # Excess return vs benchmark (corrected formula)
        "volume_profile": 10,    # Liquidity + volume dry-up (single source)
        "risk_reward": 10,       # Stop vs ADR ratio + R-multiple
    },
    # alert thresholds
    "min_score_alert": 80,
    "min_score_watchlist": 70,
    # market regime gating
    "market_regime": True,
}
