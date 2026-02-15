PARAMETERS = {
    # base
    "base_length_min": 3,
    "base_length_max": 15,
    "base_length_optimal": (5, 10),
    "range_compression_threshold": 0.05,
    # moving averages
    "sma_periods": [10, 20, 50],
    "ma_distance_optimal": 0.03,
    # volatility
    "atr_period": 20,
    # volume
    "volume_avg_period": 20,
    "volume_surge_multiplier": 1.5,
    "dollar_volume_min": 10_000_000,
    # relative strength
    "rs_lookback_periods": [20, 60, 120],
    # risk
    "stop_loss_max_pct": 0.08,
    "risk_reward_min": 3.0,
    # scoring
    "weights": {
        "base_quality": 25,
        "trend_strength": 25,
        "relative_strength": 20,
        "volume_liquidity": 15,
        "risk_reward": 15,
    },
    # filtering
    "min_daily_score": 70,
    "market_regime": True,
}
