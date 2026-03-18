PARAMETERS = {
    # base
    "base_length_min": 3,
    "vcp_windows": [10, 20, 40],             # VCP range comparison windows (short, medium, long)
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
    "volume_dryup_window": 10,               # Lookback for volume dry-up detection
    # relative strength
    "rs_lookback_periods": [20, 60, 120],
    "rs_rank_window": 60,                    # Period used for peer RS rank calculation
    # prior move
    "prior_move_window": 60,                 # Lookback for prior power-move detection
    # risk
    "stop_loss_max_pct": 0.08,
    "risk_reward_min": 3.0,
    # filtering
    "min_price": 5.0,
    "min_adr_pct": 0.05,
    "pct_from_52wk_high_max": 0.30,  # Must be within 30% of 52-week high
    # scoring — weights = max points per category (total = 100)
    # rebalanced 2026-03 based on 793 outcome correlations with 20d max gain:
    #   RS (+0.20), volume (+0.21/+0.43), prior move (+0.21) are the real signal
    #   base quality (-0.03) and trend strength (+0.01) were dead weight at old sizes
    "weights": {
        "base_quality": 15,      # VCP base structure — demoted, low empirical signal
        "trend_strength": 15,    # Stage 2 + proximity + MA — sanity check, not differentiator
        "relative_strength": 30, # RS leadership — strongest confirmed predictor
        "volume_profile": 25,    # Liquidity + dry-up + ADR — empirically dominant
        "risk_reward": 15,       # Stop vs ADR + R-multiple — foundational to the strategy
    },
    # alert thresholds
    "min_score_alert": 80,
    "min_score_watchlist": 70,
    # market regime gating
    "market_regime": True,
}
