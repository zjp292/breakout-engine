PARAMETERS = {
    # base
    "base_length_min": 3,
    "vcp_windows": [10, 20, 40],
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
    "volume_dryup_window": 10,
    "volume_historical_avg_window": 252,
    # relative strength
    "rs_lookback_periods": [20, 60, 120, 252],
    "rs_skip_days": 5,
    "rs_rank_window": 60,
    # prior move
    "prior_move_window": 60,
    # risk
    "stop_loss_max_pct": 0.08,
    "risk_reward_min": 3.0,
    "stop_adr_multiple": 5.0,
    # filtering
    "min_price": 5.0,
    "min_adr_pct": 0.08,  # raised 0.07->0.08 (2026-07): trade-sim sweep, best dev+holdout Sortino/Calmar
    "pct_from_52wk_high_max": 0.30,
    "min_prior_move_pct": 0.75,
    "min_consol_days": 5,
    "require_positive_rs_252": True,
    "weights": {
        "base_quality": 10,
        "trend_strength": 15,
        "relative_strength": 25,
        "volume_profile": 50,
        "risk_reward": 0,
    },
    # analyst coverage scoring (hong, lim & stein 2000) — requires yfinance fetch
    "score_analyst_coverage": True,
    # tsmom gate (Moskowitz, Ooi & Pedersen 2012) — uses existing price history
    "score_tsmom_gate": False,
    # momentum universe vol penalty (Barroso & Santa-Clara 2015) — uses watchlist dfs
    "score_momentum_universe_vol": True,
    # alert thresholds
    "min_score_alert": 80,
    "min_score_watchlist": 70,
    # market regime gating — disabled 2026-07-02: intent was to avoid trading in
    # downtrend eras, but the multiplier didn't track realized EV (flat-to-inverted,
    # worst in the most recent year — filter-passing stocks are already such a
    # selected population that they hold up regardless of index-level regime) and
    # a trade-simulation re-test confirmed it hurt realized Sortino/Calmar too, not
    # just uncalibrated on paper. see experiments.md Phase 4 + regime multiplier
    # follow-up. re-enable only if a redesigned version earns it back.
    "market_regime": False,
    "panic_state_floor": 0.40,
    # paper trading
    "paper_trading": {
        "enabled": True,
        "min_raw_score": 75,
        "account_equity": 100_000,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.20,
        "max_open_positions": 10,
        "min_shares": 1,
        "exit_ma_period": 10,
        "blocked_regimes": ["CAUTION", "DOWNTREND"],
        "blocked_macro_regimes": ["BEAR_TRANSITION", "DOWNTREND", "BEAR_CHOP"],
        "block_on_negative_21d_momentum": True,
    },
}
