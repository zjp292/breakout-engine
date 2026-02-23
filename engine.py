import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from models import ScoreBreakdown
from datetime import datetime, timedelta


class Engine:
    def __init__(self, config):
        self.config = config
        self.features = Features(config)
        self.scoring = Scoring(config)
        self.benchmark_df = None
        self.spy_df = None          # S&P 500 — multi-index confirmation
        self.iwm_df = None          # Russell 2000 — small-cap breadth
        self.market_condition = None  # MarketConditionResult from last run

    def load_pickle(self, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def load_benchmark(self, date_str=None):
        """
        Fetch NASDAQ Composite ($COMPX) and confirmation indices (SPY, IWM)
        from the Schwab API.

        COMPX is the primary benchmark used for RS calculations and market-condition
        scoring.  SPY and IWM provide multi-index confirmation in the market-condition
        analysis but are optional — failure to load them is handled gracefully.
        """
        from ingestion import SchwabAPIClient

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=400)   # extra buffer for 200-day SMA
        end_ts   = int(end_dt.timestamp() * 1000)
        start_ts = int(start_dt.timestamp() * 1000)

        client = SchwabAPIClient()

        print("Fetching NASDAQ Composite ($COMPX) benchmark data...")
        self.benchmark_df = client.get_index_data("$COMPX", start_ts, end_ts)
        print(f"  COMPX loaded: {len(self.benchmark_df)} trading days")

        # SPY — S&P 500 large-cap confirmation
        try:
            self.spy_df = client.get_index_data("SPY", start_ts, end_ts)
            print(f"  SPY   loaded: {len(self.spy_df)} trading days")
        except Exception as e:
            print(f"  Warning: Could not load SPY: {e}")
            self.spy_df = None

        # IWM — Russell 2000 small-cap / risk-on confirmation
        try:
            self.iwm_df = client.get_index_data("IWM", start_ts, end_ts)
            print(f"  IWM   loaded: {len(self.iwm_df)} trading days")
        except Exception as e:
            print(f"  Warning: Could not load IWM: {e}")
            self.iwm_df = None

    def analyze_market_condition(self, feature_dfs: dict) -> float:
        """
        Run the multi-factor market condition analysis and return the regime multiplier.

        Replaces the old simple SMA-based _get_regime_multiplier() with a
        comprehensive 100-point scoring system covering:
          · Index trend quality (SMA alignment + slope + SPY/IWM confirmation)
          · Distribution day count (rolling 25-session window)
          · Follow-through day recency and validity
          · Internal breadth (watchlist stocks above 50 SMA, in Stage 2, near highs)
          · Market momentum and realized volatility

        Sets self.market_condition (MarketConditionResult) for downstream use.
        Returns the regime_multiplier (0.50–1.00).
        """
        from market_condition import MarketConditionAnalyzer

        if not self.config.get("market_regime", True):
            self.market_condition = None
            return 1.0

        if self.benchmark_df is None:
            self.market_condition = None
            return 1.0

        analyzer = MarketConditionAnalyzer(self.config)
        result   = analyzer.analyze(
            compx_df          = self.benchmark_df,
            spy_df            = self.spy_df,
            iwm_df            = self.iwm_df,
            stock_feature_dfs = feature_dfs,
        )

        self.market_condition = result
        self._print_market_condition(result)
        return result.regime_multiplier

    def _print_market_condition(self, mc) -> None:
        """Print a formatted market condition report to stdout."""
        W = 66
        bar = "═" * W

        # Regime colour codes (no external dep; plain ASCII for terminal)
        regime_badges = {
            "BULL":      "▲ BULL",
            "UPTREND":   "↑ UPTREND",
            "MIXED":     "↔ MIXED",
            "CAUTION":   "↓ CAUTION",
            "DOWNTREND": "▼ DOWNTREND",
        }
        badge = regime_badges.get(mc.regime, mc.regime)

        print(f"\n{bar}")
        print(f"  MARKET CONDITION ANALYSIS")
        print(bar)
        print(
            f"  {badge:14s}  score {mc.score:.1f}/100"
            f"   →  stock-score multiplier ×{mc.regime_multiplier:.2f}"
        )
        print(f"  {'─' * (W - 4)}")

        # Index Trend
        conds = mc.details.get("index", {}).get("sma_conditions", {})
        aligned = sum(1 for v in conds.values() if v)
        spy_str = "SPY ✓" if mc.spy_above_200 else ("SPY ✗" if mc.details.get("index", {}).get("spy_above_200") is False else "SPY –")
        iwm_str = "IWM ✓" if mc.iwm_above_200 else ("IWM ✗" if mc.details.get("index", {}).get("iwm_above_200") is False else "IWM –")
        print(
            f"  Index Trend       {mc.index_trend_score:5.1f}/25"
            f"   [{aligned}/6 SMA conditions  {spy_str}  {iwm_str}]"
        )

        # Distribution Days
        d, s = mc.distribution_day_count, mc.stalling_day_count
        dist_flag = "  ⚠ heavy distribution" if d >= 5 else ""
        print(
            f"  Distribution Days {mc.distribution_score:5.1f}/20"
            f"   [{d} D-day{'s' if d != 1 else ''}  {s} stalling{dist_flag}]"
        )

        # Follow-Through Day
        if mc.ftd_found:
            validity = "valid" if mc.ftd_valid else "INVALIDATED"
            ago      = f"{mc.ftd_days_ago}d ago" if mc.ftd_days_ago is not None else "?"
            ftd_str  = f"FTD {mc.ftd_date[:10] if mc.ftd_date else '?'} ({ago}, {validity})"
        else:
            pct_hi = mc.details.get("follow_through", {}).get("pct_from_high")
            if pct_hi is not None:
                ftd_str = f"no FTD — {abs(pct_hi)*100:.1f}% from 52wk high"
            else:
                ftd_str = "no FTD detected in lookback window"
        print(
            f"  Follow-Through    {mc.follow_through_score:5.1f}/20"
            f"   [{ftd_str}]"
        )

        # Internal Breadth
        n = mc.details.get("breadth", {}).get("n_stocks", 0)
        print(
            f"  Internal Breadth  {mc.breadth_score:5.1f}/20"
            f"   [{mc.pct_above_50sma*100:.0f}% >50d  "
            f"{mc.pct_in_stage2*100:.0f}% Stage2  "
            f"{mc.pct_near_52wk_high*100:.0f}% near high  "
            f"n={n}]"
        )

        # Momentum & Volatility
        rv_pct  = mc.realized_vol_annualized * 100
        roc_pct = mc.compx_roc_21d * 100
        print(
            f"  Momentum/Vol      {mc.momentum_score:5.1f}/15"
            f"   [21d ROC {roc_pct:+.1f}%  Realized vol {rv_pct:.1f}%]"
        )

        print(bar + "\n")

    def process_stock(self, date_str=None, debug=False):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        data_dir = Path(f"data/{date_str}")

        if not data_dir.exists() or not list(data_dir.glob("*.pkl")):
            print(f"No data found for {date_str}. Running ingestion...")
            from ingestion import Ingestor

            ingestor = Ingestor()
            ingestor.mergefiles()

            if not ingestor.ticker_list:
                print("No tickers found in today's watchlist exports. Exiting.")
                return {}, None

            print(f"Fetching data for {len(ingestor.ticker_list)} tickers...")
            ingestor.get_data()

        pickle_files = list(data_dir.glob("*.pkl"))

        if not pickle_files:
            print(f"No pickle files found in {data_dir} after ingestion.")
            return {}, None

        print(f"Found {len(pickle_files)} pickle files in {data_dir}\n")

        # Load NASDAQ Composite benchmark (+ SPY, IWM) from Schwab API
        try:
            self.load_benchmark(date_str)
        except Exception as e:
            print(f"Warning: Could not load benchmark data: {e}")
            self.benchmark_df = None

        # Dictionary to store feature dataframes (scored after market condition)
        scored_dfs = {}

        # track filtered tickers
        filter_failures = {}

        for pickle_file in pickle_files:
            # get symbol
            symbol = pickle_file.stem.split("-")[0]

            print(f"Processing {symbol}...")

            try:
                df = self.load_pickle(str(pickle_file))

                # Convert Schwab ms datetime to a normalized DatetimeIndex
                if "datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
                    df = df.set_index("datetime")
                    df.index = df.index.normalize()

                # add features
                feature_df = self.features.add_all_features(
                    df, benchmark_df=self.benchmark_df
                )

                scored_dfs[symbol] = feature_df

            except Exception as e:
                print(f"- Error processing {symbol}: {e}")
                import traceback

                if debug:
                    traceback.print_exc()
                continue

        print(f"\nProcessed {len(scored_dfs)} stocks successfully")

        # Calculate RS ranks across all stocks (peer comparison)
        print("\nCalculating RS ranks vs peers...")
        rs_ranks = self.features.calculate_rs_rank(scored_dfs, self.benchmark_df)

        # ── Market Condition Analysis ──────────────────────────────────────────
        # Runs AFTER features are computed so internal breadth (% above 50 SMA,
        # % in Stage 2, % near highs) can be drawn from the full feature DFs.
        # The resulting regime_multiplier gates all stock scores downward in weak
        # markets — matching how Qullamaggie and Minervini reduce exposure.
        try:
            regime_mult = self.analyze_market_condition(scored_dfs)
        except Exception as e:
            print(f"Warning: Market condition analysis failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            regime_mult = 1.0

        self.scoring.regime_multiplier = regime_mult

        print("\nScoring stocks...")
        final_scored_dfs = {}
        for symbol, feature_df in scored_dfs.items():
            try:
                # Get RS ranks for this symbol
                symbol_rs_ranks = rs_ranks.get(symbol, {})

                # Score the dataframe
                scored_df = self.scoring.score_dataframe(
                    feature_df, symbol=symbol, rs_ranks=symbol_rs_ranks
                )

                final_scored_dfs[symbol] = scored_df

                # Debug: Check if latest row passes filters
                if debug and not scored_df.empty:
                    latest_row = scored_df.iloc[-1]
                    passes = latest_row.get("passes_filters", False)
                    if not passes:
                        # Get the failure reasons
                        _, failures = self.scoring.apply_hard_filters(latest_row)
                        filter_failures[symbol] = failures
                        print(f"  ⚠ {symbol}: Failed - {', '.join(failures)}")
                    else:
                        score = latest_row.get("total_score", 0)
                        rs_rank = latest_row.get("rs_comp_60", 0)
                        print(f"  ✓ {symbol}: Score={score:.1f}, RS_60={rs_rank:.2f}")

            except Exception as e:
                print(f"  ✗ Error scoring {symbol}: {e}")
                if debug:
                    import traceback

                    traceback.print_exc()

        # Debug: Show filter failure summary
        if debug and filter_failures:
            print("\n" + "=" * 80)
            print("FILTER FAILURE SUMMARY")
            print("=" * 80)
            failure_counts = {}
            for symbol, failures in filter_failures.items():
                for failure in failures:
                    failure_counts[failure] = failure_counts.get(failure, 0) + 1

            for failure, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                print(f"{count:3d} stocks: {failure}")
            print()

        watchlist = None
        if final_scored_dfs:
            print("\nGenerating watchlist summary...")
            watchlist = self.scoring.create_watchlist_summary(final_scored_dfs)

            if not watchlist.empty:
                print(f"✓ Watchlist created with {len(watchlist)} stocks")

            else:
                print("⚠ No stocks passed the filters")

        return final_scored_dfs, watchlist


class Features:
    def __init__(self, config):
        self.config = config

    def add_moving_averages(self, df):
        periods = self.config.get("sma_periods", [10, 20, 50])

        for period in periods:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

        return df

    def add_ma_relationships(self, df):
        df["distance_from_sma10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
        df["distance_from_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["distance_from_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

        df["ma_alignment"] = (df["sma_10"] > df["sma_20"]) & (
            df["sma_20"] > df["sma_50"]
        )

        df["ma_slope_10"] = df["sma_10"].pct_change(periods=5)
        df["ma_slope_20"] = df["sma_20"].pct_change(periods=5)
        df["ma_slope_50"] = df["sma_50"].pct_change(periods=5)

        df["mas_rising"] = (
            (df["ma_slope_10"] > 0) & (df["ma_slope_20"] > 0) & (df["ma_slope_50"] > 0)
        )

        # Stage 2 long-term structure (Minervini's primary template)
        # Requires 50 > 150 > 200 SMA, price above 150 SMA, 200 SMA trending up
        if "sma_150" in df.columns and "sma_200" in df.columns:
            df["distance_from_sma150"] = (df["close"] - df["sma_150"]) / df["sma_150"]
            df["distance_from_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
            sma200_slope = df["sma_200"].pct_change(periods=20)
            df["stage2"] = (
                (df["close"] > df["sma_150"])
                & (df["sma_50"] > df["sma_150"])
                & (df["sma_150"] > df["sma_200"])
                & (sma200_slope > 0)
            )
        else:
            df["stage2"] = False
            df["distance_from_sma150"] = np.nan
            df["distance_from_sma200"] = np.nan

        return df

    def add_atr(self, df):
        period = self.config["atr_period"]

        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.rolling(window=period).mean()

        return df

    def add_range_metrics(self, df):
        # daily range percent
        df["daily_range"] = df["high"] - df["low"]
        df["daily_range_pct"] = df["daily_range"] / df["close"]

        # adr_pct
        adr_period = self.config.get("adr_period", 20)
        df["adr_pct"] = df["daily_range_pct"].rolling(window=adr_period).mean()

        return df

    def add_volume_metrics(self, df):
        volume_period = self.config.get("volume_avg_period", 20)

        df["volume_sma_20"] = df["volume"].rolling(window=volume_period).mean()
        df["relative_volume"] = df["volume"] / df["volume_sma_20"]
        df["dollar_volume"] = df["close"] * df["volume"]

        def calculate_slope(series):
            if len(series) < 2:
                return 0

            x = np.arange(len(series))
            y = series.values

            if np.all(y == y[0]):
                return 0

            slope = np.polyfit(x, y, 1)[0]
            return slope

        df["volume_trend"] = (
            df["volume"].rolling(window=10).apply(calculate_slope, raw=False)
        )

        return df

    def detect_volume_drying(self, df, lookback):
        # recent volume average
        recent_vol = df["volume"].rolling(window=lookback).mean()

        # baseline vol avg
        baseline_vol = df["volume"].rolling(window=20).mean().shift(lookback)

        df["volume_dryup_ratio"] = recent_vol / baseline_vol

        # volume is declining if ratio < 1 and trend is negative
        df["volume_declining"] = (df["volume_dryup_ratio"] < 1.0) & (
            df["volume_trend"] < 0
        )

        return df

    def detect_consolidation_range(self, df, lookback=None):
        if lookback is None:
            lookback = self.config.get("base_length_max", 15)

        # Calculate range over rolling window
        rolling_high = df["high"].rolling(window=lookback).max()
        rolling_low = df["low"].rolling(window=lookback).min()

        df[f"consol_range_{lookback}"] = (rolling_high - rolling_low) / df["close"]

        # Top of the current base — the breakout price level.
        # Stored here so charting and backtesting can reference it directly.
        df["breakout_level"] = rolling_high

        # Detect tight consolidation (range < threshold)
        tight_threshold = self.config.get("range_compression_threshold", 0.05)
        df["is_tight_consolidation"] = df[f"consol_range_{lookback}"] < tight_threshold

        # tight range
        df["consol_days"] = (
            df["is_tight_consolidation"]
            .groupby(
                (
                    df["is_tight_consolidation"] != df["is_tight_consolidation"].shift()
                ).cumsum()  # rock_eyebrow.png
            )
            .cumsum()
        )

        # only keep count if currently in consolidation
        df.loc[~df["is_tight_consolidation"], "consol_days"] = 0

        return df

    def calculate_base_depth(self, df, lookback=20):
        # base_depth: (recent_high - current_close) / recent_high
        # days_from_high: days since recent high
        rolling_high = df["high"].rolling(window=lookback).max()
        df["base_depth"] = (rolling_high - df["close"]) / rolling_high

        # Days since high
        high_idx = (
            df["high"]
            .rolling(window=lookback)
            .apply(lambda x: lookback - x.argmax() - 1, raw=True)
        )
        df["days_from_high"] = high_idx

        return df

    def calculate_relative_strength(self, df, benchmark_df, benchmark_name="SPY"):
        """
        Calculate relative strength vs benchmark as excess return.

        RS = Stock % Change - Benchmark % Change  (excess return / alpha)

        This formulation is correct in all market conditions:
        - Bull market: stock +20%, benchmark +5% → RS = +15%  (outperforming)
        - Bear market: stock +5%, benchmark -5% → RS = +10%   (strong RS signal)
        - Laggard:     stock -5%, benchmark +5% → RS = -10%   (underperforming)

        The old ratio formula broke in bear markets: a stock that held up while
        the market sold off would produce a negative ratio and get 0 RS points —
        the exact opposite of the correct signal.

        Args:
            df: Stock dataframe
            benchmark_df: Benchmark dataframe with same date index
            benchmark_name: Name of benchmark for column naming

        Returns df with:
        - rs_{benchmark}_20: 20-day excess return vs benchmark
        - rs_{benchmark}_60: 60-day excess return vs benchmark
        - rs_{benchmark}_120: 120-day excess return vs benchmark
        """
        benchmark_aligned = benchmark_df.reindex(df.index, method="ffill")

        for period in [20, 60, 120]:
            stock_pct_change = df["close"].pct_change(periods=period)

            if "close" in benchmark_aligned.columns:
                benchmark_pct_change = benchmark_aligned["close"].pct_change(
                    periods=period
                )
            else:
                benchmark_pct_change = benchmark_aligned.iloc[:, 0].pct_change(
                    periods=period
                )

            # Excess return: positive = outperforming, negative = underperforming
            # Handles bear markets correctly — no sign inversion
            df[f"rs_{benchmark_name.lower()}_{period}"] = (
                stock_pct_change - benchmark_pct_change
            )

        return df

    def calculate_rs_rank(self, stock_dfs, benchmark_df=None):
        """
        Calculate RS rank (percentile) for each stock vs the entire watchlist.
        This shows which stocks are the strongest performers relative to peers.

        Args:
            stock_dfs: Dict of {symbol: dataframe} for all stocks in watchlist
            benchmark_df: Optional benchmark dataframe

        Returns:
            Dict of {symbol: {date: rs_rank}} where rs_rank is 0-100 percentile
        """
        # For each date, calculate all stocks' performance and rank them
        # This needs to be called at the Engine level with all stocks

        # Get common dates across all stocks
        all_dates = set()
        for df in stock_dfs.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        rs_ranks = {symbol: {} for symbol in stock_dfs.keys()}

        # For each date, rank stocks by their 60-day performance
        for date in all_dates:
            daily_performances = {}

            for symbol, df in stock_dfs.items():
                if date not in df.index:
                    continue
                # Use 60-day % change as the ranking metric
                denom = df["close"].shift(60).loc[date]
                if pd.isna(denom) or denom == 0:
                    continue  # insufficient history or data error — skip this bar
                perf = df.loc[date, "close"] / denom - 1
                if not pd.isna(perf):
                    daily_performances[symbol] = perf

            # Rank the stocks (percentile)
            if daily_performances:
                sorted_symbols = sorted(daily_performances.items(), key=lambda x: x[1])
                for rank, (symbol, _) in enumerate(sorted_symbols):
                    percentile = (rank / len(sorted_symbols)) * 100
                    rs_ranks[symbol][date] = percentile

        return rs_ranks

    # big move up before consolidation
    def detect_prior_moves(self, df, lookback=60):
        # prior_move_pct: max % gain in lookback period
        # days_since_power_move: days since 20%+ move
        rolling_low = df["low"].rolling(window=lookback).min()
        df["prior_move_pct"] = (df["close"] - rolling_low) / rolling_low

        # Detect power moves (20%+ gains)
        power_move_threshold = 0.20
        df["is_power_move"] = df["prior_move_pct"] >= power_move_threshold

        # Days since last power move
        def days_since_true(series):
            """Count days since last True value"""
            last_true_idx = np.where(series)[0]
            if len(last_true_idx) == 0:
                return len(series)
            return len(series) - last_true_idx[-1] - 1

        df["days_since_power_move"] = (
            df["is_power_move"]
            .rolling(window=lookback)
            .apply(days_since_true, raw=True)
        )

        return df

    def calculate_higher_lows(self, df, lookback=10):
        # higher_lows: boolean indicating uptrend structure
        # Find local lows (price lower than neighbors)
        df["is_swing_low"] = (df["low"] < df["low"].shift(1)) & (
            df["low"] < df["low"].shift(-1)
        )

        # Get swing low values
        swing_lows = df["low"].where(df["is_swing_low"])

        # Check if swing lows are rising
        # Compare current swing low to previous swing low
        last_swing_low = swing_lows.ffill()
        prev_swing_low = last_swing_low.shift(1)

        df["higher_lows"] = last_swing_low > prev_swing_low

        # Count higher lows in lookback period
        df["swing_low_count"] = df["higher_lows"].rolling(window=lookback).sum()

        return df

    def calculate_52wk_proximity(self, df):
        """
        Calculate proximity to 52-week high.

        Both Qullamaggie and Minervini exclusively trade stocks near their highs.
        Minervini's template: within 25% of 52-week high.
        Qullamaggie: "I only buy stocks near their highs."

        Returns df with:
        - 52wk_high: rolling 252-day high price
        - pct_from_52wk_high: % below 52wk high (0.0 = at high, -0.20 = 20% below)
        """
        df["52wk_high"] = df["high"].rolling(window=252, min_periods=100).max()
        df["pct_from_52wk_high"] = (df["close"] - df["52wk_high"]) / df["52wk_high"]
        return df

    def detect_vcp_contractions(self, df):
        """
        Detect Volatility Contraction Pattern (VCP) structure.

        Minervini's VCP requires a series of contractions, each narrower than the last.
        We measure range over three successive windows (10, 20, 40 days) as a proxy.
        True VCP: range_10 < range_20 < range_40 (each window tighter than the next).

        Returns df with:
        - range_10/20/40: price range as % of close for each window
        - vcp_contracting: True if all three windows show progressive narrowing
        - vcp_contraction_ratio: current 10d range / 40d range (lower = tighter)
        """
        range_10 = (
            df["high"].rolling(10).max() - df["low"].rolling(10).min()
        ) / df["close"]
        range_20 = (
            df["high"].rolling(20).max() - df["low"].rolling(20).min()
        ) / df["close"]
        range_40 = (
            df["high"].rolling(40).max() - df["low"].rolling(40).min()
        ) / df["close"]

        df["range_10"] = range_10
        df["range_20"] = range_20
        df["range_40"] = range_40

        # True VCP: each successive window is progressively tighter
        df["vcp_contracting"] = (range_10 < range_20) & (range_20 < range_40)

        # How tight is the current range vs the broadest recent window?
        # 0.20 means current 10d range is only 20% of the 40d range = strong contraction
        df["vcp_contraction_ratio"] = range_10 / range_40.clip(lower=0.001)

        return df

    """
    risk management stuff
    """

    def calculate_stop(self, df):
        """
        Calculate stop loss levels based on Qullamaggie methodology.

        Entry Rule: Stop at the low of the consolidation day (low of current bar)
        Trailing Rule: Once profitable, trail stop to close below 10 SMA

        Returns df with:
        - stop_level: The actual price level for the stop
        - stop_distance_pct: Distance from current price to stop as %
        - trailing_stop_triggered: Boolean if stock closed below 10 SMA
        """
        # Initial stop: Low of the current day (conservative entry)
        # If low == close, use a small buffer (0.5% below close) to avoid division by zero
        df["stop_level"] = df.apply(
            lambda row: (
                row["low"] if row["low"] < row["close"] else row["close"] * 0.995
            ),  # 0.5% below close as minimum stop
            axis=1,
        )

        # Stop distance as percentage
        # Add small epsilon to avoid division by zero
        df["stop_distance_pct"] = (df["close"] - df["stop_level"]) / (
            df["close"] + 0.0001
        )

        # Ensure stop distance is always positive and reasonable
        df["stop_distance_pct"] = df["stop_distance_pct"].clip(lower=0.001, upper=0.20)

        # Trailing stop rule: Close below 10 SMA (for position management)
        df["trailing_stop_triggered"] = df["close"] < df["sma_10"]

        return df

    def calculate_rr(self, df):
        """
        Calculate risk/reward ratio.

        Target: Based on prior move and consolidation breakout patterns
        Risk: Stop distance

        Returns df with:
        - target_level: Price target based on base depth and prior moves
        - potential_gain_pct: Potential gain to target
        - potential_r: R-multiple (reward/risk ratio)
        """
        # Target calculation based on consolidation base depth
        # Conservative: 1x the base depth from breakout
        # Aggressive: 2x the base depth or prior move high

        # Method 1: Target based on consolidation range projection
        consol_range_pct = df.get("consol_range_15", pd.Series([0.05] * len(df)))
        if isinstance(consol_range_pct, (int, float)):
            consol_range_pct = pd.Series([consol_range_pct] * len(df), index=df.index)
        base_target_1 = df["close"] * (1 + consol_range_pct)

        # Method 2: Target based on measured move (prior power move)
        lookback = 60
        rolling_high = df["high"].rolling(window=lookback).max()
        base_target_2 = rolling_high

        # Use the higher of the two targets, cap at 40% gain
        # Qullamaggie holds breakouts for 30-100%+ moves; capping at 15% was severely
        # underestimating high-quality setups and producing artificially low R-multiples
        max_target = df["close"] * 1.40
        df["target_level"] = pd.concat([base_target_1, base_target_2], axis=1).max(
            axis=1
        )
        df["target_level"] = df["target_level"].clip(upper=max_target)

        # Potential gain percentage
        df["potential_gain_pct"] = (df["target_level"] - df["close"]) / df["close"]
        df["potential_gain_pct"] = df["potential_gain_pct"].clip(lower=0)

        # Risk/Reward ratio (R-multiple)
        # Avoid division by zero - already clipped in calculate_stop
        df["potential_r"] = df["potential_gain_pct"] / df["stop_distance_pct"]
        df["potential_r"] = df["potential_r"].clip(upper=10)  # Cap at 10R

        return df

    def add_all_features(self, df, benchmark_df=None):
        """
        Add all technical features to dataframe.

        Args:
            df: Stock dataframe
            benchmark_df: Optional benchmark (SPY/QQQ) for relative strength
        """
        df = df.copy()

        # Technical indicators
        df = self.add_moving_averages(df)
        df = self.add_atr(df)
        df = self.add_range_metrics(df)
        df = self.add_volume_metrics(df)

        # Trend analysis
        df = self.add_ma_relationships(df)

        # Base structure
        df = self.detect_consolidation_range(df)
        df = self.calculate_base_depth(df)

        # Historical patterns
        df = self.detect_prior_moves(df)
        df = self.calculate_higher_lows(df)

        # 52-week proximity and VCP contraction structure
        df = self.calculate_52wk_proximity(df)
        df = self.detect_vcp_contractions(df)

        # Volume patterns
        df = self.detect_volume_drying(df, lookback=10)

        # Risk/Reward (NEW)
        df = self.calculate_stop(df)
        df = self.calculate_rr(df)

        # Relative Strength vs NASDAQ Composite - if benchmark provided
        if benchmark_df is not None:
            df = self.calculate_relative_strength(
                df, benchmark_df, benchmark_name="COMP"
            )

        return df


class Scoring:
    """
    Scores stocks based on Qullamaggie breakout + Minervini VCP principles.

    Scoring Philosophy:
    - Base Quality (25pts):      Tight VCP base — range compression + length + contraction series
    - Trend Strength (30pts):    Stage 2 MA stack + 52wk proximity + MA alignment + prior move
    - Relative Strength (25pts): Excess return vs benchmark (corrected formula, bear-safe)
    - Volume Profile (10pts):    Liquidity + volume dry-up (single, non-duplicated)
    - Risk/Reward (10pts):       Stop distance relative to ADR + R-multiple

    Market Regime: A multiplier (0.70–1.0) is applied based on benchmark trend.
    In downtrends, the same setup scores lower — matching how both traders go to cash.

    Grade Scale (after regime adjustment):
    90-100: A+ (Exceptional setup — immediate alert)
    80-89:  A  (Strong setup — high priority)
    70-79:  B  (Good setup — watch closely)
    60-69:  C  (Marginal — monitor)
    <60:    D/F (Pass)
    """

    def __init__(self, config):
        self.config = config
        self.weights = config.get(
            "weights",
            {
                "base_quality": 25,
                "trend_strength": 30,
                "relative_strength": 25,
                "volume_profile": 10,
                "risk_reward": 10,
            },
        )
        self.min_score_alert = config.get("min_score_alert", 80)
        self.min_score_watchlist = config.get("min_score_watchlist", 70)
        self.regime_multiplier = 1.0  # set by Engine before scoring

    """
    base 
    """

    def score_base_quality(self, row: pd.Series):
        """
        Score consolidation quality based on Qullamaggie + Minervini VCP criteria.

        Components:
        - Consolidation tightness (0-10pts): Range compression of the current base
        - Base length (0-8pts): Accommodates tight flags (5-15d) and VCP bases (15-45d)
        - VCP contraction structure (0-7pts): Is this base part of a narrowing series?

        Note: Volume dry-up lives exclusively in score_volume_profile to avoid
        double-counting. This category is purely about price structure.

        Perfect Score: <3% current range, 5-15 day base, clear VCP contraction series
        """
        score = 0.0
        details = {}

        # 1. CONSOLIDATION TIGHTNESS (0-10 points)
        # The tighter the base, the more energy coiled for the breakout
        consol_range = row.get("consol_range_15", 1.0)

        if consol_range <= 0.02:    # <2% — exceptional (VCP handle)
            tightness_score = 10.0
        elif consol_range <= 0.03:  # 2-3% — excellent
            tightness_score = 9.0
        elif consol_range <= 0.05:  # 3-5% — good
            tightness_score = 7.0
        elif consol_range <= 0.08:  # 5-8% — acceptable
            tightness_score = 4.0
        else:                        # >8% — too loose
            tightness_score = 0.0

        score += tightness_score
        details["tightness"] = tightness_score

        # 2. BASE LENGTH (0-8 points)
        # Qullamaggie: tight flags 5-15 days. Minervini VCPs: up to 8 weeks.
        # Extended ranges accommodated — penalize only very short or very long.
        consol_days = row.get("consol_days", 0)

        if 5 <= consol_days <= 15:     # Sweet spot: tight flag or short VCP
            length_score = 8.0
        elif 15 < consol_days <= 30:   # Classic VCP base
            length_score = 7.0
        elif 30 < consol_days <= 45:   # Longer VCP — still valid
            length_score = 6.0
        elif 3 <= consol_days < 5:     # Micro-flag — too short but acceptable
            length_score = 5.0
        elif 45 < consol_days <= 60:   # Extended — momentum may be fading
            length_score = 3.0
        elif consol_days > 60:         # Too long — institutional interest likely lost
            length_score = 1.0
        else:                           # < 3 days — not a real base
            length_score = 0.0

        score += length_score
        details["base_length"] = length_score

        # 3. VCP CONTRACTION STRUCTURE (0-7 points)
        # Is the stock in a progressively narrowing series of contractions?
        # range_10 < range_20 < range_40 = the "VC" shape of VCP
        vcp_contracting = row.get("vcp_contracting", False)
        vcp_ratio = row.get("vcp_contraction_ratio", 1.0)

        if vcp_contracting and vcp_ratio <= 0.25:   # Current range ≤25% of 40d range
            vcp_score = 7.0                          # Very strong contraction series
        elif vcp_contracting and vcp_ratio <= 0.40: # Clear multi-stage contraction
            vcp_score = 5.0
        elif vcp_contracting:                        # Contracting but modest
            vcp_score = 3.0
        elif vcp_ratio <= 0.60:                      # Partially contracted, not ordered
            vcp_score = 1.0
        else:                                         # Flat or expanding range
            vcp_score = 0.0

        score += vcp_score
        details["vcp_contraction"] = vcp_score

        return score, details

    # ============================================
    # TREND STRENGTH SCORING (0-30 points)
    # ============================================

    def score_trend_strength(self, row: pd.Series):
        """
        Score underlying trend structure — Qullamaggie + Minervini combined.

        Components:
        - Stage 2 structure (0-10pts): Long-term MA alignment (50>150>200, price>150)
        - 52-week high proximity (0-8pts): Near all-time/52wk highs
        - Short-term MA structure (0-7pts): 10>20>50 aligned + rising + price above 10 SMA
        - Prior power move (0-5pts): Strong trend leg before the current base

        The old SMA distance scoring used abs() — treating "2% below" and "2% above"
        identically. Being below the 10 SMA during a base is a red flag, not neutral.
        This version is directional.

        Perfect Score: Stage 2, within 5% of 52wk high, perfect MA structure, 40%+ prior move
        """
        score = 0.0
        details = {}

        # 1. STAGE 2 LONG-TERM STRUCTURE (0-10 points)
        # Minervini's primary template: 50>150>200, price>150 SMA, 200 trending up
        # Institutional money only flows into confirmed Stage 2 uptrends
        stage2 = row.get("stage2", False)
        dist_150 = row.get("distance_from_sma150", np.nan)
        dist_200 = row.get("distance_from_sma200", np.nan)

        if stage2:
            stage_score = 10.0  # Full: 50>150>200, price>150, 200d trending up
        elif not pd.isna(dist_150) and dist_150 > 0 and not pd.isna(dist_200) and dist_200 > 0:
            stage_score = 6.0   # Price above both 150/200 but MA stack imperfect
        elif not pd.isna(dist_200) and dist_200 > 0:
            stage_score = 3.0   # At least above 200 SMA
        elif pd.isna(dist_150):
            # Not enough history for 150/200 SMA — partial credit if short-term looks good
            ma_alignment = row.get("ma_alignment", False)
            stage_score = 4.0 if ma_alignment else 0.0
        else:
            stage_score = 0.0   # Below long-term MAs — wrong side of the trend

        score += stage_score
        details["stage2"] = stage_score

        # 2. 52-WEEK HIGH PROXIMITY (0-8 points)
        # Qullamaggie: "I only buy stocks near their highs."
        # Minervini template: within 25% of 52wk high
        # Stocks making new highs have the least overhead supply
        pct_from_high = row.get("pct_from_52wk_high", -1.0)  # 0 = at high, -0.20 = 20% below

        if pct_from_high >= -0.05:    # Within 5% — at or near breakout point
            proximity_score = 8.0
        elif pct_from_high >= -0.10:  # 5-10% below — tight base near highs
            proximity_score = 7.0
        elif pct_from_high >= -0.15:  # 10-15% below
            proximity_score = 5.0
        elif pct_from_high >= -0.20:  # 15-20% below
            proximity_score = 3.0
        elif pct_from_high >= -0.25:  # 20-25% below (Minervini's soft limit)
            proximity_score = 1.0
        else:                          # >25% below — overhead supply is a headwind
            proximity_score = 0.0

        score += proximity_score
        details["proximity_to_high"] = proximity_score

        # 3. SHORT-TERM MA STRUCTURE (0-7 points)
        # 10>20>50 aligned AND all rising AND price holding above 10 SMA
        # The old code used abs(distance) — now directional: below 10 SMA ≠ above 10 SMA
        ma_alignment = row.get("ma_alignment", False)   # sma_10 > sma_20 > sma_50
        mas_rising = row.get("mas_rising", False)
        dist_10 = row.get("distance_from_sma10", -1.0)  # positive = above, negative = below

        above_10sma = dist_10 >= -0.02  # holding above or within 2% (testing support)

        if ma_alignment and mas_rising and above_10sma:
            ma_score = 7.0   # Perfect: aligned, rising, price holding above
        elif ma_alignment and mas_rising:
            ma_score = 5.0   # Aligned and rising but price below 10 SMA
        elif ma_alignment and above_10sma:
            ma_score = 4.0   # Aligned but not all slopes positive yet
        elif ma_alignment:
            ma_score = 2.0   # Aligned but weak
        elif row.get("sma_10", 0) > row.get("sma_20", 0):
            ma_score = 1.0   # Only partial alignment (10>20)
        else:
            ma_score = 0.0   # Poor structure

        score += ma_score
        details["ma_structure"] = ma_score

        # 4. PRIOR POWER MOVE (0-5 points)
        # The "flag pole" — a strong discrete move before the consolidation
        # Without a prior move, there is no bull flag; it's just chop
        prior_move = row.get("prior_move_pct", 0.0)
        days_since_move = row.get("days_since_power_move", 999)

        if prior_move >= 0.40 and days_since_move <= 30:    # 40%+ recent — big move
            power_score = 5.0
        elif prior_move >= 0.30 and days_since_move <= 45:  # 30%+ strong
            power_score = 4.0
        elif prior_move >= 0.20 and days_since_move <= 60:  # 20%+ solid
            power_score = 3.0
        elif prior_move >= 0.15:                             # 15%+ modest
            power_score = 1.5
        else:                                                 # No clear prior move
            power_score = 0.0

        score += power_score
        details["prior_power_move"] = power_score

        return score, details

    # ============================================
    # RELATIVE STRENGTH SCORING (0-25 points)
    # ============================================

    def score_relative_strength(self, row: pd.Series, rs_rank: float = None):
        """
        Score outperformance vs NASDAQ Composite.

        RS values are now excess returns (stock% - benchmark%), not ratios.
        This correctly handles all market conditions — a stock that rises while
        the market falls shows positive RS, as it should.

        Thresholds are calibrated for excess return values:
        +0.10 = stock outperformed benchmark by 10 percentage points

        Components:
        - 20-day RS (0-7pts): Short-term leadership signal
        - 60-day RS (0-10pts): Medium-term leadership — most predictive
        - RS percentile rank (0-8pts): Rank vs all stocks in this watchlist

        Perfect Score: +10% 20d excess, +20% 60d excess, top 10% of peers
        """
        score = 0.0
        details = {}

        # 1. SHORT-TERM RS — 20 days (0-7 points)
        # Excess return: stock 20d return minus benchmark 20d return
        rs_20 = row.get("rs_comp_20", 0.0)

        if rs_20 >= 0.10:    # +10%+ excess — exceptional short-term leader
            rs_20_score = 7.0
        elif rs_20 >= 0.05:  # +5-10% excess — strong
            rs_20_score = 5.0
        elif rs_20 >= 0.02:  # +2-5% excess — moderate outperformance
            rs_20_score = 3.0
        elif rs_20 >= 0.00:  # Neutral/slight
            rs_20_score = 1.0
        else:                 # Underperforming
            rs_20_score = 0.0

        score += rs_20_score
        details["rs_20_day"] = rs_20_score

        # 2. MEDIUM-TERM RS — 60 days (0-10 points)
        # Most predictive timeframe for breakout success
        # Minervini's RS line emphasis: this should be rising and near new highs
        rs_60 = row.get("rs_comp_60", 0.0)

        if rs_60 >= 0.20:    # +20%+ excess — true market leader
            rs_60_score = 10.0
        elif rs_60 >= 0.15:  # +15-20%
            rs_60_score = 8.0
        elif rs_60 >= 0.10:  # +10-15%
            rs_60_score = 6.0
        elif rs_60 >= 0.05:  # +5-10%
            rs_60_score = 4.0
        elif rs_60 >= 0.00:  # Neutral
            rs_60_score = 1.0
        else:                 # Underperforming — wrong stock
            rs_60_score = 0.0

        score += rs_60_score
        details["rs_60_day"] = rs_60_score

        # 3. RS PERCENTILE RANK vs PEERS (0-8 points)
        # Rank within the current watchlist — buy the leaders, not the laggards
        # Analogous to IBD's RS Rating (Minervini specifically targets RS 80+)
        if rs_rank is not None:
            if rs_rank >= 90:    # Top 10% — elite leader
                rank_score = 8.0
            elif rs_rank >= 80:  # Top 20%
                rank_score = 6.0
            elif rs_rank >= 70:  # Top 30%
                rank_score = 4.0
            elif rs_rank >= 60:  # Top 40%
                rank_score = 2.0
            else:                 # Below 60th percentile — not a leader
                rank_score = 0.0

            score += rank_score
            details["rs_rank"] = rank_score
        else:
            details["rs_rank"] = 0.0

        return score, details

    # ============================================
    # VOLUME PROFILE SCORING (0-10 points)
    # ============================================

    def score_volume_profile(self, row: pd.Series):
        """
        Score liquidity and volume characteristics.

        Volume dry-up lives exclusively here — it was previously double-counted
        (also in score_base_quality for 7 pts). Removing the duplicate means
        the signal is faithfully weighted once.

        Components:
        - Dollar volume (0-3pts): Institutional-grade liquidity
        - Volume dry-up (0-4pts): Contraction into the base (single source of truth)
        - ADR % (0-3pts): Volatility above the minimum — differentiates quality

        Perfect Score: >$100M dollar volume, strong dry-up, 8%+ ADR
        """
        score = 0.0
        details = {}

        # 1. DOLLAR VOLUME (0-3 points)
        # Institutional liquidity — can we get in and out without slippage?
        dollar_vol = row.get("dollar_volume", 0)
        min_dollar_vol = self.config.get("dollar_volume_min", 10_000_000)

        if dollar_vol >= min_dollar_vol * 10:   # >$100M — institutional grade
            dv_score = 3.0
        elif dollar_vol >= min_dollar_vol * 5:  # >$50M
            dv_score = 2.5
        elif dollar_vol >= min_dollar_vol * 2:  # >$20M
            dv_score = 2.0
        elif dollar_vol >= min_dollar_vol:       # Meets minimum ($10M)
            dv_score = 1.5
        else:
            dv_score = 0.0

        score += dv_score
        details["dollar_volume"] = dv_score

        # 2. VOLUME DRY-UP (0-4 points) — SINGLE SOURCE OF TRUTH
        # Volume should contract significantly into the base.
        # This is the "V" signal in VCP: sellers exhausting themselves.
        volume_declining = row.get("volume_declining", False)
        volume_dryup_ratio = row.get("volume_dryup_ratio", 1.0)

        if volume_declining and volume_dryup_ratio < 0.60:    # Very strong dry-up
            vd_score = 4.0
        elif volume_declining and volume_dryup_ratio < 0.75:  # Solid dry-up
            vd_score = 3.0
        elif volume_declining and volume_dryup_ratio < 0.90:  # Moderate
            vd_score = 2.0
        elif volume_dryup_ratio < 1.0:                         # Slight decline
            vd_score = 1.0
        else:                                                    # No contraction
            vd_score = 0.0

        score += vd_score
        details["volume_contraction"] = vd_score

        # 3. ADR % — differentiator above the minimum (0-3 points)
        # ADR is a hard filter at 5%. Here we reward stocks that move more,
        # since bigger movers produce larger R-multiples on breakouts.
        adr_pct = row.get("adr_pct", 0.0)

        if adr_pct >= 0.10:     # 10%+ daily range — high-octane
            adr_score = 3.0
        elif adr_pct >= 0.08:   # 8%+ — very good
            adr_score = 2.5
        elif adr_pct >= 0.06:   # 6%+ — good
            adr_score = 2.0
        elif adr_pct >= 0.05:   # At minimum — adequate
            adr_score = 1.0
        else:
            adr_score = 0.0

        score += adr_score
        details["adr"] = adr_score

        return score, details

    # ============================================
    # RISK/REWARD SCORING (0-10 points)
    # ============================================

    def score_risk_reward(self, row: pd.Series):
        """
        Score risk/reward setup quality.

        The old scoring evaluated stop distance as an absolute percentage (< 3% = perfect).
        This is wrong: a 3% stop on a stock with 8% ADR will be hit on random noise.
        Stops should be sized relative to the stock's own volatility (ADR).

        Qullamaggie naturally uses stops relative to daily range — tighter than 1x ADR
        is too tight and gets shaken out; wider than 2.5x ADR is too much risk.

        Components:
        - Stop vs ADR ratio (0-7pts): How tight relative to daily volatility
        - R-multiple potential (0-3pts): Reward vs risk (target cap raised to 40%)

        Perfect Score: Stop at 0.5-1.0x ADR, 5R+ potential
        """
        score = 0.0
        details = {}

        # 1. STOP DISTANCE RELATIVE TO ADR (0-7 points)
        # Ideal: stop is 0.5–1.5x the average daily range
        # Too tight (< 0.5x ADR) → noise stops you out
        # Too wide (> 2.5x ADR) → you're taking on too much risk per trade
        stop_distance = row.get("stop_distance_pct", 0.15)
        adr_pct = row.get("adr_pct", 0.05)
        stop_in_adr = stop_distance / max(adr_pct, 0.01)

        if 0.5 <= stop_in_adr <= 1.0:    # Ideal: half to 1x daily range
            stop_score = 7.0
        elif stop_in_adr < 0.5:           # Too tight — will get hit on noise
            stop_score = 2.0
        elif stop_in_adr <= 1.5:          # Slightly wide but acceptable
            stop_score = 6.0
        elif stop_in_adr <= 2.0:          # 2x daily range — manageable
            stop_score = 4.0
        elif stop_in_adr <= 2.5:          # Getting wide
            stop_score = 2.0
        else:                              # >2.5x ADR — too much risk
            stop_score = 0.0

        score += stop_score
        details["stop_vs_adr"] = stop_score

        # 2. R-MULTIPLE POTENTIAL (0-3 points)
        # Target cap raised from 15% to 40% — breakouts can run far
        potential_r = row.get("potential_r", 0.0)
        min_r = self.config.get("risk_reward_min", 3.0)

        if potential_r >= 5.0:    # 5R+ — exceptional
            r_score = 3.0
        elif potential_r >= 4.0:  # 4R — excellent
            r_score = 2.5
        elif potential_r >= min_r:  # 3R — minimum acceptable
            r_score = 2.0
        elif potential_r >= 2.0:  # 2R — below target
            r_score = 1.0
        else:                      # <2R — poor setup
            r_score = 0.0

        score += r_score
        details["r_multiple"] = r_score

        return score, details

    # ============================================
    # AGGREGATION & FILTERING
    # ============================================

    def calculate_total_score(
        self, row: pd.Series, rs_rank: float = None
    ) -> ScoreBreakdown:
        """
        Calculate total weighted score with full breakdown.

        Returns:
            ScoreBreakdown dataclass with all components
        """
        # Calculate component scores
        base_score, base_details = self.score_base_quality(row)
        trend_score, trend_details = self.score_trend_strength(row)
        rs_score, rs_details = self.score_relative_strength(row, rs_rank)
        volume_score, volume_details = self.score_volume_profile(row)
        rr_score, rr_details = self.score_risk_reward(row)

        # raw_total reflects pure setup quality — unaffected by market regime.
        # total is regime-gated: in a downtrend (0.50-0.85x), even great setups
        # score lower, matching how Qullamaggie and Minervini go selective in weak markets.
        raw_total = base_score + trend_score + rs_score + volume_score + rr_score
        total = raw_total * self.regime_multiplier

        # Combine all details
        all_details = {
            **{f"base_{k}": v for k, v in base_details.items()},
            **{f"trend_{k}": v for k, v in trend_details.items()},
            **{f"rs_{k}": v for k, v in rs_details.items()},
            **{f"volume_{k}": v for k, v in volume_details.items()},
            **{f"rr_{k}": v for k, v in rr_details.items()},
        }

        return ScoreBreakdown(
            base_quality=base_score,
            trend_strength=trend_score,
            relative_strength=rs_score,
            volume_profile=volume_score,
            risk_reward=rr_score,
            raw_total=raw_total,
            total=total,
            details=all_details,
        )

    def apply_hard_filters(self, row: pd.Series):
        """
        Apply must-pass filters before scoring.

        Returns:
            (passes_filters, reasons_for_failure)
        """
        failures = []

        # 1. Minimum price
        min_price = self.config.get("min_price", 5.0)
        if row.get("close", 0) < min_price:
            failures.append(f"Price ${row.get('close', 0):.2f} < ${min_price}")

        # 2. Minimum dollar volume
        min_dv = self.config.get("dollar_volume_min", 10_000_000)
        if row.get("dollar_volume", 0) < min_dv:
            failures.append(
                f"Dollar volume ${row.get('dollar_volume', 0):,.0f} < ${min_dv:,.0f}"
            )

        # 3. Must be above 50 SMA
        if row.get("close", 0) < row.get("sma_50", float("inf")):
            failures.append("Price below 50 SMA")

        # 4. Minimum ADR
        min_adr = self.config.get("min_adr_pct", 0.05)
        if row.get("adr_pct", 0) < min_adr:
            failures.append(f"ADR {row.get('adr_pct', 0):.1%} < {min_adr:.1%}")

        # 5. Maximum stop distance — evaluated vs ADR for rationality
        # A very wide stop in absolute terms might still be reasonable for a volatile stock
        stop_dist = row.get("stop_distance_pct", 0)
        adr = row.get("adr_pct", 0.05)
        max_stop_adr_multiple = 3.0  # Hard cutoff: stop > 3x ADR is irrational
        if stop_dist > max_stop_adr_multiple * max(adr, 0.01):
            failures.append(
                f"Stop distance {stop_dist:.1%} > {max_stop_adr_multiple:.0f}x ADR ({adr:.1%})"
            )
        elif stop_dist < 0.001:
            failures.append(f"Stop distance {stop_dist:.1%} too tight (< 0.1%)")

        # 6. Must be within 30% of 52-week high
        # Both Qullamaggie and Minervini only trade stocks near their highs.
        # Stocks far from highs face heavy overhead supply from trapped longs.
        max_dist_from_high = self.config.get("pct_from_52wk_high_max", 0.30)
        pct_from_high = row.get("pct_from_52wk_high", -1.0)
        if not pd.isna(pct_from_high) and pct_from_high < -max_dist_from_high:
            failures.append(
                f"Price {abs(pct_from_high):.1%} below 52wk high (max {max_dist_from_high:.0%})"
            )

        return len(failures) == 0, failures

    def get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        else:
            return "D"

    def get_signal_strength(self, score: float) -> str:
        """Actionable signal based on score"""
        if score >= self.min_score_alert:
            return "STRONG BUY - Alert"
        elif score >= self.min_score_watchlist:
            return "BUY - Watch Closely"
        elif score >= 60:
            return "HOLD - Monitor"
        else:
            return "PASS"

    # ============================================
    # BATCH PROCESSING
    # ============================================

    def score_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        rs_ranks=None,
    ) -> pd.DataFrame:
        """
        Score all rows in DataFrame and add score columns.

        Args:
            df: DataFrame with features
            symbol: Stock symbol (for logging)
            rs_ranks: Dict of {date: rs_rank} for each row

        Returns:
            DataFrame with score columns added
        """
        df = df.copy()

        # Initialize score columns
        df["score_base_quality"] = 0.0
        df["score_trend_strength"] = 0.0
        df["score_relative_strength"] = 0.0
        df["score_volume_profile"] = 0.0
        df["score_risk_reward"] = 0.0
        df["raw_score"] = 0.0   # pre-regime-multiplier — used for grading setup quality
        df["total_score"] = 0.0  # regime-adjusted — used for ranking and action signals
        df["grade"] = ""
        df["signal"] = ""
        df["passes_filters"] = False

        # Score each row
        for idx, row in df.iterrows():
            # Check hard filters first
            passes, failures = self.apply_hard_filters(row)
            df.at[idx, "passes_filters"] = passes

            if not passes:
                continue  # Skip scoring if doesn't pass filters

            # Get RS rank for this date if available
            rs_rank = rs_ranks.get(idx) if rs_ranks else None

            # Calculate scores
            breakdown = self.calculate_total_score(row, rs_rank)

            df.at[idx, "score_base_quality"] = breakdown.base_quality
            df.at[idx, "score_trend_strength"] = breakdown.trend_strength
            df.at[idx, "score_relative_strength"] = breakdown.relative_strength
            df.at[idx, "score_volume_profile"] = breakdown.volume_profile
            df.at[idx, "score_risk_reward"] = breakdown.risk_reward
            df.at[idx, "raw_score"] = breakdown.raw_total
            df.at[idx, "total_score"] = breakdown.total
            # Grade reflects pure setup quality — independent of market regime.
            # A bull-market-grade setup should still show A+ even in a downtrend.
            df.at[idx, "grade"] = self.get_grade(breakdown.raw_total)
            # Signal is regime-gated — STRONG BUY requires both a great setup AND
            # a supportive market.
            df.at[idx, "signal"] = self.get_signal_strength(breakdown.total)

        return df

    def create_watchlist_summary(
        self, scored_dfs, as_of_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        Create ranked watchlist from multiple scored stocks.

        Args:
            scored_dfs: Dict of {symbol: scored_dataframe}
            as_of_date: Date to evaluate (uses latest if None)

        Returns:
            Ranked watchlist DataFrame
        """
        watchlist_data = []

        for symbol, df in scored_dfs.items():
            if as_of_date is None:
                row = df.iloc[-1]  # Latest row
                date = df.index[-1]
            else:
                try:
                    row = df.loc[as_of_date]
                    date = as_of_date
                except KeyError:
                    continue

            # Only include stocks that pass filters
            if not row.get("passes_filters", False):
                continue

            watchlist_data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "score": row["total_score"],
                    "grade": row["grade"],
                    "signal": row["signal"],
                    "price": row["close"],
                    "breakout": row.get("breakout_level"),
                    "stop": row["stop_level"],
                    "stop_distance": row["stop_distance_pct"],
                    "potential_r": row["potential_r"],
                    "base_days": row["consol_days"],
                    "base_range_%": round(row.get("consol_range_15", 0) * 100, 1),
                    "pct_from_52wk_hi": round(row.get("pct_from_52wk_high", 0) * 100, 1),
                    "stage2": row.get("stage2", False),
                    "vcp": row.get("vcp_contracting", False),
                    "rs_60_excess": round(row.get("rs_comp_60", 0.0) * 100, 1),
                    "dollar_vol": row["dollar_volume"],
                    "adr_pct": row["adr_pct"],
                    # Component scores
                    "base_quality": row["score_base_quality"],
                    "trend_strength": row["score_trend_strength"],
                    "rs_score": row["score_relative_strength"],
                    "volume_score": row["score_volume_profile"],
                    "rr_score": row["score_risk_reward"],
                }
            )

        # Create DataFrame and sort by score
        watchlist_df = pd.DataFrame(watchlist_data)

        if len(watchlist_df) == 0:
            return pd.DataFrame()  # Empty watchlist

        watchlist_df = watchlist_df.sort_values("score", ascending=False)
        watchlist_df = watchlist_df.reset_index(drop=True)
        watchlist_df.index = watchlist_df.index + 1  # Start rank at 1
        watchlist_df.index.name = "rank"

        return watchlist_df
