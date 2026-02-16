import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from models import ScoreBreakdown
from datetime import datetime


class Engine:
    def __init__(self, config):
        self.config = config
        self.features = Features(config)
        self.scoring = Scoring(config)
        self.benchmark_df = None

    def load_benchmark(self, benchmark_file):
        """
        Load benchmark data (SPY or QQQ) for relative strength calculations.

        Args:
            benchmark_file: Path to benchmark pickle file
        """
        try:
            self.benchmark_df = self.load_pickle(benchmark_file)
            print(f"✓ Loaded benchmark data from {benchmark_file}")
        except Exception as e:
            print(f"⚠ Could not load benchmark: {e}")
            self.benchmark_df = None

    def load_pickle(self, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def process_stock(self, date_str=None, debug=False):
        if date_str == None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        data_dir = Path(f"data/{date_str}")

        # Check if directory exists
        if not data_dir.exists():
            print(f"Directory {data_dir} does not exist!")
            return {}, None

        # Get all pickle files in the directory
        pickle_files = list(data_dir.glob("*.pkl"))

        if not pickle_files:
            print(f"No pickle files found in {data_dir}")
            return {}, None

        print(f"Found {len(pickle_files)} pickle files in {data_dir}\n")

        # Dictionary to store scored dataframes
        scored_dfs = {}

        # Debug: track filter failures
        filter_failures = {}

        # Process each pickle file
        for pickle_file in pickle_files:
            # Extract symbol from filename (e.g., "AAOI-2026-02-14.pkl" -> "AAOI")
            symbol = pickle_file.stem.split("-")[0]

            print(f"Processing {symbol}...")

            try:
                # Load the pickle file
                df = self.load_pickle(str(pickle_file))

                # Add features (with benchmark if available)
                feature_df = self.features.add_all_features(
                    df, benchmark_df=self.benchmark_df
                )

                # Store for RS rank calculation later
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

        # Now score each stock with RS rank data
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
                        rs_rank = latest_row.get("rs_spy_60", 0)
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
        Calculate relative strength vs benchmark (SPY or QQQ).

        RS = (Stock % Change) / (Benchmark % Change)
        RS > 1.0 means outperforming
        RS < 1.0 means underperforming

        Args:
            df: Stock dataframe
            benchmark_df: Benchmark dataframe (SPY/QQQ) with same date index
            benchmark_name: Name of benchmark for column naming

        Returns df with:
        - rs_{benchmark}_20: 20-day relative strength
        - rs_{benchmark}_60: 60-day relative strength
        - rs_{benchmark}_120: 120-day relative strength
        """
        # Ensure both dataframes are aligned by date
        # Reindex benchmark to match stock dates
        benchmark_aligned = benchmark_df.reindex(df.index, method="ffill")

        # Calculate percentage changes for different periods
        for period in [20, 60, 120]:
            # Stock % change
            stock_pct_change = df["close"].pct_change(periods=period)

            # Benchmark % change
            if "close" in benchmark_aligned.columns:
                benchmark_pct_change = benchmark_aligned["close"].pct_change(
                    periods=period
                )
            else:
                # If benchmark doesn't have 'close', assume first numeric column
                benchmark_pct_change = benchmark_aligned.iloc[:, 0].pct_change(
                    periods=period
                )

            # Relative strength ratio
            # Add small epsilon to avoid division by zero
            df[f"rs_{benchmark_name.lower()}_{period}"] = stock_pct_change / (
                benchmark_pct_change + 0.0001
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
                if date in df.index:
                    # Use 60-day % change as the ranking metric
                    perf = (
                        df.loc[date, "close"] / df["close"].shift(60).loc[date] - 1
                        if date in df.index
                        else None
                    )
                    if perf is not None and not pd.isna(perf):
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

        # Use the higher of the two targets, but cap at 15% gain
        max_target = df["close"] * 1.15
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

        # Volume patterns
        df = self.detect_volume_drying(df, lookback=10)

        # Risk/Reward (NEW)
        df = self.calculate_stop(df)
        df = self.calculate_rr(df)

        # Relative Strength (NEW) - if benchmark provided
        if benchmark_df is not None:
            df = self.calculate_relative_strength(
                df, benchmark_df, benchmark_name="SPY"
            )

        return df


class Scoring:
    """
    Scores stocks based on Qullamaggie breakout principles.

    Scoring Philosophy:
    - Base Quality (25pts): Tight consolidation with proper structure
    - Trend Strength (25pts): Strong uptrend with aligned MAs
    - Relative Strength (20pts): Outperforming market
    - Volume Profile (15pts): Liquidity + volume dry-up pattern
    - Risk/Reward (15pts): Favorable setup with tight stops

    Grade Scale:
    90-100: A+ (Exceptional setup - immediate alert)
    80-89:  A  (Strong setup - high priority)
    70-79:  B  (Good setup - watch closely)
    60-69:  C  (Marginal setup - monitor)
    <60:    D/F (Pass - doesn't meet criteria)
    """

    def __init__(self, config):
        self.config = config
        self.weights = config.get(
            "weights",
            {
                "base_quality": 25,
                "trend_strength": 25,
                "relative_strength": 20,
                "volume_profile": 15,
                "risk_reward": 15,
            },
        )

        # Thresholds for grading
        self.min_score_alert = config.get("min_score_alert", 80)
        self.min_score_watchlist = config.get("min_score_watchlist", 70)

    # ============================================
    # BASE QUALITY SCORING (0-25 points)
    # ============================================

    def score_base_quality(self, row: pd.Series):
        """
        Score consolidation quality based on Qullamaggie's tight flag criteria.

        Components:
        - Consolidation tightness (0-10pts): Range compression
        - Base length optimization (0-8pts): 3-10 days optimal
        - Volume contraction (0-7pts): Volume drying up

        Perfect Score: 3-7 day consolidation, <3% range, volume declining
        """
        score = 0.0
        details = {}

        # 1. CONSOLIDATION TIGHTNESS (0-10 points)
        # Qullamaggie looks for <5% range, tighter is better
        consol_range = row.get("consol_range_15", 1.0)

        if consol_range <= 0.02:  # <2% = exceptional
            tightness_score = 10.0
        elif consol_range <= 0.03:  # 2-3% = excellent
            tightness_score = 9.0
        elif consol_range <= 0.05:  # 3-5% = good
            tightness_score = 7.0
        elif consol_range <= 0.08:  # 5-8% = acceptable
            tightness_score = 4.0
        else:  # >8% = too loose
            tightness_score = 0.0

        score += tightness_score
        details["tightness"] = tightness_score

        # 2. BASE LENGTH (0-8 points)
        # Optimal: 5-10 days. Too short = weak base, too long = loses momentum
        consol_days = row.get("consol_days", 0)

        if 5 <= consol_days <= 10:  # Sweet spot
            length_score = 8.0
        elif 3 <= consol_days < 5:  # Acceptable but short
            length_score = 6.0
        elif 10 < consol_days <= 15:  # Getting extended
            length_score = 5.0
        elif consol_days > 15:  # Too long, likely to fail
            length_score = 2.0
        else:  # <3 days = not enough consolidation
            length_score = 1.0

        score += length_score
        details["base_length"] = length_score

        # 3. VOLUME DRY-UP (0-7 points)
        # Volume should decline during consolidation
        volume_dryup_ratio = row.get("volume_dryup_ratio", 1.0)
        volume_declining = row.get("volume_declining", False)

        if volume_declining and volume_dryup_ratio < 0.7:  # Strong dry-up
            volume_score = 7.0
        elif volume_declining and volume_dryup_ratio < 0.85:  # Moderate
            volume_score = 5.0
        elif volume_dryup_ratio < 1.0:  # Slight decline
            volume_score = 3.0
        else:  # No dry-up
            volume_score = 0.0

        score += volume_score
        details["volume_dryup"] = volume_score

        return score, details

    # ============================================
    # TREND STRENGTH SCORING (0-25 points)
    # ============================================

    def score_trend_strength(self, row: pd.Series):
        """
        Score underlying trend structure.

        Components:
        - MA alignment (0-10pts): 10>20>50, all rising
        - Distance from support (0-8pts): Price near 10/20 SMA
        - Prior power move (0-7pts): Strong move before consolidation

        Perfect Score: Bull flag structure off strong uptrend
        """
        score = 0.0
        details = {}

        # 1. MA ALIGNMENT (0-10 points)
        # Qullamaggie: Must have proper MA structure
        ma_alignment = row.get("ma_alignment", False)
        mas_rising = row.get("mas_rising", False)

        if ma_alignment and mas_rising:  # Perfect structure
            ma_score = 10.0
        elif ma_alignment:  # Aligned but not all rising
            ma_score = 7.0
        elif row.get("sma_10", 0) > row.get("sma_20", 0):  # Partial alignment
            ma_score = 4.0
        else:  # Poor structure
            ma_score = 0.0

        score += ma_score
        details["ma_alignment"] = ma_score

        # 2. DISTANCE FROM KEY MOVING AVERAGES (0-8 points)
        # Best setups consolidate near 10 or 20 SMA (1-3% away)
        dist_10 = abs(row.get("distance_from_sma10", 1.0))
        dist_20 = abs(row.get("distance_from_sma20", 1.0))

        min_distance = min(dist_10, dist_20)

        if min_distance <= 0.01:  # Within 1%
            distance_score = 8.0
        elif min_distance <= 0.03:  # Within 3%
            distance_score = 7.0
        elif min_distance <= 0.05:  # Within 5%
            distance_score = 5.0
        elif min_distance <= 0.08:  # Within 8%
            distance_score = 3.0
        else:  # Too far from support
            distance_score = 0.0

        score += distance_score
        details["ma_distance"] = distance_score

        # 3. PRIOR POWER MOVE (0-7 points)
        # Strong move (20%+) before consolidation = energy to break out
        prior_move = row.get("prior_move_pct", 0.0)
        days_since_move = row.get("days_since_power_move", 999)

        if prior_move >= 0.30 and days_since_move <= 30:  # 30%+ recent move
            power_score = 7.0
        elif prior_move >= 0.20 and days_since_move <= 45:  # 20%+ move
            power_score = 6.0
        elif prior_move >= 0.15 and days_since_move <= 60:  # 15%+ move
            power_score = 4.0
        elif prior_move >= 0.10:  # Some uptrend
            power_score = 2.0
        else:  # No prior momentum
            power_score = 0.0

        score += power_score
        details["prior_power_move"] = power_score

        return score, details

    # ============================================
    # RELATIVE STRENGTH SCORING (0-20 points)
    # ============================================

    def score_relative_strength(self, row: pd.Series, rs_rank: float = None):
        """
        Score outperformance vs market.

        Components:
        - Short-term RS (0-6pts): 1-month outperformance
        - Medium-term RS (0-7pts): 3-month outperformance
        - RS Rank (0-7pts): Percentile vs watchlist

        Perfect Score: Consistent outperformance, top quartile
        """
        score = 0.0
        details = {}

        # 1. SHORT-TERM RS - 20 days (0-6 points)
        rs_20 = row.get("rs_spy_20", 0.0)

        if rs_20 >= 0.10:  # 10%+ outperformance
            rs_20_score = 6.0
        elif rs_20 >= 0.05:  # 5%+ outperformance
            rs_20_score = 5.0
        elif rs_20 >= 0.02:  # 2%+ outperformance
            rs_20_score = 3.0
        elif rs_20 >= 0:  # Neutral/slight outperformance
            rs_20_score = 1.0
        else:  # Underperforming
            rs_20_score = 0.0

        score += rs_20_score
        details["rs_20_day"] = rs_20_score

        # 2. MEDIUM-TERM RS - 60 days (0-7 points)
        rs_60 = row.get("rs_spy_60", 0.0)

        if rs_60 >= 0.20:  # 20%+ outperformance
            rs_60_score = 7.0
        elif rs_60 >= 0.15:  # 15%+ outperformance
            rs_60_score = 6.0
        elif rs_60 >= 0.10:  # 10%+ outperformance
            rs_60_score = 5.0
        elif rs_60 >= 0.05:  # 5%+ outperformance
            rs_60_score = 3.0
        elif rs_60 >= 0:  # Neutral
            rs_60_score = 1.0
        else:  # Underperforming
            rs_60_score = 0.0

        score += rs_60_score
        details["rs_60_day"] = rs_60_score

        # 3. RS PERCENTILE RANK (0-7 points)
        if rs_rank is not None:
            if rs_rank >= 90:  # Top 10%
                rank_score = 7.0
            elif rs_rank >= 75:  # Top 25%
                rank_score = 6.0
            elif rs_rank >= 60:  # Top 40%
                rank_score = 4.0
            elif rs_rank >= 50:  # Above median
                rank_score = 2.0
            else:  # Below median
                rank_score = 0.0

            score += rank_score
            details["rs_rank"] = rank_score
        else:
            details["rs_rank"] = 0.0

        return score, details

    # ============================================
    # VOLUME PROFILE SCORING (0-15 points)
    # ============================================

    def score_volume_profile(self, row: pd.Series):
        """
        Score liquidity and volume characteristics.

        Components:
        - Dollar volume (0-5pts): Liquidity threshold
        - Volume dry-up (0-5pts): Contraction during base
        - ADR % (0-5pts): Volatility for profit potential

        Perfect Score: High liquidity, volume dry-up, good volatility
        """
        score = 0.0
        details = {}

        # 1. DOLLAR VOLUME (0-5 points)
        # Need liquidity to enter/exit positions
        dollar_vol = row.get("dollar_volume", 0)
        min_dollar_vol = self.config.get("dollar_volume_min", 10_000_000)

        if dollar_vol >= min_dollar_vol * 5:  # 5x minimum
            dv_score = 5.0
        elif dollar_vol >= min_dollar_vol * 3:  # 3x minimum
            dv_score = 4.0
        elif dollar_vol >= min_dollar_vol * 2:  # 2x minimum
            dv_score = 3.5
        elif dollar_vol >= min_dollar_vol:  # Meets minimum
            dv_score = 3.0
        else:  # Below threshold
            dv_score = 0.0

        score += dv_score
        details["dollar_volume"] = dv_score

        # 2. VOLUME DRY-UP IN BASE (0-5 points)
        # Already factored in base quality, but emphasize here
        volume_declining = row.get("volume_declining", False)
        volume_dryup_ratio = row.get("volume_dryup_ratio", 1.0)

        if volume_declining and volume_dryup_ratio < 0.65:
            vd_score = 5.0
        elif volume_declining and volume_dryup_ratio < 0.80:
            vd_score = 4.0
        elif volume_dryup_ratio < 0.90:
            vd_score = 2.5
        else:
            vd_score = 0.0

        score += vd_score
        details["volume_contraction"] = vd_score

        # 3. ADR % (0-5 points)
        # Need volatility for R-multiple expansion
        adr_pct = row.get("adr_pct", 0.0)
        min_adr = self.config.get("min_adr_pct", 0.05)

        if adr_pct >= 0.08:  # 8%+ daily range
            adr_score = 5.0
        elif adr_pct >= 0.06:  # 6%+ daily range
            adr_score = 4.0
        elif adr_pct >= min_adr:  # Meets 5% minimum
            adr_score = 3.0
        elif adr_pct >= 0.03:  # Low but tradeable
            adr_score = 1.5
        else:  # Too tight
            adr_score = 0.0

        score += adr_score
        details["adr"] = adr_score

        return score, details

    # ============================================
    # RISK/REWARD SCORING (0-15 points)
    # ============================================

    def score_risk_reward(self, row: pd.Series):
        """
        Score risk/reward setup quality.

        Components:
        - Stop distance (0-10pts): Tight stop = better
        - R-multiple potential (0-5pts): Target vs risk

        Perfect Score: <5% stop, 3R+ potential
        """
        score = 0.0
        details = {}

        # 1. STOP DISTANCE (0-10 points)
        # Qullamaggie prefers 7-8% max risk
        stop_distance = row.get("stop_distance_pct", 0.15)
        max_stop = self.config.get("stop_loss_max_pct", 0.08)

        if stop_distance <= 0.03:  # <3% = exceptional
            stop_score = 10.0
        elif stop_distance <= 0.05:  # 3-5% = excellent
            stop_score = 9.0
        elif stop_distance <= max_stop:  # Within 8% max
            stop_score = 7.0
        elif stop_distance <= 0.10:  # 8-10% = acceptable
            stop_score = 5.0
        elif stop_distance <= 0.12:  # 10-12% = marginal
            stop_score = 2.0
        else:  # >12% = too much risk
            stop_score = 0.0

        score += stop_score
        details["stop_distance"] = stop_score

        # 2. R-MULTIPLE POTENTIAL (0-5 points)
        potential_r = row.get("potential_r", 0.0)
        min_r = self.config.get("risk_reward_min", 3.0)

        if potential_r >= 5.0:  # 5R+ = home run
            r_score = 5.0
        elif potential_r >= 4.0:  # 4R = excellent
            r_score = 4.5
        elif potential_r >= min_r:  # 3R = good
            r_score = 4.0
        elif potential_r >= 2.0:  # 2R = acceptable
            r_score = 2.5
        elif potential_r >= 1.5:  # 1.5R = marginal
            r_score = 1.0
        else:  # <1.5R = poor
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

        # Total score
        total = base_score + trend_score + rs_score + volume_score + rr_score

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

        # 5. Maximum stop distance
        # Stop should be reasonable - not too tight (< 0.1%) and not too wide (> 12%)
        max_stop = self.config.get("stop_loss_max_pct", 0.08) * 1.5  # 12% hard cutoff
        stop_dist = row.get("stop_distance_pct", 0)

        if stop_dist > max_stop:
            failures.append(f"Stop distance {stop_dist:.1%} > {max_stop:.1%}")
        elif stop_dist < 0.001:  # Less than 0.1% is too tight
            failures.append(f"Stop distance {stop_dist:.1%} too tight (< 0.1%)")

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
        df["total_score"] = 0.0
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
            df.at[idx, "total_score"] = breakdown.total
            df.at[idx, "grade"] = self.get_grade(breakdown.total)
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
                    "stop": row["stop_level"],
                    "stop_distance": row["stop_distance_pct"],
                    "potential_r": row["potential_r"],
                    "base_days": row["consol_days"],
                    "base_range": row["consol_range_15"],
                    "rs_60": row.get("rs_spy_60", 0.0),  # Default to 0 if no benchmark
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
