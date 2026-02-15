import pickle
import pandas as pd
import numpy as np


class Engine:
    def __init__(self, config):
        self.config = config
        self.features = Features(config)

    def load_pickle(self, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def process_stock(self):
        df = self.load_pickle("data/2026-02-14/AAOI-2026-02-14.pkl")
        self.features.add_all_features(df)
        print(df)


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

    def detect_consolidation_range(self, df, lookback):
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

    def calculate_base_depth(self, df, lookback):
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

    # TODO - might need to change this to nasdaq comp df??
    def calculate_relative_strength(self, df, spy_df):

        # rs_spy_20: % change stock vs % change SPY (20 days)
        # rs_spy_60: % change stock vs % change SPY (60 days)
        # rs_spy_120: % change stock vs % change SPY (120 days)
        pass

    # TODO - i want to calc rs based on both the market as a whole but especially against its peers
    def calculate_rs_rank(self, symbol):
        # calculate percentile rank of this stock vs entire watchlist.
        pass

    # big move up before consolidation
    def detect_prior_moves(self, df, lookback):
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

    def calculate_higher_lows(self, df, lookback):
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
        pass

    def calculate_rr(self, df):
        pass

    def add_all_features(self, df):
        df = df.copy()

        df = self.add_moving_averages(df)
        df = self.add_atr(df)
        df = self.add_range_metrics(df)
        df = self.add_volume_metrics(df)
