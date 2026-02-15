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
        # vol declining
        # vol dryup ratio? TODO - look this up
        pass

    # TODO - lookback in config??
    def detect_consolidation_range(self, df, lookback):
        # consol_range =  (max_high - min_low) / close over lookback period
        # consol_days = consecutive days in tight range
        pass

    def calculate_base_depth(self, df, lookback):
        # base_depth: (recent_high - current_close) / recent_high
        # days_from_high: days since recent high
        pass

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
        pass

    def calculate_higher_lows(self, df, lookback):
        # higher_lows: boolean indicating uptrend structure
        pass

    """
    risk management stuff
    """

    def calculate_stop(self, df):
        pass

    def calculate_rr(self, df):
        pass

    def add_all_features(self, df):
        self.add_moving_averages(df)
        self.add_ma_relationships(df)
        self.add_atr(df)
