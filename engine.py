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
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

    def add_ma_relationships(self, df):
        df["dist_from_sma10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
        df["dist_from_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

        df["ma_aligned"] = (df["sma_10"] > df["sma_20"]) & (df["sma_20"] > df["sma_50"])
        df["sma_10_slope"] = (df["sma_10"] - df["sma_10"].shift(5)) / 5

    def add_atr(self, df):
        period = self.config["atr_period"]

        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.rolling(window=period).mean()

    def add_volume_metrics(self, df):
        pass

    def detect_volume_drying(self, df):
        pass

    def add_range_metrics(self, df):
        pass

    # TODO - lookback in config??
    def detect_consolidation_range(self, df, lookback):
        pass

    def calculate_base_depth(self, df, lookback):
        pass

    # TODO - might need to change this to nasdaq comp df??
    def calculate_relative_strength(self, df, spy_df):
        pass

    # TODO - i want to calc rs based on both the market as a whole but especially against its peers
    def calculate_rs_rank(self, symbol):
        pass

    # big move up before consolidation
    def detect_prior_moves(self, df, lookback):
        pass

    def calculate_higher_lows(self, df, lookback):
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
