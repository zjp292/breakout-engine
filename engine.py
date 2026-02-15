import pickle
import pandas as pd


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

    def add_all_features(self, df):
        self.add_moving_averages(df)
        self.add_ma_relationships(df)
