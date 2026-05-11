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
        self.spy_df = None  # S&P 500 — multi-index confirmation
        self.iwm_df = None  # Russell 2000 — small-cap breadth
        self.market_condition = None  # MarketConditionResult from last run
        self.macro_regime = None  # MacroRegimeResult — sustained environment

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

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=400)  # extra buffer for 200-day SMA
        end_ts = int(end_dt.timestamp() * 1000)
        start_ts = int(start_dt.timestamp() * 1000)

        client = SchwabAPIClient()

        print("Fetching NASDAQ Composite ($COMPX) benchmark data...")
        self.benchmark_df = client.get_index_data("$COMPX", start_ts, end_ts)
        print(f"  COMPX loaded: {len(self.benchmark_df)} trading days")

        # spy data
        try:
            self.spy_df = client.get_index_data("SPY", start_ts, end_ts)
            print(f"  SPY   loaded: {len(self.spy_df)} trading days")
        except Exception as e:
            print(f"  Warning: Could not load SPY: {e}")
            self.spy_df = None

        # russel data
        try:
            self.iwm_df = client.get_index_data("IWM", start_ts, end_ts)
            print(f"  IWM   loaded: {len(self.iwm_df)} trading days")
        except Exception as e:
            print(f"  Warning: Could not load IWM: {e}")
            self.iwm_df = None

    def analyze_market_condition(self, feature_dfs: dict) -> float:
        """
        Run both the daily market condition analysis and the macro regime analysis,
        then return a blended regime multiplier.

        Two complementary layers:

        1. MarketConditionAnalyzer (100-pt score = 0.50-1.00 multiplier)
           Short-term health: distribution days, follow-through days, SMA alignment,
           internal breadth of watchlist stocks, 21-day momentum.
           Window: roughly the last 4-6 weeks of activity.

        2. MacroRegimeAnalyzer (direction x quality = 0.60-1.00 multiplier)
           Sustained macro environment: Choppiness Index, ADX, R², Hurst Exponent,
           multi-timeframe momentum confluence, volatility regime, price structure.
           Window: 3-12 months - captures "choppy since October"-style regimes.

        Combined multiplier (weighted blend):
          final = 0.55 x daily_multiplier + 0.45 x macro_multiplier

        The daily analysis stays reactive to current conditions; the macro prevents
        over-sizing during sustained unfavorable periods even when a single day looks OK.

        Sets self.market_condition and self.macro_regime for downstream use.
        Returns the final blended multiplier (0.50-1.00).
        """
        from market_condition import MarketConditionAnalyzer
        from macro_regime import MacroRegimeAnalyzer

        if not self.config.get("market_regime", True):
            self.market_condition = None
            self.macro_regime = None
            return 1.0

        if self.benchmark_df is None:
            self.market_condition = None
            self.macro_regime = None
            return 1.0

        mc_analyzer = MarketConditionAnalyzer(self.config)
        mc_result = mc_analyzer.analyze(
            compx_df=self.benchmark_df,
            spy_df=self.spy_df,
            iwm_df=self.iwm_df,
            stock_feature_dfs=feature_dfs,
        )
        self.market_condition = mc_result
        self._print_market_condition(mc_result)

        macro_analyzer = MacroRegimeAnalyzer(self.config)
        macro_result = macro_analyzer.analyze(
            compx_df=self.benchmark_df,
            spy_df=self.spy_df,
            iwm_df=self.iwm_df,
        )
        self.macro_regime = macro_result
        self._print_macro_regime(macro_result)

        daily_mult = mc_result.regime_multiplier
        macro_mult = macro_result.macro_multiplier
        blended = round(0.55 * daily_mult + 0.45 * macro_mult, 3)

        return max(0.50, blended)

    def _print_market_condition(self, mc) -> None:
        """
        Print a formatted market condition report to stdout.

        delicious slop :)
        """
        W = 66
        bar = "═" * W

        regime_badges = {
            "BULL": "▲ BULL",
            "UPTREND": "↑ UPTREND",
            "MIXED": "↔ MIXED",
            "CAUTION": "↓ CAUTION",
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
        spy_str = (
            "SPY ✓"
            if mc.spy_above_200
            else (
                "SPY ✗"
                if mc.details.get("index", {}).get("spy_above_200") is False
                else "SPY –"
            )
        )
        iwm_str = (
            "IWM ✓"
            if mc.iwm_above_200
            else (
                "IWM ✗"
                if mc.details.get("index", {}).get("iwm_above_200") is False
                else "IWM –"
            )
        )
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
            ago = f"{mc.ftd_days_ago}d ago" if mc.ftd_days_ago is not None else "?"
            ftd_str = (
                f"FTD {mc.ftd_date[:10] if mc.ftd_date else '?'} ({ago}, {validity})"
            )
        else:
            pct_hi = mc.details.get("follow_through", {}).get("pct_from_high")
            if pct_hi is not None:
                ftd_str = f"no FTD — {abs(pct_hi) * 100:.1f}% from 52wk high"
            else:
                ftd_str = "no FTD detected in lookback window"
        print(f"  Follow-Through    {mc.follow_through_score:5.1f}/20   [{ftd_str}]")

        # Internal Breadth
        n = mc.details.get("breadth", {}).get("n_stocks", 0)
        print(
            f"  Internal Breadth  {mc.breadth_score:5.1f}/20"
            f"   [{mc.pct_above_50sma * 100:.0f}% >50d  "
            f"{mc.pct_in_stage2 * 100:.0f}% Stage2  "
            f"{mc.pct_near_52wk_high * 100:.0f}% near high  "
            f"n={n}]"
        )

        # Momentum & Volatility
        rv_pct = mc.realized_vol_annualized * 100
        roc_pct = mc.compx_roc_21d * 100
        print(
            f"  Momentum/Vol      {mc.momentum_score:5.1f}/15"
            f"   [21d ROC {roc_pct:+.1f}%  Realized vol {rv_pct:.1f}%]"
        )

        print(bar + "\n")

    def _print_macro_regime(self, mr) -> None:
        """Print a formatted macro regime report to stdout."""
        W = 66
        bar = "═" * W

        direction_badges = {
            "BULLISH": "▲ BULLISH",
            "NEUTRAL": "↔ NEUTRAL",
            "BEARISH": "▼ BEARISH",
        }
        quality_badges = {
            "TRENDING": "TRENDING",
            "TRANSITIONING": "TRANSITIONING",
            "CHOPPY": "CHOPPY ⚠",
        }
        vol_badges = {
            "CALM": "CALM",
            "NORMAL": "NORMAL",
            "ELEVATED": "ELEVATED ⚠",
            "EXTREME": "EXTREME ⛔",
        }

        dir_str = direction_badges.get(mr.trend_direction, mr.trend_direction)
        qlt_str = quality_badges.get(mr.trend_quality, mr.trend_quality)
        vol_str = vol_badges.get(mr.vol_regime, mr.vol_regime)

        print(f"{bar}")
        print("  MACRO REGIME ANALYSIS  (3-12 month sustained environment)")
        print(bar)
        print(
            f"  {mr.regime_label:<18s}  {dir_str}  ×  {qlt_str}"
            f"   →  ×{mr.macro_multiplier:.2f}"
        )
        print(f"  Vol regime: {vol_str}")
        print(f"  {'─' * (W - 4)}")

        # ── Direction signals ─────────────────────────────────────────────
        dir_bar = self._sparkbar(mr.direction_score, lo=-1.0, hi=1.0, width=20)
        print(
            f"  Direction score   {mr.direction_score:+.3f}  {dir_bar}"
            f"  [{mr.trend_direction}]"
        )
        print(
            f"    Mom confluence  {mr.mom_confluence:+.2f}"
            f"   [21d {mr.mom_21d * 100:+.1f}%"
            f"  63d {mr.mom_63d * 100:+.1f}%"
            f"  126d {mr.mom_126d * 100:+.1f}%"
            f"  252d {mr.mom_252d * 100:+.1f}%]"
        )
        di_dir = "▲" if mr.plus_di > mr.minus_di else "▼"
        print(
            f"    ADX direction   {di_dir}  +DI {mr.plus_di:.1f}"
            f"  −DI {mr.minus_di:.1f}"
            f"   Reg slope (63d) {mr.reg_slope_63d * 100:+.1f}%/yr"
        )

        # ── Quality signals ───────────────────────────────────────────────
        qlt_bar = self._sparkbar(mr.quality_score, lo=0.0, hi=1.0, width=20)
        print(
            f"  Quality score     {mr.quality_score:.3f}  {qlt_bar}"
            f"  [{mr.trend_quality}]"
        )
        ci_flag = (
            "  ⚠ choppy"
            if mr.choppiness_14 > 61.8
            else ("  ✓ trending" if mr.choppiness_14 < 38.2 else "")
        )
        print(
            f"    Choppiness(14)  {mr.choppiness_14:.1f}{ci_flag}"
            f"   Choppiness(50) {mr.choppiness_50:.1f}"
        )
        adx_note = (
            "no trend"
            if mr.adx_14 < 20
            else (
                "weak trend"
                if mr.adx_14 < 25
                else ("trending" if mr.adx_14 < 40 else "strong trend")
            )
        )
        print(
            f"    ADX(14)         {mr.adx_14:.1f}  [{adx_note}]"
            f"   R²(63d) {mr.reg_r2_63d:.2f}"
        )
        hurst_note = (
            "trending/persistent"
            if mr.hurst > 0.55
            else "mean-reverting"
            if mr.hurst < 0.45
            else "random walk"
        )
        print(f"    Hurst exp       {mr.hurst:.3f}  [{hurst_note}]")

        # ── Volatility ────────────────────────────────────────────────────
        vr_flag = "  ⚠ expanding" if mr.vol_rising else ""
        print(
            f"  Volatility        {vol_str}"
            f"   10d {mr.vol_10d * 100:.1f}%"
            f"  60d {mr.vol_60d * 100:.1f}%"
            f"  ratio ×{mr.vol_ratio:.2f}{vr_flag}"
        )

        # ── Price structure ───────────────────────────────────────────────
        spy_str = (
            "SPY ✓"
            if mr.spy_above_200 is True
            else "SPY ✗"
            if mr.spy_above_200 is False
            else "SPY –"
        )
        iwm_str = (
            "IWM ✓"
            if mr.iwm_above_200 is True
            else "IWM ✗"
            if mr.iwm_above_200 is False
            else "IWM –"
        )
        print(
            f"  3-month range     {mr.range_width_pct * 100:.1f}%"
            f"   {abs(mr.pct_from_swing_high) * 100:.1f}% below swing high"
            f"   {spy_str}  {iwm_str}"
        )

        print(bar + "\n")

    @staticmethod
    def _sparkbar(value: float, lo: float, hi: float, width: int = 20) -> str:
        """
        Render a simple ASCII progress bar showing where `value` falls in [lo, hi].
        E.g. value=0.3, lo=-1, hi=1, width=20 → '[──────────●─────────]'
        """
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        pos = int(round(frac * (width - 1)))
        bar_body = "─" * pos + "●" + "─" * (width - 1 - pos)
        return f"[{bar_body}]"

    def process_stock(self, date_str=None, debug=False):
        """
        processes ingested OHLCV data and runs the features
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        data_dir = Path(f"data/{date_str}")

        if not data_dir.exists() or not list(data_dir.glob("*.pkl")):
            print(f"No data found for {date_str}. Running ingestion...")
            from ingestion import Ingestor

            ingestor = Ingestor()
            ingestor.mergefiles(date=date_str)

            if not ingestor.ticker_list:
                print(f"No tickers found in watchlist exports for {date_str}. Exiting.")
                return {}, None

            print(f"Fetching data for {len(ingestor.ticker_list)} tickers...")
            ingestor.get_data(date=date_str)

        pickle_files = list(data_dir.glob("*.pkl"))

        if not pickle_files:
            print(f"No pickle files found in {data_dir} after ingestion.")
            return {}, None

        print(f"Found {len(pickle_files)} pickle files in {data_dir}\n")

        # Load NASDAQ Composite benchmark (+ SPY, IWM)
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
                if "datetime" in df.columns and not isinstance(
                    df.index, pd.DatetimeIndex
                ):
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
                        print(f"  {symbol}: Failed - {', '.join(failures)}")
                    else:
                        score = latest_row.get("total_score", 0)
                        rs_rank = latest_row.get("rs_comp_60", 0)
                        print(f"  {symbol}: Score={score:.1f}, RS_60={rs_rank:.2f}")

            except Exception as e:
                print(f"  Error scoring {symbol}: {e}")
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
                print(f"Watchlist created with {len(watchlist)} stocks")

            else:
                print("No stocks passed the filters")

        return final_scored_dfs, watchlist


class Features:
    def __init__(self, config):
        self.config = config

    def add_moving_averages(self, df):
        periods = self.config.get("sma_periods", [10, 20, 50])
        # the man himself uses ema for 10 and 20 day *le shrug*
        ema_periods = {10, 20}

        for period in periods:
            # keeping the ema labeled as sma to avoid rewriting everything
            if period in ema_periods:
                df[f"sma_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            else:
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

        # calc dist from mas after mas have been initalized
        if "sma_150" in df.columns and "sma_200" in df.columns:
            df["distance_from_sma150"] = (df["close"] - df["sma_150"]) / df["sma_150"]
            df["distance_from_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
            sma200_slope = df["sma_200"].pct_change(periods=20)
            df["stage2"] = (
                (df["close"] > df["sma_50"])
                & (df["close"] > df["sma_150"])
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

        # where did price close within the day's range? 1.0 = at the high, 0.0 = at the low
        # used to confirm demand during dry-up
        df["close_range_position"] = (
            (df["close"] - df["low"]) / (df["high"] - df["low"]).clip(lower=0.001)
        ).clip(0.0, 1.0)

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

        # Top of current base
        df["breakout_level"] = rolling_high

        w_short = self.config.get("vcp_windows", [10, 20, 40])[0]
        range_col = f"range_{w_short}"
        if range_col in df.columns:
            adr = df["adr_pct"].clip(lower=0.01)
            df["is_tight_consolidation"] = (df[range_col] / adr) <= 3.5
        else:
            tight_threshold = self.config.get("range_compression_threshold", 0.05)
            df["is_tight_consolidation"] = (
                df[f"consol_range_{lookback}"] < tight_threshold
            )

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

        Vectorized implementation: builds a price matrix (dates × symbols), computes
        rs_rank_window-day returns for all stocks simultaneously, then ranks across
        stocks for each date in a single pandas call.  Reduces O(N_stocks × N_dates)
        Python loops to a handful of vectorized operations.

        Args:
            stock_dfs: Dict of {symbol: dataframe} for all stocks in watchlist
            benchmark_df: Optional benchmark dataframe (unused; kept for API compat)

        Returns:
            Dict of {symbol: {date: rs_rank}} where rs_rank is 0–100 percentile
        """
        rs_window = self.config.get("rs_rank_window", 60)

        # Build a price matrix: rows=dates, columns=symbols
        price_matrix = pd.DataFrame(
            {symbol: df["close"] for symbol, df in stock_dfs.items()}
        )

        # rs_window-period returns for all stocks in one vectorized step
        returns = price_matrix.pct_change(periods=rs_window)

        # Rank across stocks for each date; pct=True gives [0, 1] → scale to [0, 100]
        # na_option="keep" leaves dates with insufficient history as NaN
        ranks = returns.rank(axis=1, pct=True, na_option="keep") * 100

        # Convert matrix back to the dict-of-dicts format expected by callers
        rs_ranks = {}
        for symbol in stock_dfs:
            if symbol in ranks.columns:
                rs_ranks[symbol] = ranks[symbol].dropna().to_dict()
            else:
                rs_ranks[symbol] = {}

        return rs_ranks

    # big move up before consolidation
    def detect_prior_moves(self, df, lookback=None):
        # prior_move_pct: max % gain in lookback period
        # days_since_power_move: days since 20%+ move
        if lookback is None:
            lookback = self.config.get("prior_move_window", 60)
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

    def calculate_higher_lows(self, df, lookback=None):
        if lookback is None:
            lookback = self.config.get("base_length_max", 60)

        df["is_swing_low"] = (
            (df["low"] < df["low"].shift(1))
            & (df["low"] < df["low"].shift(2))
            & (df["low"] < df["low"].shift(-1))
            & (df["low"] < df["low"].shift(-2))
        )

        swing_lows = df["low"].where(df["is_swing_low"])
        prev_pivot = swing_lows.ffill().shift(1)

        df["higher_lows"] = df["is_swing_low"] & (df["low"] > prev_pivot)
        df["swing_low_count"] = (
            df["higher_lows"].rolling(window=lookback, min_periods=1).sum()
        )

        return df

    def calculate_lower_highs(self, df, lookback=None):
        if lookback is None:
            lookback = self.config.get("base_length_max", 60)

        # 5-bar pivot high: higher than both the 2 bars before AND the 2 bars after
        df["is_swing_high"] = (
            (df["high"] > df["high"].shift(1))
            & (df["high"] > df["high"].shift(2))
            & (df["high"] > df["high"].shift(-1))
            & (df["high"] > df["high"].shift(-2))
        )

        swing_highs = df["high"].where(df["is_swing_high"])
        prev_pivot = swing_highs.ffill().shift(1)

        df["lower_highs"] = df["is_swing_high"] & (df["high"] < prev_pivot)
        df["swing_high_count"] = (
            df["lower_highs"].rolling(window=lookback, min_periods=1).sum()
        )

        return df

    def calculate_ema_surf(self, df):
        if "sma_10" not in df.columns:
            df["ema10_surf_ratio"] = np.nan
            return df

        dist = (df["close"] - df["sma_10"]) / df["sma_10"]
        ema_rising = df["sma_10"] > df["sma_10"].shift(1)
        surfing = (dist >= -0.03) & (dist <= 0.10) & ema_rising

        df["ema10_surf_ratio"] = surfing.rolling(window=20, min_periods=5).mean()
        return df

    def calculate_52wk_proximity(self, df):
        df["52wk_high"] = df["high"].rolling(window=252, min_periods=100).max()
        df["pct_from_52wk_high"] = (df["close"] - df["52wk_high"]) / df["52wk_high"]

        # 90-day window: for post-crash/post-move setups the 52wk high is a pre-crash
        # price and unfairly penalizes stocks near their recent flagpole top
        df["90d_high"] = df["high"].rolling(window=90, min_periods=30).max()
        df["pct_from_90d_high"] = (df["close"] - df["90d_high"]) / df["90d_high"]
        return df

    def detect_vcp_contractions(self, df):
        w_short, w_medium, w_long = self.config.get("vcp_windows", [10, 20, 40])

        h, lo, c = df["high"], df["low"], df["close"]

        # overlapping ranges
        range_short = (h.rolling(w_short).max() - lo.rolling(w_short).min()) / c
        range_medium = (h.rolling(w_medium).max() - lo.rolling(w_medium).min()) / c
        range_long = (h.rolling(w_long).max() - lo.rolling(w_long).min()) / c

        df[f"range_{w_short}"] = range_short
        df[f"range_{w_medium}"] = range_medium
        df[f"range_{w_long}"] = range_long

        # non-overlapping: three equal w_short-day windows shifted apart
        # range_now  = last w_short days
        # range_prev = w_short days before that
        # range_far  = w_short days before that
        range_now = range_short
        range_prev = (
            h.rolling(w_short).max().shift(w_short)
            - lo.rolling(w_short).min().shift(w_short)
        ) / c
        range_far = (
            h.rolling(w_short).max().shift(w_short * 2)
            - lo.rolling(w_short).min().shift(w_short * 2)
        ) / c

        # true VCP: each period strictly narrower than the one before
        df["vcp_contracting"] = (range_now < range_prev) & (range_prev < range_far)

        # ratio vs overlapping long window — useful tightness proxy for scoring
        df["vcp_contraction_ratio"] = range_short / range_long.clip(lower=0.001)

        return df

    """
    risk management stuff
    """

    def calculate_stop(self, df):
        # 60-day stop: used by the backtester and for display
        base_lookback = self.config.get("base_length_max", 60)
        df["stop_level"] = df["low"].rolling(window=base_lookback, min_periods=1).min()
        df["stop_distance_pct"] = (df["close"] - df["stop_level"]) / df["close"]
        df["stop_distance_pct"] = df["stop_distance_pct"].clip(lower=0.001, upper=0.25)

        # 20-day stop: used by the hard filter. captures actual consolidation
        # support rather than the 60-day window that includes pre-flagpole lows
        # for fresh setups (e.g. a stock 30 days into a flag still has crash lows
        # from 50 days ago in the 60-day window, making stop look artificially wide)
        df["stop_level_20d"] = df["low"].rolling(window=20, min_periods=5).min()
        df["stop_distance_20d_pct"] = (
            (df["close"] - df["stop_level_20d"]) / df["close"]
        ).clip(lower=0.001, upper=0.50)

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

        consol_range_key = f"consol_range_{self.config.get('base_length_max', 60)}"
        consol_range_pct = df.get(consol_range_key, pd.Series([0.05] * len(df)))
        if isinstance(consol_range_pct, (int, float)):
            consol_range_pct = pd.Series([consol_range_pct] * len(df), index=df.index)
        base_target_1 = df["close"] * (1 + consol_range_pct)

        lookback = 60
        rolling_high = df["high"].rolling(window=lookback).max()
        base_target_2 = rolling_high

        # Use the higher of the two targets, cap at 40% gain
        max_target = df["close"] * 1.40
        df["target_level"] = pd.concat([base_target_1, base_target_2], axis=1).max(
            axis=1
        )
        df["target_level"] = df["target_level"].clip(upper=max_target)

        # Potential gain percentage
        df["potential_gain_pct"] = (df["target_level"] - df["close"]) / df["close"]
        df["potential_gain_pct"] = df["potential_gain_pct"].clip(lower=0)

        # Risk/Reward ratio (R-multiple)
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
        df = self.calculate_ema_surf(df)

        df = self.detect_vcp_contractions(df)

        # Base structure
        df = self.detect_consolidation_range(df)
        df = self.calculate_base_depth(df)

        # Historical patterns
        df = self.detect_prior_moves(df)
        df = self.calculate_higher_lows(df)
        df = self.calculate_lower_highs(df)

        # 52-week proximity
        df = self.calculate_52wk_proximity(df)

        # Volume patterns
        df = self.detect_volume_drying(
            df, lookback=self.config.get("volume_dryup_window", 10)
        )

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

    Weights rebalanced 2026-05 to focus exclusively on consolidation quality:
    - Base Quality (20pts):      VCP base structure — tightness, length, contraction
    - Trend Strength (20pts):    Stage 2 + proximity + MA + prior move (flagpole)
    - Relative Strength (30pts): RS leadership — strongest confirmed predictor
    - Volume Profile (30pts):    Liquidity + dry-up + ADR — empirically dominant
    - Risk/Reward (excluded):    Stop info retained for display; not counted in raw_total

    Market Regime: A multiplier (0.50-1.0) applied based on benchmark trend.

    Grade Scale (based on raw_score, pre-regime):
    90-100: A+ | 80-89: A | 70-79: B | 60-69: C | <60: D
    """

    def __init__(self, config):
        self.config = config
        self.weights = config.get(
            "weights",
            {
                "base_quality": 20,
                "trend_strength": 20,
                "relative_strength": 30,
                "volume_profile": 30,
                "risk_reward": 0,
            },
        )
        self.min_score_alert = config.get("min_score_alert", 80)
        self.min_score_watchlist = config.get("min_score_watchlist", 70)
        self.regime_multiplier = 1.0

    def score_base_quality(self, row: pd.Series):
        """
        Score consolidation quality: 4 components targeting the wedge geometry that
        Qullamaggie and Minervini both require for flag/VCP entries.

        Component structure (restructured 2026-05):
        - Recent tightness   (0-6 pts): range_10 — the 10-day range, not the 60-day box.
          Using consol_range_60 inflates the reading mid-wedge because the far end of the
          lookback window contains the wide pre-consolidation swings. range_10 measures
          the actual coil at the tip of the pattern.

        - Base length        (0-4 pts): sweet-spot is 5-15 day flags; credit for VCP
          bases up to 45 days per Minervini's template.

        - VCP range contraction (0-4 pts): non-overlapping window compression confirms
          narrowing price structure. Qullamaggie's contraction-within-contraction.

        - Wedge geometry     (0-6 pts): higher pivot lows + lower pivot highs.
          Lo, Mamaysky & Wang (2000) formally define a symmetrical triangle as requiring
          E1>E3>E5 (lower highs) AND E2<E4 (higher lows). Qullamaggie's flag entry
          requires a "series of higher pivot lows". Previously computed but never scored.
          Lower highs alone (no rising support) score zero — descending channel ≠ VCP.

        Volume dry-up lives exclusively in score_volume_profile (single source of truth).
        Max total: 6 + 4 + 4 + 6 = 20 pts.
        """
        score = 0.0
        details = {}

        # 1. RECENT BASE TIGHTNESS (0-6 pts)
        # range_10 = 10-day high-to-low range as % of close (from detect_vcp_contractions).
        # falls back to consol_range_60 for rows that pre-date the VCP detection step.
        # normalized by adr_pct so that high-ADR volatile stocks (which have wider
        # absolute ranges) are judged relative to their own typical daily movement —
        # a 20% range on a 15%-ADR stock is coiling tight; on a 4%-ADR stock it is not.
        w_short = self.config.get("vcp_windows", [10, 20, 40])[0]
        recent_range = row.get(
            f"range_{w_short}",
            row.get(f"consol_range_{self.config.get('base_length_max', 60)}", 1.0),
        )
        adr = max(row.get("adr_pct", 0.05), 0.01)
        tightness_ratio = recent_range / adr  # how many average daily ranges wide?

        if tightness_ratio <= 0.75:
            tightness_score = 6.0  # coiling < 1 avg daily range over 10 days
        elif tightness_ratio <= 1.25:
            tightness_score = 5.0  # very tight flag
        elif tightness_ratio <= 2.0:
            tightness_score = 4.0  # normal consolidation
        elif tightness_ratio <= 3.5:
            tightness_score = 2.0  # loose but acceptable
        else:
            tightness_score = 0.0

        score += tightness_score
        details["tightness"] = tightness_score

        # 2. BASE LENGTH (0-4 pts)
        consol_days = row.get("consol_days", 0)

        if 5 <= consol_days <= 15:
            length_score = 4.0
        elif 15 < consol_days <= 30:
            length_score = 3.5
        elif 30 < consol_days <= 45:
            length_score = 3.0
        elif 3 <= consol_days < 5:
            length_score = 2.0
        elif 45 < consol_days <= 60:
            length_score = 1.5
        elif consol_days > 60:
            length_score = 1.0
        else:
            length_score = 0.0

        score += length_score
        details["base_length"] = length_score

        # 3. VCP RANGE CONTRACTION (0-4 pts)
        vcp_contracting = row.get("vcp_contracting", False)
        vcp_ratio = row.get("vcp_contraction_ratio", 1.0)

        if vcp_contracting and vcp_ratio <= 0.25:
            vcp_score = 4.0
        elif vcp_contracting and vcp_ratio <= 0.40:
            vcp_score = 3.0
        elif vcp_contracting:
            vcp_score = 2.0
        elif vcp_ratio <= 0.60:
            vcp_score = 0.5
        else:
            vcp_score = 0.0

        score += vcp_score
        details["vcp_contraction"] = vcp_score

        # 4. WEDGE GEOMETRY (0-6 pts)
        # swing_low_count: confirmed higher-pivot-low events in base_length_max window.
        # swing_high_count: confirmed lower-pivot-high events in base_length_max window.
        # both sides converging = symmetrical triangle (lo et al. 2000 full criterion).
        # higher lows alone = ascending base — preferred by qullamaggie for flags.
        # lower highs alone = descending pressure without rising support → 0 pts.
        hl_count = int(row.get("swing_low_count", 0))
        lh_count = int(row.get("swing_high_count", 0))

        if hl_count >= 2 and lh_count >= 2:
            wedge_score = 6.0  # textbook convergence: multiple events both sides
        elif hl_count >= 1 and lh_count >= 1 and (hl_count + lh_count) >= 3:
            wedge_score = 4.5  # well-confirmed: 3+ total pivot events both sides
        elif hl_count >= 1 and lh_count >= 1:
            wedge_score = 3.0  # early-stage wedge: one event per side
        elif hl_count >= 2:
            wedge_score = (
                2.0  # ascending base: rising support, resistance not yet compressing
            )
        elif hl_count >= 1:
            wedge_score = 1.0  # one higher low — minimal structural evidence
        else:
            wedge_score = 0.0  # no wedge structure detected

        score += wedge_score
        details["wedge_geometry"] = wedge_score

        return score, details

    # ============================================
    # TREND STRENGTH SCORING (0-20 points)
    # ============================================

    def score_trend_strength(self, row: pd.Series):
        """
        Score underlying trend structure — Qullamaggie + Minervini combined.

        Components (raised 2026-05 from 15 to 20 pts — prior move boosted to reward the flagpole):
        - Stage 2 structure (0-5pts): Long-term MA alignment
        - 52-week high proximity (0-5pts): Near highs = less overhead supply
        - Short-term MA structure (0-4pts): 10>20>50 aligned + rising
        - Prior power move (0-6pts): Flagpole before the base — Qullamaggie's #1 criterion

        Perfect Score: Stage 2, within 5% of 52wk high, perfect MA structure, 40%+ prior move
        """
        score = 0.0
        details = {}

        # 1. STAGE 2 LONG-TERM STRUCTURE (0-5 points)
        stage2 = row.get("stage2", False)
        dist_150 = row.get("distance_from_sma150", np.nan)
        dist_200 = row.get("distance_from_sma200", np.nan)

        if stage2:
            stage_score = 5.0
        elif (
            not pd.isna(dist_150)
            and dist_150 > 0
            and not pd.isna(dist_200)
            and dist_200 > 0
        ):
            stage_score = 3.0
        elif not pd.isna(dist_200) and dist_200 > 0:
            stage_score = 1.5
        elif pd.isna(dist_150):
            ma_alignment = row.get("ma_alignment", False)
            stage_score = 2.0 if ma_alignment else 0.0
        else:
            stage_score = 0.0

        score += stage_score
        details["stage2"] = stage_score

        # 2. 52-WEEK HIGH PROXIMITY (0-5 points)
        pct_from_high = row.get("pct_from_52wk_high", -1.0)

        if pct_from_high >= -0.05:
            proximity_score = 5.0
        elif pct_from_high >= -0.10:
            proximity_score = 4.5
        elif pct_from_high >= -0.15:
            proximity_score = 3.0
        elif pct_from_high >= -0.20:
            proximity_score = 2.0
        elif pct_from_high >= -0.25:
            proximity_score = 1.0
        else:
            proximity_score = 0.0

        score += proximity_score
        details["proximity_to_high"] = proximity_score

        # 3. SHORT-TERM MA STRUCTURE (0-4 points)
        # surf_ratio replaces the old single-day "above_10sma" binary.
        # it measures how consistently price hugged the rising EMA during the base —
        # the rolling signal is far more informative than one day's distance snapshot.
        ma_alignment = row.get("ma_alignment", False)
        mas_rising = row.get("mas_rising", False)
        surf_ratio = row.get("ema10_surf_ratio", 0.0) or 0.0

        if ma_alignment and mas_rising:
            if surf_ratio >= 0.75:
                ma_score = 4.0  # perfect: aligned, rising, consistently surfing EMA
            elif surf_ratio >= 0.50:
                ma_score = 3.5
            else:
                ma_score = 3.0  # aligned + rising but base is not clean
        elif ma_alignment:
            if surf_ratio >= 0.65:
                ma_score = 2.0
            else:
                ma_score = 1.0
        elif row.get("sma_10", 0) > row.get("sma_20", 0):
            ma_score = 0.5
        else:
            ma_score = 0.0

        score += ma_score
        details["ma_structure"] = ma_score

        # 4. PRIOR POWER MOVE (0-6 points) — the flagpole before the base
        prior_move = row.get("prior_move_pct", 0.0)
        days_since_move = row.get("days_since_power_move", 999)

        if prior_move >= 0.40 and days_since_move <= 30:
            power_score = 6.0
        elif prior_move >= 0.30 and days_since_move <= 45:
            power_score = 5.0
        elif prior_move >= 0.20 and days_since_move <= 60:
            power_score = 4.0
        elif prior_move >= 0.15:
            power_score = 2.0
        else:
            power_score = 0.0

        score += power_score
        details["prior_power_move"] = power_score

        return score, details

    # ============================================
    # RELATIVE STRENGTH SCORING (0-30 points)
    # ============================================

    def score_relative_strength(self, row: pd.Series, rs_rank: float = None):
        """
        Score outperformance vs NASDAQ Composite.

        RS = excess return (stock% - benchmark%). Positive = outperforming.
        Empirically the strongest confirmed predictor of breakout success.

        Components (promoted 2026-03 — raised from 25 to 30 pts):
        - 20-day RS (0-8pts): Short-term leadership signal
        - 60-day RS (0-12pts): Medium-term leadership — most predictive window
        - RS percentile rank (0-10pts): Rank vs all stocks in this watchlist

        Perfect Score: +10% 20d excess, +20% 60d excess, top 10% of peers
        """
        score = 0.0
        details = {}

        # 1. SHORT-TERM RS — 20 days (0-8 points)
        rs_20 = row.get("rs_comp_20", 0.0)

        if rs_20 >= 0.10:
            rs_20_score = 8.0
        elif rs_20 >= 0.05:
            rs_20_score = 6.0
        elif rs_20 >= 0.02:
            rs_20_score = 4.0
        elif rs_20 >= 0.00:
            rs_20_score = 1.5
        else:
            rs_20_score = 0.0

        score += rs_20_score
        details["rs_20_day"] = rs_20_score

        # 2. MEDIUM-TERM RS — 60 days (0-12 points)
        rs_60 = row.get("rs_comp_60", 0.0)

        if rs_60 >= 0.20:
            rs_60_score = 12.0
        elif rs_60 >= 0.15:
            rs_60_score = 10.0
        elif rs_60 >= 0.10:
            rs_60_score = 8.0
        elif rs_60 >= 0.05:
            rs_60_score = 5.0
        elif rs_60 >= 0.00:
            rs_60_score = 1.5
        else:
            rs_60_score = 0.0

        score += rs_60_score
        details["rs_60_day"] = rs_60_score

        # 3. RS PERCENTILE RANK vs PEERS (0-10 points)
        if rs_rank is not None:
            if rs_rank >= 90:
                rank_score = 10.0
            elif rs_rank >= 80:
                rank_score = 8.0
            elif rs_rank >= 70:
                rank_score = 6.0
            elif rs_rank >= 60:
                rank_score = 3.0
            else:
                rank_score = 0.0

            score += rank_score
            details["rs_rank"] = rank_score
        else:
            details["rs_rank"] = 0.0

        return score, details

    # ============================================
    # VOLUME PROFILE SCORING (0-30 points)
    # ============================================

    def score_volume_profile(self, row: pd.Series):
        """
        Score liquidity and volume characteristics.

        Empirically the most predictive category — volume dry-up, ADR, and
        dollar volume all correlate strongly with 20-day max gain.

        Components (raised 2026-05 from 25 to 30 pts — stop pts redistributed here):
        - Dollar volume (0-6pts): Institutional-grade liquidity
        - Volume dry-up (0-14pts): Contraction into the base (single source of truth)
        - ADR % (0-10pts): Bigger movers produce bigger breakouts

        Perfect Score: >$100M dollar volume, strong dry-up, 10%+ ADR
        """
        score = 0.0
        details = {}

        # 1. DOLLAR VOLUME (0-6 points)
        dollar_vol = row.get("dollar_volume", 0)
        min_dollar_vol = self.config.get("dollar_volume_min", 10_000_000)

        if dollar_vol >= min_dollar_vol * 10:
            dv_score = 6.0
        elif dollar_vol >= min_dollar_vol * 5:
            dv_score = 5.0
        elif dollar_vol >= min_dollar_vol * 2:
            dv_score = 4.0
        elif dollar_vol >= min_dollar_vol:
            dv_score = 2.5
        else:
            dv_score = 0.0

        score += dv_score
        details["dollar_volume"] = dv_score

        # 2. VOLUME DRY-UP (0-14 points) — SINGLE SOURCE OF TRUTH
        # the core VCP signal: sellers exhausting into the base.
        volume_declining = row.get("volume_declining", False)
        dryup_ratio = row.get("volume_dryup_ratio", 1.0)
        rel_vol = row.get("relative_volume", 1.0)
        close_pos = row.get("close_range_position", 0.5)

        # primary path: period-based dry-up
        if volume_declining and dryup_ratio < 0.60:
            vd_score = 14.0
        elif volume_declining and dryup_ratio < 0.75:
            vd_score = 10.5
        elif volume_declining and dryup_ratio < 0.90:
            vd_score = 7.0
        elif dryup_ratio < 1.0:
            vd_score = 3.5
        else:
            vd_score = 0.0

        # spot dry-up boost: today's volume clearly below its 20-day average.
        # threshold < 0.70 is below the test-fixture default (0.80) so existing
        # tests are unaffected; the boost only fires on genuinely quiet sessions.
        if rel_vol < 0.70 and vd_score < 14.0:
            vd_score = min(14.0, vd_score + (2.0 if rel_vol < 0.50 else 1.0))

        # demand signal: strong close in the day's range confirms accumulation.
        # threshold ≥ 0.70 keeps test fixture default (0.5) neutral.
        if close_pos >= 0.70 and vd_score >= 7.0:
            vd_score = min(14.0, vd_score + 0.5)

        score += vd_score
        details["volume_contraction"] = vd_score

        # 3. ADR % (0-10 points)
        # bigger movers = bigger breakout moves; qullamaggie specifically targets high-ADR
        adr_pct = row.get("adr_pct", 0.0)

        if adr_pct >= 0.10:
            adr_score = 10.0
        elif adr_pct >= 0.08:
            adr_score = 8.0
        elif adr_pct >= 0.06:
            adr_score = 6.0
        elif adr_pct >= 0.05:
            adr_score = 3.0
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
        Compute stop/RR metrics for display — NOT counted in raw_total (2026-05).

        The 60-day rolling stop penalizes early-consolidation setups unfairly: a
        stock fresh off a big move has far-back lows in its window, making the
        stop look wide even when the current base is tight. Removing from scoring
        lets base tightness and volume dry-up drive the ranking instead.

        - Stop vs ADR ratio (0-10pts)
        - R-multiple potential (0-5pts)
        """
        score = 0.0
        details = {}

        # 1. STOP DISTANCE RELATIVE TO ADR (0-10 points)
        stop_distance = row.get("stop_distance_pct", 0.15)
        adr_pct = row.get("adr_pct", 0.05)
        stop_in_adr = stop_distance / max(adr_pct, 0.01)

        if 0.5 <= stop_in_adr <= 1.0:
            stop_score = 10.0
        elif stop_in_adr < 0.5:
            stop_score = 3.0
        elif stop_in_adr <= 1.5:
            stop_score = 8.0
        elif stop_in_adr <= 2.0:
            stop_score = 5.0
        elif stop_in_adr <= 2.5:
            stop_score = 3.0
        else:
            stop_score = 0.0

        score += stop_score
        details["stop_vs_adr"] = stop_score

        # 2. R-MULTIPLE POTENTIAL (0-5 points)
        potential_r = row.get("potential_r", 0.0)
        min_r = self.config.get("risk_reward_min", 3.0)

        if potential_r >= 5.0:
            r_score = 5.0
        elif potential_r >= 4.0:
            r_score = 4.0
        elif potential_r >= min_r:
            r_score = 3.0
        elif potential_r >= 2.0:
            r_score = 1.5
        else:
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
        _, rr_details = self.score_risk_reward(
            row
        )  # score excluded from raw_total; details kept

        # normalize each component to 0-1, then apply config weights.
        # with default weights (20/20/30/30) this is equivalent to raw addition.
        # changing config weights (e.g. from optimizer output) drives live scoring.
        _maxes = {
            "base_quality": 20.0,
            "trend_strength": 20.0,
            "relative_strength": 30.0,
            "volume_profile": 30.0,
        }
        raw_total = (
            (base_score / _maxes["base_quality"])
            * self.weights.get("base_quality", 20.0)
            + (trend_score / _maxes["trend_strength"])
            * self.weights.get("trend_strength", 20.0)
            + (rs_score / _maxes["relative_strength"])
            * self.weights.get("relative_strength", 30.0)
            + (volume_score / _maxes["volume_profile"])
            * self.weights.get("volume_profile", 30.0)
        )
        total = raw_total * self.regime_multiplier

        # rr details retained in the combined dict for watchlist display
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
            risk_reward=0.0,  # excluded from scoring; use details["rr_*"] for stop info
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

        # 5. Maximum stop distance — uses 20-day stop (current consolidation support).
        # the 60-day stop includes pre-flagpole lows for fresh setups, making it
        # artificially wide. 20-day captures where a trader would actually place
        # the stop on a flag or VCP entry.
        stop_dist = row.get("stop_distance_20d_pct", row.get("stop_distance_pct", 0))
        adr = row.get("adr_pct", 0.05)
        max_stop_adr_multiple = 3.0
        if stop_dist > max_stop_adr_multiple * max(adr, 0.01):
            failures.append(
                f"Stop distance {stop_dist:.1%} > {max_stop_adr_multiple:.0f}x ADR ({adr:.1%})"
            )
        elif stop_dist < 0.001:
            failures.append(f"Stop distance {stop_dist:.1%} too tight (< 0.1%)")

        # 6. Within 30% of recent high — uses the better of 52wk and 90d windows.
        # post-crash or post-move setups (COVID recovery, sector rotation) may sit
        # 50-70% below their 52wk high while being near their 90d flagpole top.
        # a stock near EITHER window is in a legitimate uptrend for entry purposes.
        max_dist_from_high = self.config.get("pct_from_52wk_high_max", 0.30)
        pct_52wk = row.get("pct_from_52wk_high", -1.0)
        pct_90d = row.get("pct_from_90d_high", pct_52wk)
        pct_52wk = pct_52wk if not pd.isna(pct_52wk) else -1.0
        pct_90d = pct_90d if not pd.isna(pct_90d) else -1.0
        pct_from_high = max(pct_52wk, pct_90d)
        if pct_from_high < -max_dist_from_high:
            failures.append(
                f"Price {abs(pct_from_high):.1%} below 52wk/90d high (max {max_dist_from_high:.0%})"
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
        df["raw_score"] = 0.0  # pre-regime-multiplier — used for grading setup quality
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
                    "base_range_%": round(
                        row.get(
                            f"consol_range_{self.config.get('base_length_max', 60)}", 0
                        )
                        * 100,
                        1,
                    ),
                    "pct_from_52wk_hi": round(
                        row.get("pct_from_52wk_high", 0) * 100, 1
                    ),
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
