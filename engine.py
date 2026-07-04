import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd
import numpy as np
from models import ScoreBreakdown
from datetime import datetime, timedelta


def _get_analyst_coverage(symbol: str) -> Optional[int]:
    """
    Return the number of analyst opinions for a symbol via yfinance.

    Returns None on any failure — callers treat None as unknown coverage
    and skip the scoring adjustment entirely.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info
        n = info.get("numberOfAnalystOpinions")
        if n is None:
            return None
        return int(n)
    except Exception:
        return None


def _get_days_to_earnings(symbol: str, as_of_date_str: str) -> Optional[int]:
    """
    Return calendar days to the next scheduled earnings date.

    Returns None if the data is unavailable or yfinance fails — callers treat
    None as "no earnings risk known" and skip the penalty.  Only upcoming dates
    (>= as_of_date) are returned; past earnings are ignored.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        as_of = pd.Timestamp(as_of_date_str)

        # yfinance ≥0.2 returns calendar as a dict; older returns a DataFrame
        cal = ticker.calendar
        if cal is None:
            return None

        dates = []
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate") or []
            for d in raw if isinstance(raw, list) else [raw]:
                try:
                    dates.append(pd.Timestamp(d))
                except Exception:
                    pass
        elif hasattr(cal, "loc"):  # DataFrame (older yfinance)
            for key in ("Earnings Date", "earningsDate"):
                if key in cal.index:
                    for d in np.atleast_1d(cal.loc[key]):
                        try:
                            dates.append(pd.Timestamp(d))
                        except Exception:
                            pass
                    break

        future = [d for d in dates if d >= as_of]
        if future:
            return int((min(future) - as_of).days)
        return None
    except Exception:
        return None


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

    def _fetch_earnings_batch(
        self, symbols: list, date_str: str, max_workers: int = 20
    ) -> dict:
        """
        Fetch days-to-next-earnings for every symbol in parallel.

        Returns {symbol: int_or_None}.  Failures for individual symbols are
        silently swallowed — the caller treats None as "no earnings risk known".
        """
        results = {}

        def _fetch(sym):
            return sym, _get_days_to_earnings(sym, date_str)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    sym, days = future.result(timeout=10)
                    results[sym] = days
                except Exception:
                    results[futures[future]] = None

        return results

    def _fetch_analyst_coverage_batch(
        self, symbols: list, max_workers: int = 20
    ) -> dict:
        """
        Fetch analyst opinion count for every symbol in parallel.

        Returns {symbol: int_or_None}. Failures are silently swallowed —
        the caller treats None as unknown coverage and skips scoring adjustment.
        """
        results = {}

        def _fetch(sym):
            return sym, _get_analyst_coverage(sym)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    sym, count = future.result(timeout=10)
                    results[sym] = count
                except Exception:
                    results[futures[future]] = None

        return results

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

        # Fetch earnings calendars and analyst coverage for all symbols in parallel
        all_symbols = [pf.stem.split("-")[0] for pf in pickle_files]
        print(f"Fetching earnings calendars for {len(all_symbols)} symbols...")
        earnings_map = self._fetch_earnings_batch(all_symbols, date_str)
        n_with_earnings = sum(1 for v in earnings_map.values() if v is not None)
        print(f"  Earnings data found for {n_with_earnings}/{len(all_symbols)} symbols")

        coverage_map = {}
        if self.config.get("score_analyst_coverage", False):
            print(f"Fetching analyst coverage for {len(all_symbols)} symbols...")
            coverage_map = self._fetch_analyst_coverage_batch(all_symbols)
            n_with_coverage = sum(1 for v in coverage_map.values() if v is not None)
            print(
                f"  Coverage data found for {n_with_coverage}/{len(all_symbols)} symbols"
            )
        print()

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

                # stamp days_to_earnings onto the DataFrame; used by scoring for penalty
                # and by the watchlist for the ⚠ warning. None = unknown (no penalty).
                days = earnings_map.get(symbol)
                feature_df["days_to_earnings"] = days if days is not None else np.nan

                # stamp analyst_coverage; None = unknown (scoring skips adjustment)
                coverage = coverage_map.get(symbol)
                feature_df["analyst_coverage"] = (
                    float(coverage) if coverage is not None else np.nan
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

        # lee-swaminathan (2000): base volume vs 1-year historical average.
        # flagpole-relative dry-up can look great while base vol is still structurally
        # elevated. a ratio > 0.90 means the "quiet" base is still near its own long-run
        # norm — not genuine accumulation, just a lull after the flagpole.
        hist_window = self.config.get("volume_historical_avg_window", 252)
        hist_avg = (
            df["volume"]
            .rolling(window=hist_window, min_periods=hist_window // 2)
            .mean()
        )
        df["volume_vs_6m_avg"] = recent_vol / hist_avg

        return df

    def detect_consolidation_range(self, df, lookback=None):
        if lookback is None:
            lookback = self.config.get("base_length_max", 15)

        # Calculate range over rolling window
        rolling_high = df["high"].rolling(window=lookback).max()
        rolling_low = df["low"].rolling(window=lookback).min()

        df[f"consol_range_{lookback}"] = (rolling_high - rolling_low) / df["close"]

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

        # dynamic breakout level: use actual consolidation window, not the full lookback.
        # a stock 20 days into a flag after a 30-day flagpole has its 60-day high at
        # the flagpole top (10-15% above the base) — using that as breakout_level
        # means the outcome tracker never fires breakout_triggered=True for clean
        # consolidation exits. use consol_days to look back only over the base.
        highs = df["high"].values
        c_days = df["consol_days"].fillna(0).astype(int).values
        n = len(df)
        dynamic_levels = np.empty(n)

        for i in range(n):
            cd = c_days[i]
            if cd >= 5:  # confirmed consolidation: use only the base window
                window = min(cd, lookback)
                start = max(0, i - window + 1)
                dynamic_levels[i] = highs[start : i + 1].max()
            else:  # not in consolidation: fall back to full lookback
                start = max(0, i - lookback + 1)
                dynamic_levels[i] = highs[start : i + 1].max()

        df["breakout_level"] = dynamic_levels

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

    def calculate_relative_strength(
        self, df, benchmark_df, benchmark_name="SPY", skip_days=0
    ):
        """
        Calculate relative strength vs benchmark as excess return.

        RS = Stock % Change - Benchmark % Change  (excess return / alpha)

        This formulation is correct in all market conditions:
        - Bull market: stock +20%, benchmark +5% → RS = +15%  (outperforming)
        - Bear market: stock +5%, benchmark -5% → RS = +10%   (strong RS signal)
        - Laggard:     stock -5%, benchmark +5% → RS = -10%   (underperforming)

        skip_days (JT1993 skip-month): when > 0, both series are shifted so the
        measurement window ends skip_days ago rather than today. e.g. skip_days=5
        computes rs_comp_60 as (close[-6]/close[-66])-1, excluding the most recent
        week. this reduces short-term reversal noise (bid-ask bounce, microstructure)
        that attenuates the true momentum signal (Jegadeesh & Titman 1993, JF).
        """
        benchmark_aligned = benchmark_df.reindex(df.index, method="ffill")

        stock_close = df["close"].shift(skip_days) if skip_days > 0 else df["close"]

        if "close" in benchmark_aligned.columns:
            bench_close = (
                benchmark_aligned["close"].shift(skip_days)
                if skip_days > 0
                else benchmark_aligned["close"]
            )
        else:
            bench_col = benchmark_aligned.iloc[:, 0]
            bench_close = bench_col.shift(skip_days) if skip_days > 0 else bench_col

        for period in [20, 60, 120, 252]:
            stock_pct_change = stock_close.pct_change(periods=period)
            benchmark_pct_change = bench_close.pct_change(periods=period)

            # excess return: positive = outperforming, negative = underperforming
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

    def calculate_obv(self, df):
        """
        on-balance volume: cumulative volume flow in direction of price change.
        rising obv during flat price = institutional accumulation (demand side signal).
        complements volume dry-up (supply side) in the base.

        obv_trend uses a 20-day window (consolidation-length) and requires OBV to be
        rising FASTER than price — this is the accumulation divergence signal.
        a simple sign-check on the full 60-day window fires on virtually every
        uptrending stock (validated: 100% activation on filter-passing set), which
        adds no discrimination between confirmed breakouts and false positives.
        """
        direction = np.sign(df["close"].diff()).fillna(0)
        df["obv"] = (direction * df["volume"]).cumsum()

        base_len = self.config.get("base_length_max", 60)
        consol_window = 20  # short enough to reflect the current base, not the flagpole

        def _slope(s):
            if len(s) < 5:
                return 0.0
            x = np.arange(len(s))
            return float(np.polyfit(x, s, 1)[0])

        df["obv_slope"] = (
            df["obv"].rolling(window=base_len, min_periods=10).apply(_slope, raw=True)
        )

        # consolidation-aware OBV trend: rising OBV while price is flat/declining.
        # normalize OBV slope by avg daily volume to get a per-share unit.
        # normalize price slope by close to get % per day.
        # accumulation fires when OBV/vol-unit slope > price %/day slope
        avg_vol = (
            df["volume"].rolling(consol_window, min_periods=5).mean().clip(lower=1)
        )
        obv_slope_norm = (
            df["obv"].rolling(consol_window, min_periods=5).apply(_slope, raw=True)
            / avg_vol
        )
        price_slope_pct = (
            df["close"]
            .rolling(consol_window, min_periods=5)
            .apply(
                lambda s: (
                    float(np.polyfit(np.arange(len(s)), s, 1)[0]) / max(s[-1], 0.01)
                ),
                raw=True,
            )
        )
        # obv_trend=True when OBV is rising meaningfully faster than price
        # (accumulation) OR when OBV rising and price flat/declining (distribution ended)
        df["obv_trend"] = (obv_slope_norm > 0) & (
            obv_slope_norm > price_slope_pct.clip(lower=0)
        )
        return df

    def calculate_trigger_bar(self, df):
        """
        trigger bar: tightest range + lowest volume in the last 20 bars.
        qullamaggie calls this the "very tight bar" just before the breakout —
        sellers completely exhausted, range and volume both at extremes.
        """
        window = 20
        range_pct = (df["high"] - df["low"]) / df["close"].clip(lower=0.01)
        df["is_trigger_bar"] = (
            range_pct < range_pct.rolling(window, min_periods=5).quantile(0.20)
        ) & (df["volume"] < df["volume"].rolling(window, min_periods=5).quantile(0.20))
        return df

    def calculate_weekly_alignment(self, df):
        """
        weekly trend filter: close > 10-week EMA > 20-week EMA.
        a stock can look clean on the daily while distributing on the weekly —
        this prevents chasing into multi-week downtrend structures.
        soft signal only; insufficient data defaults to True (no penalty).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df["weekly_aligned"] = True
            return df

        weekly = df["close"].resample("W-FRI").last().dropna()
        if len(weekly) < 10:
            df["weekly_aligned"] = True
            return df

        w10 = weekly.ewm(span=10, adjust=False).mean()
        w20 = weekly.ewm(span=20, adjust=False).mean()
        last_close = float(weekly.iloc[-1])
        aligned = bool(
            (last_close > float(w10.iloc[-1]))
            and (float(w10.iloc[-1]) > float(w20.iloc[-1]))
        )
        df["weekly_aligned"] = aligned
        return df

    def calculate_tsmom(self, df):
        """
        time-series momentum: stock's own absolute N-month return.
        Moskowitz, Ooi & Pedersen (2012, JFE 104:228) show TSMOM is
        additive to cross-sectional RS (rs_comp_*) and independently
        predicts 1-12 month forward returns.
        abs_return_63d ≈ 3M, abs_return_126d ≈ 6M.
        NaN fills forward for short histories; scoring silently skips.
        """
        df["abs_return_63d"] = df["close"].pct_change(periods=63)
        df["abs_return_126d"] = df["close"].pct_change(periods=126)
        return df

    def calculate_52wk_proximity(self, df):
        df["52wk_high"] = df["high"].rolling(window=252, min_periods=100).max()
        df["pct_from_52wk_high"] = (df["close"] - df["52wk_high"]) / df["52wk_high"]

        # 90-day window: for post-crash/post-move setups the 52wk high is a pre-crash
        # price and unfairly penalizes stocks near their recent flagpole top
        df["90d_high"] = df["high"].rolling(window=90, min_periods=30).max()
        df["pct_from_90d_high"] = (df["close"] - df["90d_high"]) / df["90d_high"]

        # george & hwang (2004, JF 59:2145): anchoring-underreaction window
        # analysts and investors anchor to the 52wk high and systematically discount
        # further gains when a stock approaches it for the first time after consolidation.
        # alpha is strongest in the -15% to -3% zone (approaching but not yet extended).
        # stocks already above -3% (extended) or below -15% (deep base) are excluded.
        df["approaching_annual_high"] = (df["pct_from_52wk_high"] > -0.15) & (
            df["pct_from_52wk_high"] < -0.03
        )
        return df

    def detect_vcp_contractions(self, df):
        w_short, w_medium, w_long = self.config.get("vcp_windows", [10, 20, 40])

        h, lo, c = df["high"], df["low"], df["close"]

        # overlapping ranges (for ratio and tightness proxy)
        range_short = (h.rolling(w_short).max() - lo.rolling(w_short).min()) / c
        range_medium = (h.rolling(w_medium).max() - lo.rolling(w_medium).min()) / c
        range_long = (h.rolling(w_long).max() - lo.rolling(w_long).min()) / c

        df[f"range_{w_short}"] = range_short
        df[f"range_{w_medium}"] = range_medium
        df[f"range_{w_long}"] = range_long

        # non-overlapping windows going backward in time
        def _nonoverlap_range(shift_bars):
            return (
                h.rolling(w_short).max().shift(shift_bars)
                - lo.rolling(w_short).min().shift(shift_bars)
            ) / c

        range_now = range_short
        range_prev = _nonoverlap_range(w_short)
        range_far = _nonoverlap_range(w_short * 2)
        range_vfar = _nonoverlap_range(w_short * 3)

        # consecutive contraction count: each step narrower than the one before.
        # c1: most recent contraction (now vs prev)
        # c2: second contraction (prev vs far)  — only meaningful if c1 is True
        # c3: third contraction (far vs vfar)   — only meaningful if c1+c2 are True
        c1 = (range_now < range_prev).astype(int)
        c2 = ((range_prev < range_far) & (range_now < range_prev)).astype(int)
        c3 = (
            (range_far < range_vfar)
            & (range_prev < range_far)
            & (range_now < range_prev)
        ).astype(int)
        df["vcp_contraction_count"] = c1 + c2 + c3

        # boolean flag: True when at least 2 consecutive contractions confirmed
        df["vcp_contracting"] = df["vcp_contraction_count"] >= 2

        # ratio vs overlapping long window — tightness proxy kept for fallback scoring
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

        # OBV accumulation (demand side complement to volume dry-up)
        df = self.calculate_obv(df)

        # Trigger bar: tightest range + lowest volume in 20-bar window
        df = self.calculate_trigger_bar(df)

        # Weekly trend alignment filter
        df = self.calculate_weekly_alignment(df)

        # time-series momentum (Moskowitz, Ooi & Pedersen 2012, JFE 104:228)
        # stock's own absolute N-month return — additive to cross-sectional RS
        df = self.calculate_tsmom(df)

        df = self.calculate_stop(df)
        df = self.calculate_rr(df)

        # Relative Strength vs NASDAQ Composite - if benchmark provided
        if benchmark_df is not None:
            skip = self.config.get("rs_skip_days", 0)
            df = self.calculate_relative_strength(
                df, benchmark_df, benchmark_name="COMP", skip_days=skip
            )

        return df


class Scoring:
    """
    Scores stocks based on Qullamaggie breakout + Minervini VCP principles.

    Config weights (2026-06 rebalance: base anti-predictive, RS/volume raised):
    - Base Quality   (10pts contrib): raw method output 0-20; r=-0.023 anti-predictive
    - Trend Strength (15pts contrib): raw method output 0-14; prior_move is the key signal
    - Relative Strength (25pts contrib): raw method output 0-30; r=+0.076 strongest feature
    - Volume Profile (50pts contrib): raw method output 0-30; dominant predictor (+0.28r)
    - Risk/Reward    (excluded):     stop info retained for display; not in raw_total

    calculate_total_score normalizes each component: (raw_score / sub_max) * config_weight.
    Sub-component maxes (denominators): base=20, trend=14, rs=30, volume=30.
    Config weights (numerators, sum=100): base=10, trend=15, rs=25, volume=50.

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
        # DB analysis (n=9,686, 2026-06): 35-60d → mean +30.6%, 20-35d → +27.5%,
        # 5-15d → +22.9%. old scoring penalised the best-performing window (35-60d).
        # >60 days stalls: mean +5.6% as the base matures past the breakout window.
        consol_days = row.get("consol_days", 0)

        if 35 <= consol_days <= 60:
            length_score = 4.0  # optimal: mature, well-formed base
        elif 20 <= consol_days < 35:
            length_score = 3.5  # very good
        elif 10 <= consol_days < 20:
            length_score = 3.0  # normal flag
        elif 5 <= consol_days < 10:
            length_score = 2.5  # short flag
        elif consol_days > 60:
            length_score = 1.0  # too long — base stalls out
        else:
            length_score = 0.0

        score += length_score
        details["base_length"] = length_score

        # 3. VCP RANGE CONTRACTION (0-4 pts)
        # count-based scoring: each consecutive non-overlapping window that is narrower
        # than the one before it counts as one contraction. 3 = full VCP textbook pattern.
        # falls back to vcp_contracting flag or ratio when count is unavailable.
        count = int(row.get("vcp_contraction_count", -1))
        vcp_contracting = row.get("vcp_contracting", False)
        vcp_ratio = row.get("vcp_contraction_ratio", 1.0)

        if count >= 3:
            vcp_score = 4.0  # 3 consecutive contractions: textbook VCP
        elif count == 2:
            vcp_score = 3.0  # solid: 2 confirmed contractions
        elif count == 1:
            vcp_score = 2.0  # early VCP: 1 contraction confirmed
        elif count == 0:
            vcp_score = 0.0  # count available but no contraction detected
        elif vcp_contracting:
            vcp_score = 2.0  # fallback: boolean flag when count not computed
        elif vcp_ratio <= 0.60:
            vcp_score = 0.5  # ratio-only fallback: partial tightness signal
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

        # 5. TRIGGER BAR BONUS (0-1.5 pts, capped so total stays at 20)
        # the "very tight bar" just before breakout: both range and volume in the
        # bottom 20th percentile of the last 20 bars. seller exhaustion at the tip.
        if row.get("is_trigger_bar", False):
            trigger_bonus = 1.5
            score = min(20.0, score + trigger_bonus)
            details["trigger_bar"] = trigger_bonus
        else:
            details["trigger_bar"] = 0.0

        return score, details

    # ============================================
    # TREND STRENGTH SCORING (0-20 points → up to 23 with pivot proximity)
    # ============================================

    def score_trend_strength(self, row: pd.Series):
        """
        Score underlying trend structure — Qullamaggie + Minervini combined.

        Components (2026-06: stage2 removed as anti-predictive; proximity removed 2026-06):
        - Short-term MA           (0-4pts):   10>20>50 aligned + rising
        - Prior power move        (0-8pts):   Flagpole size; 100%+ produces 68.9% breakout rate
        - Approaching annual high (0-2pts):   George & Hwang (2004) anchoring-underreaction alpha
        - Weekly alignment        (penalty -5pts if weekly trend broken)
        - Analyst coverage adj    (±2/-1pts): Hong, Lim & Stein (2000) info-diffusion alpha;
                                              gated on score_analyst_coverage config flag

        Sub-max = 14 (4+8+2; analyst coverage is a ±adjustment within that range).
        Stage 2 removed 2026-06: DB analysis (n=4662, prior>=75%) showed
        stage2=True EV=0.195 vs stage2=False EV=0.276 — full Stage 2 stocks are extended;
        fresh breakout stocks (transitioning INTO Stage 2) outperform by 40%.
        """
        score = 0.0
        details = {}
        details["stage2"] = 0.0  # kept for persistence compatibility; no longer scored

        # 2. SHORT-TERM MA STRUCTURE (0-4 points)
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

        # 3. PRIOR POWER MOVE (0-8 points) — the flagpole before the base.
        # magnitude matters: 100%+ flagpoles produce 68.9% breakout rates vs 53% for 30-50%
        # (empirical finding 2026-06; prior scoring gave same 6pts for 40% and 400%).
        prior_move = row.get("prior_move_pct", 0.0)
        days_since_move = row.get("days_since_power_move", 999)

        if prior_move >= 2.0 and days_since_move <= 60:  # 200%+ flagpole
            power_score = 8.0
        elif prior_move >= 1.0 and days_since_move <= 60:  # 100-200%
            power_score = 7.0
        elif prior_move >= 0.75 and days_since_move <= 60:  # 75-100%
            power_score = 5.5
        elif (
            prior_move >= 0.75 and days_since_move <= 90
        ):  # 75%+ outside 60d window (filter minimum)
            power_score = 4.0
        else:
            power_score = 0.0

        score += power_score
        details["prior_power_move"] = power_score

        # 5. APPROACHING ANNUAL HIGH (0-2 pts) — George & Hwang (2004, JF 59:2145)
        # anchoring bias causes investors to underreact when a stock approaches its 52wk
        # high for the first time after a consolidation. the -15% to -3% zone is the
        # documented alpha window: close enough to create anchoring, far enough to not
        # be already extended. consol_days >= 10 gates out stocks that haven't formed a base.
        approaching = row.get("approaching_annual_high", False)
        consol_days_for_gh = row.get("consol_days", 0)
        if approaching and consol_days_for_gh >= 10:
            score += 2.0
            details["approaching_annual_high"] = 2.0
        else:
            details["approaching_annual_high"] = 0.0

        # 6. WEEKLY ALIGNMENT SOFT PENALTY (-5 pts if weekly trend broken)
        # prevents chasing daily-clean setups that are distributing on the weekly chart.
        # default True (no penalty) when weekly data is insufficient.
        if not row.get("weekly_aligned", True):
            score = max(0.0, score - 5.0)
            details["weekly_alignment"] = -5.0
        else:
            details["weekly_alignment"] = 0.0

        # 7. TSMOM GATE (Moskowitz, Ooi & Pedersen 2012, JFE 104:228)
        # stock's own absolute 3-month return < 0 means it has fully given back its
        # gains — negative TSMOM is the weakest momentum bin and highest crash-risk
        # cohort (Barroso & Santa-Clara 2015 confirm). penalty -3 pts, floored at 0.
        # gated on score_tsmom_gate config flag; None/NaN = graceful skip.
        details["tsmom_gate"] = 0.0
        if self.config.get("score_tsmom_gate", False):
            abs_63d = row.get("abs_return_63d")
            if abs_63d is not None and not (
                isinstance(abs_63d, float) and np.isnan(abs_63d)
            ):
                if float(abs_63d) < 0.0:
                    score = max(0.0, score - 3.0)
                    details["tsmom_gate"] = -3.0

        # 8. ANALYST COVERAGE ADJUSTMENT (hong, lim & stein 2000, JF 55:265)
        # information diffuses more slowly through under-covered stocks → longer
        # underreaction window → stronger and more persistent momentum.
        # 0 analysts → +2 pts | 1-2 → +1 pt | 3-5 → 0 pts | 6+ → -1 pt
        # gated on score_analyst_coverage config flag; None/NaN = skip silently
        details["analyst_coverage"] = 0.0
        if self.config.get("score_analyst_coverage", False):
            raw_coverage = row.get("analyst_coverage")
            if raw_coverage is not None and not (
                isinstance(raw_coverage, float) and np.isnan(raw_coverage)
            ):
                n = int(raw_coverage)
                if n == 0:
                    coverage_adj = 2.0
                elif n <= 2:
                    coverage_adj = 1.0
                elif n <= 5:
                    coverage_adj = 0.0
                else:
                    coverage_adj = -1.0
                score = max(0.0, score + coverage_adj)
                details["analyst_coverage"] = coverage_adj

        # cap at sub-max so adjustments (analyst coverage +2, etc.) can't inflate
        # the normalized contribution above the allocated config weight
        return min(score, 14.0), details

    # ============================================
    # RELATIVE STRENGTH SCORING (0-30 points)
    # ============================================

    def score_relative_strength(self, row: pd.Series, rs_rank: float = None):
        """
        Score outperformance vs NASDAQ Composite.

        RS = excess return (stock% - benchmark%). Positive = outperforming.
        Empirically the strongest confirmed predictor of breakout success.

        Components (restructured 2026-05 — 20d replaced by 120d):
        - 120-day RS (0-12pts): captures the full flagpole cleanly pre-consolidation
        - 60-day RS (0-12pts): PRIMARY signal — captures the flagpole itself cleanly
        - 120-day RS (0-8pts): secondary; positive=trending, but very high 120d = extended
        - RS percentile rank (0-10pts): rank vs all stocks in this watchlist

        DB EV analysis (n=4662, prior>=75% cohort 2026-06):
        strong60+weak120: EV=0.301 (best); weak60+strong120: EV=0.186 (worst).
        60d rs_comp_60 Q4 EV=0.303 vs Q1 EV=0.231 (monotonic, positive).
        120d rs_comp_120 is non-monotonic (Q3 is lowest), so reduced to secondary.

        Peer rank recalibration: the 95th+ percentile stock is often the most
        widely-watched, most distributed name at the pivot. the 85-95th percentile
        range captures strong leaders before the crowd finds them.

        Perfect Score: 25%+ 120d excess, 20%+ 60d excess, 85-95th peer rank
        """
        score = 0.0
        details = {}

        # 1. MEDIUM-TERM RS — 60 days (0-12 points) — PRIMARY signal
        # DB: strong60d rs EV=0.303 (Q4) vs weak60d EV=0.231 (Q1). Monotonic positive.
        # 60d window captures the recent flagpole cleanly for a 10-30d base.
        rs_60 = row.get("rs_comp_60", 0.0)

        if rs_60 >= 0.50:  # top quartile in filter-passing cohort (EV=0.303)
            rs_60_score = 12.0
        elif rs_60 >= 0.25:
            rs_60_score = 9.0
        elif rs_60 >= 0.12:
            rs_60_score = 6.0
        elif rs_60 >= 0.00:
            rs_60_score = 2.0
        else:
            rs_60_score = 0.0

        score += rs_60_score
        details["rs_60_day"] = rs_60_score

        # 2. LONG-TERM RS — 120 days (0-8 points) — secondary, non-monotonic
        # DB: Q1 EV=0.240, Q2 EV=0.256, Q3 EV=0.199, Q4 EV=0.254. Non-monotonic.
        # Very high 120d = extended stock (flagpole was 3-4 months ago). Moderate = fresh.
        rs_120 = row.get("rs_comp_120", 0.0)

        if rs_120 >= 0.20:
            rs_120_score = 8.0
        elif rs_120 >= 0.10:
            rs_120_score = 6.0
        elif rs_120 >= 0.04:
            rs_120_score = 3.0
        elif rs_120 >= 0.00:
            rs_120_score = 1.0
        else:
            rs_120_score = 0.0

        score += rs_120_score
        details["rs_120_day"] = rs_120_score

        # 3. RS PERCENTILE RANK vs PEERS (0-10 points)
        # the 85-95th percentile sweet spot: strong leaders before they become
        # crowded; max 8 pts for the very top (95th+) to reduce distribution risk.
        if rs_rank is not None:
            if rs_rank >= 95:
                rank_score = 8.0
            elif rs_rank >= 85:
                rank_score = 10.0
            elif rs_rank >= 75:
                rank_score = 7.0
            elif rs_rank >= 65:
                rank_score = 4.0
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
        # only valid for stocks in an active base (consol_days >= 5).
        # for non-consolidating stocks, declining volume = momentum exhaustion (bearish),
        # not accumulation. gating here protects the backtester/optimizer on historical
        # data that predates the consol_days hard filter.
        consol_days = int(row.get("consol_days", 0) or 0)
        volume_declining = row.get("volume_declining", False)
        dryup_ratio = row.get("volume_dryup_ratio", 1.0)
        rel_vol = row.get("relative_volume", 1.0)
        close_pos = row.get("close_range_position", 0.5)

        if consol_days >= 5:
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
            if rel_vol < 0.70 and vd_score < 14.0:
                vd_score = min(14.0, vd_score + (2.0 if rel_vol < 0.50 else 1.0))

            # demand signal: strong close in the day's range confirms accumulation.
            if close_pos >= 0.70 and vd_score >= 7.0:
                vd_score = min(14.0, vd_score + 0.5)

            # OBV accumulation bonus (0-2 pts, capped within the 14-pt max).
            if row.get("obv_trend", False):
                obv_bonus = 2.0 if vd_score >= 7.0 else 1.0
                vd_score = min(14.0, vd_score + obv_bonus)

            # lee-swaminathan (2000) historical vol penalty: if base volume is still
            # >= 90% of the 1-year average, the "quiet" base is not structurally quiet —
            # just quiet relative to the flagpole spike. genuine accumulation has base
            # vol well below the stock's own historical norm.
            vol_vs_hist = row.get("volume_vs_6m_avg", None)
            if (
                vol_vs_hist is not None
                and not np.isnan(vol_vs_hist)
                and vol_vs_hist > 0.90
            ):
                vd_score = max(0.0, vd_score - 2.0)
        else:
            vd_score = 0.0

        score += vd_score
        details["volume_contraction"] = vd_score

        # 3. ADR % (0-10 points) — non-monotonic: peak at 15-20%, soft penalty above 20%.
        # DB analysis (n=9,686, 2026-06): 15-20% ADR → mean +46.3% 20d gain vs
        # >20% ADR → mean +23.2% (drops back to near baseline of 7-10% bucket).
        # very high ADR stocks are too volatile to hold reliably through the breakout.
        adr_pct = row.get("adr_pct", 0.0)

        # DB EV analysis (n=9686, prior>=75% subset): 12-15% EV=0.429 (peak),
        # 15-20% EV=0.378, 10-12% EV=0.331, 7-10% EV=0.213, 20-25% EV=0.222, 25%+ EV=0.055
        if adr_pct >= 0.12 and adr_pct < 0.15:  # peak: EV=0.429
            adr_score = 10.0
        elif adr_pct >= 0.15 and adr_pct <= 0.20:  # still great: EV=0.378
            adr_score = 9.0
        elif adr_pct > 0.20 and adr_pct < 0.25:  # above peak, decent: EV=0.222
            adr_score = 5.5
        elif adr_pct >= 0.25:  # very high vol, poor EV=0.055
            adr_score = 2.0
        elif adr_pct >= 0.10:  # good: EV=0.331
            adr_score = 7.5
        elif adr_pct >= 0.08:
            adr_score = 5.0
        elif adr_pct >= 0.07:
            adr_score = 2.5
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
        # sub-max is the actual highest possible raw score from each method:
        #   base_quality:  6+4+4+6 = 20 (trigger bar capped at 20)
        #   trend_strength: 4+8+2 = 14 (approaching_annual_high added 2026-06: GH2004 anchoring alpha)
        #   relative_strength: 12+8+10 = 30 (60d now primary, swapped from 120d)
        #   volume_profile: 6+14+10 = 30 (OBV bonus capped within vd 14)
        _maxes = {
            "base_quality": 20.0,
            "trend_strength": 14.0,
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

        # earnings proximity penalty: upcoming earnings = coin-flip risk
        # threshold ≤ 5 → -10 pts; ≤ 10 → -5 pts. None/NaN = no penalty.
        days_to_earnings = row.get("days_to_earnings")
        if days_to_earnings is not None and not (
            isinstance(days_to_earnings, float) and np.isnan(days_to_earnings)
        ):
            dte = int(days_to_earnings)
            if 0 <= dte <= 5:
                raw_total = max(0.0, raw_total - 10.0)
            elif 0 <= dte <= 10:
                raw_total = max(0.0, raw_total - 5.0)

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
        max_stop_adr_multiple = self.config.get("stop_adr_multiple", 5.0)
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

        # 7. Minimum prior move — Qullamaggie's strategy requires a genuine flagpole.
        # raised from 25% → 30% → 50% → 75% (2026-06): DB EV analysis (n=9,686):
        # >=50% EV=0.203, >=75% EV=0.236, >=100% EV=0.262. monotonically improving.
        # 75% balances quality vs scan universe size (4,662 vs 7,124 stocks at 50%).
        min_prior_move = self.config.get("min_prior_move_pct", 0.25)
        prior_move = row.get("prior_move_pct", 0.0) or 0.0
        if prior_move < min_prior_move:
            failures.append(
                f"Prior move {prior_move:.1%} < {min_prior_move:.0%} minimum flagpole"
            )

        # 8. Minimum consolidation days — must be in an active base.
        # 96% of historical records have consol_days=0; stocks with consol_days=0
        # averaged +29% 20d gain vs +10.7% for stocks in a detected base — the gap
        # exists because consol_days=0 stocks are already mid-breakout (the engine
        # was scoring breakout stocks rather than pre-breakout setups). this filter
        # ensures we only score stocks actually building a base.
        min_consol = self.config.get("min_consol_days", 5)
        consol = int(row.get("consol_days", 0) or 0)
        if consol < min_consol:
            failures.append(
                f"Consolidation {consol}d < {min_consol}d minimum — not in a base"
            )

        # 9. 12-month RS must not underperform NASDAQ — AQR momentum universe gate.
        # Moskowitz et al. (2012) and AQR live indices require stocks to have beaten
        # their benchmark over the prior year to qualify as momentum candidates.
        # NaN = fewer than 252 bars of history → silently skipped (graceful degradation).
        if self.config.get("require_positive_rs_252", True):
            rs_252 = row.get("rs_comp_252", np.nan)
            if not pd.isna(rs_252) and rs_252 < 0.0:
                failures.append(
                    f"12-month RS {rs_252:.1%} vs NASDAQ — not a 12M leader"
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

            dte = row.get("days_to_earnings")
            if dte is not None and not (isinstance(dte, float) and np.isnan(dte)):
                dte_int = int(dte)
                if 0 <= dte_int <= 5:
                    earn_warn = f"⚠ EARN {dte_int}d"
                elif dte_int <= 10:
                    earn_warn = f"earn {dte_int}d"
                else:
                    earn_warn = ""
            else:
                earn_warn = ""

            watchlist_data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "score": row["total_score"],
                    "grade": row["grade"],
                    "signal": row["signal"],
                    "earnings": earn_warn,
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
