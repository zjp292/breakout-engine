"""
Performance Analyzer — derives actionable insights from historical scan outcomes.

Reports produced:
  1. Outcome rate by score bracket  — do higher scores predict better results?
  2. Market regime vs outcomes      — does the regime multiplier add real value?
  3. Feature correlation            — which computed features are most predictive?
  4. Filter effectiveness           — are any hard filters eliminating winners?
  5. Scoring weight suggestions     — which weights deserve more / less emphasis?

Requires scan + outcome data from persistence.py / outcome_tracker.py.
The analysis becomes meaningful after ~3-4 weeks of scans and outcomes.
Scoring weight suggestions should not be acted on until 50+ outcomes exist.

Run directly:
    python analyze.py
    python analyze.py --min-outcomes 5
    python analyze.py --db results/breakout.db
"""

import argparse

import pandas as pd

from persistence import ScanPersistence

W = 70  # report width


class PerformanceAnalyzer:
    def __init__(self, db: ScanPersistence):
        self.db = db

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self, min_outcomes: int = 5) -> None:
        """Run the full analysis suite and print results to stdout."""
        scans = self.db.load_scans()
        outcomes = self.db.load_outcomes()
        mc = self.db.load_market_conditions()

        if outcomes.empty:
            print(
                "\nNo outcome data yet.\n"
                "Run outcome_tracker.py after your scans are at least 10 days old."
            )
            return

        # Join scans ↔ outcomes on (scan_date, symbol)
        merged = scans.merge(
            outcomes, on=["scan_date", "symbol"], how="inner", suffixes=("", "_out")
        )

        if len(merged) < min_outcomes:
            print(
                f"\nOnly {len(merged)} scans with outcomes "
                f"(minimum required: {min_outcomes}).\n"
                "Collect more data before running the full analysis."
            )
            return

        # Attach market regime label for regime report
        if not mc.empty:
            merged = merged.merge(
                mc[["scan_date", "regime"]].rename(columns={"regime": "mc_regime"}),
                on="scan_date",
                how="left",
            )

        print(f"\n{'═' * W}")
        print(f"  PERFORMANCE ANALYSIS   ({len(merged)} scans with outcomes)")
        print(f"{'═' * W}")

        self._report_score_brackets(merged)
        self._report_regime_vs_outcomes(merged)
        self._report_feature_correlation(merged)
        self._report_filter_effectiveness(scans, outcomes)
        self._report_weight_suggestions(merged)

        print(f"\n{'═' * W}\n")

    # ── Report 1: Score brackets ──────────────────────────────────────────────

    def _report_score_brackets(self, df: pd.DataFrame) -> None:
        _header("1. OUTCOME RATE BY SCORE BRACKET")

        passed = df[df["passes_filters"] == 1].copy()
        if passed.empty:
            print("  No data with passes_filters = 1")
            return

        brackets = [
            ("A+  90–100", 90, 101),
            ("A   80–89 ", 80, 90),
            ("B   70–79 ", 70, 80),
            ("C   60–69 ", 60, 70),
            ("D   <60   ", 0, 60),
        ]

        print(
            f"  {'Grade':10s}  {'N':>4}  {'Breakout%':>9}  {'Stop%':>6}  "
            f"{'Target%':>7}  {'AvgChg':>7}  {'AvgMax20':>9}  {'AvgDd20':>8}"
        )
        _divider()

        for label, lo, hi in brackets:
            sub = passed[(passed["score"] >= lo) & (passed["score"] < hi)]
            if sub.empty:
                continue
            n = len(sub)
            print(
                f"  {label:10s}  {n:>4}"
                f"  {sub['breakout_triggered'].mean() * 100:>8.0f}%"
                f"  {sub['stop_triggered'].mean() * 100:>5.0f}%"
                f"  {sub['target_reached'].mean() * 100:>6.0f}%"
                f"  {sub['pct_change'].mean() * 100:>+6.1f}%"
                f"  {sub['max_gain_20d'].mean() * 100:>+8.1f}%"
                f"  {sub['max_drawdown_20d'].mean() * 100:>+7.1f}%"
            )

    # ── Report 2: Market regime ───────────────────────────────────────────────

    def _report_regime_vs_outcomes(self, df: pd.DataFrame) -> None:
        if "mc_regime" not in df.columns:
            return

        passed = df[df["passes_filters"] == 1].dropna(subset=["mc_regime"])
        if passed.empty:
            return

        _header("2. MARKET REGIME VS OUTCOMES")
        print(
            f"  {'Regime':12s}  {'N':>4}  {'Breakout%':>9}  "
            f"{'AvgChg':>7}  {'AvgMax20':>9}  {'Multiplier note'}"
        )
        _divider()

        regime_order = ["BULL", "UPTREND", "MIXED", "CAUTION", "DOWNTREND"]
        multipliers = {
            "BULL": "×1.00",
            "UPTREND": "×0.95",
            "MIXED": "×0.85",
            "CAUTION": "×0.70",
            "DOWNTREND": "×0.50",
        }
        for regime in regime_order:
            sub = passed[passed["mc_regime"] == regime]
            if sub.empty:
                continue
            n = len(sub)
            print(
                f"  {regime:12s}  {n:>4}"
                f"  {sub['breakout_triggered'].mean() * 100:>8.0f}%"
                f"  {sub['pct_change'].mean() * 100:>+6.1f}%"
                f"  {sub['max_gain_20d'].mean() * 100:>+8.1f}%"
                f"  {multipliers.get(regime, '')}"
            )

    # ── Report 3: Feature correlation ────────────────────────────────────────

    def _report_feature_correlation(self, df: pd.DataFrame) -> None:
        passed = df[df["passes_filters"] == 1].copy()
        if passed.empty or "max_gain_20d" not in passed.columns:
            return

        _header("3. FEATURE CORRELATION WITH 20-DAY MAX GAIN")

        # Features to evaluate: (column, human label, interpretation note)
        features = [
            ("score", "Total score", ""),
            ("base_quality", "Base quality", ""),
            ("trend_strength", "Trend strength", ""),
            ("relative_strength_score", "Relative strength", ""),
            ("volume_score", "Volume profile", ""),
            ("rr_score", "Risk/reward", ""),
            ("rs_comp_60", "RS vs COMP 60d", "higher = stronger leader"),
            ("vcp_contraction_ratio", "VCP contraction ratio", "lower = tighter base"),
            (
                "pct_from_52wk_high",
                "% from 52wk high",
                "less negative = closer to high",
            ),
            ("prior_move_pct", "Prior move %", ""),
            ("adr_pct", "ADR %", ""),
        ]

        rows = []
        target = passed["max_gain_20d"]
        for col, label, note in features:
            if col not in passed.columns:
                continue
            sub = passed[[col]].join(target).dropna()
            if len(sub) < 5:
                continue
            corr = sub[col].corr(sub["max_gain_20d"])
            if not pd.isna(corr):
                rows.append((label, corr, len(sub), note))

        rows.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  {'Feature':28s}  {'Corr':>6}  {'Visual':12s}  N    Note")
        _divider()
        for label, corr, n, note in rows:
            filled = int(abs(corr) * 10)
            bar = ("▲" if corr > 0 else "▼") + "█" * filled + "░" * (10 - filled)
            print(f"  {label:28s}  {corr:>+.3f}  {bar}  {n:<4} {note}")

    # ── Report 4: Filter effectiveness ───────────────────────────────────────

    def _report_filter_effectiveness(
        self, scans: pd.DataFrame, outcomes: pd.DataFrame
    ) -> None:
        _header("4. FILTER EFFECTIVENESS (failed-filter stocks)")

        failed = scans[scans["passes_filters"] == 0].copy()
        if failed.empty:
            print("  No failed-filter stocks in database.")
            return

        failed_outcomes = failed.merge(
            outcomes, on=["scan_date", "symbol"], how="inner"
        )
        if failed_outcomes.empty:
            print("  No outcomes yet for failed-filter stocks.")
            return

        # Infer the primary filter failure from stored feature values.
        # "Price below 50 SMA" and "stop distance" filters can't be re-derived
        # from stored columns, so they fall into the "other" bucket.
        def primary_failure(row) -> str:
            if (row.get("price") or 999) < 5:
                return "price < $5"
            dv = row.get("dollar_volume") or 0
            if dv < 10_000_000:
                return "dollar volume < $10M"
            adr = row.get("adr_pct") or 1
            if adr < 0.05:
                return "ADR < 5%"
            hi = row.get("pct_from_52wk_high") or 0
            if hi < -0.30:
                return ">30% from 52-week high"
            return "other (SMA / stop-distance)"

        failed_outcomes["inferred_filter"] = failed_outcomes.apply(
            primary_failure, axis=1
        )

        # Baseline: what passed-filter stocks averaged for comparison
        passed_outcomes = scans[scans["passes_filters"] == 1].merge(
            outcomes, on=["scan_date", "symbol"], how="inner"
        )
        baseline_max20 = (
            passed_outcomes["max_gain_20d"].mean() * 100
            if not passed_outcomes.empty
            else None
        )

        print(
            f"  {'Filter failed':32s}  {'N':>4}  {'AvgChg':>7}  "
            f"{'AvgMax20':>9}  Verdict"
        )
        _divider()

        for filter_name, group in failed_outcomes.groupby("inferred_filter"):
            n = len(group)
            avg_chg = group["pct_change"].mean() * 100
            avg_max = group["max_gain_20d"].mean() * 100
            if baseline_max20 is not None and avg_max > baseline_max20 * 0.8:
                verdict = "⚠  reconsider — similar gains to passing stocks"
            else:
                verdict = "filter justified"
            print(
                f"  {filter_name:32s}  {n:>4}"
                f"  {avg_chg:>+6.1f}%"
                f"  {avg_max:>+8.1f}%"
                f"  {verdict}"
            )

        if baseline_max20 is not None:
            print(f"\n  Baseline (passed-filter avg max20d): {baseline_max20:+.1f}%")

    # ── Report 5: Weight suggestions ─────────────────────────────────────────

    def _report_weight_suggestions(self, df: pd.DataFrame) -> None:
        passed = df[df["passes_filters"] == 1].copy()
        if len(passed) < 10:
            return

        _header("5. SCORING WEIGHT SUGGESTIONS")

        # Map: display name → (db column, current weight in config.py)
        weight_map = [
            ("base_quality", "base_quality", 25),
            ("trend_strength", "trend_strength", 30),
            ("relative_strength", "relative_strength_score", 25),
            ("volume_profile", "volume_score", 10),
            ("risk_reward", "rr_score", 10),
        ]

        target = passed["max_gain_20d"].fillna(0)
        corrs = {}
        for name, col, current in weight_map:
            if col not in passed.columns:
                continue
            c = passed[col].corr(target)
            if not pd.isna(c):
                corrs[name] = (abs(c), current)

        if not corrs:
            print("  Insufficient data.")
            return

        # Normalise absolute correlations to sum to 100
        total_corr = sum(v[0] for v in corrs.values())
        if total_corr == 0:
            print("  Zero correlation — not enough variance in outcomes.")
            return

        insufficient = len(passed) < 50

        print(
            f"  {'Component':22s}  {'Current':>8}  "
            f"{'Predictiveness':>14}  {'Suggested':>10}  {'Change':>7}"
        )
        _divider()

        for name, (abs_corr, current) in corrs.items():
            suggested = round((abs_corr / total_corr) * 100)
            diff = suggested - current
            diff_str = f"({diff:+d})" if abs(diff) >= 3 else "—"
            bar = "█" * int(abs_corr * 20)
            print(
                f"  {name:22s}  {current:>7}pt  "
                f"  {abs_corr:>.3f}  {bar:<10}  "
                f"{suggested:>9}pt  {diff_str:>7}"
            )

        if insufficient:
            print(
                f"\n  ⚠  Only {len(passed)} outcomes recorded. "
                "Suggestions are preliminary.\n"
                "     Aim for 50+ before editing config.py weights."
            )
        else:
            print(
                "\n  These weights are derived from empirical outcome data.\n"
                "  Review before applying — update config.py weights manually."
            )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n  {'─' * (W - 2)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 2)}")


def _divider() -> None:
    print(f"  {'·' * (W - 2)}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse historical scan performance")
    parser.add_argument(
        "--min-outcomes",
        type=int,
        default=5,
        help="Minimum joined scan+outcome rows required (default: 5)",
    )
    parser.add_argument(
        "--db",
        default="results/breakout.db",
        help="Path to SQLite database (default: results/breakout.db)",
    )
    args = parser.parse_args()

    db = ScanPersistence(db_path=args.db)
    analyzer = PerformanceAnalyzer(db)
    analyzer.run(min_outcomes=args.min_outcomes)
