import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine import Engine
from config import PARAMETERS
from persistence import ScanPersistence


def rerun_all_dates(engine, db, debug=False, dry_run=False):
    data_root = Path("data")
    if not data_root.exists():
        print("data/ directory not found. Nothing to rerun.")
        return

    date_dirs = sorted(
        [
            d
            for d in data_root.iterdir()
            if d.is_dir()
            and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)
            and any(d.glob("*.pkl"))
        ]
    )

    if not date_dirs:
        print("No scan data found in data/. Nothing to rerun.")
        return

    print(
        f"Found {len(date_dirs)} scan date(s): "
        f"{date_dirs[0].name} → {date_dirs[-1].name}"
    )

    print("\nFetching benchmark data (once for all dates)...")
    try:
        engine.load_benchmark()
        full_compx = engine.benchmark_df
        full_spy = engine.spy_df
        full_iwm = engine.iwm_df
        print(
            f"  COMPX spans {full_compx.index[0].date()} → {full_compx.index[-1].date()}\n"
        )
    except Exception as e:
        print(
            f"Warning: benchmark unavailable ({e}) — regime_multiplier=1.0 for all dates\n"
        )
        full_compx = full_spy = full_iwm = None

    # suppress per-date regime report walls; one compact line per date is enough
    engine._print_market_condition = lambda _: None
    engine._print_macro_regime = lambda _: None

    total_saved = 0

    for date_dir in date_dirs:
        date_str = date_dir.name
        date_ts = pd.Timestamp(date_str)

        # slice benchmark to this date so market condition reflects what was knowable then
        if full_compx is not None:
            compx_slice = full_compx[full_compx.index <= date_ts]
            engine.benchmark_df = compx_slice if not compx_slice.empty else None
            engine.spy_df = (
                full_spy[full_spy.index <= date_ts] if full_spy is not None else None
            )
            engine.iwm_df = (
                full_iwm[full_iwm.index <= date_ts] if full_iwm is not None else None
            )
        else:
            engine.benchmark_df = engine.spy_df = engine.iwm_df = None

        pickle_files = list(date_dir.glob("*.pkl"))
        scored_dfs = {}

        for pf in pickle_files:
            symbol = pf.stem.split("-")[0]
            try:
                df = engine.load_pickle(str(pf))
                if "datetime" in df.columns and not isinstance(
                    df.index, pd.DatetimeIndex
                ):
                    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
                    df = df.set_index("datetime")
                    df.index = df.index.normalize()
                feature_df = engine.features.add_all_features(
                    df, benchmark_df=engine.benchmark_df
                )
                scored_dfs[symbol] = feature_df
            except Exception as exc:
                if debug:
                    print(f"    {symbol}: feature error — {exc}")

        if not scored_dfs:
            print(f"  {date_str}: no stocks loaded, skipping")
            continue

        rs_ranks = engine.features.calculate_rs_rank(scored_dfs, engine.benchmark_df)

        try:
            regime_mult = engine.analyze_market_condition(scored_dfs)
        except Exception:
            regime_mult = 1.0

        engine.scoring.regime_multiplier = regime_mult

        final_scored_dfs = {}
        for symbol, feature_df in scored_dfs.items():
            try:
                scored_df = engine.scoring.score_dataframe(
                    feature_df, symbol=symbol, rs_ranks=rs_ranks.get(symbol, {})
                )
                final_scored_dfs[symbol] = scored_df
            except Exception as exc:
                if debug:
                    print(f"    {symbol}: scoring error — {exc}")

        mc = engine.market_condition
        mc_label = getattr(mc, "regime", "N/A") if mc else "N/A"
        mult_label = f"×{regime_mult:.2f}"

        if dry_run:
            print(
                f"  {date_str}: {len(final_scored_dfs)} stocks  "
                f"regime={mc_label} {mult_label}  [dry-run]"
            )
        else:
            n = db.save_scan(date_str, final_scored_dfs, mc)
            print(f"  {date_str}: {n} stocks saved  regime={mc_label} {mult_label}")
            total_saved += n

    print()
    if dry_run:
        print(
            f"[dry-run] Would rerun {len(date_dirs)} date(s) with current scoring code."
        )
    else:
        print(
            f"Rerun complete — {total_saved} stock records updated across {len(date_dirs)} date(s)."
        )
        db.summary()


def main():
    parser = argparse.ArgumentParser(
        description="Daily breakout scanner — Qullamaggie / Minervini methodology"
    )
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Scan date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scan but do not save results to the database",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help=(
            "Rescore every date in data/ using current code. "
            "Fetches benchmark once; overwrites DB records in-place."
        ),
    )
    parser.add_argument(
        "--no-paper",
        action="store_true",
        help="Skip paper trading even if enabled in config",
    )
    args = parser.parse_args()

    eg = Engine(PARAMETERS)
    db = ScanPersistence()

    if args.rerun_all:
        rerun_all_dates(eg, db, debug=args.debug, dry_run=args.dry_run)
        return

    scored_dfs, watchlist = eg.process_stock(args.date, debug=args.debug)

    if watchlist is not None:
        print(watchlist.head(20))

    if scored_dfs and not args.dry_run:
        n = db.save_scan(args.date, scored_dfs, eg.market_condition)
        print(f"\nSaved {n} stock(s) for {args.date}")
        db.summary()

        paper_cfg = PARAMETERS.get("paper_trading", {})
        if paper_cfg.get("enabled") and not args.no_paper:
            try:
                from paper_trader import AlpacaClient, PaperTradeManager

                manager = PaperTradeManager(AlpacaClient(), db, PARAMETERS)
                manager.run(args.date, scored_dfs, eg.market_condition, eg.macro_regime)
            except EnvironmentError as exc:
                print(f"\n⚠ paper trading: {exc}")
            except Exception as exc:
                print(f"\n⚠ paper trading error (scanner unaffected): {exc}")

    elif args.dry_run:
        print("\n[dry-run] Results not saved to database.")


if __name__ == "__main__":
    main()
