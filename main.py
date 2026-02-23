import argparse
from datetime import datetime

from engine import Engine
from config import PARAMETERS
from persistence import ScanPersistence


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
    args = parser.parse_args()

    eg = Engine(PARAMETERS)
    scored_dfs, watchlist = eg.process_stock(args.date, debug=args.debug)

    if watchlist is not None:
        print(watchlist.head(20))

    if scored_dfs and not args.dry_run:
        db = ScanPersistence()
        n = db.save_scan(args.date, scored_dfs, eg.market_condition)
        print(f"\nSaved {n} stock(s) for {args.date}")
        db.summary()
    elif args.dry_run:
        print("\n[dry-run] Results not saved to database.")


if __name__ == "__main__":
    main()
