from engine import Engine
from config import PARAMETERS
from persistence import ScanPersistence

DATE = "2026-02-19"

eg = Engine(PARAMETERS)
scored_dfs, watchlist = eg.process_stock(DATE, debug=True)

if watchlist is not None:
    print(watchlist.head(20))

if scored_dfs:
    db = ScanPersistence()
    n = db.save_scan(DATE, scored_dfs, eg.market_condition)
    db.summary()
