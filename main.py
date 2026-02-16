from ingestion import Ingestor, SchwabAPIClient
import pickle
from datetime import datetime
from engine import Engine
from config import PARAMETERS


eg = Engine(PARAMETERS)
scored_dfs, watchlist = eg.process_stock("2026-02-14", debug=True)

if watchlist is not None:
    print(watchlist.head(20))
