from ingestion import Ingestor, SchwabAPIClient
import pickle
from datetime import datetime
from engine import Engine
from config import PARAMETERS


print("test: ", datetime.today().strftime("%Y-%m-%d"))

# init objects
ig = Ingestor()
# ig.mergefiles()
# ig.get_data()

eg = Engine(PARAMETERS)

# print(eg.load_pickle("data/2026-02-14/AAOI-2026-02-14.pkl"))
eg.process_stock()
