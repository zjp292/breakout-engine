from ingestion import Ingestor, SchwabAPIClient
import pickle
from datetime import datetime


print("test: ", datetime.today().strftime("%Y-%m-%d"))

# init objects
ig = Ingestor()
ig.mergefiles()
ig.get_data()
