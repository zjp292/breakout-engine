from ingestion import Ingestor, SchwabAPIClient
import pickle

# init objects
ig = Ingestor()
ig.mergefiles()
ig.get_data()

with open('data/UAMY-2026-02-14.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)