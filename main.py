from ingestion import Ingestor, SchwabAPIClient

# init objects
ig = Ingestor()

ig.mergefiles()
ig.get_data()
# api.initial_auth_flow()
