from ingestion import Ingestor, SchwabAPIClient

# init objects
ig = Ingestor()
api = SchwabAPIClient()

ig.mergefiles()
# api.initial_auth_flow()
api.get_access_token()
