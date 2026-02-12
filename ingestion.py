"""
this file handles the ingestion of data from think or swim.
"""

from pathlib import Path
from datetime import datetime


class Ingestor:
    """
    this class takes in 1 or multiple csv files and should merge them into a single list.
    """

    def __init__(self):
        self.watchlist_path = Path("watchlists")

    def mergefiles(
        self,
    ):
        """
        loops though all the files in self.watchlist_path and
        """

        today = datetime.today().strftime("%Y-%m-%d")
