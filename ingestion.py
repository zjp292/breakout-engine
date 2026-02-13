"""
this file handles the ingestion of data from think or swim.
"""

from pathlib import Path
import os
import csv
from datetime import datetime


class Ingestor:
    """
    this class takes in 1 or multiple csv files and should merge them into a single list.
    """

    def __init__(self):
        self.watchlist_path = Path("watchlists")
        self.ticker_list = set()

    def mergefiles(
        self,
    ):
        """
        loops though all the files in self.watchlist_path and
        """

        today = datetime.today().strftime("%Y-%m-%d")
        print(today)
        headers = [
            "",
            "symbol",
            "Watchlist '1 month gainer - KK'",
            "1 month gainer - KK",
            "Watchlist '3 month gainers - KK'",
            "3 month gainers - KK",
            "Watchlist '6 month gainers - KK'",
            "6 month gainers - KK",
        ]

        # loop through files in watchlist directory
        for file in os.listdir(self.watchlist_path):
            # print("file: ", file)

            # check for today's watchlist exports
            if file.startswith(today):
                with open(self.watchlist_path / file, "r") as f:
                    csv_reader = csv.reader(f)

                    # skips header of hte csv
                    for _ in range(4):
                        next(csv_reader)

                    # add row to ticker list set
                    for row in csv_reader:
                        self.ticker_list.add(row[0])

        print(self.ticker_list)
