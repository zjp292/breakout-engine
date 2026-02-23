"""
this file handles the ingestion of data from think or swim.
"""

import os
import json
import base64
import requests
import webbrowser
import csv
import pickle
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


class Ingestor:
    """
    this class takes in 1 or multiple csv files and should merge them into a single list.
    """

    def __init__(self):
        self.watchlist_path = Path("watchlists")
        self.ticker_list = set()
        self.api = SchwabAPIClient()
        # self.api.initial_auth_flow()

    def mergefiles(
        self,
    ):
        """
        loops though all the files in self.watchlist_path and
        """
        today = datetime.today().strftime("%Y-%m-%d")

        # loop through files in watchlist directory
        for file in os.listdir(self.watchlist_path):
            # check for today's watchlist exports
            if file.startswith(today):
                with open(self.watchlist_path / file, "r") as f:
                    csv_reader = csv.reader(f)

                    # skips header of hte csv
                    for _ in range(4):
                        next(csv_reader)

                    # add row to ticker list set; allow hyphens for tickers like BRK-B
                    for row in csv_reader:
                        ticker = row[0].strip()
                        if not ticker or not ticker.replace("-", "").isalnum():
                            continue
                        self.ticker_list.add(ticker)
        return

    def get_data(self):
        if len(self.ticker_list) == 0:
            raise ValueError("ticker list not initialized")

        today = datetime.today().strftime("%Y-%m-%d")
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365)
        end_dt = int(end_dt.timestamp() * 1000)
        start_dt = int(start_dt.timestamp() * 1000)

        data_dir = f"data/{today}"
        os.makedirs(data_dir, exist_ok=True)

        for ticker in self.ticker_list:
            self.api.get_historical_prices(
                data_dir=data_dir,
                symbol=ticker,
                periodType="year",
                frequencyType="daily",
                frequency=1,
                startDate=start_dt,
                endDate=end_dt,
                needExtendedHoursData=False,
            )


class SchwabAPIClient:
    """
    this class handles the schwab trader api to get stock data

    source: https://medium.com/@carstensavage/the-unofficial-guide-to-charles-schwabs-trader-apis-14c1f5bc1d57
    """

    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
    TOKEN_FILE = "schwab_tokens.json"

    def __init__(self):
        load_dotenv()
        self.app_key = os.getenv("TOS_APP_KEY")
        self.app_secret = os.getenv("TOS_SECRET")
        self.redirect_uri = "https://127.0.0.1"
        self.base_url_market_data = "https://api.schwabapi.com/marketdata/v1"
        self.base_url_accounts = "https://api.schwabapi.com/trader/v1"
        self.tokens = self.load_tokens()

    def load_tokens(self):
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, "r") as f:
                tokens = json.load(f)
            return tokens
        return {}

    def save_tokens(self):
        with open(self.TOKEN_FILE, "w") as f:
            json.dump(self.tokens, f)

    def is_token_expired(self):
        # Schwab returns 'expires_in' in seconds; store 'expires_at' in tokens
        expires_at = self.tokens.get("expires_at")
        if not expires_at:
            return True
        dt = datetime.fromisoformat(expires_at)
        # Handle legacy tokens stored without timezone info (treat as UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= dt

    def set_expiry(self):
        # Set 'expires_at' as a UTC-aware ISO string
        expires_in = self.tokens.get("expires_in", 0)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in) - 60)
        self.tokens["expires_at"] = expires_at.isoformat()
        self.save_tokens()

    def refresh_tokens(self):
        refresh_token = self.tokens.get("refresh_token")
        if not refresh_token:
            raise ValueError("No refresh token available. Please re-authenticate.")

        credentials = f"{self.app_key}:{self.app_secret}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {b64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(self.TOKEN_URL, headers=headers, data=data, timeout=15)
                resp.raise_for_status()
                self.tokens = resp.json()
                self.set_expiry()
                print("Tokens refreshed automatically.")
                return self.tokens
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"  Token refresh attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    def get_access_token(self):
        if not self.tokens or self.is_token_expired():
            print("Access token expired or missing, refreshing...")
            self.refresh_tokens()
        return self.tokens.get("access_token")

    def initial_auth_flow(self):
        """
        Run this ONCE to get your first refresh token.
        """
        auth_url = (
            f"https://api.schwabapi.com/v1/oauth/authorize"
            f"?client_id={self.app_key}&redirect_uri={self.redirect_uri}"
        )
        print("Opening browser for Schwab authentication...")
        webbrowser.open(auth_url)
        print("After logging in, paste the full redirected URL here:")
        redirected_url = input().strip()
        # Extract code from redirect URL safely
        parsed = urlparse(redirected_url)
        params = parse_qs(parsed.query)
        if "code" not in params:
            raise ValueError(f"No 'code' parameter found in redirect URL: {redirected_url}")
        code = params["code"][0]

        credentials = f"{self.app_key}:{self.app_secret}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Basic {b64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        resp = requests.post(self.TOKEN_URL, headers=headers, data=data)
        resp.raise_for_status()
        self.tokens = resp.json()
        self.set_expiry()
        print("Initial tokens obtained and saved.")

    def get_account_info(self):
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = "https://api.schwabapi.com/trader/v1/accounts"
        resp = requests.get(url, headers=headers)
        return resp.json()

    def get_nasdaq_benchmark(self, start_ts, end_ts):
        """
        Fetch NASDAQ Composite ($COMPX) daily price history from Schwab API.

        Returns a DataFrame indexed by normalized datetime (date-only),
        with columns: open, high, low, close, volume.
        """
        return self.get_index_data("$COMPX", start_ts, end_ts)

    def get_index_data(self, symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """
        Fetch daily OHLCV price history for any symbol (index, ETF, or stock).

        Used to load COMPX, SPY, IWM and other market-condition indices.

        Returns a DataFrame indexed by normalized datetime (date-only),
        with columns: open, high, low, close, volume.

        Raises ValueError if no candle data is returned.
        """
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url_market_data}/pricehistory"

        params = {
            "symbol": symbol,
            "periodType": "year",
            "frequencyType": "daily",
            "frequency": 1,
            "needExtendedHoursData": "false",
            "needPreviousClose": "false",
            "startDate": int(start_ts),
            "endDate": int(end_ts),
        }

        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()

        if "candles" not in data:
            raise ValueError(f"No candle data returned for {symbol}: {data}")

        df = pd.DataFrame(data["candles"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")
        df.index = df.index.normalize()

        return df

    def get_historical_prices(
        self,
        data_dir,
        symbol,
        periodType="month",
        period=1,
        frequencyType="daily",
        frequency=1,
        startDate=None,
        endDate=None,
        needExtendedHoursData=False,
        needPreviousClose=False,
    ):
        end = datetime.today()
        start = end - timedelta(days=365)
        end = end.strftime("%Y-%m-%d")
        start = start.strftime("%Y-%m-%d")

        path = f"{data_dir}/{symbol}-{end}.pkl"

        if os.path.exists(path):
            return

        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url_market_data}/pricehistory"

        params = {
            "symbol": symbol,
            "periodType": periodType,
            "period": period,
            "frequencyType": frequencyType,
            "frequency": frequency,
            "needExtendedHoursData": str(needExtendedHoursData).lower(),
            "needPreviousClose": str(needPreviousClose).lower(),
        }

        if startDate:
            params["startDate"] = int(startDate)
        if endDate:
            params["endDate"] = int(endDate)

        res = requests.get(url, headers=headers, params=params)
        try:
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"  ⚠ {symbol}: API error — {e}")
            return

        if "candles" not in data:
            print(f"  ⚠ {symbol}: no candle data in response — {data}")
            return

        df = pd.DataFrame(data["candles"])
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print(f"{symbol} data saved")
