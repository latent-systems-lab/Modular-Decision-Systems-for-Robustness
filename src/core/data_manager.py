"""
Data Manager for the hedge fund simulation system.
Responsible for loading and managing stock and options data.
"""

import os
import pandas as pd
import sqlite3
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import models
from src.models.pydantic_models import StockData, OptionData


class DataManager:
    """
    Manages data loading and processing for the hedge fund simulation.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the DataManager.

        Args:
            data_dir: Directory containing data files.
        """
        self.data_dir = data_dir
        self.stock_data_cache = {}  # Cache for stock data
        self.options_data_cache = {}  # Cache for options data
        self.options_db_path = os.path.join(data_dir, "options.db")

    def load_stock_data(
        self, ticker: str, start_date: date = None, end_date: date = None
    ) -> List[StockData]:
        """
        Load stock data for a specific ticker.

        Args:
            ticker: Stock ticker symbol.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            List of StockData objects.
        """
        # Check if data is already in cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.stock_data_cache:
            return self.stock_data_cache[cache_key]

        # Construct file path
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            df["Date"] = pd.to_datetime(df["Date"])

            # Filter by date range if provided
            if start_date:
                df = df[df["Date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["Date"] <= pd.Timestamp(end_date)]

            # Sort by date
            df = df.sort_values("Date")

            # Convert to list of StockData objects
            stock_data_list = []
            for _, row in df.iterrows():
                stock_data = StockData(
                    date=row["Date"].date(),
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                )
                stock_data_list.append(stock_data)

            # Cache the result
            self.stock_data_cache[cache_key] = stock_data_list

            return stock_data_list

        except Exception as e:
            print(f"Error loading stock data for {ticker}: {str(e)}")
            return []

    def load_stock_data_as_dataframe(
        self, ticker: str, start_date: date = None, end_date: date = None
    ) -> pd.DataFrame:
        """
        Load stock data for a specific ticker as a pandas DataFrame.

        Args:
            ticker: Stock ticker symbol.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            Pandas DataFrame containing stock data.
        """
        # Construct file path
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            df["Date"] = pd.to_datetime(df["Date"])

            # Filter by date range if provided
            if start_date:
                df = df[df["Date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["Date"] <= pd.Timestamp(end_date)]

            # Sort by date
            df = df.sort_values("Date")

            return df

        except Exception as e:
            print(f"Error loading stock data for {ticker}: {str(e)}")
            return pd.DataFrame()

    # QUOTE_DATE,option_id,is_call,STRIKE,EXPIRE_DATE,UNDERLYING_LAST,DTE,STRIKE_DISTANCE,STRIKE_DISTANCE_PCT,DELTA,GAMMA,VEGA,THETA,RHO,IV,VOLUME,LAST,SIZE,BID,ASK,time_to_maturity,moneyness

    def load_options_data(
        self,
        ticker: str,
        quote_date: date,
        min_dte: int = 0,
        max_dte: int = 60,
        min_delta: float = 0.1,
        max_delta: float = 0.9,
    ) -> List[OptionData]:
        """
        Load options data for a specific ticker and date.

        Args:
            ticker: Stock ticker symbol.
            quote_date: Date for which to load options data.
            min_dte: Minimum days to expiration.
            max_dte: Maximum days to expiration.
            min_delta: Minimum delta value.
            max_delta: Maximum delta value.

        Returns:
            List of OptionData objects.
        """
        # Check if data is already in cache
        cache_key = f"{ticker}_{quote_date}_{min_dte}_{max_dte}_{min_delta}_{max_delta}"
        if cache_key in self.options_data_cache:
            return self.options_data_cache[cache_key]
        # choose difrfernt file asbed on ticker in style tocker_optionts.db
        options_db_path = os.path.join(self.data_dir, f"{ticker.lower()}_options.db")
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(options_db_path)

            quote_date_str = quote_date.strftime("%Y-%m-%d")
            quote_date_dt = pd.to_datetime(quote_date_str)
            # Use parameterized queries instead of f-strings
            query = f"""
            SELECT QUOTE_DATE, option_id, is_call, STRIKE, EXPIRE_DATE, DTE, LAST, BID, ASK, VOLUME, IV, DELTA, GAMMA, VEGA, THETA, RHO, moneyness, UNDERLYING_LAST
            FROM options
            WHERE QUOTE_DATE = '{quote_date_dt}'
            AND DTE BETWEEN {min_dte} AND {max_dte}
            AND LAST > 0.1
            ORDER BY QUOTE_DATE, is_call, STRIKE, DTE
            """

            df = pd.read_sql_query(query, conn)
            # Close connection
            conn.close()

            # Convert to list of OptionData objects
            options_data_list = []
            for _, row in df.iterrows():
                # Check for NaN values in the VOLUME field
                if pd.isna(row["VOLUME"]) or (
                    isinstance(row["VOLUME"], np.float64) and np.isnan(row["VOLUME"])
                ):
                    row["VOLUME"] = 0
                option_data = OptionData(
                    quote_date=datetime.strptime(
                        row["QUOTE_DATE"].split()[0], "%Y-%m-%d"
                    ).date(),
                    option_id=str(row["option_id"]),
                    option_type=int(row["is_call"]),
                    strike=float(row["STRIKE"]),
                    expire_date=datetime.strptime(
                        row["EXPIRE_DATE"].split()[0], "%Y-%m-%d"
                    ).date(),
                    dte=int(row["DTE"]),
                    last=float(row["LAST"]),
                    bid=float(row["BID"]),
                    ask=float(row["ASK"]),
                    delta=float(row["DELTA"]),
                    gamma=float(row["GAMMA"]),
                    vega=float(row["VEGA"]),
                    theta=float(row["THETA"]),
                    volume=float(row["VOLUME"]),
                    rho=float(row["RHO"]),
                    moneyness=float(row["moneyness"]),
                    underlying_last=float(row["UNDERLYING_LAST"]),
                    iv=float(row["IV"]),
                )
                options_data_list.append(option_data)

            # Cache the result
            self.options_data_cache[cache_key] = options_data_list

            return options_data_list

        except Exception as e:
            print(f"Error loading options data for {ticker} on {quote_date}: {str(e)}")
            return []

    def get_stock_price(self, ticker: str, target_date: date) -> Optional[float]:
        """
        Get the closing price for a stock on a specific date.

        Args:
            ticker: Stock ticker symbol.
            target_date: Date for which to get the price.

        Returns:
            Closing price or None if not available.
        """
        # Load stock data
        stock_data = self.load_stock_data(ticker)

        # Find the data point for the target date
        for data_point in stock_data:
            if data_point.date == target_date:
                return data_point.close

        # If exact date not found, find the closest previous date
        closest_data_point = None
        closest_date_diff = timedelta.max

        for data_point in stock_data:
            if data_point.date <= target_date:
                date_diff = target_date - data_point.date
                if date_diff < closest_date_diff:
                    closest_date_diff = date_diff
                    closest_data_point = data_point

        return closest_data_point.close if closest_data_point else None

    def get_technical_indicators(
        self, ticker: str, target_date: date
    ) -> Dict[str, float]:
        """
        Calculate technical indicators for a stock on a specific date.

        Args:
            ticker: Stock ticker symbol.
            target_date: Date for which to calculate indicators.

        Returns:
            Dictionary of technical indicators.
        """
        # Import financial utilities
        from src.utils.financial_utils import (
            calculate_rsi,
            calculate_moving_average,
            calculate_bollinger_bands,
            calculate_macd,
            calculate_atr,
        )

        # Load stock data up to target date
        stock_data = self.load_stock_data(ticker, end_date=target_date)

        # Extract price lists
        closes = [data.close for data in stock_data]
        highs = [data.high for data in stock_data]
        lows = [data.low for data in stock_data]

        # Calculate indicators
        indicators = {}

        # RSI
        indicators["rsi_14"] = calculate_rsi(closes, period=14)

        # Moving Averages
        indicators["sma_20"] = calculate_moving_average(closes, period=20)
        indicators["sma_50"] = calculate_moving_average(closes, period=50)
        indicators["sma_200"] = calculate_moving_average(closes, period=200)

        # Bollinger Bands
        lower_band, middle_band, upper_band = calculate_bollinger_bands(closes)
        indicators["bollinger_lower"] = lower_band
        indicators["bollinger_middle"] = middle_band
        indicators["bollinger_upper"] = upper_band

        # MACD
        macd_line, signal_line, histogram = calculate_macd(closes)
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = histogram

        # ATR
        indicators["atr_14"] = calculate_atr(highs, lows, closes)

        # Price vs Moving Averages
        if len(closes) > 0:
            current_price = closes[-1]
            indicators["price_vs_sma_20"] = (
                current_price / indicators["sma_20"] - 1
                if indicators["sma_20"] > 0
                else 0
            )
            indicators["price_vs_sma_50"] = (
                current_price / indicators["sma_50"] - 1
                if indicators["sma_50"] > 0
                else 0
            )
            indicators["price_vs_sma_200"] = (
                current_price / indicators["sma_200"] - 1
                if indicators["sma_200"] > 0
                else 0
            )

        # reutrn in the last 7, 30, 90, 180 days
        if len(closes) > 7:
            indicators["price_change_7"] = (closes[-1] - closes[-8]) / closes[-8]
        if len(closes) > 30:
            indicators["price_change_30"] = (closes[-1] - closes[-31]) / closes[-31]
        if len(closes) > 90:
            indicators["price_change_90"] = (closes[-1] - closes[-91]) / closes[-91]
        if len(closes) > 180:
            indicators["price_change_180"] = (closes[-1] - closes[-181]) / closes[-181]
        return indicators

    def get_available_tickers(self) -> List[str]:
        """
        Get a list of available stock tickers based on CSV files in the data directory.

        Returns:
            List of ticker symbols.
        """
        tickers = []

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".csv"):
                ticker = file_name.split(".")[0]
                tickers.append(ticker)

        return tickers

    def get_date_range(self, ticker: str) -> Tuple[date, date]:
        """
        Get the available date range for a specific ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Tuple of (start_date, end_date).
        """
        # Load stock data
        stock_data = self.load_stock_data(ticker)

        if not stock_data:
            return (date.today(), date.today())

        start_date = stock_data[0].date
        end_date = stock_data[-1].date

        return (start_date, end_date)

    def load_vix_data(self, file_path: str = "data/vix.csv") -> pd.DataFrame:
        """
        Load VIX data from a CSV file.
        Expected columns: Date, Open, High, Low, Close, Volume.
        Date is assumed to be in YYYY-MM-DD format.
        """
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def load_fear_greed_data(
        self, file_path: str = "data/fear-greed-2011-2023.csv"
    ) -> pd.DataFrame:
        """
        Load Fear-Greed Index data from a CSV file.
        Expected columns: Date, Fear Greed.
        Date is parsed with the appropriate format (e.g. M/D/YYYY).
        """
        df = pd.read_csv(file_path)
        # Here we assume dates are like "1/3/2011"; adjust the format if needed.
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        return df

    def get_common_date_range(self, tickers: List[str]) -> Tuple[date, date]:
        """
        Get the common date range available for all specified tickers.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            Tuple of (start_date, end_date).
        """
        if not tickers:
            return (date.today(), date.today())

        # Initialize with the date range of the first ticker
        start_date, end_date = self.get_date_range(tickers[0])

        # Find the common date range across all tickers
        for ticker in tickers[1:]:
            ticker_start, ticker_end = self.get_date_range(ticker)

            # Update common start and end dates
            start_date = max(start_date, ticker_start)
            end_date = min(end_date, ticker_end)

        return (start_date, end_date)

    def check_stock_data_exists(self, ticker: str, target_date: date) -> bool:
        """
        Check if stock data exists for given ticker and date
        """
        try:
            df = pd.read_csv(
                os.path.join(self.data_dir, f"{ticker}.csv"),
                parse_dates=["Date"],
                usecols=["Date"],
            )
            return pd.to_datetime(target_date) in df["Date"].values
        except FileNotFoundError:
            return False

    def get_option_data_by_id(
        self, option_id: str, ticker: str, quote_date: date
    ) -> Optional[Dict]:
        """
        Get option data for a specific option ID on a specific quote date.

        Args:
            option_id: The option ID to retrieve.
            ticker: The ticker symbol of the underlying asset.
            quote_date: The quote date for which to get option data.

        Returns:
            Dictionary containing option data, or None if not found.
        """
        import os, sqlite3

        options_db_path = os.path.join(self.data_dir, f"{ticker.lower()}_options.db")

        if not os.path.exists(options_db_path):
            print(f"Options database for {ticker} does not exist")
            return None

        # Format the date to match the stored format (with time as "00:00:00")
        quote_date_str = quote_date.strftime("%Y-%m-%d 00:00:00")

        try:
            # Using a context manager to ensure the connection is closed
            with sqlite3.connect(options_db_path) as conn:
                # Use a row factory to access columns by name
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Query filtering by both option_id and QUOTE_DATE
                query = """
                SELECT *
                FROM options
                WHERE option_id = ? AND QUOTE_DATE = ?
                """
                cursor.execute(query, (option_id, quote_date_str))
                row = cursor.fetchone()

                if row is None:
                    return None
                row = dict(row)
                # Return the result as a dictionary
                if row["LAST"] == 0.0:
                    row["LAST"] = (row["BID"] + row["ASK"]) / 2.0
                return row

        except Exception as e:
            print(
                f"Error querying option data for ID {option_id} on {quote_date}: {str(e)}"
            )
            return None

    def load_cpi_timeline(self, current_date: date = None) -> pd.DataFrame:
        """
        Load the daily CPI timeline CSV with columns:
        date, inflation, forecast, actual_state, last_release_date,
        next_release_date, days_until_next_release

        Args:
            current_date: Date for which to load CPI data. If None, returns all data.

        Returns:
            DataFrame with CPI data for the specified date or all data.
        """
        fn = "data/cpi.csv"
        df = pd.read_csv(
            fn, parse_dates=["date", "last_release_date", "next_release_date"]
        )

        # Filter by current_date if provided
        if current_date is not None:
            df = df[df["date"].dt.date == current_date]

        return df

    def load_earnings_data(
        self, current_date: date = None
    ) -> Dict[str, Dict[date, Dict[str, Optional[float]]]]:
        """
        Load earnings countdown and growth data from CSV and organize by stock symbol.

        Args:
            file_path: Path to the earnings CSV file (with columns Date, aapl, tsla, nvda, and growth columns).
            current_date: Date for which to load earnings data. If None, returns all data.

        Returns:
            A dict mapping each symbol ('aapl', 'tsla', 'nvda') to another dict that maps
            dates to a dict containing days_to_next, eps_growth, and rev_growth.
        """
        df = pd.read_csv("data/earnings.csv", parse_dates=["Date"])
        df["Date"] = df["Date"].dt.date

        # Filter by current_date if provided
        if current_date is not None:
            df = df[df["Date"] == current_date]

        symbols = ["aapl", "tsla", "nvda"]
        result: Dict[str, Dict[date, Dict[str, Optional[float]]]] = {}

        for sym in symbols:
            symbol_data: Dict[date, Dict[str, Optional[float]]] = {}
            for _, row in df.iterrows():
                d = row["Date"]
                days_to_next = int(row[sym]) if pd.notna(row[sym]) else None
                eps_growth = (
                    float(row[f"{sym}_eps_growth"])
                    if pd.notna(row[f"{sym}_eps_growth"])
                    else None
                )
                rev_growth = (
                    float(row[f"{sym}_revenue_growth"])
                    if pd.notna(row[f"{sym}_revenue_growth"])
                    else None
                )
                symbol_data[d] = {
                    "days_to_next": days_to_next,
                    "eps_growth": eps_growth,
                    "rev_growth": rev_growth,
                }
            result[sym.upper()] = symbol_data

        return result
