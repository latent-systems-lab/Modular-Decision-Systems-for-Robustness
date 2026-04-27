import os
import random
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

# Ensure deterministic sample generation
np.random.seed(42)
random.seed(42)

# --- Configuration ---
DATA_DIR = "data"
TICKERS = ["AAPL", "NVDA", "TSLA"]
START_DATE = "2023-01-01"
END_DATE = "2023-03-30"  # Generating 3 months for speed; extend if needed
TRADING_DAYS = 252

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

# --- 1. Generate Stock Data (CSV) ---
def generate_stock_data():
    print("Generating Stock CSVs...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    
    for ticker in TICKERS:
        # Random walk parameters
        start_price = 150 if ticker == "AAPL" else (200 if ticker == "TSLA" else 400)
        volatility = 0.02
        
        prices = [start_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
            
        df = pd.DataFrame({
            "Date": dates,
            "Open": [p * 0.99 for p in prices],
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, len(dates))
        })
        
        # Save as required by DataManager: data/{TICKER}.csv
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}")
        
    return dates

# --- 2. Generate Options Data (SQLite DB) ---
def black_scholes(S, K, T, r, sigma, is_call):
    if T <= 0: return max(0, S - K) if is_call else max(0, K - S), 0, 0, 0, 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) / 365
    rho = 0 # Simplified
    
    return price, delta, gamma, vega, theta, rho

def generate_options_db(dates):
    print("Generating Options SQLite DBs (this may take a moment)...")
    
    for ticker in TICKERS:
        # Load the stock price we just generated to ensure consistency
        stock_df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        all_options = []
        
        # Generate options for every 5th day to save time/space (simulation will interpolate or skip)
        for _, row in stock_df.iloc[::5].iterrows():
            current_date = row['Date']
            spot = row['Close']
            
            # Create expirations: 30 days and 60 days out
            expirations = [current_date + timedelta(days=30), current_date + timedelta(days=60)]
            
            for expire_date in expirations:
                dte = (expire_date - current_date).days
                T = dte / 365.0
                
                # Strikes: +/- 10% of spot
                strikes = np.linspace(spot * 0.9, spot * 1.1, 5)
                
                for strike in strikes:
                    for is_call in [0, 1]: # 0=Put, 1=Call
                        iv = 0.3
                        price, delta, gamma, vega, theta, rho = black_scholes(spot, strike, T, 0.04, iv, is_call)
                        
                        if price < 0.01: continue

                        # Column format matching the expected options schema and the DataManager SQL query
                        opt_data = {
                            "QUOTE_DATE": current_date.strftime("%Y-%m-%d"),
                            "option_id": f"{is_call}_{strike:.2f}_{expire_date.strftime('%Y-%m-%d')}",
                            "is_call": is_call,
                            "STRIKE": strike,
                            "EXPIRE_DATE": expire_date.strftime("%Y-%m-%d"),
                            "UNDERLYING_LAST": spot,
                            "DTE": dte,
                            "STRIKE_DISTANCE": abs(spot - strike),
                            "STRIKE_DISTANCE_PCT": abs(spot - strike)/spot,
                            "DELTA": delta,
                            "GAMMA": gamma,
                            "VEGA": vega,
                            "THETA": theta,
                            "RHO": rho,
                            "IV": iv,
                            "VOLUME": np.random.randint(10, 1000),
                            "LAST": price,
                            "SIZE": "10x10",
                            "BID": price * 0.98,
                            "ASK": price * 1.02,
                            "time_to_maturity": T,
                            "moneyness": spot/strike
                        }
                        all_options.append(opt_data)

        # Save to SQLite
        db_path = os.path.join(DATA_DIR, f"{ticker.lower()}_options.db")
        conn = sqlite3.connect(db_path)
        df_opt = pd.DataFrame(all_options)
        
        # Important: DataManager expects specific column types, especially dates
        df_opt.to_sql("options", conn, if_exists="replace", index=False)
        
        # Create index for speed (DataManager queries by QUOTE_DATE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON options (QUOTE_DATE)")
        conn.close()
        print(f"  Saved DB {db_path} ({len(df_opt)} rows)")

# --- 3. Generate Auxiliary Data (Earnings, CPI) ---
def generate_aux_data(dates):
    print("Generating Auxiliary CSVs...")
    
    # Earnings.csv
    # Structure based on core/market_analyzer.py: Date, aapl, tsla, nvda (days to next), growth columns
    earnings_data = []
    for d in dates:
        row = {"Date": d}
        for t in ["aapl", "tsla", "nvda"]:
            row[t] = np.random.randint(1, 90) # Days to next earnings
            row[f"{t}_eps_growth"] = np.random.uniform(0.05, 0.20)
            row[f"{t}_revenue_growth"] = np.random.uniform(0.05, 0.20)
        earnings_data.append(row)
    pd.DataFrame(earnings_data).to_csv(os.path.join(DATA_DIR, "earnings.csv"), index=False)
    
    # CPI.csv
    # Structure: date, inflation, forecast, actual_state, last_release_date, next_release_date, days_until...
    cpi_data = []
    for d in dates:
        cpi_data.append({
            "date": d,
            "inflation": 3.5 + np.random.normal(0, 0.1),
            "forecast": 3.5,
            "actual_state": "neutral",
            "last_release_date": d - timedelta(days=15),
            "next_release_date": d + timedelta(days=15),
            "days_until_next_release": 15
        })
    pd.DataFrame(cpi_data).to_csv(os.path.join(DATA_DIR, "cpi.csv"), index=False)
    
    # VIX.csv (Used in market_analyzer)
    vix_data = []
    for d in dates:
        vix_data.append({
            "Date": d,
            "Open": 20, "High": 22, "Low": 18, "Close": 19 + np.random.normal(0, 1)
        })
    pd.DataFrame(vix_data).to_csv(os.path.join(DATA_DIR, "vix.csv"), index=False)
    
    # Fear Greed
    fg_data = []
    for d in dates:
        fg_data.append({
            "Date": d.strftime("%m/%d/%Y"), # Code expects specific format
            "Fear Greed": np.random.randint(20, 80)
        })
    pd.DataFrame(fg_data).to_csv(os.path.join(DATA_DIR, "fear-greed-2011-2023.csv"), index=False)
    
    print("  Saved earnings.csv, cpi.csv, vix.csv, fear-greed.csv")

if __name__ == "__main__":
    ensure_data_dir()
    dates = generate_stock_data()
    generate_aux_data(dates)
    generate_options_db(dates)
    print("\nSUCCESS: All datasets generated. You can now run 'main.py' (or the coordinator).")