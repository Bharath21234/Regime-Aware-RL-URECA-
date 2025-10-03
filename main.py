import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

##Fetch S&P 500 tickers
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet Class A
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "NFLX",  # Netflix
    "AMD",   # Advanced Micro Devices
    "INTC",  # Intel
    "ORCL",  # Oracle
    "IBM",   # IBM
    "CSCO",  # Cisco Systems
    "QCOM",  # Qualcomm
    "AVGO",  # Broadcom
    "ADBE",  # Adobe
    "CRM",   # Salesforce
    "PYPL",  # PayPal
    "SHOP",  # Shopify
]

def fetch_and_preprocess_data(number, start_date, end_date):
    # Randomly pick number of stocks from universe
    chosen = random.sample(tickers, number)

    results = {}

    for ticker in chosen:
        try:
            # Download price data
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            data = data.dropna()

            #Compute daily returns and 5-day rolling return
            data["5d_return"] = data["Close"].pct_change(5)

            # Build matrix: [[date, price, 5d_return], ...]
            matrix = [
                [str(idx.date()), float(row["Close"]), float(row["5d_return"]) if pd.notna(row["5d_return"]) else None]
                for idx, row in data.iterrows()
            ]

            results[ticker] = matrix

        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    return results

    



def RL_agent(returns) : 
    pass




if __name__ == "__main__":
    data = fetch_and_preprocess_data(10, "2020-01-01", "2023-01-01")
    print(data)