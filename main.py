import yfinance as yf
import pandas as pd
import random
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym


# Universe of tickers
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "ORCL", "IBM", "CSCO", "QCOM", "AVGO",
    "ADBE", "CRM", "PYPL", "SHOP",
]

def fetch_and_preprocess_data(number, start_date, end_date):
    """
    Fetch stock data for a random selection of tickers and compute 5-day returns.
    
    Args:
        number (int): Number of stocks to randomly select.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
    
    Returns:
        dict: {ticker: [[date, close, 5d_return], ...]}
    """
    chosen = random.sample(tickers, number)
    results = {}

    for ticker in chosen:
        try:
            # Download price data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            # Skip if no data
            if data.empty:
                print(f"Skipping {ticker}: no data")
                continue

            # Compute 5-day rolling returns
            data["5d_return"] = data["Close"].pct_change(5)

            # Convert DataFrame to list of [date, close, 5d_return]
            matrix = []
            for idx, row in data.iterrows():
                # Ensure scalar values (handle possible Series)
                close = row["Close"].iloc[0] if isinstance(row["Close"], pd.Series) else row["Close"]
                ret5d = row["5d_return"].iloc[0] if isinstance(row["5d_return"], pd.Series) else row["5d_return"]
                
                # Convert to float safely, handle NaN
                close = float(close)
                ret5d = float(ret5d) if pd.notna(ret5d) else None

                matrix.append([str(idx.date()), close, ret5d])

            results[ticker] = matrix

        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    return results



def RL_agent(returns) : 
    class PolicyNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(PolicyNetwork, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.fc(x)
        
        def train(self) :
            pass
    


if __name__ == "__main__":
    data = fetch_and_preprocess_data(number=5, start_date="2023-01-01", end_date="2023-12-31")
    for ticker, matrix in data.items():
        print(f"\n{ticker}: ")
        print(matrix[5:10])


