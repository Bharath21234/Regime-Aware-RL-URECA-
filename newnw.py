import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------
# Compute annualized Sharpe ratio for S&P 500 from yfinance
# Using ^GSPC as the S&P 500 index
# Risk-free proxy: 13-week Treasury bill (^IRX), quoted in percent
# ------------------------------------------------------------

# Parameters
ticker = "^GSPC"
rf_ticker = "^IRX"
start_date = "2022-01-01"
end_date = "2025-01-01"   # end-exclusive style for full 2022-2024 coverage
trading_days = 252

# ------------------------------------------------------------
# Download data
# ------------------------------------------------------------
price_df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

rf_df = yf.download(
    rf_ticker,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

# Keep only adjusted close-equivalent series
spx = price_df["Close"].copy()
rf = rf_df["Close"].copy()

# If yfinance returns a DataFrame with a ticker level, squeeze it
if isinstance(spx, pd.DataFrame):
    spx = spx.squeeze()
if isinstance(rf, pd.DataFrame):
    rf = rf.squeeze()

# ------------------------------------------------------------
# Compute daily returns
# ------------------------------------------------------------
spx_returns = spx.pct_change().dropna()

# ^IRX is annualized yield in percent, e.g. 5.2 means 5.2%
# Convert to decimal annual yield
rf_annual = (rf / 100.0).copy()

# Convert annual yield to approximate daily risk-free rate
rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1

# Align risk-free series to stock return dates
rf_daily = rf_daily.reindex(spx_returns.index).ffill().bfill()

# ------------------------------------------------------------
# Build DataFrame
# ------------------------------------------------------------
df = pd.DataFrame({
    "spx_return": spx_returns,
    "rf_daily": rf_daily
}).dropna()

df["excess_return"] = df["spx_return"] - df["rf_daily"]
df["year"] = df.index.year

# ------------------------------------------------------------
# Compute annualized Sharpe by calendar year
# ------------------------------------------------------------
results = []

for year, grp in df.groupby("year"):
    mean_excess_daily = grp["excess_return"].mean()
    std_daily = grp["spx_return"].std(ddof=1)

    # Guard against divide-by-zero
    if std_daily == 0 or pd.isna(std_daily):
        sharpe_annualized = np.nan
    else:
        sharpe_annualized = (mean_excess_daily / std_daily) * np.sqrt(trading_days)

    annual_return = (1 + grp["spx_return"]).prod() - 1
    annual_rf = (1 + grp["rf_daily"]).prod() - 1
    annual_vol = grp["spx_return"].std(ddof=1) * np.sqrt(trading_days)

    results.append({
        "Year": year,
        "Annual Return": annual_return,
        "Annual Risk-Free Return": annual_rf,
        "Annual Volatility": annual_vol,
        "Annualized Sharpe Ratio": sharpe_annualized,
        "Trading Days": len(grp)
    })

results_df = pd.DataFrame(results)

# Format for display
display_df = results_df.copy()
for col in ["Annual Return", "Annual Risk-Free Return", "Annual Volatility", "Annualized Sharpe Ratio"]:
    display_df[col] = display_df[col].astype(float)

pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print(display_df)

# Optional: save to CSV
display_df.to_csv("sp500_annualized_sharpe_2022_2024.csv", index=False)

# ------------------------------------------------------------
# Also compute full-period Sharpe for 2022-2024 combined
# ------------------------------------------------------------
mean_excess_daily_full = df["excess_return"].mean()
std_daily_full = df["spx_return"].std(ddof=1)

if std_daily_full == 0 or pd.isna(std_daily_full):
    sharpe_full = np.nan
else:
    sharpe_full = (mean_excess_daily_full / std_daily_full) * np.sqrt(trading_days)

full_return = (1 + df["spx_return"]).prod() - 1
full_rf = (1 + df["rf_daily"]).prod() - 1
full_vol = df["spx_return"].std(ddof=1) * np.sqrt(trading_days)

print("\nFull period (2022-2024 combined):")
print(f"Return: {full_return:.6f}")
print(f"Risk-free return: {full_rf:.6f}")
print(f"Volatility: {full_vol:.6f}")
print(f"Annualized Sharpe Ratio: {sharpe_full:.6f}")