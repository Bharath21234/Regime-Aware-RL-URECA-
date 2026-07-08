"""Regenerates a clean, paper-ready version of 3_Agent_Select_1/results/regimes.png:
HMM-detected market regimes overlaid on cumulative market returns. Replicates
hmm.py's MarketRegimeHMM exactly (fit on EXO_TICKERS train period only, predict
on the full train+test span), but fixes the unreadable date axis.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn import hmm

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TRAIN_START, TRAIN_END = "2015-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2024-01-01"

print("Downloading macro ETF panel...")
raw = yf.download(EXO_TICKERS, start=TRAIN_START, end=TEST_END, auto_adjust=True, progress=False)
close = raw["Close"].dropna(how="all")
df_exo = close.reset_index().melt(id_vars="Date", var_name="tic", value_name="close")
df_exo = df_exo.rename(columns={"Date": "date"}).dropna(subset=["close"])
df_exo["date"] = pd.to_datetime(df_exo["date"])


def prepare_data(df):
    prices_df = df.pivot(index="date", columns="tic", values="close")
    returns_df = prices_df.pct_change().fillna(0)
    window = 20
    mean_return = returns_df.mean(axis=1)
    rolling_vol = returns_df.rolling(window).std().mean(axis=1)
    rolling_momentum = returns_df.rolling(window).sum().mean(axis=1)
    rolling_corr = returns_df.rolling(window).corr()
    avg_corr = []
    for date in returns_df.index:
        try:
            corr_matrix = rolling_corr.loc[date].values
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            pairwise = corr_matrix[mask]
            pairwise = pairwise[np.isfinite(pairwise)]
            avg_corr.append(pairwise.mean() if len(pairwise) > 0 else 0.0)
        except Exception:
            avg_corr.append(0.0)
    avg_corr = pd.Series(avg_corr, index=returns_df.index)
    features = pd.DataFrame({
        "mean_return": mean_return, "rolling_vol": rolling_vol,
        "rolling_momentum": rolling_momentum, "avg_correlation": avg_corr,
    }, index=returns_df.index).dropna()
    means, stds = features.mean(), features.std() + 1e-8
    return ((features - means) / stds).values, features.index


X_train, _ = prepare_data(df_exo[(df_exo.date >= TRAIN_START) & (df_exo.date <= TRAIN_END)])
model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X_train)

state_means = model.means_
regime_scores = state_means[:, 2] - state_means[:, 1]  # momentum - volatility
sorted_states = np.argsort(regime_scores)
print(f"Regime order (Bear -> Bull): {sorted_states}, scores: {regime_scores[sorted_states]}")

X_full, dates_full = prepare_data(df_exo)
raw_regimes = model.predict(X_full)
mapped = np.zeros_like(raw_regimes)
for i, s in enumerate(sorted_states):
    mapped[raw_regimes == s] = i
regime_df = pd.DataFrame({"date": dates_full, "regime": mapped})

# Market cumulative return proxy (SPY, matches the all-equity universe direction)
spy = df_exo[df_exo.tic == "SPY"].set_index("date")["close"].sort_index()
spy_ret = spy.pct_change().dropna()
cum_returns = (1 + spy_ret).cumprod()
cum_returns = cum_returns.loc[cum_returns.index >= regime_df.date.min()]

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(cum_returns.index, cum_returns.values, color="black", alpha=0.75, lw=1.2, label="SPY Cum. Return")

regime_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
labels = ["Bear", "Sideways / Low Vol (Bear-leaning)", "Sideways / Low Vol (Bull-leaning)", "Bull"]

# Build contiguous regime spans for clean axvspan fills (avoids the
# one-axvspan-per-day "black smear" artifact in the original plot).
regime_df = regime_df.sort_values("date").reset_index(drop=True)
regime_df["block"] = (regime_df["regime"] != regime_df["regime"].shift()).cumsum()
for _, block in regime_df.groupby("block"):
    r = int(block["regime"].iloc[0])
    ax.axvspan(block["date"].iloc[0], block["date"].iloc[-1], color=regime_colors[r], alpha=0.25, lw=0)

from matplotlib.patches import Patch
handles = [Patch(facecolor=regime_colors[i], alpha=0.5, label=labels[i]) for i in range(4)]
handles.append(plt.Line2D([0], [0], color="black", lw=1.2, label="SPY Cum. Return"))
ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

ax.set_title("HMM-Detected Market Regimes (4-state Gaussian HMM, fitted on macro ETF panel)", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Return (SPY, normalised)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.axvline(pd.Timestamp(TEST_START), color="black", ls="--", lw=1, alpha=0.6)
ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[1] * 0.97, "  Test period starts",
        fontsize=8.5, va="top", ha="left", style="italic", color="#333")
fig.tight_layout()
fig.savefig("hmm_regimes_timeline.png", dpi=300, bbox_inches="tight")
fig.savefig("hmm_regimes_timeline.pdf", bbox_inches="tight")
print("Saved hmm_regimes_timeline.{png,pdf}")

regime_counts = regime_df["regime"].value_counts(normalize=True).sort_index()
print("Regime day-share:", {labels[i]: f"{regime_counts.get(i, 0):.1%}" for i in range(4)})
