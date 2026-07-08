"""Gate-interpretability figure: what the Soft RA-RL blend weights actually do.

For Soft RA-RL the blending weights ARE the HMM posterior probabilities
(no learned gate), so this figure is exact and needs no model checkpoints:
it replicates hmm.py's MarketRegimeHMM pipeline (fit on macro-ETF train
period 2015-2021, posterior over the full 2015-2024 span) and plots the
per-day posterior as a stacked area, with SPY cumulative return above for
context. A second figure zooms on the COVID-19 crash (H1 2020), the window
where the paper's regime-transition claim lives.

Colors/labels/state-ordering match make_regime_plot.py (Figure 1) exactly.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from hmmlearn import hmm

plt.rcParams["font.family"] = "DejaVu Sans"

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TRAIN_START, TRAIN_END = "2015-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2024-01-01"

REGIME_COLORS = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
REGIME_LABELS = ["Bear", "Sideways (Bear-leaning)", "Sideways (Bull-leaning)", "Bull"]

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
sorted_states = np.argsort(regime_scores)               # Bear -> Bull
print(f"Regime order (Bear -> Bull): {sorted_states}")

X_full, dates_full = prepare_data(df_exo)
posterior_raw = model.predict_proba(X_full)
posterior = posterior_raw[:, sorted_states]              # columns ordered Bear -> Bull

spy = df_exo[df_exo.tic == "SPY"].set_index("date")["close"].sort_index()
spy_ret = spy.pct_change().dropna()
cum = (1 + spy_ret).cumprod()
cum = cum.loc[cum.index >= dates_full.min()]


def plot_span(x0, x1, fname, title, tick_fmt="%Y", tick_loc=None):
    m = (dates_full >= x0) & (dates_full <= x1)
    d = dates_full[m]
    post = posterior[m]
    c = cum.loc[(cum.index >= x0) & (cum.index <= x1)]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                   height_ratios=[1, 1.4],
                                   gridspec_kw={"hspace": 0.08})
    ax0.plot(c.index, c.values / c.values[0], color="black", lw=1.2)
    ax0.set_ylabel("SPY cum. return")
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.set_title(title, fontsize=12.5, fontweight="bold")

    ax1.stackplot(d, post.T, colors=REGIME_COLORS, alpha=0.85, lw=0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("HMM posterior\n= Soft RA-RL blend weight")
    ax1.spines[["top", "right"]].set_visible(False)

    for ax in (ax0, ax1):
        if pd.Timestamp(TEST_START) >= x0 and pd.Timestamp(TEST_START) <= x1:
            ax.axvline(pd.Timestamp(TEST_START), color="black", ls="--", lw=1, alpha=0.6)

    ax1.xaxis.set_major_locator(tick_loc or mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(tick_fmt))

    handles = [Patch(facecolor=REGIME_COLORS[i], label=REGIME_LABELS[i]) for i in range(4)]
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=4, fontsize=9, frameon=False)
    fig.savefig(fname + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}.{{png,pdf}}")


plot_span(pd.Timestamp("2015-03-01"), pd.Timestamp(TEST_END), "gate_posterior",
          "Soft RA-RL Gate: HMM Posterior Used Directly as Expert Blend Weights (2015–2024)")

plot_span(pd.Timestamp("2019-11-01"), pd.Timestamp("2020-07-01"), "gate_posterior_covid",
          "Gate Behaviour Through the COVID-19 Crash (Nov 2019 – Jun 2020)",
          tick_fmt="%b %Y", tick_loc=mdates.MonthLocator())

# Console diagnostics for the paper text: transition sharpness during COVID
covid = (dates_full >= "2020-02-01") & (dates_full <= "2020-04-30")
bear_w = posterior[covid][:, 0]
print(f"\nCOVID window (Feb-Apr 2020): max Bear weight {bear_w.max():.3f}, "
      f"days with Bear weight > 0.5: {(bear_w > 0.5).sum()}")
