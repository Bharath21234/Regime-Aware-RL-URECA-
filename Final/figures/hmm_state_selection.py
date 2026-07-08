"""HMM state-count model selection: fits GaussianHMM with K=2..6 states on the
same macro-ETF training features used by the paper's regime detector
(replicating hmm.py's MarketRegimeHMM pipeline exactly, fit on train period
2015-2021 only) and reports log-likelihood, AIC, and BIC per K.

Justifies the paper's K=4 choice (currently asserted without evidence) and
informs which K values are worth an end-to-end RL ablation.

Parameter count for a GaussianHMM with K states, D features, full covariance:
  initial distribution: K-1
  transition matrix:    K*(K-1)
  means:                K*D
  covariances:          K*D*(D+1)/2
"""
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TRAIN_START, TRAIN_END = "2015-01-01", "2021-12-31"
TEST_END = "2024-01-01"

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
n, D = X_train.shape
print(f"Training features: n={n} days, D={D} features\n")

N_RESTARTS = 10  # EM is sensitive to init; keep best logL across restarts per K

rows = []
for K in range(2, 7):
    best = None
    for rs in range(N_RESTARTS):
        m = hmm.GaussianHMM(n_components=K, covariance_type="full",
                            n_iter=1000, random_state=42 + rs)
        try:
            m.fit(X_train)
            ll = m.score(X_train)
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, m)
    ll, model = best
    p = (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll
    # occupancy: fraction of train days assigned to each state (degeneracy check)
    occ = np.bincount(model.predict(X_train), minlength=K) / n
    rows.append((K, p, ll, aic, bic, occ.min(), sorted(occ * 100)))

print(f"{'K':>2} {'params':>7} {'logL':>12} {'AIC':>12} {'BIC':>12} {'min occ %':>10}   occupancy per state (%)")
for K, p, ll, aic, bic, mo, occs in rows:
    occ_str = ", ".join(f"{o:.1f}" for o in occs)
    print(f"{K:>2} {p:>7} {ll:>12.1f} {aic:>12.1f} {bic:>12.1f} {mo*100:>9.1f}%   [{occ_str}]")

best_aic = min(rows, key=lambda r: r[3])[0]
best_bic = min(rows, key=lambda r: r[4])[0]
print(f"\nBest by AIC: K={best_aic}   Best by BIC: K={best_bic}")
