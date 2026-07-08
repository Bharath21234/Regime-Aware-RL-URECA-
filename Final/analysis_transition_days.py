"""Regime-transition-day analysis (ICAIF idea #4, results_log.md section 6e).

Splits every walk-forward test day into "transition" vs "stable" days using
the HMM's own out-of-sample regime path (a transition day = the most-likely
HMM state differs from the previous day's, plus a +/-BUFFER-day halo), then
reports each method's annualised Sharpe separately on the two subsets,
pooled across all 8 windows. Turns the paper's single COVID anecdote into a
paper-wide quantified claim ("regime-aware methods add their value on
transition days").

For each window the HMM is refit on that window's training span only
(2015-01-01 -> train_end), exactly as the walk-forward pipeline does, so the
regime path over the test span is out-of-sample.

Usage:
  python3 analysis_transition_days.py [results_dir]
  (default results_dir: Walkforward/results_gae_2seed -- NOTE: superseded,
   old-mechanics data; rerun with Walkforward/results_batchfix_2seed when
   job wf_bf_2seed lands. The pipeline is data-source-agnostic.)

Alignment note: asset_memory has one value per test trading day (verified:
125 for W1 = H1-2020), starting at the initial account value, so daily
returns = pct_change() aligned to test days [1:]. SPY B&H returns from this
pipeline were cross-checked against yfinance SPY daily returns to confirm
the offset.
"""
import json
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm as hmmlib
from scipy import stats

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "Walkforward/results_gae_2seed"
BUFFER = 2  # days on each side of a label switch also count as "transition"

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
WINDOWS = [
    ("W1", "2015-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
    ("W2", "2015-01-01", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("W3", "2015-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
    ("W4", "2015-01-01", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("W5", "2015-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("W6", "2015-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("W7", "2015-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("W8", "2015-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
]

print("Downloading macro ETF panel...")
raw = yf.download(EXO_TICKERS, start="2014-06-01", end="2024-01-01",
                  auto_adjust=True, progress=False)
close = raw["Close"].dropna(how="all")
df_exo = close.reset_index().melt(id_vars="Date", var_name="tic", value_name="close")
df_exo = df_exo.rename(columns={"Date": "date"}).dropna(subset=["close"])
df_exo["date"] = pd.to_datetime(df_exo["date"])

spy_close = close["SPY"].dropna()


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


def regime_path_for_window(train_start, train_end, test_start, test_end):
    """Fit HMM on train span; return most-likely state per test-span day."""
    tr = df_exo[(df_exo.date >= train_start) & (df_exo.date <= train_end)]
    X_tr, _ = prepare_data(tr)
    model = hmmlib.GaussianHMM(n_components=4, covariance_type="full",
                               n_iter=1000, random_state=42)
    model.fit(X_tr)
    # predict over train+test span so the test-period path is conditioned on history
    full = df_exo[(df_exo.date >= train_start) & (df_exo.date <= test_end)]
    X_full, dates_full = prepare_data(full)
    states = model.predict(X_full)
    m = (dates_full >= test_start) & (dates_full <= test_end)
    return pd.Series(states[m], index=dates_full[m])


def transition_mask(states: pd.Series, buffer: int) -> pd.Series:
    switch = states.ne(states.shift()).fillna(False)
    switch.iloc[0] = False
    mask = switch.copy()
    idx = np.where(switch.values)[0]
    for i in idx:
        lo, hi = max(0, i - buffer), min(len(mask), i + buffer + 1)
        mask.iloc[lo:hi] = True
    return mask


def ann_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 5 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


# ── Load per-run daily returns ───────────────────────────────────────────────
path = os.path.join(RESULTS_DIR, "per_run_metrics.json")
rows = json.load(open(path))
print(f"Loaded {len(rows)} runs from {path}\n")

# ── Alignment sanity check: SPY B&H vs yfinance SPY, W1 ─────────────────────
w1 = WINDOWS[0]
spy_test = spy_close[(spy_close.index >= w1[3]) & (spy_close.index <= w1[4])]
spy_rets_ref = spy_test.pct_change().dropna().values
spy_row = next(r for r in rows if r["window"] == "W1" and r["method"] == "S&P 500 B&H")
am = np.array(spy_row["asset_memory"])
spy_rets_run = am[1:] / am[:-1] - 1
n = min(len(spy_rets_ref), len(spy_rets_run))
corr = np.corrcoef(spy_rets_ref[-n:], spy_rets_run[-n:])[0, 1]
print(f"[alignment check] W1 SPY: run has {len(spy_rets_run)} daily returns, "
      f"yfinance has {len(spy_rets_ref)}; corr(last {n} aligned) = {corr:.4f}")
if corr < 0.98:
    print("  !! alignment correlation below 0.98 — inspect offset before trusting results")

# ── Build masks per window, evaluate all runs ────────────────────────────────
results = {}
tstats = {}
for wid, tr_s, tr_e, te_s, te_e in WINDOWS:
    states = regime_path_for_window(tr_s, tr_e, te_s, te_e)
    tmask = transition_mask(states, BUFFER)
    n_trans = int(tmask.sum())
    test_days = states.index

    for r in [x for x in rows if x["window"] == wid]:
        am = np.array(r["asset_memory"])
        rets = am[1:] / am[:-1] - 1
        # align returns to the last len(rets) test days (dropna offsets at start)
        days = test_days[-len(rets):] if len(rets) <= len(test_days) else test_days
        rets = rets[-len(days):]
        msk = tmask.loc[days].values
        key = r["method"]
        results.setdefault(key, {"trans": [], "stable": []})
        results[key]["trans"].append(rets[msk])
        results[key]["stable"].append(rets[~msk])
    print(f"  {wid}: {n_trans}/{len(states)} transition days "
          f"({states.ne(states.shift()).sum() - 1} switches)")

print(f"\n{'Method':<22} {'Sharpe (transition)':>20} {'Sharpe (stable)':>17} "
      f"{'n_trans':>8} {'n_stable':>9} {'Welch p (2t)':>13}")
order = ["Equal-Weight", "Markowitz MVO", "S&P 500 B&H",
         "Baseline (no gate)", "Hard Routing", "Soft MoE (ours)"]
for m in order:
    if m not in results:
        continue
    tr = np.concatenate(results[m]["trans"])
    st = np.concatenate(results[m]["stable"])
    _, p = stats.ttest_ind(tr, st, equal_var=False)
    print(f"{m:<22} {ann_sharpe(tr):>20.3f} {ann_sharpe(st):>17.3f} "
          f"{len(tr):>8} {len(st):>9} {p:>13.4f}")

print(f"\n(BUFFER = +/-{BUFFER} days around each most-likely-state switch; "
      f"RL methods pool all seeds; Sharpe annualised sqrt(252), rf=0)")
