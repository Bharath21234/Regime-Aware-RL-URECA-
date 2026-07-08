"""HMM-gated Equal-Weight rule baseline (ICAIF item #22, added 2026-07-04).

Answers the reviewer question: "if the HMM signal is good, why RL at all?
What does a simple rule -- Bear regime -> de-risk, otherwise 1/N -- achieve?"

Two rule variants, both zero-training:
  hard-gated EW: equity exposure = 0 when the HMM's most-likely state is Bear,
                 else 1.0 (hold cash on Bear days; rf = 0 per paper convention)
  soft-gated EW: equity exposure = 1 - P(Bear) each day (fractional de-risking)

Evaluated on both protocols:
  1. Walk-forward (8 windows): per-window HMM refit on that window's train
     span (out-of-sample regime path), EW daily returns taken from the
     walk-forward backtest's own Equal-Weight rows (deterministic, identical
     across mechanics versions -- so these numbers are FINAL and unaffected by
     the old-code walkforward caveat). Aggregates comparable to Table III.
  2. Single-period (train 2015-2021, test 2022-2024): EW = daily-rebalanced
     mean return over the 38-ticker universe from yfinance; HMM fit on train
     span only. Comparable to Tables I/II.
"""
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm as hmmlib
from scipy import stats

WF_DIR = "Walkforward/results_gae_2seed"   # EW rows only (deterministic)

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TICKERS = ['AAPL','ABBV','ABT','AMD','AMZN','BA','BAC','C','CAT','COST','CRM',
           'CVX','DIS','GE','GOOGL','GS','HD','HON','INTC','JNJ','JPM','KO',
           'MCD','META','MMM','MRK','MS','MSFT','NFLX','NVDA','PEP','PFE','PG',
           'TXN','UNH','V','WMT','XOM']
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
            cm = rolling_corr.loc[date].values
            mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
            pw = cm[mask]; pw = pw[np.isfinite(pw)]
            avg_corr.append(pw.mean() if len(pw) > 0 else 0.0)
        except Exception:
            avg_corr.append(0.0)
    avg_corr = pd.Series(avg_corr, index=returns_df.index)
    feats = pd.DataFrame({
        "mean_return": mean_return, "rolling_vol": rolling_vol,
        "rolling_momentum": rolling_momentum, "avg_correlation": avg_corr,
    }, index=returns_df.index).dropna()
    m, s = feats.mean(), feats.std() + 1e-8
    return ((feats - m) / s).values, feats.index


def hmm_posterior(train_start, train_end, span_end):
    """Fit on train span; return (dates, posterior[:, Bear..Bull]) over
    train_start..span_end."""
    tr = df_exo[(df_exo.date >= train_start) & (df_exo.date <= train_end)]
    X_tr, _ = prepare_data(tr)
    model = hmmlib.GaussianHMM(n_components=4, covariance_type="full",
                               n_iter=1000, random_state=42)
    model.fit(X_tr)
    order = np.argsort(model.means_[:, 2] - model.means_[:, 1])  # Bear -> Bull
    full = df_exo[(df_exo.date >= train_start) & (df_exo.date <= span_end)]
    X_f, dates = prepare_data(full)
    post = model.predict_proba(X_f)[:, order]
    return dates, post


def metrics(rets):
    rets = np.asarray(rets)
    wealth = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(wealth)
    mdd = ((peak - wealth) / peak).max() * 100
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std() > 0 else np.nan
    return sharpe, (wealth[-1] - 1) * 100, mdd


def jk_memmel(r1, r2):
    n = len(r1)
    s1, s2 = r1.mean() / r1.std(ddof=1), r2.mean() / r2.std(ddof=1)
    rho = np.corrcoef(r1, r2)[0, 1]
    theta = (2 * (1 - rho) + 0.5 * (s1**2 + s2**2) - s1 * s2 * (1 + rho**2)) / n
    z = (s1 - s2) / np.sqrt(theta)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# ═════ 1. Walk-forward ═══════════════════════════════════════════════════════
rows = json.load(open(os.path.join(WF_DIR, "per_run_metrics.json")))
agg = {"Equal-Weight": [], "hard-gated EW": [], "soft-gated EW": []}
daily_all = {"Equal-Weight": [], "hard-gated EW": [], "soft-gated EW": []}

print("\nWALK-FORWARD (8 windows) — per-window Sharpe")
print(f"  {'Win':<4} {'EW':>7} {'hard-gated':>11} {'soft-gated':>11}   Bear-day share")
for wid, tr_s, tr_e, te_s, te_e in WINDOWS:
    ew_row = next(r for r in rows if r["window"] == wid and r["method"] == "Equal-Weight")
    am = np.array(ew_row["asset_memory"])
    ew = am[1:] / am[:-1] - 1

    dates, post = hmm_posterior(tr_s, tr_e, te_e)
    m = (dates >= te_s) & (dates <= te_e)
    p_bear = pd.Series(post[m][:, 0], index=dates[m]).iloc[-len(ew):].values
    is_bear = (post[m].argmax(axis=1) == 0)[-len(ew):]

    hard_g = np.where(is_bear, 0.0, ew)
    soft_g = (1 - p_bear) * ew

    res = {}
    for name, r in (("Equal-Weight", ew), ("hard-gated EW", hard_g),
                    ("soft-gated EW", soft_g)):
        res[name] = metrics(r)
        agg[name].append(res[name])
        daily_all[name].append(r)
    print(f"  {wid:<4} {res['Equal-Weight'][0]:>7.3f} {res['hard-gated EW'][0]:>11.3f} "
          f"{res['soft-gated EW'][0]:>11.3f}   {is_bear.mean() * 100:>5.1f}%")

print(f"\n  {'Aggregate (mean of 8 windows)':<32} {'Sharpe':>8} {'Return %':>9} {'MaxDD %':>8}")
for name, vals in agg.items():
    v = np.array(vals)
    print(f"  {name:<32} {v[:, 0].mean():>8.3f} {v[:, 1].mean():>9.2f} {v[:, 2].mean():>8.2f}")
print("  (compare Table III: EW 1.645 | Soft RA-RL 1.352 | Hard RA-RL 1.176 — "
      "old-mechanics RL values, superseded)")

r_ew = np.concatenate(daily_all["Equal-Weight"])
for g in ("hard-gated EW", "soft-gated EW"):
    z, p = jk_memmel(np.concatenate(daily_all[g]), r_ew)
    print(f"  JK-Memmel {g} vs EW (n={len(r_ew)} daily): z={z:+.2f}, p={p:.4f}")

# ═════ 2. Single-period (2022-2024 test) ═════════════════════════════════════
print("\nSINGLE-PERIOD (train 2015-2021, test 2022-2024)")
print("Downloading 38-ticker universe...")
raw_u = yf.download(TICKERS, start="2021-12-15", end="2024-01-01",
                    auto_adjust=True, progress=False)
uclose = raw_u["Close"].dropna(how="all")
urets = uclose.pct_change().dropna(how="all")
urets = urets[(urets.index >= "2022-01-01") & (urets.index <= "2024-01-01")]
ew_sp = urets.mean(axis=1)   # daily-rebalanced 1/N

dates, post = hmm_posterior("2015-01-01", "2021-12-31", "2024-01-01")
post_s = pd.DataFrame(post, index=dates).reindex(ew_sp.index).ffill()
p_bear = post_s[0].values
is_bear = post_s.values.argmax(axis=1) == 0

variants = {
    "Equal-Weight": ew_sp.values,
    "hard-gated EW": np.where(is_bear, 0.0, ew_sp.values),
    "soft-gated EW": (1 - p_bear) * ew_sp.values,
}
print(f"  {'Variant':<32} {'Sharpe':>8} {'Return %':>9} {'MaxDD %':>8}")
for name, r in variants.items():
    s, ret, mdd = metrics(r)
    print(f"  {name:<32} {s:>8.3f} {ret:>9.2f} {mdd:>8.2f}")
print(f"  Bear-day share of test period: {is_bear.mean() * 100:.1f}%")
print("  (compare Table II means: Soft RA-RL Sharpe 0.568, Return 22.97%, "
      "MaxDD 20.70% | Hard 0.222 | Router -0.060)")
for g in ("hard-gated EW", "soft-gated EW"):
    z, p = jk_memmel(variants[g], variants["Equal-Weight"])
    print(f"  JK-Memmel {g} vs EW (n={len(ew_sp)} daily): z={z:+.2f}, p={p:.4f}")
