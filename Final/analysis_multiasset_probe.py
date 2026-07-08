"""Multi-asset feasibility probe (zero training) — scopes the follow-up paper.

Question: in an equities-only universe the HMM-gated rule DESTROYS value
(results_log 7c) because the only defense is cash. Does the same signal
become valuable when Bear-regime capital can rotate into assets that rally
in bear markets (long Treasuries, gold)?

Rule strategies, all zero-training, daily rebalanced, rf=0:
  EW-equity        100% equal-weight over the 38-stock universe (baseline)
  static 80/20     80% EW-equity + 20% defensive at all times
                   (controls for "always holding some bonds/gold helps")
  hard-rotate      Bear (most-likely HMM state) -> 100% defensive, else EW-equity
  soft-rotate      (1 - P(Bear)) EW-equity + P(Bear) defensive
  cash-gated       Bear -> cash (the 7c rule, for reference)

defensive = 50% TLT + 50% GLD.
NOTE: TLT is also in the HMM's macro feature basket — acceptable for a
feasibility probe, but the follow-up paper should redesign the regime basket
to stay disjoint from the tradable universe.

Protocols: 8-window walk-forward (per-window HMM refit, out-of-sample regime
path) + single-period (fit 2015-2021, test 2022-2024). All return legs from
yfinance for consistent date alignment (7c verified yfinance vs backtest EW
returns align at corr=1.0000).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm as hmmlib
from scipy import stats

EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TICKERS = ['AAPL','ABBV','ABT','AMD','AMZN','BA','BAC','C','CAT','COST','CRM',
           'CVX','DIS','GE','GOOGL','GS','HD','HON','INTC','JNJ','JPM','KO',
           'MCD','META','MMM','MRK','MS','MSFT','NFLX','NVDA','PEP','PFE','PG',
           'TXN','UNH','V','WMT','XOM']
DEFENSIVE = ['TLT', 'GLD']
WINDOWS = [
    ("W1", "2015-01-01", "2019-12-31", "2020-01-01", "2020-06-30", "COVID crash"),
    ("W2", "2015-01-01", "2020-06-30", "2020-07-01", "2020-12-31", "Recovery"),
    ("W3", "2015-01-01", "2020-12-31", "2021-01-01", "2021-06-30", "Bull"),
    ("W4", "2015-01-01", "2021-06-30", "2021-07-01", "2021-12-31", "Bull"),
    ("W5", "2015-01-01", "2021-12-31", "2022-01-01", "2022-06-30", "Bear onset"),
    ("W6", "2015-01-01", "2022-06-30", "2022-07-01", "2022-12-31", "Bear/recov"),
    ("W7", "2015-01-01", "2022-12-31", "2023-01-01", "2023-06-30", "Bull"),
    ("W8", "2015-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "Bull"),
]

print("Downloading macro ETF panel + trading universe + defensive assets...")
raw_exo = yf.download(EXO_TICKERS, start="2014-06-01", end="2024-01-01",
                      auto_adjust=True, progress=False)
close_exo = raw_exo["Close"].dropna(how="all")
df_exo = close_exo.reset_index().melt(id_vars="Date", var_name="tic", value_name="close")
df_exo = df_exo.rename(columns={"Date": "date"}).dropna(subset=["close"])
df_exo["date"] = pd.to_datetime(df_exo["date"])

raw_all = yf.download(TICKERS + DEFENSIVE, start="2019-10-01", end="2024-01-01",
                      auto_adjust=True, progress=False)
close_all = raw_all["Close"]
rets_all = close_all.pct_change().dropna(how="all")
eq_ret = rets_all[TICKERS].mean(axis=1)                 # daily-rebalanced 1/N equities
def_ret = rets_all[DEFENSIVE].mean(axis=1)              # 50/50 TLT+GLD


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
    tr = df_exo[(df_exo.date >= train_start) & (df_exo.date <= train_end)]
    X_tr, _ = prepare_data(tr)
    model = hmmlib.GaussianHMM(n_components=4, covariance_type="full",
                               n_iter=1000, random_state=42)
    model.fit(X_tr)
    order = np.argsort(model.means_[:, 2] - model.means_[:, 1])
    full = df_exo[(df_exo.date >= train_start) & (df_exo.date <= span_end)]
    X_f, dates = prepare_data(full)
    return pd.DataFrame(model.predict_proba(X_f)[:, order], index=dates)


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


def build_strategies(idx, p_bear, is_bear):
    eq = eq_ret.reindex(idx).values
    dfv = def_ret.reindex(idx).values
    return {
        "EW-equity":     eq,
        "static 80/20":  0.8 * eq + 0.2 * dfv,
        "hard-rotate":   np.where(is_bear, dfv, eq),
        "soft-rotate":   (1 - p_bear) * eq + p_bear * dfv,
        "cash-gated":    np.where(is_bear, 0.0, eq),
    }


STRATS = ["EW-equity", "static 80/20", "hard-rotate", "soft-rotate", "cash-gated"]

# ═════ Walk-forward ══════════════════════════════════════════════════════════
print("\nWALK-FORWARD — per-window Sharpe")
print(f"  {'Win':<4} {'context':<12} " + " ".join(f"{s:>13}" for s in STRATS) + "   Bear%")
agg = {s: [] for s in STRATS}
daily_all = {s: [] for s in STRATS}
for wid, tr_s, tr_e, te_s, te_e, ctx in WINDOWS:
    post = hmm_posterior(tr_s, tr_e, te_e)
    post = post[(post.index >= te_s) & (post.index <= te_e)]
    idx = post.index.intersection(eq_ret.index)
    post = post.loc[idx]
    p_bear = post[0].values
    is_bear = post.values.argmax(axis=1) == 0

    strat = build_strategies(idx, p_bear, is_bear)
    line = f"  {wid:<4} {ctx:<12} "
    for s in STRATS:
        m = metrics(strat[s])
        agg[s].append(m)
        daily_all[s].append(strat[s])
        line += f"{m[0]:>13.3f} "
    print(line + f"  {is_bear.mean()*100:>4.1f}%")

print(f"\n  {'Aggregate (mean of 8 windows)':<30} {'Sharpe':>8} {'Return %':>9} {'MaxDD %':>8}")
for s in STRATS:
    v = np.array(agg[s])
    print(f"  {s:<30} {v[:, 0].mean():>8.3f} {v[:, 1].mean():>9.2f} {v[:, 2].mean():>8.2f}")

r_ew = np.concatenate(daily_all["EW-equity"])
print(f"\n  JK-Memmel vs EW-equity (n={len(r_ew)} daily):")
for s in STRATS[1:]:
    z, p = jk_memmel(np.concatenate(daily_all[s]), r_ew)
    star = " *" if p < 0.05 else ""
    print(f"    {s:<30} z={z:+.2f}  p={p:.4f}{star}")

# ═════ Single-period (test 2022-2024) ════════════════════════════════════════
print("\nSINGLE-PERIOD (train 2015-2021, test 2022-2024)")
post = hmm_posterior("2015-01-01", "2021-12-31", "2024-01-01")
post = post[(post.index >= "2022-01-01") & (post.index <= "2024-01-01")]
idx = post.index.intersection(eq_ret.index)
post = post.loc[idx]
p_bear = post[0].values
is_bear = post.values.argmax(axis=1) == 0
strat = build_strategies(idx, p_bear, is_bear)

print(f"  {'Strategy':<30} {'Sharpe':>8} {'Return %':>9} {'MaxDD %':>8}")
for s in STRATS:
    m = metrics(strat[s])
    print(f"  {s:<30} {m[0]:>8.3f} {m[1]:>9.2f} {m[2]:>8.2f}")
print(f"  Bear-day share: {is_bear.mean()*100:.1f}%")
print(f"\n  JK-Memmel vs EW-equity (n={len(idx)} daily):")
for s in STRATS[1:]:
    z, p = jk_memmel(strat[s], strat["EW-equity"])
    star = " *" if p < 0.05 else ""
    print(f"    {s:<30} z={z:+.2f}  p={p:.4f}{star}")

# context: how did the defensive leg itself behave in the key windows?
print("\n  Defensive leg (50/50 TLT+GLD) sanity check, key windows:")
for wid, _, _, te_s, te_e, ctx in [WINDOWS[0], WINDOWS[4], WINDOWS[5]]:
    d = def_ret[(def_ret.index >= te_s) & (def_ret.index <= te_e)]
    m = metrics(d.values)
    print(f"    {wid} ({ctx:<11}): Sharpe {m[0]:>6.3f}  Return {m[1]:>+7.2f}%")
