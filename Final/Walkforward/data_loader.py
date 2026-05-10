"""
Self-contained data loader for the walk-forward evaluation.

Downloads ALL data once at module load (using the global date range), then
exposes window-splitting + per-window HMM fitting so each walk-forward
window gets its own train-only HMM (no test-period leakage).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Asset universe — must match the existing portfolio code
# =============================================================================

TICKER_LIST = sorted(list(set([
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JNJ',  'UNH',  'PFE',
    'JPM',  'BAC',  'GS',
    'XOM',  'CVX',
    'WMT',  'PG',
    'BA',   'CAT',
    'AMZN', 'AMD',  'NFLX',
    'V',    'HD',   'MCD',
    'KO',   'PEP',
    'DIS',  'COST',
    'CRM',  'INTC', 'TXN',
    'GE',   'MMM',  'HON',
    'C',    'MS',
    'ABT',  'ABBV', 'MRK',
])))

EXO_TICKERS     = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
TECH_INDICATORS = ["macd", "rsi", "cci", "adx"]

GLOBAL_START = "2014-06-01"     # leave headroom for the 20-day rolling window
GLOBAL_END   = "2024-01-01"


# =============================================================================
# 8 walk-forward windows  (matches the research plan exactly)
# =============================================================================

WALK_FORWARD_WINDOWS = [
    # (label,  train_start,  train_end,    test_start,   test_end)
    ("W1", "2015-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
    ("W2", "2015-01-01", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("W3", "2015-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
    ("W4", "2015-01-01", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("W5", "2015-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("W6", "2015-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("W7", "2015-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("W8", "2015-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
]


# =============================================================================
# Utilities
# =============================================================================

def add_covariance_matrix(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df    = df.sort_values(['date', 'tic'], ignore_index=True)
    pivot = df.pivot_table(index='date', columns='tic', values='close').ffill().bfill()
    rets  = pivot.pct_change()
    rets_np = rets.values
    dates   = rets.index.tolist()
    cov_list = []
    for i in range(len(dates)):
        win = rets_np[max(0, i - lookback + 1): i + 1]
        win = win[~np.isnan(win).any(axis=1)]
        cov = (np.cov(win, rowvar=False)
               if win.shape[0] >= 2 else np.zeros((rets_np.shape[1],) * 2))
        cov_list.append(np.nan_to_num(cov))
    df_cov = pd.DataFrame({'date': dates, 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    return df.dropna(subset=['cov_list']).sort_values(['date', 'tic']).reset_index(drop=True)


# =============================================================================
# One-time data download + feature engineering
# =============================================================================

print("=" * 70)
print(f"[DataLoader] Downloading data for {len(TICKER_LIST)} stocks "
      f"+ {len(EXO_TICKERS)} macro ETFs   ({GLOBAL_START} → {GLOBAL_END})")
print("=" * 70)

_df_raw = YahooDownloader(start_date=GLOBAL_START, end_date=GLOBAL_END,
                            ticker_list=TICKER_LIST).fetch_data()
_fe = FeatureEngineer(use_technical_indicator=True,
                       tech_indicator_list=TECH_INDICATORS,
                       use_vix=False, use_turbulence=False,
                       user_defined_feature=False)
_df = _fe.preprocess_data(_df_raw)
_df = _df.dropna().drop_duplicates(subset=["date", "tic"])
_counts = _df.groupby('tic').size()
_df = _df[_df.tic.isin(_counts[_counts == _counts.max()].index)]
_df = _df.sort_values(["date", "tic"]).reset_index(drop=True)
_df = add_covariance_matrix(_df, lookback=20)

ASSETS    = sorted(_df.tic.unique().tolist())
N_ASSETS  = len(ASSETS)

DF_PRICES = _df.copy()
print(f"[DataLoader] Stock data: {DF_PRICES.shape}   "
      f"{N_ASSETS} assets   "
      f"dates {DF_PRICES.date.min()} → {DF_PRICES.date.max()}")

DF_EXO = YahooDownloader(start_date=GLOBAL_START, end_date=GLOBAL_END,
                           ticker_list=EXO_TICKERS).fetch_data()
DF_EXO = DF_EXO.sort_values(["date", "tic"]).reset_index(drop=True)
print(f"[DataLoader] Macro ETFs:  {DF_EXO.shape}")
print("=" * 70)


# =============================================================================
# Per-window HMM (fit on TRAIN portion of macro ETFs, predict on full window)
# =============================================================================

class WindowedRegimeHMM:
    """
    GaussianHMM with 4 components, trained per-window on the macro ETFs.
    Provides both hard regime labels and soft probabilities.

    Features (4 daily, computed on EXO_TICKERS pivot):
      0  cross-asset mean return       (overall direction)
      1  cross-asset return volatility (stress)
      2  SPY 20-day momentum
      3  SPY-TLT rolling correlation   (risk-on / risk-off)
    """

    def __init__(self, n_regimes: int = 4, lookback: int = 20, seed: int = 42):
        self.n_regimes = n_regimes
        self.lookback  = lookback
        self.model     = GaussianHMM(n_components=n_regimes, covariance_type="diag",
                                       n_iter=200, tol=1e-4, random_state=seed)
        self.scaler        = StandardScaler()
        self.regime_order  = None
        self.inverse_order = None

    def _features(self, df_exo: pd.DataFrame) -> np.ndarray:
        pivot = (df_exo.pivot_table(index="date", columns="tic", values="close")
                       .sort_index().ffill().bfill())
        rets  = pivot.pct_change().fillna(0)
        spy_r = rets["SPY"] if "SPY" in rets.columns else rets.iloc[:, 0]
        tlt_r = rets["TLT"] if "TLT" in rets.columns else rets.iloc[:, 0]
        mean_ret  = rets.mean(axis=1).rolling(self.lookback).mean().fillna(0)
        vol       = rets.std(axis=1).rolling(self.lookback).mean().fillna(0)
        spy_mom   = spy_r.rolling(self.lookback).mean().fillna(0)
        spy_tlt_c = spy_r.rolling(self.lookback).corr(tlt_r).fillna(0)
        X = np.column_stack([mean_ret.values, vol.values, spy_mom.values, spy_tlt_c.values])
        return np.nan_to_num(X), pivot.index

    def fit(self, train_exo: pd.DataFrame):
        X, _      = self._features(train_exo)
        Xs        = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        means     = self.model.means_
        scores    = means[:, 2] - means[:, 1]              # momentum − volatility
        self.regime_order  = np.argsort(scores)
        self.inverse_order = np.argsort(self.regime_order)

    def predict(self, df_exo: pd.DataFrame) -> pd.DataFrame:
        X, dates  = self._features(df_exo)
        Xs        = self.scaler.transform(X)
        raw_lab   = self.model.predict(Xs)
        raw_pr    = self.model.predict_proba(Xs)
        labels    = self.regime_order[raw_lab]
        proba     = raw_pr[:, self.inverse_order]
        out = pd.DataFrame({"date": list(dates), "regime": labels})
        for i in range(self.n_regimes):
            out[f"prob_{i}"] = proba[:, i]
        return out


# =============================================================================
# Window splitter
# =============================================================================

def split_window(window: tuple) -> dict:
    """
    For a window tuple (label, train_start, train_end, test_start, test_end):
      - filter price data to train/test
      - fit a fresh HMM on the train-only macro ETFs
      - predict regime labels + probabilities for the FULL window range

    Returns a dict containing all dataframes the per-window pipeline needs.
    """
    label, ts, te, vs, ve = window

    train_df = DF_PRICES[(DF_PRICES.date >= ts) & (DF_PRICES.date <= te)].reset_index(drop=True)
    test_df  = DF_PRICES[(DF_PRICES.date >= vs) & (DF_PRICES.date <= ve)].reset_index(drop=True)

    train_exo = DF_EXO[(DF_EXO.date >= ts) & (DF_EXO.date <= te)].reset_index(drop=True)
    full_exo  = DF_EXO[(DF_EXO.date >= ts) & (DF_EXO.date <= ve)].reset_index(drop=True)

    hmm = WindowedRegimeHMM(n_regimes=4, lookback=20)
    hmm.fit(train_exo)

    regime_full   = hmm.predict(full_exo)
    train_regime  = regime_full[(regime_full.date >= ts) & (regime_full.date <= te)].reset_index(drop=True)
    test_regime   = regime_full[(regime_full.date >= vs) & (regime_full.date <= ve)].reset_index(drop=True)

    # Align dates: keep only dates present in both prices and regimes
    common_train = set(train_df.date.unique()) & set(train_regime.date.tolist())
    common_test  = set(test_df.date.unique())  & set(test_regime.date.tolist())
    train_df    = train_df[train_df.date.isin(common_train)].reset_index(drop=True)
    test_df     = test_df[test_df.date.isin(common_test)].reset_index(drop=True)
    train_regime = train_regime[train_regime.date.isin(common_train)].reset_index(drop=True)
    test_regime  = test_regime[test_regime.date.isin(common_test)].reset_index(drop=True)

    return {
        "label":         label,
        "train_start":   ts,  "train_end":  te,
        "test_start":    vs,  "test_end":   ve,
        "train_df":      train_df,
        "test_df":       test_df,
        "train_regime":  train_regime,    # has 'regime' (int) and 'prob_0..3'
        "test_regime":   test_regime,
        "assets":        ASSETS,
        "n_assets":      N_ASSETS,
    }
