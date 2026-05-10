"""
Non-RL portfolio baselines for the walk-forward evaluation.

Implemented:
  - Equal-Weight (1/N)         simple, monthly rebalance
  - Markowitz MVO              max-Sharpe weights from train data, monthly rebalance
  - S&P 500 Buy-and-Hold       SPY allocation, no rebalance
  - LSTM-Context A2C           regime-AGNOSTIC RL baseline (separate path; trained later
                                via the same train_a2c loop in train.py)

Each non-RL baseline takes a window dict (from data_loader.split_window) and
returns the same metrics format used by the RL variants:
  {'asset_memory': [...], 'portfolio_return_memory': [...], 'date_memory': [...]}
so compute_metrics() can be applied uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from data_loader import ASSETS, GLOBAL_START, GLOBAL_END

INITIAL_AMOUNT = 1_000_000


# =============================================================================
# Helpers
# =============================================================================

def _close_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return (df.pivot_table(index="date", columns="tic", values="close")
              .sort_index().ffill().bfill())


def _simulate_with_target_weights(price_pivot: pd.DataFrame,
                                    target_fn,
                                    rebalance_days: int = 21,
                                    initial: float = INITIAL_AMOUNT) -> dict:
    """
    Generic portfolio simulator.

    target_fn(t_idx, date) -> ndarray of length N (target weights, sum=1).
    Rebalances every `rebalance_days` to the target; otherwise weights drift
    with realised returns.

    Returns dict with asset_memory, portfolio_return_memory, date_memory.
    """
    dates    = list(price_pivot.index)
    prices   = price_pivot.values                              # (T, N)
    T, N     = prices.shape
    rets     = price_pivot.pct_change().fillna(0).values      # (T, N)

    portfolio_value         = initial
    asset_memory            = [initial]
    portfolio_return_memory = [0.0]
    date_memory             = [dates[0]]

    weights = np.full(N, 1.0 / N, dtype=np.float64)
    for t in range(1, T):
        # Drift weights with previous-day returns
        weights = weights * (1 + rets[t])
        weights = weights / max(weights.sum(), 1e-12)
        # Rebalance if hit
        if (t - 1) % rebalance_days == 0:
            target  = target_fn(t, dates[t])
            weights = np.asarray(target, dtype=np.float64)

        port_ret = float(np.dot(weights, rets[t]))
        portfolio_value *= (1.0 + port_ret)
        asset_memory.append(portfolio_value)
        portfolio_return_memory.append(port_ret)
        date_memory.append(dates[t])

    return {
        "asset_memory":             asset_memory,
        "portfolio_return_memory":  portfolio_return_memory,
        "date_memory":              date_memory,
    }


# =============================================================================
# 1. Equal-Weight (1/N)
# =============================================================================

def equal_weight(window: dict) -> dict:
    pivot = _close_pivot(window["test_df"])
    N     = pivot.shape[1]
    target = np.full(N, 1.0 / N, dtype=np.float64)
    return _simulate_with_target_weights(pivot, lambda t, d: target,
                                          rebalance_days=21)


# =============================================================================
# 2. Markowitz MVO  (max-Sharpe weights from train data)
# =============================================================================

def markowitz_mvo(window: dict) -> dict:
    """
    Estimate μ, Σ from the TRAIN-period prices of the same asset universe;
    solve max-Sharpe under [w_min, w_max] bounds + sum-to-1 constraint;
    apply those weights monthly during the test window.
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    train_pivot = _close_pivot(window["train_df"])
    test_pivot  = _close_pivot(window["test_df"])

    # Some windows might have a slightly different asset list — intersect
    common = sorted(set(train_pivot.columns) & set(test_pivot.columns))
    train_pivot = train_pivot[common]
    test_pivot  = test_pivot[common]

    mu  = expected_returns.mean_historical_return(train_pivot)
    S   = risk_models.sample_cov(train_pivot)

    ef  = EfficientFrontier(mu, S, weight_bounds=(-0.05, 0.20))
    try:
        ef.max_sharpe()
        cleaned = ef.clean_weights()
    except Exception:
        # Fall back to min-variance if max-Sharpe fails to converge
        ef2     = EfficientFrontier(mu, S, weight_bounds=(-0.05, 0.20))
        ef2.min_volatility()
        cleaned = ef2.clean_weights()

    target = np.array([cleaned[t] for t in test_pivot.columns], dtype=np.float64)
    target = target / max(target.sum(), 1e-12)                # safety renormalise

    return _simulate_with_target_weights(test_pivot,
                                          lambda t, d: target,
                                          rebalance_days=21)


# =============================================================================
# 3. S&P 500 Buy-and-Hold  (SPY)
# =============================================================================

# Fetch SPY once at module load (covers the global range)
print("[Baselines] Downloading SPY (S&P 500 buy-hold benchmark)...")
_spy_df = YahooDownloader(start_date=GLOBAL_START, end_date=GLOBAL_END,
                            ticker_list=["SPY"]).fetch_data()
_spy_df = _spy_df.sort_values("date").reset_index(drop=True)


def sp500_buy_hold(window: dict) -> dict:
    """Allocate 100 % to SPY at test_start; hold to test_end."""
    spy_test = _spy_df[(_spy_df.date >= window["test_start"])
                        & (_spy_df.date <= window["test_end"])].reset_index(drop=True)
    if spy_test.empty:
        # Empty test window — return flat
        return {"asset_memory":            [INITIAL_AMOUNT, INITIAL_AMOUNT],
                "portfolio_return_memory": [0.0, 0.0],
                "date_memory":             [window["test_start"], window["test_end"]]}

    closes  = spy_test.close.values
    rets    = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])
    portfolio_value         = INITIAL_AMOUNT
    asset_memory            = [INITIAL_AMOUNT]
    portfolio_return_memory = [0.0]
    date_memory             = [spy_test.date.iloc[0]]

    for t in range(1, len(closes)):
        portfolio_value *= (1.0 + rets[t])
        asset_memory.append(portfolio_value)
        portfolio_return_memory.append(float(rets[t]))
        date_memory.append(spy_test.date.iloc[t])

    return {
        "asset_memory":             asset_memory,
        "portfolio_return_memory":  portfolio_return_memory,
        "date_memory":              date_memory,
    }


# =============================================================================
# Registry
# =============================================================================

NON_RL_BASELINES = {
    "Equal-Weight":  equal_weight,
    "Markowitz MVO": markowitz_mvo,
    "S&P 500 B&H":   sp500_buy_hold,
}
