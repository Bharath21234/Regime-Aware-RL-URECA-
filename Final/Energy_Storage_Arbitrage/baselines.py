"""
Non-RL baseline strategies for the Energy Storage Arbitrage problem.

All strategies operate on the SAME test_df loaded by main.py.
Each returns an asset_memory list (portfolio value over time) and a
daily_pnl list, which are fed into main.compute_metrics.

Strategies
----------
DoNothing     — never charge or discharge; profit = 0
Threshold     — charge when price < p20 of train prices;
                discharge when price > p80 of train prices
TimeOfDay     — charge in spring/autumn months (low price);
                discharge in summer/winter peak months
Momentum      — charge when 10-day trend is falling;
                discharge when trend is rising
OracleLPstrategy — LP optimal with PERFECT future price knowledge (upper bound)
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Import shared constants from main (data already loaded there)
from main import (
    test_df, train_df,
    BATTERY_CAPACITY_MWH, MAX_ENERGY_MWH, ONE_WAY_EFF,
    DEGRADATION_PER_MWH, INITIAL_AMOUNT,
)


# =============================================================================
# Simulation engine
# =============================================================================

def simulate(action_fn, test_df: pd.DataFrame,
             initial_soc: float = BATTERY_CAPACITY_MWH / 2) -> dict:
    """
    Run a strategy defined by action_fn over the test period.

    action_fn(day_idx, row, soc, context) -> float  in [-1, 1]
      day_idx : integer step
      row     : pandas Series (current day's test_df row)
      soc     : current battery state-of-charge (MWh)
      context : dict for strategy-specific carry-overs

    Returns dict with 'daily_pnl', 'asset_memory', 'date_memory'.
    """
    soc           = initial_soc
    portfolio     = INITIAL_AMOUNT
    daily_pnl     = []
    asset_memory  = [INITIAL_AMOUNT]
    date_memory   = [test_df.iloc[0]["date"]]
    context: dict = {}

    for idx in range(len(test_df) - 1):
        row    = test_df.iloc[idx]
        a      = float(np.clip(action_fn(idx, row, soc, context), -1.0, 1.0))
        price  = float(row["price"])
        energy = a * MAX_ENERGY_MWH

        if energy > 0:
            energy_in    = min(energy * ONE_WAY_EFF, BATTERY_CAPACITY_MWH - soc)
            energy_drawn = energy_in / ONE_WAY_EFF
            cost         = energy_drawn * price
            rev          = 0.0
            soc         += energy_in
        else:
            energy_out  = min(-energy / ONE_WAY_EFF, soc)
            energy_sold  = energy_out * ONE_WAY_EFF
            rev          = energy_sold * price
            cost         = 0.0
            soc         -= energy_out

        pnl = rev - cost - DEGRADATION_PER_MWH * abs(energy)
        portfolio    += pnl
        daily_pnl.append(pnl)
        asset_memory.append(portfolio)
        date_memory.append(test_df.iloc[idx + 1]["date"])

    return {"daily_pnl": daily_pnl, "asset_memory": asset_memory,
            "date_memory": date_memory}


# =============================================================================
# Strategy definitions
# =============================================================================

def _do_nothing(idx, row, soc, ctx):
    return 0.0


def _threshold(idx, row, soc, ctx):
    """Charge when price < train p20; discharge when price > train p80."""
    if "p20" not in ctx:
        ctx["p20"] = float(train_df["price"].quantile(0.20))
        ctx["p80"] = float(train_df["price"].quantile(0.80))
    price = row["price"]
    if price < ctx["p20"] and soc < BATTERY_CAPACITY_MWH * 0.95:
        return 1.0   # charge
    if price > ctx["p80"] and soc > BATTERY_CAPACITY_MWH * 0.05:
        return -1.0  # discharge
    return 0.0


def _time_of_day(idx, row, soc, ctx):
    """
    Spring/autumn (Mar–May, Sep–Nov): charge (historically cheaper).
    Summer/winter peak (Jun–Aug, Dec–Jan): discharge (historically expensive).
    """
    m = int(row["month"])
    if m in (3, 4, 5, 9, 10, 11) and soc < BATTERY_CAPACITY_MWH * 0.95:
        return 0.8
    if m in (6, 7, 8, 12, 1) and soc > BATTERY_CAPACITY_MWH * 0.05:
        return -0.8
    return 0.0


def _momentum(idx, row, soc, ctx):
    """
    Charge when the 10-day price trend is falling (buy the dip).
    Discharge when trend is rising (sell the rally).
    Uses pre-computed 10-day momentum from test_df if available.
    """
    if "momentum" not in ctx:
        prices = test_df["price"].values
        trend  = pd.Series(prices).rolling(10).mean()
        lag    = pd.Series(prices).rolling(10).mean().shift(10)
        ctx["momentum"] = (trend - lag).fillna(0).values

    mom = ctx["momentum"][idx]
    if mom < -0.5 and soc < BATTERY_CAPACITY_MWH * 0.95:   # price falling → charge
        return min(1.0, -mom / 5.0)
    if mom > 0.5 and soc > BATTERY_CAPACITY_MWH * 0.05:    # price rising → discharge
        return -min(1.0, mom / 5.0)
    return 0.0


# =============================================================================
# Oracle LP  (perfect foresight upper bound)
# =============================================================================

def run_oracle_lp(test_prices: np.ndarray,
                  initial_soc: float = BATTERY_CAPACITY_MWH / 2) -> dict:
    """
    Solve the exact LP:
      max  sum_t(revenue_t - cost_t - degradation_t)
    Variables: charge_t, discharge_t ≥ 0  (separate, so |a_t| is linear)
    Constraints:
      SoC_t = SoC_{t-1} + charge_t * one_way_eff - discharge_t / one_way_eff
      0 ≤ SoC_t ≤ CAPACITY
      0 ≤ charge_t, discharge_t ≤ MAX_ENERGY_MWH

    Returns dict with 'daily_pnl', 'asset_memory', 'date_memory'.
    """
    T   = len(test_prices) - 1      # decisions for days 0 … T-1
    p   = test_prices[:T]
    eff = ONE_WAY_EFF

    # Variables: [charge_0 … charge_{T-1}, discharge_0 … discharge_{T-1}]
    # Minimise cost (linprog minimises):
    #   c . x  where c_i = price_i + deg (charge side)
    #                 c_{T+i} = -(price_i - deg) (discharge side)
    deg = DEGRADATION_PER_MWH
    c   = np.concatenate([p + deg, -p + deg])

    # SoC dynamics encoded as cumulative sum inequalities
    # For each t: 0 ≤ initial_soc + cumsum(charge*eff - discharge/eff) ≤ CAPACITY
    # → cumsum(ch*eff - dis/eff) ≤ CAPACITY - initial_soc
    # → cumsum(-ch*eff + dis/eff) ≤ initial_soc
    A_rows = []
    b_rows = []
    for t in range(T):
        # Upper SoC bound
        row = np.zeros(2 * T)
        row[:t + 1]        =  eff      # charge columns
        row[T: T + t + 1]  = -1 / eff  # discharge columns
        A_rows.append(row)
        b_rows.append(BATTERY_CAPACITY_MWH - initial_soc)
        # Lower SoC bound (SoC ≥ 0)
        A_rows.append(-row)
        b_rows.append(initial_soc)

    A_ub  = np.array(A_rows)
    b_ub  = np.array(b_rows)
    bounds = [(0.0, MAX_ENERGY_MWH)] * (2 * T)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        print(f"[Oracle LP] solver status: {res.message}. Returning zeros.")
        charge    = np.zeros(T)
        discharge = np.zeros(T)
    else:
        charge    = res.x[:T]
        discharge = res.x[T:]

    # Reconstruct P&L
    pnl_arr   = discharge * p - charge * p - deg * (charge + discharge)
    portfolio = INITIAL_AMOUNT
    asset_mem = [INITIAL_AMOUNT]
    daily_pnl = []
    dates     = test_df["date"].values

    for t in range(T):
        portfolio += pnl_arr[t]
        daily_pnl.append(pnl_arr[t])
        asset_mem.append(portfolio)

    return {"daily_pnl": daily_pnl, "asset_memory": asset_mem,
            "date_memory": list(dates[: len(asset_mem)])}


# =============================================================================
# Run all baselines
# =============================================================================

def run_all_baselines() -> dict:
    """
    Returns {strategy_name: result_dict} where each result_dict has
    keys 'daily_pnl', 'asset_memory', 'date_memory'.
    """
    print("\nRunning baseline strategies...")
    prices = test_df["price"].values

    results = {}

    results["Do Nothing"] = simulate(_do_nothing, test_df)
    print("  [✓] Do Nothing")

    results["Threshold"] = simulate(_threshold, test_df)
    print("  [✓] Threshold (p20/p80)")

    results["Time-of-Day"] = simulate(_time_of_day, test_df)
    print("  [✓] Time-of-Day seasonal")

    results["Momentum"] = simulate(_momentum, test_df)
    print("  [✓] Momentum (10-day trend)")

    print("  [~] Oracle LP (solving ~730-day LP)...")
    results["Oracle (LP)"] = run_oracle_lp(prices)
    print("  [✓] Oracle LP solved")

    return results
