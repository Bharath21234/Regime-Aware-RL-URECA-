"""
Portfolio Allocation via A2C — Hard Regime Routing
4 specialist heads; the HMM's predicted next-regime index activates one exclusively.

State  : flatten(cov_matrix [N×N] | tech_indicators [4×N]) + [regime_label]
         shape: (N²+4N+1,)   ← flat 1D, no lookback window
Gating : hard argmax — x[:, -1].long() selects which head fires
Reward : mean-variance penalised return − turnover penalty − concentration penalty

Controlled ablation: identical env/reward/training to Baseline and Select_3.
Only the appended state signal and head-selection mechanism differ.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from hmm import MarketRegimeHMM, plot_regimes

# ============================================================================
# Portfolio constraint — IDENTICAL across all three variants
# ============================================================================
MIN_WEIGHT = -0.05
MAX_WEIGHT =  0.20

def enforce_portfolio_constraints(weights):
    """Euclidean projection onto the bounded simplex via bisection search.
    Guarantees: MIN_WEIGHT <= w_i <= MAX_WEIGHT and sum(w) == 1.
    """
    weights = np.array(weights, dtype=np.float32)
    mu_min = float(np.min(weights)) - MAX_WEIGHT - 1.0
    mu_max = float(np.max(weights)) - MIN_WEIGHT + 1.0
    mu = 0.0
    for _ in range(50):
        mu = (mu_min + mu_max) / 2.0
        clipped = np.clip(weights - mu, MIN_WEIGHT, MAX_WEIGHT)
        s = clipped.sum()
        if abs(s - 1.0) < 1e-6:
            break
        if s > 1.0:
            mu_min = mu
        else:
            mu_max = mu
    return np.clip(weights - mu, MIN_WEIGHT, MAX_WEIGHT).astype(np.float32)

# ============================================================================
# Environment — flat state + ONE scalar regime label at the end
# ============================================================================
class HardRegimePortfolioEnv(gym.Env):
    """
    Flat-state portfolio environment with a hard HMM regime label appended.
    State = flatten(cov_matrix || tech_indicators) + [regime_label]
            shape: (N²+4N+1,)
    The regime_label is a single float (0.0, 1.0, 2.0, or 3.0).
    The actor reads x[:, -1].long() to route to the correct specialist head.
    Reward = (risk-adj return − turnover − concentration) × reward_scaling
    """
    def __init__(self, df, stock_dim, initial_amount, tech_indicator_list,
                 regime_df, reward_scaling=1000.0):
        """
        regime_df : DataFrame with columns ['date', 'future_regime']
                    where future_regime is the HMM-predicted next-step regime (int 0-3).
        """
        super().__init__()
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.tech_indicators = tech_indicator_list
        self.regime_df = regime_df
        self.reward_scaling = reward_scaling

        self.unique_dates = sorted(df.date.unique())
        self.day = 0

        self.action_space = spaces.Box(
            low=MIN_WEIGHT, high=MAX_WEIGHT,
            shape=(self.stock_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._get_state()),), dtype=np.float32
        )

    def _get_state(self):
        date = self.unique_dates[self.day]
        data = self.df[self.df.date == date].sort_values('tic')
        covs  = np.nan_to_num(np.array(data["cov_list"].iloc[0])).astype(np.float32)
        techs = np.nan_to_num(
            np.array([data[t].values for t in self.tech_indicators])
        ).astype(np.float32)
        base = np.vstack([covs, techs]).flatten()

        match = self.regime_df.loc[self.regime_df.date == date, 'future_regime']
        regime_label = float(match.values[0]) if not match.empty else 0.0
        return np.concatenate([base, [regime_label]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.date_memory = [self.unique_dates[0]]
        self.actions_memory = []
        self.state = self._get_state()
        return self.state, {}

    def step(self, actions):
        weights = enforce_portfolio_constraints(actions)

        last_data = self.df[self.df.date == self.unique_dates[self.day]].sort_values('tic')
        covs = np.nan_to_num(np.array(last_data["cov_list"].iloc[0]))

        turnover = np.sum(np.abs(weights - self.actions_memory[-1])) \
                   if self.actions_memory else 0.0
        self.actions_memory.append(weights)

        self.day += 1
        self.terminal = self.day >= len(self.unique_dates) - 1
        new_data = self.df[self.df.date == self.unique_dates[self.day]].sort_values('tic')

        ret = np.sum(((new_data.close.values / last_data.close.values) - 1) * weights)
        var = float(np.dot(weights, np.dot(covs, weights)))

        # Mean-variance utility  − turnover cost  − HHI concentration penalty
        reward = (
            ret
            - 0.5 * 0.5 * var       # risk aversion λ=0.5, factor 0.5 from MV utility
            - 0.0001 * turnover
            - 0.005 * np.sum(weights ** 2)
        ) * self.reward_scaling

        self.portfolio_value *= (1 + ret)
        self.portfolio_return_memory.append(ret)
        self.asset_memory.append(self.portfolio_value)
        self.date_memory.append(self.unique_dates[self.day])
        self.state = self._get_state()
        return self.state, reward, self.terminal, False, {}

# ============================================================================
# Configuration
# ============================================================================
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

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-01-01"
INITIAL_AMOUNT = 1_000_000

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

# ============================================================================
# Data Pipeline
# ============================================================================
df = YahooDownloader(
    start_date=TRAIN_START, end_date=TEST_END, ticker_list=TICKER_LIST
).fetch_data()

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    use_vix=False, use_turbulence=False, user_defined_feature=False
)
df = fe.preprocess_data(df)

# Exogenous macro ETFs for HMM training
print("\nFetching exogenous macro ETFs for HMM...")
EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
df_exo = YahooDownloader(
    start_date=TRAIN_START, end_date=TEST_END, ticker_list=EXO_TICKERS
).fetch_data()
df_exo = df_exo.sort_values(["date", "tic"]).reset_index(drop=True)

df = df.dropna().drop_duplicates(subset=["date", "tic"])
counts = df.groupby('tic').size()
df = df[df.tic.isin(counts[counts == counts.max()].index)]
df = df.sort_values(["date", "tic"]).reset_index(drop=True)
TICKER_LIST = sorted(df.tic.unique().tolist())
print(f"Using {len(TICKER_LIST)} tickers after data cleaning.")


def add_covariance_matrix(df, lookback=20):
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    pivot = df.pivot_table(index='date', columns='tic', values='close').ffill().bfill()
    rets = pivot.pct_change()
    rets_np = rets.values
    dates = rets.index.tolist()
    cov_list = []
    for i in range(len(dates)):
        window = rets_np[max(0, i - lookback + 1):i + 1]
        window = window[~np.isnan(window).any(axis=1)]
        cov = (np.cov(window, rowvar=False)
               if window.shape[0] >= 2
               else np.zeros((rets_np.shape[1],) * 2))
        cov_list.append(np.nan_to_num(cov))
    df = df.merge(pd.DataFrame({'date': dates, 'cov_list': cov_list}), on='date')
    return df.dropna(subset=['cov_list']).sort_values(['date', 'tic']).reset_index(drop=True)


print("Computing covariance matrices...")
df = add_covariance_matrix(df)

# ---- HMM regime detection (fit on TRAIN only, predict on full dataset) ----
print("\nFitting HMM on train data (4 regimes)...")
train_exo = df_exo[(df_exo.date >= TRAIN_START) & (df_exo.date <= TRAIN_END)].reset_index(drop=True)
hmm_model = MarketRegimeHMM(n_regimes=4)
hmm_model.fit(train_exo)

print("Predicting regimes for full dataset...")
regime_df = hmm_model.predict(df_exo)
regime_df['future_regime'] = regime_df['regime'].apply(hmm_model.predict_next_regime)

plot_regimes(df, regime_df)

# ---- Train / test split ----
train_df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = df[(df.date >= TEST_START)  & (df.date <= TEST_END)].reset_index(drop=True)
train_regime = regime_df[
    (regime_df.date >= TRAIN_START) & (regime_df.date <= TRAIN_END)
].reset_index(drop=True)
test_regime  = regime_df[
    (regime_df.date >= TEST_START)  & (regime_df.date <= TEST_END)
].reset_index(drop=True)

print(f"Train: {train_df.shape}  Test: {test_df.shape}")

env_train = HardRegimePortfolioEnv(
    train_df, len(TICKER_LIST), INITIAL_AMOUNT, ["macd","rsi","cci","adx"], train_regime
)
env_test  = HardRegimePortfolioEnv(
    test_df,  len(TICKER_LIST), INITIAL_AMOUNT, ["macd","rsi","cci","adx"], test_regime
)

# ============================================================================
# Actor — 4 specialist heads, hard-routed by x[:, -1]
# ============================================================================
class Actor(nn.Module):
    """
    Hard-routing actor.
    Reads the regime index from x[:, -1] (the single appended scalar),
    then activates exactly the corresponding specialist head.
    Shared feature extractor and log-std across all heads.
    """
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.heads   = nn.ModuleList([nn.Linear(hidden, num_assets) for _ in range(4)])
        self.log_std = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        regime_idx = x[:, -1].long()                  # shape [batch]
        features   = self.feature_extractor(x)         # shape [batch, hidden]
        raw = torch.zeros(x.shape[0], self.heads[0].out_features, device=x.device)
        for i in range(4):
            mask = (regime_idx == i)
            if mask.any():
                raw[mask] = self.heads[i](features[mask])
        mean = raw * 0.1
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================================
# A2C Training — slide-by-1 rolling buffer (IDENTICAL to Baseline and Select_3)
# ============================================================================
def train_a2c(env, epochs=200, gamma=0.99, lr=1e-4,
              value_coef=0.5, entropy_coef=0.01, batch_size=20):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor  = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    opt    = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr
    )
    history = []

    for ep in range(epochs):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(
                    mean.cpu(), std.cpu()
                ).sample().to(DEVICE)

            next_state, reward, done, _, _ = env.step(w_raw.cpu().numpy()[0])

            s_buf.append(s_t)
            w_buf.append(w_raw)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            mean_buf.append(mean)
            state = next_state
            ep_reward += reward

            if len(r_buf) >= batch_size:
                bs = torch.cat(s_buf)
                bw = torch.cat(w_buf)
                br = torch.tensor(r_buf, dtype=torch.float32, device=DEVICE)
                bm = torch.tensor(m_buf, dtype=torch.float32, device=DEVICE)

                mean_b, std_b = actor(bs)
                vals          = critic(bs).squeeze()
                dist_b        = torch.distributions.Normal(mean_b, std_b)
                lp            = dist_b.log_prob(bw).sum(-1)
                ent           = dist_b.entropy().sum(-1)

                with torch.no_grad():
                    ns_t = torch.tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)
                    nv = (critic(ns_t).squeeze()
                          if not done
                          else torch.zeros(1, device=DEVICE))

                rets, R = [], nv
                for r, m in zip(reversed(r_buf), reversed(m_buf)):
                    R = r + gamma * R * m
                    rets.insert(0, R)
                rets = torch.stack(rets).squeeze()
                adv  = rets - vals

                loss = (
                    -(lp * adv.detach()).mean()
                    + value_coef  * adv.pow(2).mean()
                    - entropy_coef * ent.mean()
                    + 0.5 * (mean_b ** 2).mean()   # L2 penalty on raw logits
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                opt.step()

                # slide buffer by 1
                for buf in (s_buf, w_buf, r_buf, m_buf, mean_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        history.append(ep_reward)
        if ep % 10 == 0:
            print(f"[Hard] Ep {ep:04d} | Reward: {ep_reward:.4f}")

    return actor, critic, history

# ============================================================================
# Plotting helpers
# ============================================================================
def plot_training_progress(rewards, path='results/hard_training.png'):
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    if len(rewards) >= 20:
        plt.plot(pd.Series(rewards).rolling(20).mean(), lw=2, label='20-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C Training — Hard Regime Routing')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_wealth_over_time(env, initial, path='results/hard_wealth.png'):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    os.makedirs('results', exist_ok=True)
    dates = env.date_memory[:len(env.asset_memory)]
    vals  = np.array(env.asset_memory)
    norm  = vals / initial

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(dates, vals, color='steelblue', lw=1.5)
    axes[0].fill_between(dates, initial, vals,
                         where=(vals >= initial), alpha=0.25, color='green', label='Gain')
    axes[0].fill_between(dates, initial, vals,
                         where=(vals <  initial), alpha=0.25, color='red',   label='Loss')
    axes[0].axhline(initial, color='grey', ls='--', lw=0.8)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )
    axes[0].set_ylabel('Portfolio Value (USD)')
    axes[0].set_title('Hard Routing Wealth (Out-of-Sample)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(dates, norm, color='darkorange', lw=1.5)
    axes[1].axhline(1.0, color='grey', ls='--', lw=0.8)
    axes[1].set_ylabel('Normalised Wealth')
    axes[1].set_xlabel('Date')
    axes[1].grid(alpha=0.3)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

# ============================================================================
# Metrics helpers
# ============================================================================
def compute_metrics(portfolio_return_memory, asset_memory, initial_amount,
                    periods_per_year=252):
    """
    Compute scalar performance metrics for the test period.

    Parameters
    ----------
    portfolio_return_memory : list  [0, r1, r2, ...] — first element is 0 (reset day)
    asset_memory            : list  [v0, v1, v2, ...] — v0 = initial_amount
    initial_amount          : float
    periods_per_year        : int   trading days per year (default 252)

    Returns
    -------
    dict of scalar metrics (all rounded for readability)
    """
    returns = np.array(portfolio_return_memory[1:])   # drop the initial 0
    values  = np.array(asset_memory)

    total_return_pct = (values[-1] / initial_amount - 1) * 100

    mean_r = returns.mean()
    std_r  = returns.std(ddof=1) + 1e-8
    annualised_sharpe = (mean_r / std_r) * np.sqrt(periods_per_year)

    peak        = np.maximum.accumulate(values)
    drawdowns   = (peak - values) / (peak + 1e-8)
    max_drawdown_pct = drawdowns.max() * 100

    downside = returns[returns < 0]
    down_std = (np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8) + 1e-8
    sortino_ratio = (mean_r / down_std) * np.sqrt(periods_per_year)

    return {
        "Total Return (%)":      round(float(total_return_pct),   4),
        "Annualised Sharpe":     round(float(annualised_sharpe),  4),
        "Max Drawdown (%)":      round(float(max_drawdown_pct),   4),
        "Sortino Ratio":         round(float(sortino_ratio),      4),
        "N Trading Days":        int(len(returns)),
        "Final Value ($)":       round(float(values[-1]),         2),
    }


def plot_metrics_over_time(portfolio_return_memory, asset_memory, dates,
                           initial_amount, title_prefix='',
                           save_path='results/metrics_over_time.png',
                           rolling_window=20):
    """
    4-panel figure showing performance metrics across the test period:
      Panel 1 — Cumulative Return (%)
      Panel 2 — Rolling Annualised Sharpe Ratio   (rolling_window-day)
      Panel 3 — Running Maximum Drawdown (%)
      Panel 4 — Rolling Annualised Sortino Ratio  (rolling_window-day)

    Does NOT replace any existing plot; saved to a separate file.
    """
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)

    returns_arr = np.array(portfolio_return_memory[1:])   # length N
    values      = np.array(asset_memory)                  # length N+1
    plot_dates  = list(dates)[:len(values)]               # length N+1

    # ── 1. Cumulative return (%) ─────────────────────────────────────────
    cum_return = (values / initial_amount - 1) * 100      # length N+1

    # ── 2. Rolling annualised Sharpe ──────────────────────────────────────
    ret_s       = pd.Series(returns_arr)
    roll_mean   = ret_s.rolling(rolling_window).mean()
    roll_std    = ret_s.rolling(rolling_window).std() + 1e-8
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    # Prepend NaN so length matches values (N → N+1)
    roll_sharpe_full = np.concatenate([[np.nan], roll_sharpe.values])

    # ── 3. Running maximum drawdown (%) ──────────────────────────────────
    peak     = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-8) * 100      # length N+1, positive = loss

    # ── 4. Rolling annualised Sortino ─────────────────────────────────────
    def _rolling_sortino(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w - 1, len(arr)):
            window  = arr[i - w + 1: i + 1]
            m       = window.mean()
            d       = window[window < 0]
            d_std   = (np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 1e-8) + 1e-8
            out[i]  = (m / d_std) * np.sqrt(252)
        return out

    roll_sortino_full = np.concatenate(
        [[np.nan], _rolling_sortino(returns_arr, rolling_window)]
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"{title_prefix} — Performance Metrics Over Time (Out-of-Sample Test)",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Cumulative return
    ax = axes[0]
    ax.plot(plot_dates, cum_return, color='steelblue', lw=1.5)
    ax.fill_between(plot_dates, 0, cum_return,
                    where=(cum_return >= 0), alpha=0.20, color='green', label='Gain')
    ax.fill_between(plot_dates, 0, cum_return,
                    where=(cum_return <  0), alpha=0.20, color='red',   label='Loss')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_ylabel('Return (%)', fontsize=10)
    ax.set_title('Cumulative Return (%)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Rolling Sharpe
    ax = axes[1]
    ax.plot(plot_dates, roll_sharpe_full, color='purple', lw=1.5,
            label=f'{rolling_window}-day rolling')
    ax.axhline(0,   color='grey',  ls='--', lw=0.8)
    ax.axhline(1.0, color='green', ls=':',  lw=0.8, alpha=0.7, label='Sharpe = 1')
    ax.set_ylabel('Sharpe (ann.)', fontsize=10)
    ax.set_title(f'Rolling {rolling_window}-Day Annualised Sharpe Ratio',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: Running max drawdown (plotted as negative so valleys go down)
    ax = axes[2]
    ax.fill_between(plot_dates, 0, -drawdown,
                    color='red', alpha=0.35, label='Drawdown depth')
    ax.plot(plot_dates, -drawdown, color='darkred', lw=1, alpha=0.8)
    ax.set_ylabel('Drawdown (%)', fontsize=10)
    ax.set_title('Running Maximum Drawdown % (from rolling peak)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: Rolling Sortino
    ax = axes[3]
    ax.plot(plot_dates, roll_sortino_full, color='darkorange', lw=1.5,
            label=f'{rolling_window}-day rolling')
    ax.axhline(0,   color='grey',  ls='--', lw=0.8)
    ax.axhline(1.0, color='green', ls=':',  lw=0.8, alpha=0.7, label='Sortino = 1')
    ax.set_ylabel('Sortino (ann.)', fontsize=10)
    ax.set_title(f'Rolling {rolling_window}-Day Annualised Sortino Ratio',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics over time plot saved to {save_path}")


# ============================================================================
# run_experiment — callable by multi-seed runner or standalone
# ============================================================================
def run_experiment(seed: int = 0, out_dir: str = "results/hard") -> dict:
    """
    One full train + evaluate cycle for a given random seed.

    Returns a dict with scalar performance metrics, 'seed', and 'rewards' list.
    Per-seed plots are written to out_dir/seed_{seed}_*.png.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    actor, critic, rewards = train_a2c(env_train)
    plot_training_progress(rewards, path=f"{out_dir}/seed_{seed}_training.png")

    # ── Greedy evaluation ─────────────────────────────────────────────────
    print("\n[Evaluating Hard-Routing on Test Set (Out-of-Sample)...]")
    state, _ = env_test.reset()
    done = False
    all_weights, final_weights = [], None
    actor.eval()
    with torch.no_grad():
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _ = actor(s_t)
            weights_raw = mean.cpu().numpy()[0]
            final_weights = enforce_portfolio_constraints(weights_raw)
            all_weights.append(final_weights)
            state, _, done, _, _ = env_test.step(weights_raw)

    # ── Scalar metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(
        env_test.portfolio_return_memory, env_test.asset_memory, INITIAL_AMOUNT
    )
    metrics['seed']    = seed
    metrics['rewards'] = rewards

    # ── Metrics-over-time 4-panel figure ─────────────────────────────────
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        env_test.date_memory, INITIAL_AMOUNT,
        title_prefix=f"Hard Routing (Select_1) — Seed {seed}",
        save_path=f"{out_dir}/seed_{seed}_metrics_over_time.png",
        rolling_window=20,
    )

    return metrics


# ============================================================================
# Entry point (standalone run)
# ============================================================================
if __name__ == "__main__":
    results = run_experiment(seed=0)

    # Additional standalone-only outputs
    plot_wealth_over_time(env_test, INITIAL_AMOUNT)

    print("=" * 60)
    print("HARD ROUTING OUT-OF-SAMPLE RESULTS")
    print("=" * 60)
    print(f"Final Portfolio Value : ${env_test.portfolio_value:,.2f}")
    print(f"Total Return          : {(env_test.portfolio_value / INITIAL_AMOUNT - 1) * 100:.2f}%")
    ret_df = pd.DataFrame(env_test.portfolio_return_memory, columns=['ret'])
    sharpe = (252 ** 0.5) * ret_df['ret'].mean() / (ret_df['ret'].std() + 1e-8)
    print(f"Sharpe Ratio          : {sharpe:.4f}")
    print("\nCOMPREHENSIVE PERFORMANCE METRICS (Hard Routing)")
    print("-" * 44)
    for k, v in results.items():
        if k not in ('seed', 'rewards'):
            print(f"  {k:<25s}: {v}")
    print("-" * 44)
