"""
Portfolio Allocation via A2C — Flat-State Baseline
Single Gaussian head, no regime gating, no lookback history.

State  : flatten(cov_matrix [N×N] | tech_indicators [4×N])  →  shape (N²+4N,)
Gating : none (single head)
Reward : mean-variance penalised return − turnover penalty − concentration penalty

This is the ablation baseline: identical env/reward/training to Select_1 and Select_3.
Only the gating mechanism and what is appended to the state differs across variants.
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
# Environment — flat state, NO regime information
# ============================================================================
class FlatPortfolioEnv(gym.Env):
    """
    Flat-state portfolio environment.
    State = flatten(cov_matrix || tech_indicators)   shape: (N²+4N,)
    No regime signal, no lookback window.
    Reward = (risk-adj return − turnover − concentration) × reward_scaling
    """
    def __init__(self, df, stock_dim, initial_amount, tech_indicator_list,
                 reward_scaling=1000.0):
        super().__init__()
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.tech_indicators = tech_indicator_list
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
        return np.vstack([covs, techs]).flatten()

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

train_df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = df[(df.date >= TEST_START)  & (df.date <= TEST_END)].reset_index(drop=True)
print(f"Train: {train_df.shape}  Test: {test_df.shape}")

env_train = FlatPortfolioEnv(train_df, len(TICKER_LIST), INITIAL_AMOUNT, ["macd","rsi","cci","adx"])
env_test  = FlatPortfolioEnv(test_df,  len(TICKER_LIST), INITIAL_AMOUNT, ["macd","rsi","cci","adx"])

# ============================================================================
# Actor — single Gaussian head (no gating)
# ============================================================================
class Actor(nn.Module):
    """Single specialist head — no regime gating."""
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.head    = nn.Linear(hidden, num_assets)
        self.log_std = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        mean = self.head(self.feature_extractor(x)) * 0.1
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
# A2C Training — slide-by-1 rolling buffer (IDENTICAL to Select_1 and Select_3)
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
            print(f"[Baseline] Ep {ep:04d} | Reward: {ep_reward:.4f}")

    return actor, critic, history

# ============================================================================
# Plotting helpers
# ============================================================================
def plot_training_progress(rewards, path='results/baseline_training.png'):
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    if len(rewards) >= 20:
        plt.plot(pd.Series(rewards).rolling(20).mean(), lw=2, label='20-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C Training — Flat-State Baseline (No Gating)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_wealth_over_time(env, initial, path='results/baseline_wealth.png'):
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
    axes[0].set_title('Baseline Wealth (Out-of-Sample)')
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
# Train
# ============================================================================
actor, critic, rewards = train_a2c(env_train)
plot_training_progress(rewards)

# ============================================================================
# Evaluation
# ============================================================================
print("\n[Evaluating Baseline on Test Set (Out-of-Sample)...]")
state, _ = env_test.reset()
done = False
all_weights = []
final_weights = None

actor.eval()
with torch.no_grad():
    while not done:
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mean, _ = actor(s_t)
        weights_raw = mean.cpu().numpy()[0]
        final_weights = enforce_portfolio_constraints(weights_raw)
        all_weights.append(final_weights)
        state, _, done, _, _ = env_test.step(weights_raw)

plot_wealth_over_time(env_test, INITIAL_AMOUNT)

print("=" * 60)
print("BASELINE OUT-OF-SAMPLE RESULTS")
print("=" * 60)
print(f"Final Portfolio Value : ${env_test.portfolio_value:,.2f}")
print(f"Total Return          : {(env_test.portfolio_value / INITIAL_AMOUNT - 1) * 100:.2f}%")
ret_df = pd.DataFrame(env_test.portfolio_return_memory, columns=['ret'])
sharpe = (252 ** 0.5) * ret_df['ret'].mean() / (ret_df['ret'].std() + 1e-8)
print(f"Sharpe Ratio          : {sharpe:.4f}")
if final_weights is not None:
    print("\nFinal Portfolio Allocation:")
    for i, t in enumerate(TICKER_LIST):
        print(f"  {t:5s}: {final_weights[i] * 100:.2f}%")
print("=" * 60)
