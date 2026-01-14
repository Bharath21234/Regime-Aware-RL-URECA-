"""
Portfolio Allocation using A2C with FinRL Official Environment
Author: (Your Name)

- Uses FinRL's StockPortfolioEnv (official)
- Custom A2C implementation
- Continuous portfolio weights (simplex-safe)
"""

# ============================================================================
# Imports
# ============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

# ============================================================================
# Configuration
# ============================================================================
TICKER_LIST = [
    'AAPL','MSFT','NVDA','GOOGL','META',
    'JNJ','UNH','PFE',
    'JPM','BAC','GS',
    'XOM','CVX',
    'WMT','PG',
    'BA','CAT'
]

START_DATE = "2020-01-01"
END_DATE   = "2023-12-31"
INITIAL_AMOUNT = 1_000_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# FinRL Data Pipeline (OFFICIAL)
# ============================================================================
df = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=TICKER_LIST
).fetch_data()

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False
)

df = fe.preprocess_data(df)
df = df.sort_values(["date", "tic"]).reset_index(drop=True)

# ============================================================================
# FinRL Official Portfolio Environment
# ============================================================================
env = StockPortfolioEnv(
    df=df,
    stock_dim=len(TICKER_LIST),
    initial_amount=INITIAL_AMOUNT,
    transaction_cost_pct=0.001,
    reward_scaling=1.0
)

# ============================================================================
# Actor-Critic Networks
# ============================================================================
class Actor(nn.Module):
    """Actor network outputs portfolio weights on simplex"""
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_assets)
        )

    def forward(self, x):
        logits = self.net(x)
        weights = torch.softmax(logits, dim=-1)
        return weights


class Critic(nn.Module):
    """Critic network estimates state value"""
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================================
# A2C Training Loop (Simplex-Safe)
# ============================================================================
def train_a2c(
    env,
    epochs=200,
    gamma=0.99,
    lr=3e-4,
    value_coef=0.5,
    entropy_coef=0.01
):
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )

    rewards_history = []

    for ep in range(epochs):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            s = torch.tensor(
                state.flatten(),
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            # Actor
            weights = actor(s)

            # Critic
            value = critic(s)

            # Environment step
            next_state, reward, done, _, _ = env.step(
                weights.detach().cpu().numpy()[0]
            )

            s_next = torch.tensor(
                next_state.flatten(),
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                next_value = critic(s_next) if not done else torch.zeros_like(value)

            # TD target and advantage
            td_target = reward + gamma * next_value
            advantage = td_target - value

            # ---- Losses ----
            # Deterministic policy gradient surrogate
            log_prob = torch.sum(torch.log(weights + 1e-8) * weights.detach())
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))

            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)

            loss = (
                actor_loss
                + value_coef * critic_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_reward += reward
            state = next_state

        rewards_history.append(ep_reward)

        if ep % 20 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:.6f}")

    return actor, critic, rewards_history

# ============================================================================
# Train Agent
# ============================================================================
actor, critic, rewards = train_a2c(env)

# ============================================================================
# Evaluation
# ============================================================================
state, _ = env.reset()
done = False

while not done:
    s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        weights = actor(s).cpu().numpy()[0]
    state, _, done, _, _ = env.step(weights)

print("=" * 60)
print(f"Final Portfolio Value: ${env.portfolio_value:,.2f}")
print(f"Total Return: {(env.portfolio_value / INITIAL_AMOUNT - 1) * 100:.2f}%")
print("=" * 60)
