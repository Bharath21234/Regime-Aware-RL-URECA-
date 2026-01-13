import yfinance as yf
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

# -------------------------------
# 1. Fetch and preprocess data
# -------------------------------
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "ORCL", "IBM", "CSCO", "QCOM", "AVGO",
    "ADBE", "CRM", "PYPL", "SHOP",
]

def fetch_and_preprocess_data(number, start_date, end_date):
    chosen = random.sample(tickers, number)
    results = {}

    for ticker in chosen:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                print(f"Skipping {ticker}: no data")
                continue
            data["return"] = data["Close"].pct_change().fillna(0)
            results[ticker] = data["return"].values
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    min_len = min(len(v) for v in results.values())
    for k in results:
        results[k] = results[k][-min_len:]

    df = pd.DataFrame(results)
    return df

# -------------------------------
# 2. Portfolio Environment
# -------------------------------
class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio environment.
    State: window_size x num_assets of past returns
    Action: continuous weight vector (sum to 1)
    Reward: clipped portfolio return
    """

    def __init__(self, returns_df, window_size=5):
        super().__init__()
        self.returns = returns_df.values
        self.num_assets = returns_df.shape[1]
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_assets), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        return self._get_obs(), {}

    def _get_obs(self):
        return self.returns[self.current_step - self.window_size : self.current_step, :].astype(np.float32)

    def step(self, action):
        weights = np.clip(action, 0, np.inf)
        weights /= np.sum(weights) + 1e-8

        current_returns = self.returns[self.current_step]
        # Clip daily portfolio return to ±10% to avoid explosion
        reward = float(np.dot(weights, current_returns))
        reward = np.clip(reward, -0.10, 0.10)

        self.current_step += 1
        done = self.current_step >= len(self.returns)
        obs = self._get_obs()
        return obs, reward, done, False, {}

# -------------------------------
# 3. Policy Network
# -------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_assets):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_assets)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.fc(x)
        weights = self.softmax(logits)
        return weights

# -------------------------------
# 4. REINFORCE Training
# -------------------------------
def train_agent(env, epochs=500, lr=0.001, gamma=0.99):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    model = PolicyNetwork(input_dim, env.num_assets)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        log_probs = []
        rewards = []

        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
            weights = model(state_tensor).squeeze(0)

            dist = torch.distributions.Dirichlet(weights + 1e-3)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, _ = env.step(action.detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state

        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)

        discounted_returns = torch.FloatTensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        loss = -torch.sum(torch.stack(log_probs) * discounted_returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward:.6f}")

    return model

# -------------------------------
# 5. Random Portfolio Baseline
# -------------------------------
def random_portfolio_baseline(df):
    returns = df.values
    num_assets = df.shape[1]
    cumulative_log_return = 0.0

    for t in range(len(returns)):
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)
        step_return = np.dot(weights, returns[t])
        step_return = np.clip(step_return, -0.10, 0.10)  # clip to ±10%
        cumulative_log_return += np.log1p(step_return)

    return np.expm1(cumulative_log_return)  # convert log return back to linear

# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    df = fetch_and_preprocess_data(5, "2023-01-01", "2024-01-01")
    print(f"\nTraining continuous portfolio agent on {list(df.columns)} ...\n")

    env = PortfolioEnv(df, window_size=5)
    trained_model = train_agent(env, epochs=500)

    # Random baseline
    rand_ret = random_portfolio_baseline(df)
    print(f"\nRandom Portfolio Total Return: {rand_ret*100:.2f}%")

    # Evaluate RL agent
    state, _ = env.reset()
    done = False
    cumulative_log_return = 0.0

    while not done:
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        weights = trained_model(state_tensor).detach().numpy().squeeze()
        next_state, reward, done, _, _ = env.step(weights)
        cumulative_log_return += np.log1p(reward)
        state = next_state

    rl_total_return = np.expm1(cumulative_log_return)
    print(f"RL Agent Total Return: {rl_total_return*100:.2f}%")
