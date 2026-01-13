import yfinance as yf
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces


# --- (reuse your data and environment code) ---
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


class PortfolioEnv(gym.Env):
    """Continuous portfolio environment with returns as observations."""
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
        reward = float(np.dot(weights, current_returns))
        reward = np.clip(reward, -0.10, 0.10)

        self.current_step += 1
        done = self.current_step >= len(self.returns)
        obs = self._get_obs()
        return obs, reward, done, False, {}


# --- Actor & Critic networks ---
class Actor(nn.Module):
    def __init__(self, input_dim, num_assets, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_assets),
        )

    def forward(self, x):
        logits = self.net(x)
        # Softplus ensures positive Dirichlet concentrations
        concentrations = torch.nn.functional.softplus(logits) + 1e-3
        return concentrations


class Critic(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --- TD(0) Actor–Critic training loop ---
def train_actor_critic_td(env,
                          epochs=500,
                          lr=3e-4,
                          gamma=0.99,
                          value_coef=0.5,
                          entropy_coef=0.01,
                          print_every=50,
                          device='cpu'):

    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    num_assets = env.num_assets

    actor = Actor(input_dim, num_assets).to(device)
    critic = Critic(input_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            s_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)

            # Actor: sample from Dirichlet
            concentrations = actor(s_t)
            dist = torch.distributions.Dirichlet(concentrations.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Critic: value estimate
            value = critic(s_t)

            # Step environment
            next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            s_next = torch.FloatTensor(next_state.flatten()).unsqueeze(0).to(device)

            # Critic's next value
            with torch.no_grad():
                next_value = critic(s_next) if not done else torch.zeros_like(value)

            # TD target (bootstrapped Bellman target)
            td_target = torch.tensor(reward, device=device) + gamma * next_value
            advantage = td_target - value

            # Actor and Critic losses
            actor_loss = -log_prob * advantage.detach()  # detach advantage so critic doesn’t affect actor gradient
            critic_loss = advantage.pow(2)
            entropy_loss = -entropy

            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            state = next_state

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Total Reward: {total_reward:.6f}")

    return actor, critic


# --- Evaluation ---
def evaluate_actor(env, actor, device='cpu'):
    state, _ = env.reset()
    done = False
    cumulative_log_return = 0.0

    while not done:
        s_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
        concentrations = actor(s_t)
        conc = concentrations.squeeze(0).detach().cpu().numpy()
        weights = conc / (conc.sum() + 1e-8)
        next_state, reward, done, _, _ = env.step(weights)
        cumulative_log_return += np.log1p(float(reward))
        state = next_state

    return np.expm1(cumulative_log_return)


# --- Random baseline ---
def random_portfolio_baseline(df):
    returns = df.values
    num_assets = df.shape[1]
    cumulative_log_return = 0.0

    for t in range(len(returns)):
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)
        step_return = np.dot(weights, returns[t])
        step_return = np.clip(step_return, -0.10, 0.10)
        cumulative_log_return += np.log1p(step_return)

    return np.expm1(cumulative_log_return)


# --- Run ---
if __name__ == "__main__":
    df = fetch_and_preprocess_data(5, "2023-01-01", "2024-01-01")
    env = PortfolioEnv(df, window_size=5)

    actor, critic = train_actor_critic_td(env, epochs=400, gamma=0.99, print_every=50)

    rand_ret = random_portfolio_baseline(df)
    print(f"\nRandom Portfolio Total Return: {rand_ret*100:.2f}%")

    rl_ret = evaluate_actor(env, actor)
    print(f"A2C (TD(0)) Actor Total Return: {rl_ret*100:.2f}%")
