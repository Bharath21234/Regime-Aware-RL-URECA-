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
import gym
from gym import spaces
from gym.utils import seeding

# ============================================================================
# Helper Functions
# ============================================================================
def enforce_portfolio_constraints(weights, max_weight=0.20):
    """
    Robustly enforce maximum weight constraint by iteratively capping and redistributing.
    Guarantees that no weight exceeds max_weight (unless max_weight < 1/N).
    """
    weights = np.array(weights).copy()
    weights = np.clip(weights, 0, 1)
    weights = weights / (weights.sum() + 1e-8)
    
    # 5 iterations is usually plenty for convergence
    for _ in range(5):
        over = weights > max_weight
        if not over.any():
            break
            
        # Calculate excess mass
        excess = weights[over] - max_weight
        total_excess = excess.sum()
        
        # Cap the overweight stocks
        weights[over] = max_weight
        
        # Distribute excess to under-weight stocks
        under = weights < max_weight
        if under.any():
            # Distribute proportionally to preserve relative preferences
            current_mass = weights[under].sum()
            if current_mass > 0:
                weights[under] += total_excess * (weights[under] / current_mass)
            else:
                weights[under] += total_excess / under.sum()
            
    # Final safety normalization
    return weights / (weights.sum() + 1e-8)

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

# Resolving FinRL compatibility issues with local overrides
def _init_override(self, df, stock_dim, hmax, initial_amount, transaction_cost_pct, reward_scaling, state_space, action_space, tech_indicator_list, turbulence_threshold=None, lookback=252, day=0):
    self.day = day
    self.lookback = lookback
    self.df = df
    self.stock_dim = stock_dim
    self.hmax = hmax
    self.initial_amount = initial_amount
    self.transaction_cost_pct = transaction_cost_pct
    self.reward_scaling = reward_scaling
    self.state_space = state_space
    self.action_space_dim = action_space
    self.tech_indicator_list = tech_indicator_list
    self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space + len(self.tech_indicator_list), self.state_space), dtype=np.float32)
    self.unique_dates = self.df.date.unique()
    self.data = self.df[self.df.date == self.unique_dates[self.day]]
    self.covs = self.data["cov_list"].iloc[0] # Fixed access
    tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
    self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
    self.terminal = False
    self.turbulence_threshold = turbulence_threshold
    self.portfolio_value = self.initial_amount
    self.asset_memory = [self.initial_amount]
    self.portfolio_return_memory = [0]
    self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
    self.date_memory = [self.unique_dates[self.day]]

def _step_override(self, actions):
    self.terminal = self.day >= len(self.unique_dates) - 1
    if self.terminal:
        df = pd.DataFrame(self.portfolio_return_memory)
        df.columns = ['daily_return']
        if df['daily_return'].std() != 0:
            self.sharpe = (252**0.5) * df['daily_return'].mean() / df['daily_return'].std()
        return self.state, self.reward, self.terminal, False, {}
    else:
        try:
            weights = enforce_portfolio_constraints(actions, max_weight=0.20)
        except:
            weights = actions
        self.actions_memory.append(weights)
        last_day_memory = self.data
        self.day += 1
        self.data = self.df[self.df.date == self.unique_dates[self.day]]
        self.covs = self.data["cov_list"].iloc[0] # Fixed
        tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
        self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
        portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.unique_dates[self.day])
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        self.reward = portfolio_return * self.reward_scaling
        return self.state, self.reward, self.terminal, False, {}

def _reset_override(self, seed=None, options=None):
    if seed is not None:
        self._seed(seed)
    self.asset_memory = [self.initial_amount]
    self.day = 0
    self.data = self.df[self.df.date == self.unique_dates[self.day]]
    self.covs = self.data["cov_list"].iloc[0]
    tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
    self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
    self.portfolio_value = self.initial_amount
    self.terminal = False
    self.portfolio_return_memory = [0]
    self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
    self.date_memory = [self.unique_dates[self.day]]
    return self.state, {}

def _seed_override(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

# Inject overrides
StockPortfolioEnv.__init__ = _init_override
StockPortfolioEnv.step = _step_override
StockPortfolioEnv.reset = _reset_override
StockPortfolioEnv._seed = _seed_override

# ============================================================================
# Configuration
# ============================================================================
TICKER_LIST = [
    'AAPL','MSFT','NVDA','GOOGL','META',
    'JNJ','UNH','PFE',
    'JPM','BAC','GS',
    'XOM','CVX',
    'WMT','PG',
    'BA','CAT',
    # Added even more stocks for a broader universe (~40)
    'AMZN', 'AMD', 'NFLX',  # Tech/Growth
    'V', 'HD', 'MCD',       # Consumer/Finance
    'KO', 'PEP',            # Defensive/Staples
    'DIS', 'COST',          # Entertainment/Retail
    'CRM', 'INTC', 'TXN',   # Semiconductors/Software
    'GE', 'MMM', 'HON',     # Industrials
    'C', 'GS', 'MS',        # Investment Banking
    'ABT', 'ABBV', 'MRK'    # More Healthcare
]

START_DATE = "2015-01-01"
END_DATE   = "2024-01-01"
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

# Add Covariance Matrix (Required for StockPortfolioEnv)
def add_covariance_matrix(df, lookback=252):
    """Add covariance matrix to the dataframe"""
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]
    
    cov_list = []
    
    # Iterate over unique dates
    unique_dates = df.date.unique()
    for i in range(lookback, len(unique_dates)):
        # Select lookback window
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date': unique_dates[lookback:], 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df

print("\nComputing covariance matrix...")
df = add_covariance_matrix(df, lookback=252)
print(f"Data shape with covariance: {df.shape}")

# Debug prints before env instantiation
print("="*50)
print(f"DEBUG: Initializing StockPortfolioEnv")
print(f"DEBUG: df type: {type(df)}")
print(f"DEBUG: df shape: {df.shape}")
print(f"DEBUG: df head:\n{df.head()}")
print(f"DEBUG: Ticker list length: {len(TICKER_LIST)}")
print("="*50)

env = StockPortfolioEnv(
    df=df,
    stock_dim=len(TICKER_LIST),
    hmax=100,  # Required but not used in portfolio allocation
    initial_amount=INITIAL_AMOUNT,
    transaction_cost_pct=0.001,
    reward_scaling=1000.0, # Increased massively to force learning (was 100.0)
    state_space=len(TICKER_LIST),  # Simplified state space (dim)
    action_space=len(TICKER_LIST), # Action is weights vector
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    lookback=252, # Required argument
    day=0 # Required argument
)

# ============================================================================
# Actor-Critic Networks
# ============================================================================
class Actor(nn.Module):
    """Actor network outputs Dirichlet concentration parameters (alpha)"""
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
        # Softplus ensures alpha > 0. We add 1.0 to bias towards uniform initially (concentration > 1)
        logits = self.net(x)
        alpha = torch.nn.functional.softplus(logits) + 1.0
        return alpha


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
    epochs=200, # Reduced to 200 as requested
    gamma=0.99,
    lr=1e-4,    # Slightly lower for Dirichlet stability
    value_coef=0.5,
    entropy_coef=0.01,
    batch_size=64
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

        # Rollout buffer
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []

        while not done:
            s = torch.tensor(
                state.flatten(),
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            # Actor & Critic
            alpha = actor(s)
            value = critic(s)

            # Dirichlet distribution for exploration
            dist = torch.distributions.Dirichlet(alpha)
            # Sample weights (sum to 1)
            weights = dist.sample()
            log_prob = dist.log_prob(weights)
            entropy = dist.entropy()

            # Environment step
            action = weights.detach().cpu().numpy()[0]
            next_state, reward, done, _, _ = env.step(action)

            # Store in buffer
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=DEVICE))
            masks.append(torch.tensor([1 - float(done)], dtype=torch.float32, device=DEVICE))
            entropies.append(entropy)

            state = next_state
            ep_reward += reward

            # Update if batch is full or episode ends
            if len(rewards) >= batch_size or done:
                with torch.no_grad():
                    s_next = torch.tensor(
                        next_state.flatten(),
                        dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)
                    next_value = critic(s_next) if not done else torch.zeros(1, device=DEVICE)

                # Calculate returns (bootstrapped)
                returns = []
                R = next_value.squeeze()
                for r, m in zip(reversed(rewards), reversed(masks)):
                    R = r + gamma * R * m
                    returns.insert(0, R)
                
                returns = torch.stack(returns).squeeze()
                values_tensor = torch.cat(values).squeeze()
                log_probs_tensor = torch.cat(log_probs).squeeze()
                entropies_tensor = torch.cat(entropies).squeeze()
                
                # Advantage
                advantages = returns - values_tensor
                
                # Losses
                actor_loss = -(log_probs_tensor * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropies_tensor.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clear buffer
                log_probs, values, rewards, masks, entropies = [], [], [], [], []

        rewards_history.append(ep_reward)

        if ep % 20 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:.6f}")

    return actor, critic, rewards_history

def plot_training_progress(rewards, save_path='results/finrlmain_training.png'):
    """Plot training progress"""
    import os
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Add moving average
    window = 20
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(moving_avg, linewidth=2, label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C Training Progress (FinRL Main)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")

# ============================================================================
# Train Agent
# ============================================================================
actor, critic, rewards = train_a2c(env)
plot_training_progress(rewards)

# ============================================================================
# Evaluation
# ============================================================================
print("\n[Evaluating trained agent...]")
state, _ = env.reset()
done = False
final_weights = None

while not done:
    s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Use Softmax Actor logic if applicable, otherwise default
    with torch.no_grad():
        # Use Dirichlet Actor logic (concentration parameters)
        alpha = actor(s)
        # Use MEAN of Dirichlet for evaluation (deterministic)
        weights = (alpha / alpha.sum(dim=-1, keepdim=True)).cpu().numpy()[0]
             
        # Apply constraints for evaluation
        weights = enforce_portfolio_constraints(weights, max_weight=0.20)
        final_weights = weights # Keep track of last weights
        
    state, _, done, _, _ = env.step(weights)

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Final Portfolio Value: ${env.portfolio_value:,.2f}")
print(f"Total Return: {(env.portfolio_value / INITIAL_AMOUNT - 1) * 100:.2f}%")

# Calculate Sharpe
returns_df = pd.DataFrame(env.portfolio_return_memory)
if returns_df.std().iloc[0] != 0:
    sharpe = (252**0.5) * returns_df.mean().iloc[0] / returns_df.std().iloc[0]
    print(f"Sharpe Ratio: {sharpe:.4f}")

print("\nFinal Portfolio Allocation:")
if final_weights is not None:
    for i, ticker in enumerate(TICKER_LIST):
        print(f"  {ticker}: {final_weights[i]*100:.2f}%")
print("=" * 60)
