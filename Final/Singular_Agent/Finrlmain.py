"""
Portfolio Allocation using A2C with FinRL Official Environment
Single Agent Baseline (No HMM, No Multi-Head Routing)

- Uses FinRL's StockPortfolioEnv (official)
- Custom A2C implementation (single actor head)
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
from gym import spaces
from gym.utils import seeding
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================================================
# Helper Functions
# ============================================================================
def enforce_portfolio_constraints(weights, max_weight=0.60): # Lowered from 0.20 to 0.10 for spread
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
def _init_override(self, df, stock_dim, hmax, initial_amount, transaction_cost_pct, reward_scaling, state_space, action_space, tech_indicator_list, turbulence_threshold=None, lookback=252, day=0, **kwargs):
    self.day = day
    self.lookback = lookback
    self.stock_dim = stock_dim
    self.hmax = hmax
    self.initial_amount = initial_amount
    self.transaction_cost_pct = transaction_cost_pct
    self.reward_scaling = reward_scaling
    self.state_space = state_space
    self.action_space_dim = action_space
    self.tech_indicator_list = tech_indicator_list
    self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.float32)
    self.terminal = False
    self.turbulence_threshold = turbulence_threshold
    
    # Pre-process data into NumPy arrays for MUCH faster lookup
    self.unique_dates = df.date.unique()
    
    self.data_dict = {}
    self.covs_dict = {}
    self.state_dict = {}
    
    # Group by date once
    for date, group in df.groupby('date'):
        # Sort by tic to ensure consistency
        group = group.sort_values('tic')
        self.covs_dict[date] = group["cov_list"].iloc[0]
        self.data_dict[date] = group
        # Technical indicators as matrix
        tech_array = np.array([group[tech].values for tech in self.tech_indicator_list])
        tech_array = np.nan_to_num(tech_array) # Safety check
        
        # Combine covs and tech safely (NO regime row for single agent)
        covs = np.nan_to_num(self.covs_dict[date])
        state_base = np.vstack([covs, tech_array]).astype(np.float32)
        self.state_dict[date] = state_base

    # Initial state
    date = self.unique_dates[self.day]
    self.state = self.state_dict[date]
    self.state_memory = [self.state] * self.lookback
    self.portfolio_value = self.initial_amount
    self.asset_memory = [self.initial_amount]
    self.portfolio_return_memory = [0]
    self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
    self.date_memory = [date]

def _step_override(self, actions):
    self.terminal = self.day >= len(self.unique_dates) - 1
    if self.terminal:
        df_rets = pd.DataFrame(self.portfolio_return_memory)
        df_rets.columns = ['daily_return']
        if df_rets['daily_return'].std() != 0:
            self.sharpe = (252**0.5) * df_rets['daily_return'].mean() / df_rets['daily_return'].std()
        return np.array(self.state_memory), self.reward, self.terminal, False, {}
    else:
        try:
            weights = enforce_portfolio_constraints(actions, max_weight=0.30)
        except:
            weights = actions
        self.actions_memory.append(weights)
        
        last_date = self.unique_dates[self.day]
        last_day_data = self.data_dict[last_date]
        
        self.day += 1
        current_date = self.unique_dates[self.day]
        current_day_data = self.data_dict[current_date]
        
        self.state = self.state_dict[current_date]
        
        # Update state memory
        self.state_memory.pop(0)
        self.state_memory.append(self.state)

        # Vectorized portfolio return calculation
        # last_day_data and current_day_data are sorted by tic
        portfolio_return = np.sum(((current_day_data.close.values / last_day_data.close.values) - 1) * weights)
        
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(current_date)
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        self.reward = portfolio_return * self.reward_scaling
        return np.array(self.state_memory), self.reward, self.terminal, False, {}

def _reset_override(self, seed=None, options=None):
    if seed is not None:
        self._seed(seed)
    self.asset_memory = [self.initial_amount]
    self.day = 0
    date = self.unique_dates[self.day]
    self.state = self.state_dict[date]
    self.state_memory = [self.state] * self.lookback
    self.portfolio_value = self.initial_amount
    self.terminal = False
    self.portfolio_return_memory = [0]
    self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
    self.date_memory = [date]
    return np.array(self.state_memory), {}

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
TICKER_LIST = sorted(list(set(TICKER_LIST))) # Deduplicate and sort

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-01-01"
INITIAL_AMOUNT = 1_000_000

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# FinRL Data Pipeline (OFFICIAL)
# ============================================================================
df = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TEST_END,
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

# --- Robust Data Cleaning ---
print("\nCleaning data for consistency...")
df = df.dropna()
df = df.drop_duplicates(subset=["date", "tic"])
# Ensure all tickers have data for the same set of dates
df_counts = df.groupby('tic').size()
max_counts = df_counts.max()
# Filter tickers that have the maximum number of observations
tickers_to_keep = df_counts[df_counts == max_counts].index.tolist()
if len(tickers_to_keep) < len(df_counts):
    print(f"Dropping {len(df_counts) - len(tickers_to_keep)} tickers with incomplete data.")
df = df[df.tic.isin(tickers_to_keep)]
df = df.sort_values(["date", "tic"]).reset_index(drop=True)
TICKER_LIST = sorted(df.tic.unique().tolist()) # Update list to match actual data
# ---------------------------------

# Add Covariance Matrix (Required for StockPortfolioEnv)
def add_covariance_matrix(df, lookback=20):
    """
    Optimized covariance matrix calculation.
    Pivots once and uses a rolling window on NumPy arrays.
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    
    # Pivot to get a matrix of prices: dates as rows, tickers as columns
    price_pivot = df.pivot_table(index='date', columns='tic', values='close')
    # Use ffill and bfill to handle potential missing values
    price_pivot = price_pivot.ffill().bfill()
    
    # Calculate daily returns
    returns = price_pivot.pct_change()
    
    unique_dates = returns.index.tolist()
    cov_list = []
    
    # Pre-calculate covariance matrices using rolling window on NumPy array
    returns_np = returns.values
    
    for i in range(len(unique_dates)):
        # Ensure we have at least 2 rows for covariance, otherwise use identity or zero
        if i < 2:
            cov = np.zeros((returns_np.shape[1], returns_np.shape[1]))
        else:
            start_idx = max(0, i - lookback + 1)
            window = returns_np[start_idx:i+1]
            # Filter out NaNs in the window if any
            window = window[~np.isnan(window).any(axis=1)]
            
            if window.shape[0] < 2:
                cov = np.zeros((returns_np.shape[1], returns_np.shape[1]))
            else:
                cov = np.cov(window, rowvar=False)
        
        # Replace any remaining NaNs (e.g. from constant returns) with 0
        cov = np.nan_to_num(cov)
        cov_list.append(cov)
    
    # Create a mapping of date to covariance matrix
    df_cov = pd.DataFrame({'date': unique_dates, 'cov_list': cov_list})
    
    # Merge back to original dataframe (drops the first row since returns[0] is NaN)
    df = df.merge(df_cov, on='date')
    df = df.dropna(subset=['cov_list'])
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df

print("\nComputing covariance matrix...")
df = add_covariance_matrix(df, lookback=20)
print(f"Data shape with covariance: {df.shape}")

print("="*50)
print(f"DEBUG: Initializing StockPortfolioEnv")
print(f"DEBUG: df type: {type(df)}")
print(f"DEBUG: df shape: {df.shape}")
print(f"DEBUG: df head:\n{df.head()}")
print(f"DEBUG: Ticker list length: {len(TICKER_LIST)}")
print("="*50)

print("\nSplitting into Train and Test Data...")
train_df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df = df[(df.date >= TEST_START) & (df.date <= TEST_END)].reset_index(drop=True)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape:  {test_df.shape}")

# Create separate environments for training and testing
env_train = StockPortfolioEnv(
    df=train_df,
    stock_dim=len(TICKER_LIST),
    hmax=100,  # Required but not used in portfolio allocation
    initial_amount=INITIAL_AMOUNT,
    transaction_cost_pct=0.001,
    reward_scaling=1000.0,
    state_space=len(TICKER_LIST),
    action_space=len(TICKER_LIST),
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    lookback=20,
    day=0,
)

env_train.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, 
    shape=(env_train.lookback, env_train.state.shape[0], env_train.state.shape[1]), 
    dtype=np.float32
)

env_test = StockPortfolioEnv(
    df=test_df,
    stock_dim=len(TICKER_LIST),
    hmax=100,
    initial_amount=INITIAL_AMOUNT,
    transaction_cost_pct=0.001,
    reward_scaling=1000.0,
    state_space=len(TICKER_LIST),
    action_space=len(TICKER_LIST),
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    lookback=20,
    day=0,
)

env_test.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, 
    shape=(env_test.lookback, env_test.state.shape[0], env_test.state.shape[1]), 
    dtype=np.float32
)

# ============================================================================
# Actor-Critic Networks
# ============================================================================
class Actor(nn.Module):
    """Single Actor: One head for all market conditions (no regime routing)"""
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        # Feature extractor (same architecture as multi-agent version)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Single output head (no regime-based routing)
        self.head = nn.Linear(hidden, num_assets)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.head(features)
        
        # Normalize across stocks to amplify relative differences
        # (raw linear outputs are clustered too close for softplus to differentiate)
        out_mean = out.mean(dim=-1, keepdim=True)
        out_std = out.std(dim=-1, keepdim=True) + 1e-6
        out_normalized = (out - out_mean) / out_std
        out_normalized = torch.clamp(out_normalized, -3, 3)  # Prevent extreme values
        alpha = torch.exp(out_normalized) + 0.1  # alpha range: ~0.15 to ~20.2
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
    entropy_coef=0.001, # Low: with 38 stocks, entropy bonus pushes toward uniform if too high
    batch_size=20
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

        # Rollout buffer (Data storage)
        states_buffer = []
        weights_buffer = []
        rewards_buffer = []
        masks_buffer = []

        while not done:
            # Efficiently convert state to tensor
            s_input = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(DEVICE)

            # Actor & Critic (No grad for sampling)
            with torch.no_grad():
                alpha = actor(s_input)
                # Dirichlet sampling fallback (MPS doesn't support it yet)
                dist = torch.distributions.Dirichlet(alpha.cpu())
                weights = dist.sample().to(DEVICE)
            
            # Environment step
            action = weights.cpu().numpy()[0]
            next_state, reward, done, _, _ = env.step(action)

            # Store in buffer
            states_buffer.append(s_input)
            weights_buffer.append(weights)
            rewards_buffer.append(reward)
            masks_buffer.append(1.0 - float(done))

            state = next_state
            ep_reward += reward

            # Update if buffer is full (Rolling Update)
            if len(rewards_buffer) >= batch_size:
                # Prepare trajectory tensors (Minimize concat)
                b_states = torch.cat(states_buffer) # [batch, flattened_features]
                b_weights = torch.cat(weights_buffer) # [batch, num_assets]
                
                # Convert list to tensor on device once
                b_rewards = torch.tensor(rewards_buffer, dtype=torch.float32, device=DEVICE)
                b_masks = torch.tensor(masks_buffer, dtype=torch.float32, device=DEVICE)

                # Re-evaluate graph for the entire window
                alpha_batch = actor(b_states)
                values_tensor = critic(b_states).squeeze()
                
                dist_batch = torch.distributions.Dirichlet(alpha_batch)
                log_probs_tensor = dist_batch.log_prob(b_weights)
                entropies_tensor = dist_batch.entropy()

                # Calculate returns (bootstrapped)
                with torch.no_grad():
                    s_next = torch.from_numpy(next_state.flatten()).float().unsqueeze(0).to(DEVICE)
                    next_value = critic(s_next) if not done else torch.zeros(1, 1, device=DEVICE)
                
                returns = []
                R = next_value.squeeze()
                # Use reversed loop efficiently
                for r, m in zip(reversed(rewards_buffer), reversed(masks_buffer)):
                    R = r + gamma * R * m
                    returns.insert(0, R)
                
                returns = torch.stack(returns).squeeze()
                
                # Advantage (Normalize for stability)
                advantages = returns - values_tensor
                
                # Losses
                actor_loss = -(log_probs_tensor * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropies_tensor.mean()

                optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients (stability)
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()

                # SLIDE BY 1: Remove the oldest entry
                states_buffer.pop(0)
                weights_buffer.pop(0)
                rewards_buffer.pop(0)
                masks_buffer.pop(0)
            
            if done:
                # Clear buffer at end of episode
                states_buffer, weights_buffer, rewards_buffer, masks_buffer = [], [], [], []

        rewards_history.append(ep_reward)

        if ep % 10 == 0:
            print(f"Episode {ep:04d} | Reward: {ep_reward:.6f} | Value: {R.item():.4f}")

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
    plt.title('A2C Training Progress (Single Agent Baseline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")
    
def plot_portfolio_allocation(df_weights, save_path='results/allocation_history.png'):
    """Plot portfolio allocation over time using a stacked area chart"""
    import os
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    # Filter tickers that had at least some allocation (to keep legend clean)
    ticker_cols = df_weights.columns
    # Sort by average allocation to make the plot look better
    avg_alloc = df_weights.mean().sort_values(ascending=False)
    sorted_tickers = avg_alloc.index.tolist()
    
    plt.stackplot(df_weights.index, 
                  [df_weights[tic] for tic in sorted_tickers], 
                  labels=sorted_tickers, 
                  alpha=0.8)
    
    plt.title('Portfolio Allocation History (Single Agent Baseline)', fontsize=14)
    plt.xlabel('Date / Step', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    # Put legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Allocation history plot saved to {save_path}")

# ============================================================================
# Train Agent
# ============================================================================
actor, critic, rewards = train_a2c(env_train)
plot_training_progress(rewards)

# ============================================================================
# Evaluation
# ============================================================================
print("\n[Evaluating trained agent on Test Set (Out-of-Sample)...]")
state, _ = env_test.reset()
done = False
final_weights = None
all_weights = []

while not done:
    s = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Use Dirichlet Actor logic (concentration parameters)
        alpha = actor(s)
        # Temperature-sharpened allocation (Dirichlet mean is too uniform with many stocks)
        log_alpha = torch.log(alpha)
        weights = torch.softmax(log_alpha / 0.5, dim=-1).cpu().numpy()[0]  # temp=0.5 (alphas now naturally varied)
             
        # Apply constraints for evaluation
        weights = enforce_portfolio_constraints(weights, max_weight=0.30)
        final_weights = weights # Keep track of last weights
        all_weights.append(weights)
        
    state, _, done, _, _ = env_test.step(weights)

# Plot allocation history
df_weights = pd.DataFrame(all_weights, columns=TICKER_LIST)
# If dates are available in env, use them
if hasattr(env_test, 'date_memory') and len(env_test.date_memory) > len(all_weights):
    df_weights.index = env_test.date_memory[1:len(all_weights)+1]
plot_portfolio_allocation(df_weights)

print("=" * 60)
print("OUT-OF-SAMPLE RESULTS")
print("=" * 60)
print(f"Final Portfolio Value: ${env_test.portfolio_value:,.2f}")
print(f"Total Return: {(env_test.portfolio_value / INITIAL_AMOUNT - 1) * 100:.2f}%")

# Calculate Sharpe
returns_df = pd.DataFrame(env_test.portfolio_return_memory)
if returns_df.std().iloc[0] != 0:
    sharpe = (252**0.5) * returns_df.mean().iloc[0] / returns_df.std().iloc[0]
    print(f"Sharpe Ratio: {sharpe:.4f}")

print("\nFinal Portfolio Allocation:")
if final_weights is not None:
    for i, ticker in enumerate(TICKER_LIST):
        print(f"  {ticker}: {final_weights[i]*100:.2f}%")
print("=" * 60)
