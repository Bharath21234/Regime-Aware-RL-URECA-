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
from gym import spaces
from gym.utils import seeding
from hmm import MarketRegimeHMM, plot_regimes
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
    
    # Store prices and technical indicators in a dictionary of matrices keyed by date
    # Or better yet, a 3D NumPy array [num_dates, num_indicators, num_stocks]
    # For now, let's use a dictionary of pre-extracted rows to keep it simple but fast
    self.data_dict = {}
    self.covs_dict = {}
    self.regime_dict = {}
    
    regime_df = kwargs.get('regime_df', None)
    
    # Group by date once
    for date, group in df.groupby('date'):
        # Sort by tic to ensure consistency
        group = group.sort_values('tic')
        self.covs_dict[date] = group["cov_list"].iloc[0]
        self.data_dict[date] = group
        # Technical indicators as matrix
        tech_array = np.array([group[tech].values for tech in self.tech_indicator_list])
        tech_array = np.nan_to_num(tech_array) # Safety check
        
        # Combine covs and tech safely
        covs = np.nan_to_num(self.covs_dict[date])
        state_base = np.vstack([covs, tech_array]).astype(np.float32)
        
        # Add regime info
        current_regime = 0
        if regime_df is not None:
            regime_match = regime_df.loc[regime_df.date == date, 'future_regime']
            if not regime_match.empty:
                current_regime = regime_match.values[0]
        
        regime_row = np.full((1, state_base.shape[1]), current_regime).astype(np.float32)
        self.regime_dict[date] = np.vstack([state_base, regime_row])

    # Initial state
    date = self.unique_dates[self.day]
    self.state = self.regime_dict[date]
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
            weights = enforce_portfolio_constraints(actions, max_weight=0.60)
        except:
            weights = actions
        self.actions_memory.append(weights)
        
        last_date = self.unique_dates[self.day]
        last_day_data = self.data_dict[last_date]
        
        self.day += 1
        current_date = self.unique_dates[self.day]
        current_day_data = self.data_dict[current_date]
        
        self.state = self.regime_dict[current_date]
        
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
    self.state = self.regime_dict[date]
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

START_DATE = "2015-01-01"
END_DATE   = "2024-01-01"
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

# --- NEW: Fetch Exogenous Features (Benchmarks) ---
print("\nFetching Exogenous Benchmarks for HMM...")
EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
df_exo = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=EXO_TICKERS
).fetch_data()
# Clean and align exogenous data
df_exo = df_exo.sort_values(["date", "tic"]).reset_index(drop=True)
# --------------------------------------------------

# --- NEW: Robust Data Cleaning ---
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

# --- NEW: HMM Regime Fitting ---
print("\nFitting HMM for regime detection (4-Regime Setup)...")
hmm_model = MarketRegimeHMM(n_regimes=4)
hmm_model.fit(df_exo)
regime_df = hmm_model.predict(df_exo)

# --- NEW: Predict Future Regime ---
print("Predicting future regimes...")
regime_df['future_regime'] = regime_df['regime'].apply(lambda x: hmm_model.predict_next_regime(x))
# ----------------------------------

plot_regimes(df, regime_df)
# -------------------------------

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
    reward_scaling=1000.0,
    state_space=len(TICKER_LIST),
    action_space=len(TICKER_LIST),
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    lookback=20,
    day=0,
    regime_df=regime_df # Pass regimes to env
)

# Update observation space shape in env object manually due to augmentation
env.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, 
    shape=(env.lookback, env.state.shape[0], env.state.shape[1]), 
    dtype=np.float32
)

# ============================================================================
# Actor-Critic Networks
# ============================================================================
class Actor(nn.Module):
    """Multi-Agent Actor: Switches between 3 regime-specialized heads"""
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        # Share the feature extractor (lower layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Specialist heads for 4 regimes (Bull, Sideways Up, Sideways Down, Bear)
        self.heads = nn.ModuleList([
            nn.Linear(hidden, num_assets) for _ in range(4)
        ])

    def forward(self, x):
        # Extract regime from state (last row, first element - we filled it with regime index)
        # Note: x shape is [batch, obs_dim]
        # Our obs_dim mapping: state matrix flattened
        # Let's be explicit: regime is stored in the last element of each flattened batch entry (rough approximation if flattened)
        # Better: keep track of where the regime info is. 
        # In our env, it's the last row of the matrix.
        
        # Selection logic (Multi-agent routing)
        # Regime info is in the last row of the LAST state in the sequence
        # x shape: [batch, window, features, assets] -> flattened to [batch, window*features*assets]
        # In our case, features_per_day = features * assets
        # The very last element of the flattened vector for each batch is the regime of the current day
        regime_indices = x[:, -1].long() 
        
        features = self.feature_extractor(x)
        
        # Selection logic (Multi-agent routing)
        # To support batching, we process each item according to its regime
        out = torch.zeros(x.shape[0], self.heads[0].out_features, device=x.device)
        for i in range(4):
            mask = (regime_indices == i)
            if mask.any():
                out[mask] = self.heads[i](features[mask])
        
        alpha = torch.nn.functional.softplus(out) + 1.0
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
    entropy_coef=0.01, # Increased from 0.01 to 0.05 to encourage spread
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
    plt.title('A2C Training Progress (FinRL Main)')
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
    
    plt.title('Portfolio Allocation History (Weight Evolution)', fontsize=14)
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
actor, critic, rewards = train_a2c(env)
plot_training_progress(rewards)

# ============================================================================
# Evaluation
# ============================================================================
print("\n[Evaluating trained agent...]")
state, _ = env.reset()
done = False
final_weights = None
all_weights = []

while not done:
    s = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Use Dirichlet Actor logic (concentration parameters)
        alpha = actor(s)
        # Use MEAN of Dirichlet for evaluation (deterministic)
        weights = (alpha / alpha.sum(dim=-1, keepdim=True)).cpu().numpy()[0]
             
        # Apply constraints for evaluation
        weights = enforce_portfolio_constraints(weights, max_weight=0.60)
        final_weights = weights # Keep track of last weights
        all_weights.append(weights)
        
    state, _, done, _, _ = env.step(weights)

# Plot allocation history
df_weights = pd.DataFrame(all_weights, columns=TICKER_LIST)
# If dates are available in env, use them
if hasattr(env, 'date_memory') and len(env.date_memory) > len(all_weights):
    df_weights.index = env.date_memory[1:len(all_weights)+1]
plot_portfolio_allocation(df_weights)

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
