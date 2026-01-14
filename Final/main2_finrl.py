"""
Portfolio Allocation using A2C with FinRL Environment
This script uses the FinRL StockPortfolioEnv with a custom A2C implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym import spaces
import gym
from gym.utils import seeding
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


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
            # Distribute equally to all valid receivers
            weights[under] += total_excess / under.sum()
            
    # Final safety normalization
    return weights / (weights.sum() + 1e-8)
# ============================================================================
# FinRL StockPortfolioEnv (from FinRL documentation)
# ============================================================================
class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment for OpenAI gym
    
    Attributes
    ----------
    df: DataFrame
        input data with stock prices, technical indicators, and covariance
    stock_dim : int
        number of unique stocks
    hmax : int
        maximum number of shares to trade (not used in portfolio allocation)
    initial_amount : int
        start money
    transaction_cost_pct: float
        transaction cost percentage per trade
    reward_scaling: float
        scaling factor for reward, good for training
    state_space: int
        the dimension of input features (number of stocks)
    action_space: int
        equals stock dimension
    tech_indicator_list: list
        a list of technical indicator names
    turbulence_threshold: int
        a threshold to control risk aversion
    lookback: int
        lookback period for covariance calculation
    day: int
        an increment number to control date
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 df,
                 stock_dim,
                 hmax,
                 initial_amount,
                 transaction_cost_pct,
                 reward_scaling,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 turbulence_threshold=None,
                 lookback=252,
                 day=0):
        
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
        
        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        
        # observation_space: covariance matrix + technical indicators
        # Shape = (num_indicators + stock_dim, stock_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
            dtype=np.float32
        )
        
        # load data from a pandas dataframe
        # Group by unique dates to get all stocks for each day
        self.unique_dates = self.df.date.unique()
        self.data = self.df[self.df.date == self.unique_dates[self.day]]
        
        # Handle cov_list - it might be a Series or already an array
        cov_data = self.data['cov_list'].iloc[0]  # All rows for same day have same covariance
        self.covs = cov_data
        
        # Build state: covariance matrix (stock_dim x stock_dim) + technical indicators (num_indicators x stock_dim)
        # self.data is a DataFrame with stock_dim rows (one per ticker), so self.data[tech] gives a Series
        tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
        self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
        
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        
        # memorize portfolio values
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.date_memory = [self.unique_dates[self.day]]
    
    def step(self, actions):
        self.terminal = self.day >= len(self.unique_dates) - 1
        
        if self.terminal:
            # Episode ended - calculate final metrics (don't print, too spammy)
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            
            # Store final sharpe for later retrieval if needed
            if df['daily_return'].std() != 0:
                self.final_sharpe = (252**0.5) * df['daily_return'].mean() / df['daily_return'].std()
            else:
                self.final_sharpe = 0.0
            
            return self.state, self.reward, self.terminal, False, {}
        
        else:
            # actions are the portfolio weights
            # Enforce constraints robustly
            weights = enforce_portfolio_constraints(actions, max_weight=0.20)
            
            self.actions_memory.append(weights)
            
            last_day_memory = self.data
            
            # load next state
            self.day += 1
            self.data = self.df[self.df.date == self.unique_dates[self.day]]
            
            # Handle cov_list - all rows for same day have same covariance
            cov_data = self.data['cov_list'].iloc[0]
            self.covs = cov_data
            
            # Build state: covariance matrix (stock_dim x stock_dim) + technical indicators (num_indicators x stock_dim)
            tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
            self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
            
            # calculate portfolio return
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value
            
            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.unique_dates[self.day])
            self.asset_memory.append(new_portfolio_value)

            # ===== SHARPE RATIO REWARD =====
            sharpe_window = 30
            risk_free_rate = 0.0  # Assumed 0 daily for simplicity
            
            if len(self.portfolio_return_memory) >= sharpe_window:
                rolling_returns = np.array(self.portfolio_return_memory[-sharpe_window:])
                std_return = np.std(rolling_returns)
                mean_return = np.mean(rolling_returns)
                
                if std_return > 1e-6:
                    sharpe = (mean_return - risk_free_rate) / std_return
                    # We use the Sharpe Ratio directly as reward
                    # But we scale it/transform it to be a good per-step signal
                    # Option: Use Sharpe × (current_return / mean_return) to make it depend on current action
                    
                    # Simpler approach: Reward is return normalized by RECENT volatility
                    # This directly optimizes for Sharpe gradient
                    self.reward = portfolio_return / (std_return + 1e-6)
                else:
                    self.reward = portfolio_return
            else:
                # Not enough data for rolling window yet
                # Use simple return - volatility penalty
                if len(self.portfolio_return_memory) > 5:
                    vol = np.std(self.portfolio_return_memory)
                    self.reward = portfolio_return - 0.5 * vol
                else:
                    self.reward = portfolio_return
            
            # Add small concentration bonus to encourage decision making
            # (kept from previous successful fix)
            concentration_bonus = 0.05
            uniform_weight = 1.0 / self.stock_dim
            weight_deviation = np.sum(np.abs(weights - uniform_weight))
            self.reward += concentration_bonus * weight_deviation
            
            self.reward *= self.reward_scaling

            
            return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df[self.df.date == self.unique_dates[self.day]]
        
        # load states
        # Handle cov_list - all rows for same day have same covariance
        cov_data = self.data['cov_list'].iloc[0]
        self.covs = cov_data
        
        # Build state: covariance matrix (stock_dim x stock_dim) + technical indicators (num_indicators x stock_dim)
        tech_array = np.array([self.data[tech].values for tech in self.tech_indicator_list])
        self.state = np.vstack([self.covs, tech_array]).astype(np.float32)
        
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.date_memory = [self.unique_dates[self.day]]
        self.reward = 0
        
        return self.state, {}
    
    def render(self, mode='human'):
        return self.state
    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value
    
    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# ============================================================================
# Data Processing Functions (FinRL-style)
# ============================================================================
def download_data(ticker_list, start_date, end_date):
    """Download stock data using yfinance"""
    data_df = pd.DataFrame()
    
    for tic in ticker_list:
        try:
            temp_df = yf.download(tic, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # Reset index to make date a column
            temp_df = temp_df.reset_index()
            
            # Flatten multi-level columns if they exist
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in temp_df.columns.values]
            
            # Rename columns to lowercase
            temp_df.columns = temp_df.columns.str.lower()
            
            # Add ticker column
            temp_df['tic'] = tic
            
            # Standardize column names
            column_mapping = {}
            for col in temp_df.columns:
                if 'date' in col.lower():
                    column_mapping[col] = 'date'
                elif 'open' in col.lower():
                    column_mapping[col] = 'open'
                elif 'high' in col.lower():
                    column_mapping[col] = 'high'
                elif 'low' in col.lower():
                    column_mapping[col] = 'low'
                elif 'close' in col.lower() and 'adj' not in col.lower():
                    column_mapping[col] = 'close'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'volume'
            
            temp_df = temp_df.rename(columns=column_mapping)
            
            # Select only the columns we need
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
            temp_df = temp_df[[col for col in required_cols if col in temp_df.columns]]
            
            data_df = pd.concat([data_df, temp_df], ignore_index=True)
            
        except Exception as e:
            print(f"Error downloading {tic}: {e}")
            continue
    
    # Ensure date is datetime
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return data_df


def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    df = df.sort_values(by=['tic', 'date'])
    
    # Simple Moving Average
    df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
    
    # Exponential Moving Average
    df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20).mean())
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = df.groupby('tic')['close'].transform(lambda x: calculate_rsi(x))
    
    # MACD
    df['ema_12'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=12).mean())
    df['ema_26'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd'] = df['ema_12'] - df['ema_26']
    
    # Fill NaN values (bfill then forward fill any remaining)
    df = df.fillna(method='ffill').fillna(0)
    
    return df


def add_covariance_matrix(df, lookback=252):
    """Add covariance matrix to the dataframe"""
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]
    
    cov_list = []
    
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return df


# ============================================================================
# Actor-Critic Networks
# ============================================================================
class Actor(nn.Module):
    """Actor network outputs portfolio weights using Softmax (replaces Dirichlet)"""
    def __init__(self, input_dim, num_assets, hidden=256, temperature=2.5):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_assets),
        )
        
        # Break symmetry with random initialization
        with torch.no_grad():
            self.net[-1].bias.data = torch.randn(num_assets) * 0.3
    
    def forward(self, x):
        logits = self.net(x)
        # Softmax: more sensitive to differences than Dirichlet
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs


class Critic(nn.Module):
    """Critic network estimates state value"""
    def __init__(self, input_dim, hidden=256):
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


# ============================================================================
# A2C Training Loop
# ============================================================================
def train_a2c(env,
              epochs=300,
              lr=3e-2,
              gamma=0.99,
              value_coef=0.5,
              entropy_coef=0.01,
              print_every=50,
              device='cpu'):
    """
    Train A2C agent for portfolio allocation
    
    Parameters:
    -----------
    env: gym.Env
        The portfolio environment
    epochs: int
        Number of training episodes
    lr: float
        Learning rate
    gamma: float
        Discount factor
    value_coef: float
        Coefficient for value loss
    entropy_coef: float
        Coefficient for entropy bonus (encourages exploration)
    print_every: int
        Print progress every N epochs
    device: str
        'cpu' or 'cuda'
    """
    
    # Get dimensions from environment
    state_shape = env.observation_space.shape
    input_dim = np.prod(state_shape)  # Flatten the state
    num_assets = env.stock_dim
    
    # Initialize networks
    actor = Actor(input_dim, num_assets).to(device)
    critic = Critic(input_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    
    # Training statistics
    epoch_rewards = []
    
    for epoch in range(1, epochs + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Flatten state and convert to tensor
            s_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            
            # Actor: get portfolio weights from Softmax
            probs = actor(s_t).squeeze(0)
            
            # Use probabilities directly as portfolio weights
            action = probs
            
            # For policy gradient, treat as categorical distribution
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(dist.sample())
            entropy = dist.entropy()
            
            # Critic: estimate value of current state
            value = critic(s_t)
            
            # Step environment
            next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            s_next = torch.FloatTensor(next_state.flatten()).unsqueeze(0).to(device)
            
            # Critic's next value (bootstrap)
            with torch.no_grad():
                next_value = critic(s_next) if not done else torch.zeros_like(value)
            
            # TD target and advantage
            td_target = torch.tensor(reward, device=device) + gamma * next_value
            advantage = td_target - value
            
            # Compute losses
            actor_loss = -log_prob * advantage.detach()  # Policy gradient
            critic_loss = advantage.pow(2)  # MSE for TD error
            entropy_loss = -entropy  # Negative because we want to maximize entropy
            
            # Combined loss
            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
            
            # Update networks
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            
            total_reward += reward
            state = next_state
            steps += 1
        
        epoch_rewards.append(total_reward)
        
        if epoch % print_every == 0 or epoch == 1:
            avg_reward = np.mean(epoch_rewards[-print_every:])
            print(f"Epoch {epoch}/{epochs}, Total Reward: {total_reward:.6f}, "
                  f"Avg Reward (last {print_every}): {avg_reward:.6f}")
    
    return actor, critic, epoch_rewards


# ============================================================================
# Evaluation and Visualization
# ============================================================================
def evaluate_actor(env, actor, device='cpu'):
    """Evaluate the trained actor on the environment"""
    state, _ = env.reset()
    done = False
    total_return = 0.0
    actions_history = []
    
    while not done:
        s_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get probabilities directly from Softmax
            probs = actor(s_t).squeeze(0).cpu().numpy()
            # Apply constraints to evaluation weights too!
            weights = enforce_portfolio_constraints(probs, max_weight=0.20)
        
        actions_history.append(weights)
        next_state, reward, done, _, _ = env.step(weights)
        total_return += reward
        state = next_state
    
    return total_return, actions_history


def plot_training_progress(rewards, save_path='results/training_progress.png'):
    """Plot training progress"""
    import os
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
    plt.title('A2C Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    # Configuration - Diversified Portfolio Across Sectors
    # This provides lower correlation and allows agent to learn meaningful allocations
    TICKER_LIST = [
        # Technology (30%)
        'AAPL',   # Apple - Consumer Electronics
        'MSFT',   # Microsoft - Software
        'NVDA',   # Nvidia - Semiconductors
        'GOOGL',  # Google - Internet
        'META',   # Meta - Social Media
        
        # Healthcare (20%)
        'JNJ',    # Johnson & Johnson - Pharmaceuticals
        'UNH',    # UnitedHealth - Health Insurance
        'PFE',    # Pfizer - Biotech
        
        # Finance (20%)
        'JPM',    # JPMorgan - Banking
        'BAC',    # Bank of America - Banking
        'GS',     # Goldman Sachs - Investment Banking
        
        # Energy (10%)
        'XOM',    # Exxon Mobil - Oil & Gas
        'CVX',    # Chevron - Oil & Gas
        
        # Consumer (10%)
        'WMT',    # Walmart - Retail
        'PG',     # Procter & Gamble - Consumer Goods
        
        # Industrials (10%)
        'BA',     # Boeing - Aerospace
        'CAT',    # Caterpillar - Heavy Machinery
    ]
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    LOOKBACK = 120  # Increased to reduce dataset size for faster training
    INITIAL_AMOUNT = 1000000
    
    print("=" * 60)
    print("Portfolio Allocation with A2C using FinRL Environment")
    print("=" * 60)
    
    # Step 1: Download data
    print("\n[1/6] Downloading stock data...")
    df = download_data(TICKER_LIST, START_DATE, END_DATE)
    print(f"Downloaded data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Add technical indicators
    print("\n[2/6] Adding technical indicators...")
    df = add_technical_indicators(df)
    tech_indicators = ['sma_20', 'ema_20', 'rsi', 'macd']
    print(f"Technical indicators: {tech_indicators}")
    
    # Step 3: Add covariance matrix
    print("\n[3/6] Computing covariance matrices...")
    df = add_covariance_matrix(df, lookback=LOOKBACK)
    df = df.dropna().reset_index(drop=True)
    print(f"Data shape after covariance: {df.shape}")
    
    # Step 4: Create environment
    print("\n[4/6] Creating portfolio environment...")
    stock_dim = len(TICKER_LIST)
    state_space = stock_dim
    
    env = StockPortfolioEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=100,  # not used in portfolio allocation
        initial_amount=INITIAL_AMOUNT,
        transaction_cost_pct=0.001,
        reward_scaling=1.0,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=tech_indicators,
        turbulence_threshold=None,
        lookback=LOOKBACK
    )
    
    print(f"Environment created:")
    print(f"  - Stock dimension: {stock_dim}")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.shape}")
    print(f"  - Number of trading days: {len(df.index.unique())}")
    
    # Step 5: Train A2C agent
    print("\n[5/6] Training A2C agent...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    actor, critic, rewards = train_a2c(
        env=env,
        epochs=200,  # Increased from 300 for better convergence with 17 stocks
        lr=5e-4,  # Reduced further for stable learning (from 1e-3)
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,  # Re-enabled entropy to prevent premature convergence!
        print_every=50,
        device=device
    )
    
    # Step 6: Evaluate and visualize
    print("\n[6/6] Evaluating trained agent...")
    total_return, actions = evaluate_actor(env, actor, device=device)
    
    # Get portfolio performance metrics
    df_return = env.save_asset_memory()
    df_actions = env.save_action_memory()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial Portfolio Value: ${INITIAL_AMOUNT:,.2f}")
    print(f"Final Portfolio Value: ${env.portfolio_value:,.2f}")
    print(f"Total Return: {((env.portfolio_value / INITIAL_AMOUNT) - 1) * 100:.2f}%")
    print(f"Total Reward (scaled): {total_return:.6f}")
    
    # Calculate Sharpe ratio
    returns_series = pd.Series(env.portfolio_return_memory[1:])
    if returns_series.std() != 0:
        sharpe = (252 ** 0.5) * returns_series.mean() / returns_series.std()
        print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Plot results
    plot_training_progress(rewards)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Save final portfolio weights
    print("\nFinal Portfolio Allocation:")
    final_weights = actions[-1]
    for i, ticker in enumerate(TICKER_LIST):
        print(f"  {ticker}: {final_weights[i]*100:.2f}%")
