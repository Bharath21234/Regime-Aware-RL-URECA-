"""
Main Entry point for Probabilistic MoE FinRL System.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from hmm_probabilistic import ProbabilisticHMM, plot_regime_probs
from agents_moe import ActorMoE, Critic
from env_mixture import MixturePortfolioEnv, enforce_portfolio_constraints

# --- Config ---
# Restoring full TICKER_LIST from baseline
TICKER_LIST = [
    'AAPL','MSFT','NVDA','GOOGL','META',
    'JNJ','UNH','PFE','JPM','BAC','GS',
    'XOM','CVX','WMT','PG','BA','CAT',
    'AMZN', 'AMD', 'NFLX', 'V', 'HD', 'MCD',
    'KO', 'PEP', 'DIS', 'COST', 'CRM', 'INTC', 'TXN',
    'GE', 'MMM', 'HON', 'C', 'MS', 'ABT', 'ABBV', 'MRK'
]
TICKERS = sorted(list(set(TICKER_LIST)))

TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-01-01"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# Data Pipeline
# ============================================================================
print(f"Fetching data for {len(TICKERS)} stocks...")
df = YahooDownloader(start_date=TRAIN_START, end_date=TEST_END, ticker_list=TICKERS).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=["macd", "rsi", "cci", "adx"])
df = fe.preprocess_data(df)

# Dedup and Clean
df = df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)

# Ensure data consistency (matching stock_dim)
df_counts = df.groupby('tic').size()
max_counts = df_counts.max()
tickers_to_keep = df_counts[df_counts == max_counts].index.tolist()
df = df[df.tic.isin(tickers_to_keep)]
df = df.sort_values(["date", "tic"]).reset_index(drop=True)
TICKERS = sorted(df.tic.unique().tolist())

# Covariance
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

print("Computing covariance matrices...")
df = add_covariance_matrix(df, lookback=20)

# --- NEW: Fetch Exogenous Features (Macro Benchmarks for HMM) ---
print("\nFetching Exogenous Benchmarks for Macro HMM...")
EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
df_exo = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TEST_END,
    ticker_list=EXO_TICKERS
).fetch_data()
df_exo = df_exo.sort_values(["date", "tic"]).reset_index(drop=True)

print("\nSplitting into Train and Test Data...")
train_df_exo = df_exo[(df_exo.date >= TRAIN_START) & (df_exo.date <= TRAIN_END)].reset_index(drop=True)

# HMM
print("Fitting Probabilistic Macro HMM on TRAIN DATA ONLY...")
hmm = ProbabilisticHMM(n_regimes=4)
hmm.fit(train_df_exo)

print("Predicting probabilities for the entire dataset...")
prob_df = hmm.predict_proba(df_exo)
plot_regime_probs(prob_df, save_path='results/regime_probabilities.png')

# Align dates between df and prob_df to prevent IndexError
common_dates = set(df['date']).intersection(set(prob_df['date']))
df = df[df['date'].isin(common_dates)].reset_index(drop=True)
prob_df = prob_df[prob_df['date'].isin(common_dates)].reset_index(drop=True)

train_df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = df[(df.date >= TEST_START)  & (df.date <= TEST_END)].reset_index(drop=True)
prob_df_train = prob_df[(prob_df.date >= TRAIN_START) & (prob_df.date <= TRAIN_END)].reset_index(drop=True)
prob_df_test  = prob_df[(prob_df.date >= TEST_START)  & (prob_df.date <= TEST_END)].reset_index(drop=True)

print(f"DEBUG: train_df shape: {train_df.shape}")
print(f"DEBUG: test_df shape:  {test_df.shape}")

# Env Train
env_train = MixturePortfolioEnv(
    df=train_df, 
    stock_dim=len(TICKERS), 
    initial_amount=1000000, 
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df_train,
    reward_scaling=1000.0,
    lookback=20
)

# Env Test
env_test = MixturePortfolioEnv(
    df=test_df, 
    stock_dim=len(TICKERS), 
    initial_amount=1000000, 
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df_test,
    reward_scaling=1000.0,
    lookback=20
)

# ============================================================================
# Training (A2C Core)
# ============================================================================
def train(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = ActorMoE(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    epochs     = 200
    gamma      = 0.99
    batch_size = 20
    value_coef    = 0.5
    entropy_coef  = 0.01
    l2_coef       = 0.5   # aligned with Baseline and Select_1

    rewards_history = []

    print(f"Starting MoE Training on {DEVICE} for {epochs} epochs...")
    for ep in range(epochs):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        # slide-by-1 rolling buffers — IDENTICAL structure to Baseline and Select_1
        s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(
                    mean.cpu(), std.cpu()
                ).sample().to(DEVICE)

            action_np = w_raw.cpu().numpy()[0]
            next_state, reward, done, _, _ = env.step(action_np)

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
                    + value_coef   * adv.pow(2).mean()
                    - entropy_coef * ent.mean()
                    + l2_coef      * (mean_b ** 2).mean()
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                optimizer.step()

                # slide buffer by 1
                for buf in (s_buf, w_buf, r_buf, m_buf, mean_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        rewards_history.append(ep_reward)
        if ep % 10 == 0:
            print(f"[MoE] Ep {ep:04d} | Reward: {ep_reward:.4f} | PortVal: ${env.portfolio_value:,.2f}")

    return actor, critic, rewards_history

def plot_training_progress(rewards, save_path='results/finrlmain_training.png'):
    """Plot training progress"""
    import os
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    
    window = 20
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(moving_avg, linewidth=2, label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('MoE A2C Training Progress')
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
    ticker_cols = df_weights.columns
    avg_alloc = df_weights.mean().sort_values(ascending=False)
    sorted_tickers = avg_alloc.index.tolist()
    
    plt.stackplot(df_weights.index, 
                  [df_weights[tic] for tic in sorted_tickers], 
                  labels=sorted_tickers, 
                  alpha=0.8)
    
    plt.title('Portfolio Allocation History (MoE Evolution)', fontsize=14)
    plt.xlabel('Date / Step', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Allocation history plot saved to {save_path}")

if __name__ == "__main__":
    actor, critic, rewards = train(env_train)
    plot_training_progress(rewards)

    # --- EVALUATION ---
    print("\n[Evaluating MoE trained agent on Test Set (Out-of-Sample)...]")
    state, _ = env_test.reset()
    done = False
    final_weights = None
    all_weights = []
    
    actor.eval()
    with torch.no_grad():
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _ = actor(s_tensor)   # greedy mean at eval time
            weights_raw = mean.detach().cpu().numpy()[0]
            final_weights = enforce_portfolio_constraints(weights_raw)
            all_weights.append(final_weights)
            state, _, done, _, _ = env_test.step(weights_raw)
            
    # Plot allocation history
    df_weights = pd.DataFrame(all_weights, columns=TICKERS)
    if hasattr(env_test, 'unique_dates') and len(env_test.unique_dates) > len(all_weights):
        df_weights.index = env_test.unique_dates[1:len(all_weights)+1]
    plot_portfolio_allocation(df_weights)

    # ---- NEW: Wealth over time plot ----
    def plot_wealth_over_time(asset_memory, date_memory, initial_amount, save_path='results/wealth_over_time.png'):
        """
        Plot portfolio wealth over the test period and save as an image.
        Two panels: absolute portfolio value and normalised growth of $1.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        os.makedirs('results', exist_ok=True)

        dates = date_memory[:len(asset_memory)] if date_memory else list(range(len(asset_memory)))
        values = np.array(asset_memory)
        normalised = values / initial_amount

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top panel: absolute portfolio value
        axes[0].plot(dates, values, color='steelblue', linewidth=1.5)
        axes[0].fill_between(dates, initial_amount, values,
                             where=(values >= initial_amount),
                             alpha=0.25, color='green', label='Gain')
        axes[0].fill_between(dates, initial_amount, values,
                             where=(values < initial_amount),
                             alpha=0.25, color='red', label='Loss')
        axes[0].axhline(initial_amount, color='grey', linestyle='--', linewidth=0.8, label='Initial')
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        axes[0].set_ylabel('Portfolio Value (USD)', fontsize=11)
        axes[0].set_title('Portfolio Wealth over Time — MoE (Out-of-Sample Test)', fontsize=13)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Bottom panel: normalised (growth of $1)
        axes[1].plot(dates, normalised, color='darkorange', linewidth=1.5)
        axes[1].axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
        axes[1].set_ylabel('Normalised Wealth (Growth of $1)', fontsize=11)
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Wealth over time plot saved to {save_path}")

    # Build a date_memory list from unique_dates (index 0 is reset day, returns start from day 1)
    date_memory = list(env_test.unique_dates[:len(env_test.asset_memory)])
    plot_wealth_over_time(env_test.asset_memory, date_memory, 1_000_000)

    print("="*60)
    print("FINAL MOE OUT-OF-SAMPLE PERFORMANCE")
    print("="*60)
    print(f"Final Portfolio Value: ${env_test.portfolio_value:,.2f}")
    ret_df = pd.DataFrame(env_test.portfolio_return_memory, columns=['return'])
    sharpe = (252**0.5) * ret_df['return'].mean() / (ret_df['return'].std() + 1e-8)
    print(f"Total Return: {(env_test.portfolio_value/1000000 - 1)*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    print("\nFinal Portfolio Allocation (All Stocks):")
    for i, tic in enumerate(TICKERS):
        print(f"  {tic:5s}: {final_weights[i]*100:6.2f}%")
    print("="*60)
