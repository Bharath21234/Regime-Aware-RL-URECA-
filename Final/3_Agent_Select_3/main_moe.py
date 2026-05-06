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
def run_experiment(seed: int = 0, out_dir: str = "results/moe") -> dict:
    """
    One full train + evaluate cycle for a given random seed.

    Returns a dict with scalar performance metrics, 'seed', and 'rewards' list.
    Per-seed plots are written to out_dir/seed_{seed}_*.png.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────
    actor, critic, rewards = train(env_train)
    plot_training_progress(rewards, save_path=f"{out_dir}/seed_{seed}_training.png")

    # ── Greedy evaluation ─────────────────────────────────────────────────
    print("\n[Evaluating MoE trained agent on Test Set (Out-of-Sample)...]")
    state, _ = env_test.reset()
    done = False
    final_weights = None
    actor.eval()
    with torch.no_grad():
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _ = actor(s_tensor)
            weights_raw = mean.detach().cpu().numpy()[0]
            final_weights = enforce_portfolio_constraints(weights_raw)
            state, _, done, _, _ = env_test.step(weights_raw)

    # ── Scalar metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(
        env_test.portfolio_return_memory, env_test.asset_memory, 1_000_000
    )
    metrics['seed']    = seed
    metrics['rewards'] = rewards

    # ── Metrics-over-time 4-panel figure ─────────────────────────────────
    date_memory = list(env_test.unique_dates[:len(env_test.asset_memory)])
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        date_memory, 1_000_000,
        title_prefix=f"Soft MoE (Select_3) — Seed {seed}",
        save_path=f"{out_dir}/seed_{seed}_metrics_over_time.png",
        rolling_window=20,
    )

    return metrics


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

    metrics = compute_metrics(env_test.portfolio_return_memory, env_test.asset_memory, 1_000_000)
    print("\nCOMPREHENSIVE PERFORMANCE METRICS (Soft MoE)")
    print("-" * 44)
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v}")
    print("-" * 44)
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        date_memory, 1_000_000,
        title_prefix="Soft MoE (Select_3)",
        save_path="results/moe_metrics_over_time.png", rolling_window=20,
    )
