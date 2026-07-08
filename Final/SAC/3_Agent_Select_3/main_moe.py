"""
Main Entry point for Probabilistic MoE FinRL System — SAC version.

Identical environment, reward, and actor/critic architecture to the A2C
version in 3_Agent_Select_3/main_moe.py. ONLY the training algorithm differs
(PPO clipped-surrogate, multi-epoch minibatch updates over a full-episode
rollout, vs A2C's single-pass sliding-window update).
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

df = df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)

df_counts = df.groupby('tic').size()
max_counts = df_counts.max()
tickers_to_keep = df_counts[df_counts == max_counts].index.tolist()
df = df[df.tic.isin(tickers_to_keep)]
df = df.sort_values(["date", "tic"]).reset_index(drop=True)
TICKERS = sorted(df.tic.unique().tolist())

def add_covariance_matrix(df, lookback=20):
    """
    Optimized covariance matrix calculation.
    Pivots once and uses a rolling window on NumPy arrays.
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    price_pivot = df.pivot_table(index='date', columns='tic', values='close')
    price_pivot = price_pivot.ffill().bfill()
    returns = price_pivot.pct_change()
    unique_dates = returns.index.tolist()
    cov_list = []
    returns_np = returns.values

    for i in range(len(unique_dates)):
        if i < 2:
            cov = np.zeros((returns_np.shape[1], returns_np.shape[1]))
        else:
            start_idx = max(0, i - lookback + 1)
            window = returns_np[start_idx:i+1]
            window = window[~np.isnan(window).any(axis=1)]
            if window.shape[0] < 2:
                cov = np.zeros((returns_np.shape[1], returns_np.shape[1]))
            else:
                cov = np.cov(window, rowvar=False)
        cov = np.nan_to_num(cov)
        cov_list.append(cov)

    df_cov = pd.DataFrame({'date': unique_dates, 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    df = df.dropna(subset=['cov_list'])
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df

print("Computing covariance matrices...")
df = add_covariance_matrix(df, lookback=20)

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

print("Fitting Probabilistic Macro HMM on TRAIN DATA ONLY...")
hmm = ProbabilisticHMM(n_regimes=4)
hmm.fit(train_df_exo)

print("Predicting probabilities for the entire dataset...")
prob_df = hmm.predict_proba(df_exo)
plot_regime_probs(prob_df, save_path='results/regime_probabilities.png')

common_dates = set(df['date']).intersection(set(prob_df['date']))
df = df[df['date'].isin(common_dates)].reset_index(drop=True)
prob_df = prob_df[prob_df['date'].isin(common_dates)].reset_index(drop=True)

train_df = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = df[(df.date >= TEST_START)  & (df.date <= TEST_END)].reset_index(drop=True)
prob_df_train = prob_df[(prob_df.date >= TRAIN_START) & (prob_df.date <= TRAIN_END)].reset_index(drop=True)
prob_df_test  = prob_df[(prob_df.date >= TEST_START)  & (prob_df.date <= TEST_END)].reset_index(drop=True)

print(f"DEBUG: train_df shape: {train_df.shape}")
print(f"DEBUG: test_df shape:  {test_df.shape}")

env_train = MixturePortfolioEnv(
    df=train_df,
    stock_dim=len(TICKERS),
    initial_amount=1000000,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df_train,
    reward_scaling=1000.0,
    lookback=20
)

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
# Training (SAC Core) — full-episode on-policy rollout, GAE-lambda advantages,
# K-epoch clipped-surrogate minibatch updates. See 3_Agent_Select_1's PPO
# version for the detailed comparison-to-A2C note.
# ============================================================================
def train(env, epochs=300):
    """SAC training — delegates to the shared trainer in SAC/sac_core.py.
    Identical env/reward/trunk architecture to the A2C and SAC versions;
    only the algorithm differs (off-policy, twin-Q, auto-alpha, squashed
    Gaussian). Default 300 epochs: SAC reuses transitions via the replay
    buffer, so it needs fewer passes than on-policy A2C/PPO (1000);
    calibrate before full runs.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from sac_core import train_sac
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    trunk = ActorMoE(obs_dim, act_dim).to(DEVICE)
    return train_sac(env, trunk, epochs=epochs, device=DEVICE, tag="MOE-SAC")

def plot_training_progress(rewards, save_path='results/finrlmain_sac_training.png'):
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
    plt.title('MoE SAC Training Progress')
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
    avg_alloc = df_weights.mean().sort_values(ascending=False)
    sorted_tickers = avg_alloc.index.tolist()

    plt.stackplot(df_weights.index,
                  [df_weights[tic] for tic in sorted_tickers],
                  labels=sorted_tickers,
                  alpha=0.8)

    plt.title('Portfolio Allocation History (MoE PPO Evolution)', fontsize=14)
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
    returns = np.array(portfolio_return_memory[1:])
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
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)

    returns_arr = np.array(portfolio_return_memory[1:])
    values      = np.array(asset_memory)
    plot_dates  = list(dates)[:len(values)]

    cum_return = (values / initial_amount - 1) * 100

    ret_s       = pd.Series(returns_arr)
    roll_mean   = ret_s.rolling(rolling_window).mean()
    roll_std    = ret_s.rolling(rolling_window).std() + 1e-8
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    roll_sharpe_full = np.concatenate([[np.nan], roll_sharpe.values])

    peak     = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-8) * 100

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

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"{title_prefix} — Performance Metrics Over Time (Out-of-Sample Test)",
        fontsize=13, fontweight='bold'
    )

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

    ax = axes[1]
    ax.plot(plot_dates, roll_sharpe_full, color='purple', lw=1.5,
            label=f'{rolling_window}-day rolling')
    ax.axhline(0,   color='grey',  ls='--', lw=0.8)
    ax.axhline(1.0, color='green', ls=':',  lw=0.8, alpha=0.7, label='Sharpe = 1')
    ax.set_ylabel('Sharpe (ann.)', fontsize=10)
    ax.set_title(f'Rolling {rolling_window}-Day Annualised Sharpe Ratio',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.fill_between(plot_dates, 0, -drawdown,
                    color='red', alpha=0.35, label='Drawdown depth')
    ax.plot(plot_dates, -drawdown, color='darkred', lw=1, alpha=0.8)
    ax.set_ylabel('Drawdown (%)', fontsize=10)
    ax.set_title('Running Maximum Drawdown % (from rolling peak)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
def run_experiment(seed: int = 0, out_dir: str = "results/moe_sac",
                    reward_mode: str = 'mv', dsr_eta: float = 0.01,
                    epochs: int = 300) -> dict:
    """
    One full train + evaluate cycle for a given random seed.

    reward_mode: 'mv' (default) or 'dsr' (Differential Sharpe Ratio). Applied to
    both env_train and env_test (module-level globals built at import time).

    Returns a dict with scalar performance metrics, 'seed', and 'rewards' list.
    Per-seed plots are written to out_dir/seed_{seed}_*.png.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    env_train.reward_mode = reward_mode
    env_train.dsr_eta = dsr_eta
    env_test.reward_mode = reward_mode
    env_test.dsr_eta = dsr_eta

    actor, critic, rewards = train(env_train, epochs=epochs)
    plot_training_progress(rewards, save_path=f"{out_dir}/seed_{seed}_training.png")

    print("\n[Evaluating MoE (PPO) trained agent on Test Set (Out-of-Sample)...]")
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

    metrics = compute_metrics(
        env_test.portfolio_return_memory, env_test.asset_memory, 1_000_000
    )
    metrics['seed']    = seed
    metrics['rewards'] = rewards

    date_memory = list(env_test.unique_dates[:len(env_test.asset_memory)])
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        date_memory, 1_000_000,
        title_prefix=f"Soft MoE PPO (Select_3) — Seed {seed}",
        save_path=f"{out_dir}/seed_{seed}_metrics_over_time.png",
        rolling_window=20,
    )

    return metrics


if __name__ == "__main__":
    actor, critic, rewards = train(env_train)
    plot_training_progress(rewards)

    print("\n[Evaluating MoE (PPO) trained agent on Test Set (Out-of-Sample)...]")
    state, _ = env_test.reset()
    done = False
    final_weights = None
    all_weights = []

    actor.eval()
    with torch.no_grad():
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _ = actor(s_tensor)
            weights_raw = mean.detach().cpu().numpy()[0]
            final_weights = enforce_portfolio_constraints(weights_raw)
            all_weights.append(final_weights)
            state, _, done, _, _ = env_test.step(weights_raw)

    df_weights = pd.DataFrame(all_weights, columns=TICKERS)
    if hasattr(env_test, 'unique_dates') and len(env_test.unique_dates) > len(all_weights):
        df_weights.index = env_test.unique_dates[1:len(all_weights)+1]
    plot_portfolio_allocation(df_weights)

    def plot_wealth_over_time(asset_memory, date_memory, initial_amount, save_path='results/wealth_over_time_sac.png'):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        os.makedirs('results', exist_ok=True)

        dates = date_memory[:len(asset_memory)] if date_memory else list(range(len(asset_memory)))
        values = np.array(asset_memory)
        normalised = values / initial_amount

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

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
        axes[0].set_title('Portfolio Wealth over Time — MoE PPO (Out-of-Sample Test)', fontsize=13)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

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

    date_memory = list(env_test.unique_dates[:len(env_test.asset_memory)])
    plot_wealth_over_time(env_test.asset_memory, date_memory, 1_000_000)

    print("="*60)
    print("FINAL MOE (PPO) OUT-OF-SAMPLE PERFORMANCE")
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
    print("\nCOMPREHENSIVE PERFORMANCE METRICS (Soft MoE, PPO)")
    print("-" * 44)
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v}")
    print("-" * 44)
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        date_memory, 1_000_000,
        title_prefix="Soft MoE PPO (Select_3)",
        save_path="results/moe_sac_metrics_over_time.png", rolling_window=20,
    )
