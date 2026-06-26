"""
Main entry point for the Learned-Router MoE FinRL variant (Select_4).

Identical data pipeline and A2C training loop as Select_3 (Soft MoE),
with ActorLearnedRouter replacing ActorMoE.  The only architectural
difference: gating weights come from a small MLP that jointly conditions
on extracted state features AND raw HMM probabilities, instead of using
HMM probabilities directly.
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
from agents_router import ActorLearnedRouter, Critic
from env_router import MixturePortfolioEnv, enforce_portfolio_constraints

# ── Config ────────────────────────────────────────────────────────────────────
TICKER_LIST = [
    'AAPL','MSFT','NVDA','GOOGL','META',
    'JNJ','UNH','PFE','JPM','BAC','GS',
    'XOM','CVX','WMT','PG','BA','CAT',
    'AMZN','AMD','NFLX','V','HD','MCD',
    'KO','PEP','DIS','COST','CRM','INTC','TXN',
    'GE','MMM','HON','C','MS','ABT','ABBV','MRK',
]
TICKERS = sorted(list(set(TICKER_LIST)))

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-01-01"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ── Data pipeline (runs once at import) ───────────────────────────────────────
print(f"Fetching data for {len(TICKERS)} stocks...")
df = YahooDownloader(
    start_date=TRAIN_START, end_date=TEST_END, ticker_list=TICKERS
).fetch_data()
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
)
df = fe.preprocess_data(df)

df = df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)
df_counts      = df.groupby('tic').size()
max_counts     = df_counts.max()
tickers_to_keep = df_counts[df_counts == max_counts].index.tolist()
df  = df[df.tic.isin(tickers_to_keep)].sort_values(["date", "tic"]).reset_index(drop=True)
TICKERS = sorted(df.tic.unique().tolist())


def add_covariance_matrix(df, lookback=20):
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    price_pivot = df.pivot_table(index='date', columns='tic', values='close')
    price_pivot = price_pivot.ffill().bfill()
    returns     = price_pivot.pct_change()
    unique_dates = returns.index.tolist()
    returns_np   = returns.values
    cov_list = []
    for i in range(len(unique_dates)):
        if i < 2:
            cov = np.zeros((returns_np.shape[1], returns_np.shape[1]))
        else:
            start_idx = max(0, i - lookback + 1)
            window    = returns_np[start_idx:i+1]
            window    = window[~np.isnan(window).any(axis=1)]
            cov = np.cov(window, rowvar=False) if window.shape[0] >= 2 else np.zeros((returns_np.shape[1], returns_np.shape[1]))
        cov_list.append(np.nan_to_num(cov))
    df_cov = pd.DataFrame({'date': unique_dates, 'cov_list': cov_list})
    df = df.merge(df_cov, on='date').dropna(subset=['cov_list'])
    return df.sort_values(['date', 'tic']).reset_index(drop=True)


print("Computing covariance matrices...")
df = add_covariance_matrix(df, lookback=20)

print("\nFetching Exogenous Benchmarks for Macro HMM...")
EXO_TICKERS = ['SPY', 'DBC', 'LQD', 'EMB', 'TLT', 'TIP']
df_exo = YahooDownloader(
    start_date=TRAIN_START, end_date=TEST_END, ticker_list=EXO_TICKERS
).fetch_data().sort_values(["date", "tic"]).reset_index(drop=True)

train_df_exo = df_exo[(df_exo.date >= TRAIN_START) & (df_exo.date <= TRAIN_END)].reset_index(drop=True)

print("Fitting Probabilistic Macro HMM on TRAIN DATA ONLY...")
hmm_model = ProbabilisticHMM(n_regimes=4)
hmm_model.fit(train_df_exo)

print("Predicting probabilities for the entire dataset...")
prob_df = hmm_model.predict_proba(df_exo)
os.makedirs('results', exist_ok=True)
plot_regime_probs(prob_df, save_path='results/regime_probabilities_router.png')

common_dates = set(df['date']).intersection(set(prob_df['date']))
df      = df[df['date'].isin(common_dates)].reset_index(drop=True)
prob_df = prob_df[prob_df['date'].isin(common_dates)].reset_index(drop=True)

train_df      = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].reset_index(drop=True)
test_df       = df[(df.date >= TEST_START)  & (df.date <= TEST_END)].reset_index(drop=True)
prob_df_train = prob_df[(prob_df.date >= TRAIN_START) & (prob_df.date <= TRAIN_END)].reset_index(drop=True)
prob_df_test  = prob_df[(prob_df.date >= TEST_START)  & (prob_df.date <= TEST_END)].reset_index(drop=True)

print(f"DEBUG: train_df shape: {train_df.shape}")
print(f"DEBUG: test_df shape:  {test_df.shape}")

env_train = MixturePortfolioEnv(
    df=train_df, stock_dim=len(TICKERS), initial_amount=1_000_000,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df_train, reward_scaling=1000.0, lookback=20,
)
env_test = MixturePortfolioEnv(
    df=test_df, stock_dim=len(TICKERS), initial_amount=1_000_000,
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df_test, reward_scaling=1000.0, lookback=20,
)

# ── A2C training loop ─────────────────────────────────────────────────────────
def train(env, epochs=1000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor  = ActorLearnedRouter(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-5
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
    )

    gamma         = 0.99
    gae_lambda    = 0.95
    batch_size    = 20
    value_coef    = 0.5
    entropy_coef  = 0.01
    l2_coef       = 0.01

    rewards_history = []
    print(f"Starting Learned-Router Training on {DEVICE} for {epochs} epochs...")

    for ep in range(epochs):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0
        s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(mean.cpu(), std.cpu()).sample().to(DEVICE)

            action_np             = w_raw.cpu().numpy()[0]
            next_state, reward, done, _, _ = env.step(action_np)

            s_buf.append(s_t)
            w_buf.append(w_raw)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            mean_buf.append(mean)
            state      = next_state
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
                    ns_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    nv   = critic(ns_t).squeeze() if not done else torch.zeros(1, device=DEVICE)

                with torch.no_grad():
                    vals_d    = vals.detach()
                    next_vals = torch.cat([vals_d[1:], nv.reshape(1)])
                    deltas    = br + gamma * next_vals * bm - vals_d
                    adv_raw   = torch.zeros_like(deltas)
                    gae       = torch.zeros((), device=DEVICE)
                    for i in reversed(range(len(deltas))):
                        gae = deltas[i] + gamma * gae_lambda * bm[i] * gae
                        adv_raw[i] = gae
                    value_target = adv_raw + vals_d

                value_loss = (value_target - vals).pow(2).mean()
                adv_norm   = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

                loss = (
                    -(lp * adv_norm).mean()
                    + value_coef   * value_loss
                    - entropy_coef * ent.mean()
                    + l2_coef      * (mean_b ** 2).mean()
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                optimizer.step()

                for buf in (s_buf, w_buf, r_buf, m_buf, mean_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        scheduler.step()
        rewards_history.append(ep_reward)
        if ep % 10 == 0:
            print(f"[Router] Ep {ep:04d} | Reward: {ep_reward:.4f} | PortVal: ${env.portfolio_value:,.2f}")

    return actor, critic, rewards_history


def plot_training_progress(rewards, save_path='results/router_training.png'):
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
    plt.title('Learned-Router A2C Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {save_path}")


def compute_metrics(portfolio_return_memory, asset_memory, initial_amount,
                    periods_per_year=252):
    returns = np.array(portfolio_return_memory[1:])
    values  = np.array(asset_memory)
    total_return_pct  = (values[-1] / initial_amount - 1) * 100
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1) + 1e-8
    annualised_sharpe = (mean_r / std_r) * np.sqrt(periods_per_year)
    peak              = np.maximum.accumulate(values)
    drawdowns         = (peak - values) / (peak + 1e-8)
    max_drawdown_pct  = drawdowns.max() * 100
    downside          = returns[returns < 0]
    down_std = (np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8) + 1e-8
    sortino_ratio     = (mean_r / down_std) * np.sqrt(periods_per_year)
    return {
        "Total Return (%)":  round(float(total_return_pct),  4),
        "Annualised Sharpe": round(float(annualised_sharpe), 4),
        "Max Drawdown (%)":  round(float(max_drawdown_pct),  4),
        "Sortino Ratio":     round(float(sortino_ratio),     4),
        "N Trading Days":    int(len(returns)),
        "Final Value ($)":   round(float(values[-1]),        2),
    }


def plot_metrics_over_time(portfolio_return_memory, asset_memory, dates,
                           initial_amount, title_prefix='',
                           save_path='results/router_metrics_over_time.png',
                           rolling_window=20):
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)

    returns_arr = np.array(portfolio_return_memory[1:])
    values      = np.array(asset_memory)
    plot_dates  = list(dates)[:len(values)]
    cum_return  = (values / initial_amount - 1) * 100

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
            window = arr[i - w + 1: i + 1]
            m      = window.mean()
            d      = window[window < 0]
            d_std  = (np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 1e-8) + 1e-8
            out[i] = (m / d_std) * np.sqrt(252)
        return out

    roll_sortino_full = np.concatenate([[np.nan], _rolling_sortino(returns_arr, rolling_window)])

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"{title_prefix} — Performance Metrics Over Time (Out-of-Sample Test)",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(plot_dates, cum_return, color='steelblue', lw=1.5)
    ax.fill_between(plot_dates, 0, cum_return, where=(cum_return >= 0), alpha=0.20, color='green', label='Gain')
    ax.fill_between(plot_dates, 0, cum_return, where=(cum_return <  0), alpha=0.20, color='red',   label='Loss')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_ylabel('Return (%)', fontsize=10); ax.set_title('Cumulative Return (%)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(plot_dates, roll_sharpe_full, color='purple', lw=1.5, label=f'{rolling_window}-day rolling')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.axhline(1.0, color='green', ls=':', lw=0.8, alpha=0.7, label='Sharpe = 1')
    ax.set_ylabel('Sharpe (ann.)', fontsize=10)
    ax.set_title(f'Rolling {rolling_window}-Day Annualised Sharpe Ratio', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.fill_between(plot_dates, 0, -drawdown, color='red', alpha=0.35, label='Drawdown depth')
    ax.plot(plot_dates, -drawdown, color='darkred', lw=1, alpha=0.8)
    ax.set_ylabel('Drawdown (%)', fontsize=10)
    ax.set_title('Running Maximum Drawdown % (from rolling peak)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(plot_dates, roll_sortino_full, color='darkorange', lw=1.5, label=f'{rolling_window}-day rolling')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.axhline(1.0, color='green', ls=':', lw=0.8, alpha=0.7, label='Sortino = 1')
    ax.set_ylabel('Sortino (ann.)', fontsize=10); ax.set_xlabel('Date', fontsize=10)
    ax.set_title(f'Rolling {rolling_window}-Day Annualised Sortino Ratio', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics over time plot saved to {save_path}")


# ── run_experiment — called by multi-seed runner ──────────────────────────────
def run_experiment(seed: int = 0, out_dir: str = "results/router",
                    reward_mode: str = 'mv', dsr_eta: float = 0.01,
                    epochs: int = 1000) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    env_train.reward_mode = reward_mode
    env_train.dsr_eta = dsr_eta
    env_test.reward_mode = reward_mode
    env_test.dsr_eta = dsr_eta

    actor, critic, rewards = train(env_train, epochs=epochs)
    plot_training_progress(rewards, save_path=f"{out_dir}/seed_{seed}_training.png")

    print("\n[Evaluating Learned-Router agent on Test Set (Out-of-Sample)...]")
    state, _ = env_test.reset()
    done = False
    actor.eval()
    with torch.no_grad():
        while not done:
            s_tensor    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _     = actor(s_tensor)
            weights_raw = mean.detach().cpu().numpy()[0]
            enforce_portfolio_constraints(weights_raw)
            state, _, done, _, _ = env_test.step(weights_raw)

    metrics = compute_metrics(env_test.portfolio_return_memory, env_test.asset_memory, 1_000_000)
    metrics['seed']    = seed
    metrics['rewards'] = rewards

    date_memory = list(env_test.unique_dates[:len(env_test.asset_memory)])
    plot_metrics_over_time(
        env_test.portfolio_return_memory, env_test.asset_memory,
        date_memory, 1_000_000,
        title_prefix=f"Learned Router (Select_4) — Seed {seed}",
        save_path=f"{out_dir}/seed_{seed}_metrics_over_time.png",
        rolling_window=20,
    )
    return metrics


if __name__ == "__main__":
    metrics = run_experiment(seed=0, out_dir="results/router")
    print("\nCOMPREHENSIVE PERFORMANCE METRICS (Learned Router)")
    print("-" * 44)
    for k, v in metrics.items():
        if k != 'rewards':
            print(f"  {k:<25s}: {v}")
    print("-" * 44)
