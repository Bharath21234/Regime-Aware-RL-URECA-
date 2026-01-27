"""
Main Entry point for Probabilistic MoE FinRL System.
"""
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from hmm_probabilistic import ProbabilisticHMM, plot_regime_probs
from agents_moe import ActorMoE, Critic
from env_mixture import MixturePortfolioEnv

# --- Config ---
TICKERS = ['AAPL','MSFT','NVDA','GOOGL','META','AMZN','AMD','NFLX','JPM','GS'] # Trimmed for speed/verif
START_DATE = "2016-01-01"
END_DATE = "2024-01-01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Data Pipeline
# ============================================================================
df = YahooDownloader(start_date=START_DATE, end_date=END_DATE, ticker_list=TICKERS).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=["macd", "rsi", "cci", "adx"])
df = fe.preprocess_data(df)

# Dedup and Clean
df = df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)

# Covariance
def add_cov(df, lookback=252):
    df.index = df.date.factorize()[0]
    unique_dates = df.date.unique()
    cov_list = []
    for i in range(lookback, len(unique_dates)):
        data_window = df.loc[i - lookback:i, :]
        price_pivot = data_window.pivot_table(index='date', columns='tic', values='close')
        covs = price_pivot.pct_change().dropna().cov().values
        cov_list.append(covs)
    df_cov = pd.DataFrame({'date': unique_dates[lookback:], 'cov_list': cov_list})
    return df.merge(df_cov, on='date')

df = add_cov(df)

# HMM
hmm = ProbabilisticHMM(n_regimes=3)
hmm.fit(df)
prob_df = hmm.predict_proba(df)
plot_regime_probs(prob_df)

# Env
env = MixturePortfolioEnv(
    df=df, 
    stock_dim=len(TICKERS), 
    initial_amount=1000000, 
    tech_indicator_list=["macd", "rsi", "cci", "adx"],
    regime_probs_df=prob_df,
    risk_penalty=0.05
)

# ============================================================================
# Training (A2C Core)
# ============================================================================
def train():
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    actor = ActorMoE(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
    
    epochs = 200
    gamma = 0.99
    
    print(f"Starting MoE Training on {DEVICE}...")
    batch_size = 64
    for ep in range(epochs):
        state = env.reset()
        done = False
        ep_reward = 0
        
        # Rollout buffers
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []
        
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            alpha = actor(s_tensor)
            value = critic(s_tensor)
            
            dist = torch.distributions.Dirichlet(alpha)
            weights = dist.sample()
            log_prob = dist.log_prob(weights)
            
            action = weights.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            
            # Store in buffers
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=DEVICE))
            masks.append(torch.tensor([1 - float(done)], dtype=torch.float32, device=DEVICE))
            entropies.append(dist.entropy())
            
            state = next_state
            ep_reward += reward
            
            # Batch Update
            if len(rewards) >= batch_size or done:
                # Calculate returns and advantages
                with torch.no_grad():
                    ns_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    next_value = critic(ns_tensor) if not done else torch.zeros(1, device=DEVICE)
                
                returns = []
                R = next_value
                for r, m in zip(reversed(rewards), reversed(masks)):
                    R = r + gamma * R * m
                    returns.insert(0, R)
                
                returns = torch.cat(returns)
                vals = torch.cat(values)
                advantages = returns - vals
                
                l_probs = torch.cat(log_probs)
                ents = torch.cat(entropies)
                
                actor_loss = -(l_probs * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.05 * ents.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Clear buffers
                log_probs, values, rewards, masks, entropies = [], [], [], [], []
                
        if ep % 20 == 0:
            print(f"Episode {ep:03d} | Reward: {ep_reward:.2f} | Final Val: {env.portfolio_value:.2f}")

if __name__ == "__main__":
    train()
