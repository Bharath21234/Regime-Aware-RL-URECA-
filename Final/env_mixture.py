import numpy as np
import pandas as pd
import gym
from gym import spaces

class MixturePortfolioEnv(gym.Env):
    """
    Portfolio Environment with Risk-Adjusted Rewards 
    and HMM Probability State Augmentation.
    """
    def __init__(self, df, stock_dim, initial_amount, tech_indicator_list, 
                 regime_probs_df, risk_penalty=0.1, transaction_cost_pct=0.001,
                 reward_scaling=1e3, lookback=252):
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.tech_indicators = tech_indicator_list
        self.regime_probs_df = regime_probs_df
        self.risk_penalty = risk_penalty
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.lookback = lookback
        
        self.unique_dates = sorted(df.date.unique())
        self.day = 0
        
        # Spaces
        # State: Covariance + Tech + 3 Regime Probs
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dim,))
        
        # Observation space visualization (internal only)
        # Final shape depends on tick list and tech indicators
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state.flatten().shape[0],))

    def _get_state(self):
        date = self.unique_dates[self.day]
        data = self.df[self.df.date == date]
        
        covs = data["cov_list"].iloc[0]
        techs = np.array([data[t].values for t in self.tech_indicators])
        
        # Baseline state
        state = np.vstack([covs, techs]).astype(np.float32).flatten()
        
        # Append probabilities
        probs = self.regime_probs_df[self.regime_probs_df.date == date].iloc[0]
        prob_vec = probs[['regime_p_0', 'regime_p_1', 'regime_p_2']].values.astype(np.float32)
        
        return np.concatenate([state, prob_vec])

    def reset(self):
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.state = self._get_state()
        return self.state

    def step(self, actions):
        # Normalize actions (just in case)
        weights = actions / (np.sum(actions) + 1e-8)
        
        last_day_data = self.df[self.df.date == self.unique_dates[self.day]]
        self.day += 1
        self.terminal = self.day >= len(self.unique_dates) - 1
        
        new_day_data = self.df[self.df.date == self.unique_dates[self.day]]
        
        # Calculate Return
        portfolio_return = sum(((new_day_data.close.values / last_day_data.close.values) - 1) * weights)
        
        # Risk-Adjusted Reward
        # We use daily volatility proxy from the HMM input as the risk baseline
        # or calculate it from recent portfolio returns.
        # Here we use the HMM's med/high prob as a proxy for risk level.
        probs = self.regime_probs_df[self.regime_probs_df.date == self.unique_dates[self.day]].iloc[0]
        risk_level = probs['regime_p_1'] * 0.5 + probs['regime_p_2'] * 1.0 # Weighted risk sentiment
        
        reward = (portfolio_return - (self.risk_penalty * risk_level)) * self.reward_scaling
        
        # Update Portfolio
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        
        self.state = self._get_state()
        return self.state, reward, self.terminal, {}
