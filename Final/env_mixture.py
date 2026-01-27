import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class MixturePortfolioEnv(gym.Env):
    """
    Portfolio Environment with Action-Based Risk-Adjusted Rewards 
    and HMM Probability State Augmentation.
    """
    def __init__(self, df, stock_dim, initial_amount, tech_indicator_list, 
                 regime_probs_df, risk_penalty=0.1, transaction_cost_pct=0.001,
                 reward_scaling=1e3, lookback=252):
        super().__init__()
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
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dim,))
        
        # Determine obs_dim dynamically
        test_state = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(test_state),))

    def _get_state(self):
        date = self.unique_dates[self.day]
        data = self.df[self.df.date == date]
        
        # Fixed: Support list/series from the dataframe
        covs = data["cov_list"].iloc[0]
        if isinstance(covs, list):
            covs = np.array(covs)
            
        techs = np.array([data[t].values for t in self.tech_indicators])
        
        # Baseline state
        state = np.vstack([covs, techs]).astype(np.float32).flatten()
        
        # Append probabilities
        probs = self.regime_probs_df[self.regime_probs_df.date == date].iloc[0]
        prob_vec = probs[['regime_p_0', 'regime_p_1', 'regime_p_2']].values.astype(np.float32)
        
        return np.concatenate([state, prob_vec])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.state = self._get_state()
        return self.state, {}

    def step(self, actions):
        # Normalize actions
        weights = actions / (np.sum(actions) + 1e-8)
        
        last_day_data = self.df[self.df.date == self.unique_dates[self.day]]
        
        # Action-Based Risk Calculation (Portfolio Variance)
        # var = w.T @ Cov @ w
        covs = last_day_data["cov_list"].iloc[0]
        if isinstance(covs, list):
            covs = np.array(covs)
        port_variance = np.dot(weights.T, np.dot(covs, weights))

        self.day += 1
        self.terminal = self.day >= len(self.unique_dates) - 1
        
        new_day_data = self.df[self.df.date == self.unique_dates[self.day]]
        
        # Calculate Return
        portfolio_return = sum(((new_day_data.close.values / last_day_data.close.values) - 1) * weights)
        
        # REFINED REWARD: penalize variance (risk) directly chosen by agent
        reward = (portfolio_return - (self.risk_penalty * port_variance)) * self.reward_scaling
        
        # Update Portfolio
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        
        self.state = self._get_state()
        return self.state, reward, self.terminal, False, {}
