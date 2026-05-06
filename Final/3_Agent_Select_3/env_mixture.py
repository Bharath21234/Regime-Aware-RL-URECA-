import numpy as np
import pandas as pd
import gym
from gym import spaces

MIN_WEIGHT = -0.05   # Allow up to 5% short
MAX_WEIGHT =  0.20   # Cap any single position at 20%

def enforce_portfolio_constraints(weights, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT):
    """
    Exact Euclidean projection onto the bounded simplex.
    Finds weights that minimize ||w - raw||^2 subject to sum(w)=1 and min <= w <= max.
    Uses continuous bisection search for strict constraint satisfaction.
    """
    weights = np.array(weights, dtype=np.float32)
    
    # Bisection search bounds for the Lagrange multiplier
    mu_min = np.min(weights) - max_weight - 1.0
    mu_max = np.max(weights) - min_weight + 1.0
    
    for _ in range(50):
        mu = (mu_min + mu_max) / 2.0
        clipped = np.clip(weights - mu, min_weight, max_weight)
        current_sum = np.sum(clipped)
        
        if abs(current_sum - 1.0) < 1e-6:
            break
        elif current_sum > 1.0:
            mu_min = mu  # Need to decrease sum -> increase mu
        else:
            mu_max = mu  # Need to increase sum -> decrease mu
            
    return np.clip(weights - mu, min_weight, max_weight).astype(np.float32)

class MixturePortfolioEnv(gym.Env):
    """
    Portfolio Environment with Action-Based Risk-Adjusted Rewards 
    and HMM Probability State Augmentation.
    """
    def __init__(self, df, stock_dim, initial_amount, tech_indicator_list, 
                 regime_probs_df, transaction_cost_pct=0.001,
                 reward_scaling=1e3, lookback=20):
        super().__init__()
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.tech_indicators = tech_indicator_list
        self.regime_probs_df = regime_probs_df
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.lookback = lookback
        
        self.unique_dates = sorted(df.date.unique())
        self.day = 0
        
        # Spaces
        self.action_space = spaces.Box(low=MIN_WEIGHT, high=MAX_WEIGHT, shape=(self.stock_dim,))
        
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
        prob_vec = probs[['regime_p_0', 'regime_p_1', 'regime_p_2', 'regime_p_3']].values.astype(np.float32)
        
        return np.concatenate([state, prob_vec])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.state = self._get_state()
        return self.state, {}

    def step(self, actions):
        # Constraints to match 3_Agent_Select_1
        try:
            weights = enforce_portfolio_constraints(actions)
        except:
            weights = actions / (np.sum(actions) + 1e-8)
            
        if len(self.actions_memory) > 0:
            previous_weights = self.actions_memory[-1]
            turnover = np.sum(np.abs(weights - previous_weights))
        else:
            turnover = 0.0
            
        self.actions_memory.append(weights)
        
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
        
        # MEAN-VARIANCE REWARD: Penalise volatility to improve Sharpe Ratio
        risk_aversion = 0.5  # lambda (reduced to focus on return over vol)
        portfolio_return_penalised = portfolio_return - 0.5 * risk_aversion * port_variance
        
        # TURNOVER PENALTY
        turnover_penalty = 0.0001 * turnover  # 0.01% transaction cost penalty (lowered to encourage learning)
        
        # CONCENTRATION PENALTY (Reduced to 0.005 to match daily return magnitude)
        concentration_penalty = 0.005 * np.sum(weights ** 2)
        
        # REFINED REWARD: match 3_Agent_Select_1
        reward = (portfolio_return_penalised - turnover_penalty - concentration_penalty) * self.reward_scaling
        
        # Update Portfolio
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        
        self.state = self._get_state()
        return self.state, reward, self.terminal, False, {}





