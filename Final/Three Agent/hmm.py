import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

class MarketRegimeHMM:
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="diag", 
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False

    def prepare_data(self, df):
        """
        Prepares data for HMM. 
        Uses equal-weight returns of all tickers as the primary signal.
        """
        # Pivot to get returns for all tickers
        returns_df = df.pivot(index='date', columns='tic', values='close').pct_change().fillna(0)
        # Equal-weight market return
        market_return = returns_df.mean(axis=1).values.reshape(-1, 1)
        # Volatility signal (rolling std)
        market_vol = returns_df.rolling(window=20).std().mean(axis=1).fillna(0).values.reshape(-1, 1)
        
        X = np.column_stack([market_return, market_vol])
        return X, returns_df.index

    def fit(self, df):
        X, dates = self.prepare_data(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Determine regime "meaning" (sort by volatility)
        # We want to know which state is Bear (high vol) vs Bull (low vol)
        state_means = self.model.means_ # [n_regimes, n_features]
        vol_index = 1 # index of market_vol in X
        self.vol_states = np.argsort(state_means[:, vol_index]) 
        # Low index in vol_states = Low vol (Bull), High index = High vol (Bear)
        
        print(f"HMM fitted. Volatility order (Low->High): {self.vol_states}")
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("HMM not fitted yet.")
        X, dates = self.prepare_data(df)
        regimes = self.model.predict(X)
        
        # Map raw states to common scale (0=Low Vol, 1=Med Vol, 2=High Vol)
        mapped_regimes = np.zeros_like(regimes)
        for i, state in enumerate(self.vol_states):
            mapped_regimes[regimes == state] = i
            
        return pd.DataFrame({'date': dates, 'regime': mapped_regimes})

    def predict_next_regime(self, current_regime):
        """
        Predicts the next regime based on the transition matrix.
        Returns the regime with the highest transition probability.
        """
        if not self.is_fitted:
            raise ValueError("HMM not fitted yet.")
        
        # current_regime is the mapped regime index (0, 1, 2)
        # We need to map it back to the raw state index used by hmmlearn
        raw_state = self.vol_states[current_regime]
        
        # Get transition probabilities from this state
        trans_probs = self.model.transmat_[raw_state]
        
        # Find raw state with highest probability
        next_raw_state = np.argmax(trans_probs)
        
        # Map back to 0, 1, 2 scale
        next_mapped_regime = np.where(self.vol_states == next_raw_state)[0][0]
        
        return next_mapped_regime

def plot_regimes(df, regime_df, save_path='results/regimes.png'):
    import os
    os.makedirs('results', exist_ok=True)
    
    # Calculate cumulative returns of the equal weight market for visualization
    returns_df = df.pivot(index='date', columns='tic', values='close').pct_change().dropna()
    market_return = returns_df.mean(axis=1)
    cum_returns = (1 + market_return).cumprod()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot cum returns
    ax.plot(cum_returns.index, cum_returns.values, color='black', alpha=0.3, label='Market Cum. Returns')
    
    # Overlay colors based on regime
    regime_colors = ['green', 'blue', 'red'] # Bull, Sideways, Bear (based on vol sorting)
    labels = ['Low Vol (Bull)', 'Med Vol', 'High Vol (Bear)']
    
    for i in range(3):
        mask = regime_df['regime'] == i
        dates = regime_df.loc[mask, 'date']
        # Highlight these segments
        for d in dates:
            ax.axvspan(d, d, color=regime_colors[i], alpha=0.2)
    
    # Proxy artists for legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=regime_colors[i], lw=4, alpha=0.5) for i in range(3)]
    ax.legend(custom_lines + [Line2D([0], [0], color='black', alpha=0.3)], labels + ['Market'])
    
    plt.title('Market Regimes Detected by HMM')
    plt.savefig(save_path)
    plt.close()
    print(f"Regime plot saved to {save_path}")
