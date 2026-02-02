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
        Uses the returns of all tickers in the pivot table as exogenous features.
        """
        # Pivot to get returns for all specified exogenous tickers
        returns_df = df.pivot(index='date', columns='tic', values='close').pct_change().fillna(0)
        
        # Use all available tickers as features (multi-dimensional)
        X = returns_df.values
        return X, returns_df.index

    def fit(self, df):
        """
        Fit HMM using the provided exogenous data.
        """
        X, dates = self.prepare_data(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Determine regime "meaning" (sort by average return across all features)
        # In a multi-dimensional setup, we can sort by the mean of the means
        state_means = self.model.means_ # [n_regimes, n_features]
        avg_returns = state_means.mean(axis=1)
        self.sorted_states = np.argsort(avg_returns) 
        # Low index in sorted_states = Lowest average return, High index = Highest average return
        
        print(f"HMM fitted. Regime order (Low Return -> High Return): {self.sorted_states}")
        return self

    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("HMM not fitted yet.")
        X, dates = self.prepare_data(df)
        regimes = self.model.predict(X)
        
        # Map raw states to sorted scale (0 to n_regimes-1)
        mapped_regimes = np.zeros_like(regimes)
        for i, state in enumerate(self.sorted_states):
            mapped_regimes[regimes == state] = i
            
        return pd.DataFrame({'date': dates, 'regime': mapped_regimes})

    def predict_next_regime(self, current_regime):
        """
        Predicts the next regime based on the transition matrix.
        Returns the regime with the highest transition probability.
        """
        if not self.is_fitted:
            raise ValueError("HMM not fitted yet.")
        
        # current_regime is the mapped regime index (0, 1, 2, 3)
        # We need to map it back to the raw state index used by hmmlearn
        raw_state = self.sorted_states[current_regime]
        
        # Get transition probabilities from this state
        trans_probs = self.model.transmat_[raw_state]
        
        # Find raw state with highest probability
        next_raw_state = np.argmax(trans_probs)
        
        # Map back to sorted scale
        next_mapped_regime = np.where(self.sorted_states == next_raw_state)[0][0]
        
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
    regime_colors = ['red', 'orange', 'blue', 'green'] # Bear, Sideways Down, Sideways Up, Bull
    labels = ['High Bear', 'Sideways/Low Bear', 'Sideways/Low Bull', 'High Bull']
    
    for i in range(len(regime_colors)):
        mask = regime_df['regime'] == i
        dates = regime_df.loc[mask, 'date']
        # Highlight these segments
        for d in dates:
            ax.axvspan(d, d, color=regime_colors[i], alpha=0.15)
    
    # Proxy artists for legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=regime_colors[i], lw=4, alpha=0.5) for i in range(len(regime_colors))]
    ax.legend(custom_lines + [Line2D([0], [0], color='black', alpha=0.3)], labels + ['Market'])
    
    plt.title('Market Regimes Detected by HMM')
    plt.savefig(save_path)
    plt.close()
    print(f"Regime plot saved to {save_path}")
