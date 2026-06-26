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
            covariance_type="full",  # Full covariance to capture feature interactions
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False

    def prepare_data(self, df):
        """
        Prepares enriched features for HMM regime detection.
        Raw daily returns alone are too noisy — regimes are better characterized by:
          1. Rolling volatility (risk level)
          2. Rolling momentum (trend direction)
          3. Average cross-asset correlation (crisis vs calm)
        """
        # Pivot to get prices and compute returns
        prices_df = df.pivot(index='date', columns='tic', values='close')
        returns_df = prices_df.pct_change().fillna(0)
        
        window = 20  # ~1 month lookback
        
        # Feature 1: Mean return across assets (market direction)
        mean_return = returns_df.mean(axis=1)
        
        # Feature 2: Rolling volatility per asset, then average (market risk)
        rolling_vol = returns_df.rolling(window).std().mean(axis=1)
        
        # Feature 3: Rolling momentum (cumulative return over window)
        rolling_momentum = returns_df.rolling(window).sum().mean(axis=1)
        
        # Feature 4: Average cross-asset correlation (contagion/crisis indicator)
        rolling_corr = returns_df.rolling(window).corr()
        # Average pairwise correlation per date (excluding self-correlation)
        n_assets = returns_df.shape[1]
        avg_corr = []
        for date in returns_df.index:
            try:
                corr_matrix = rolling_corr.loc[date].values
                # Extract upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                pairwise = corr_matrix[mask]
                pairwise = pairwise[np.isfinite(pairwise)]
                avg_corr.append(pairwise.mean() if len(pairwise) > 0 else 0.0)
            except:
                avg_corr.append(0.0)
        avg_corr = pd.Series(avg_corr, index=returns_df.index)
        
        # Combine features into a matrix
        features = pd.DataFrame({
            'mean_return': mean_return,
            'rolling_vol': rolling_vol,
            'rolling_momentum': rolling_momentum,
            'avg_correlation': avg_corr
        }, index=returns_df.index)
        
        # Drop initial NaN rows from rolling window
        features = features.dropna()
        
        # Standardize features so HMM treats them equally
        self.feature_means = features.mean()
        self.feature_stds = features.std() + 1e-8
        features_scaled = (features - self.feature_means) / self.feature_stds
        
        X = features_scaled.values
        return X, features.index

    def fit(self, df):
        """
        Fit HMM using the provided exogenous data.
        """
        X, dates = self.prepare_data(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Determine regime "meaning" using momentum vs volatility
        # Features: [0]=mean_return, [1]=rolling_vol, [2]=rolling_momentum, [3]=avg_correlation
        # Sort by (momentum - volatility): Bull = high momentum + low vol, Bear = low momentum + high vol
        state_means = self.model.means_ # [n_regimes, n_features]
        regime_scores = state_means[:, 2] - state_means[:, 1]  # momentum - volatility
        self.sorted_states = np.argsort(regime_scores) 
        # Low index = Bear (low momentum, high vol), High index = Bull (high momentum, low vol)
        
        print(f"HMM fitted. Regime order (Bear -> Bull): {self.sorted_states}")
        print(f"  Regime scores (momentum - vol): {regime_scores[self.sorted_states]}")
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
