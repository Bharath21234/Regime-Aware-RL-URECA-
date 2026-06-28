import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os

class ProbabilisticHMM:
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full",  # Full covariance to capture feature interactions
            n_iter=1000,
            random_state=42
        )
        self.vol_states = None
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
        avg_corr = []
        for date in returns_df.index:
            try:
                corr_matrix = rolling_corr.loc[date].values
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
        X, dates = self.prepare_data(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Determine regime "meaning" using momentum vs volatility
        # Features: [0]=mean_return, [1]=rolling_vol, [2]=rolling_momentum, [3]=avg_correlation
        # Sort by (momentum - volatility): Bull = high momentum + low vol, Bear = low momentum + high vol
        state_means = self.model.means_ # [n_regimes, n_features]
        regime_scores = state_means[:, 2] - state_means[:, 1]  # momentum - volatility
        self.vol_states = np.argsort(regime_scores) 
        # Low index = Bear (low momentum, high vol), High index = Bull (high momentum, low vol)
        
        print(f"HMM fitted. Regime order (Bear -> Bull): {self.vol_states}")
        print(f"  Regime scores (momentum - vol): {regime_scores[self.vol_states]}")
        return self

    def predict_proba(self, df):
        if not self.is_fitted:
            raise ValueError("HMM not fitted.")
        X, dates = self.prepare_data(df)
        probs = self.model.predict_proba(X)
        
        # Reorder probabilities to match vol_states sorting
        ordered_probs = probs[:, self.vol_states]
        
        cols = [f'regime_p_{i}' for i in range(self.n_regimes)]
        prob_df = pd.DataFrame(ordered_probs, columns=cols)
        prob_df['date'] = dates.values
        return prob_df

def plot_regime_probs(prob_df, save_path='results/regime_probabilities.png'):
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.stackplot(prob_df['date'], 
                  prob_df['regime_p_0'], prob_df['regime_p_1'], 
                  prob_df['regime_p_2'], prob_df['regime_p_3'],
                  labels=['Bear', 'Sideways Down', 'Sideways Up', 'Bull'],
                  colors=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'], alpha=0.6)
    plt.title('Market Regime Probabilities (HMM)')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()
