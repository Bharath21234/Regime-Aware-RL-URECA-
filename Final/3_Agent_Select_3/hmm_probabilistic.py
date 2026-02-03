import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os

class ProbabilisticHMM:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="diag", 
            n_iter=1000,
            random_state=42
        )
        self.vol_states = None
        self.is_fitted = False

    def prepare_data(self, df):
        """
        Prepares data for HMM using multi-dimensional benchmark returns.
        """
        returns_df = df.pivot(index='date', columns='tic', values='close').pct_change().fillna(0)
        X = returns_df.values
        return X, returns_df.index

    def fit(self, df):
        X, dates = self.prepare_data(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Sort states by average return across all features
        state_means = self.model.means_ # [n_regimes, n_features]
        avg_returns = state_means.mean(axis=1)
        self.vol_states = np.argsort(avg_returns) 
        print(f"HMM fitted. Regime order (Low Return -> High Return): {self.vol_states}")
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
                  prob_df['regime_p_0'], prob_df['regime_p_1'], prob_df['regime_p_2'],
                  labels=['Bull', 'Sideways', 'Bear'],
                  colors=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.6)
    plt.title('Market Regime Probabilities (HMM)')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()
