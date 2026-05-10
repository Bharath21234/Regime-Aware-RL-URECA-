"""
Synthetic Hidden-Regime MDP (HR-MDP) for portfolio allocation.

Four latent regimes: Bear | Sideways-Low | Sideways-High | Bull
Five assets ranging from defensive (low vol, weak Bull) to high-beta (very
volatile, strong Bull).  At every step the environment is in exactly one
regime; asset returns are drawn from a regime-specific multivariate Gaussian.
The regime evolves via a Markov chain with controllable per-step switch
probability `p_switch`.

The agent observes the past `window` days of asset returns (and, depending on
variant, an HMM-detected regime label or probability vector) but never the
ground-truth regime.

This file also contains:
  * SyntheticRegimeHMM       — Gaussian HMM fitted to observed returns
  * Oracle                   — analytical mean-variance optimal weights per regime
                               (computable because we know μ_k and Σ_k exactly)
  * MarkovRegimeChain        — handles the latent regime transitions
"""

import numpy as np
import gym
from gym import spaces
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize

# =============================================================================
# Asset universe — 5 assets with different regime sensitivities
# =============================================================================

N_ASSETS    = 5
N_REGIMES   = 4
REGIME_NAMES = ["Bear", "Sideways-Low", "Sideways-High", "Bull"]

# Annual mean returns (rows = regime, cols = asset).
# Asset 0 = defensive (bond-like), Asset 4 = high-beta cyclical.
ANNUAL_MU = np.array([
    # asset:  0      1       2       3       4
    [+0.05, +0.10, -0.10, -0.20, -0.40],   # Bear
    [+0.03, +0.04, +0.02, +0.01, +0.02],   # Sideways-Low
    [+0.04, +0.02, +0.06, +0.10, +0.15],   # Sideways-High
    [+0.02, -0.05, +0.10, +0.20, +0.40],   # Bull
])
ANNUAL_SIGMA = np.array([
    [0.10, 0.12, 0.20, 0.30, 0.50],        # Bear (high vol everywhere)
    [0.05, 0.07, 0.10, 0.12, 0.15],        # Sideways-Low (calm)
    [0.06, 0.08, 0.12, 0.16, 0.22],        # Sideways-High
    [0.08, 0.10, 0.14, 0.18, 0.25],        # Bull
])

DAILY_MU    = ANNUAL_MU / 252.0
DAILY_SIGMA = ANNUAL_SIGMA / np.sqrt(252.0)

# Per-regime cross-asset correlation: higher in Bear (everything sells off together)
REGIME_CORRELATION = [0.55, 0.20, 0.25, 0.30]


def _regime_cov(sigma_vec: np.ndarray, corr: float) -> np.ndarray:
    """Build N×N covariance matrix from a sigma vector + scalar correlation."""
    n    = len(sigma_vec)
    C    = corr * np.ones((n, n))
    np.fill_diagonal(C, 1.0)
    return np.outer(sigma_vec, sigma_vec) * C


REGIME_COV = np.array([
    _regime_cov(DAILY_SIGMA[k], REGIME_CORRELATION[k]) for k in range(N_REGIMES)
])   # shape (4, 5, 5)


# =============================================================================
# Markov regime chain
# =============================================================================

class MarkovRegimeChain:
    """
    Symmetric Markov chain over `n_regimes` states with a per-step switch
    probability.  P[stay] = 1 − p_switch ; remaining mass spread equally
    across the other states.
    """
    def __init__(self, n_regimes: int = N_REGIMES, p_switch: float = 0.05,
                 rng: np.random.Generator | None = None):
        self.n_regimes  = n_regimes
        self.p_switch   = p_switch
        self.rng        = rng if rng is not None else np.random.default_rng()
        self.T          = self._build_transition_matrix()
        self.state      = self.rng.integers(0, n_regimes)

    def _build_transition_matrix(self) -> np.ndarray:
        n      = self.n_regimes
        T      = np.eye(n) * (1 - self.p_switch)
        T     += (self.p_switch / (n - 1)) * (1 - np.eye(n))
        return T

    def reset(self, initial_state: int | None = None):
        self.state = (self.rng.integers(0, self.n_regimes)
                      if initial_state is None else initial_state)
        return self.state

    def step(self) -> int:
        self.state = int(self.rng.choice(self.n_regimes, p=self.T[self.state]))
        return self.state


# =============================================================================
# HMM regime detector
# =============================================================================

class SyntheticRegimeHMM:
    """
    Gaussian HMM fitted directly to (T, N_assets) return sequences.

    The fit recovers the four latent components.  Internal labels are
    permuted into a canonical order based on the per-component asset-mean
    average, so label 0 = lowest mean (Bear) … label 3 = highest (Bull).
    """
    def __init__(self, n_regimes: int = N_REGIMES, seed: int = 42):
        self.n_regimes      = n_regimes
        self.model          = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=seed,
        )
        self.regime_order   = None     # [internal_label] -> sorted_label
        self.inverse_order  = None     # [sorted_label]   -> internal_label
        self._fitted        = False

    def fit(self, returns: np.ndarray):
        """returns : ndarray of shape (T, N_assets)"""
        self.model.fit(returns)
        # Sort by average asset mean so 0 ↔ Bear and 3 ↔ Bull
        avg_means          = self.model.means_.mean(axis=1)
        self.regime_order  = np.argsort(avg_means)            # internal -> sorted
        self.inverse_order = np.argsort(self.regime_order)    # sorted   -> internal
        self._fitted       = True

    def _check(self):
        if not self._fitted:
            raise RuntimeError("HMM has not been fitted; call .fit() first.")

    def predict_label(self, recent_returns: np.ndarray) -> int:
        """Return canonical regime label for the *last* observation in window."""
        self._check()
        raw = self.model.predict(recent_returns)
        return int(self.regime_order[raw[-1]])

    def predict_proba(self, recent_returns: np.ndarray) -> np.ndarray:
        """Return canonical-ordered (4,) probability vector for the last step."""
        self._check()
        proba_raw = self.model.predict_proba(recent_returns)         # (T, 4)
        proba_sorted = proba_raw[:, self.inverse_order]              # re-order cols
        return proba_sorted[-1].astype(np.float32)


# =============================================================================
# Oracle policy — analytical MV-optimal weights per regime
# =============================================================================

LAMBDA_RISK_AVERSION = 0.5
W_MIN, W_MAX         = -0.05, 0.40


def mv_optimal_weights(mu: np.ndarray, sigma: np.ndarray,
                       lam: float = LAMBDA_RISK_AVERSION,
                       w_min: float = W_MIN, w_max: float = W_MAX) -> np.ndarray:
    """
    Mean-variance optimal weights:
      max  w·μ − 0.5·λ·w·Σ·w
      s.t. sum(w)=1,   w_min ≤ w_i ≤ w_max
    Uses SLSQP (closed-form not available with the box + simplex constraint).
    """
    n = len(mu)
    def neg_utility(w):
        return -(w @ mu - 0.5 * lam * w @ sigma @ w)

    def grad_neg(w):
        return -(mu - lam * sigma @ w)

    cons   = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(w_min, w_max)] * n
    x0     = np.full(n, 1.0 / n)

    res = minimize(neg_utility, x0, jac=grad_neg,
                   bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-9, "maxiter": 200})
    return res.x.astype(np.float32)


class Oracle:
    """
    Mean-variance optimal policy given perfect regime knowledge.
    Pre-computes optimal weights for each regime once.
    """
    def __init__(self):
        self.weights_per_regime = np.stack([
            mv_optimal_weights(DAILY_MU[k], REGIME_COV[k])
            for k in range(N_REGIMES)
        ])   # shape (4, N_assets)

    def get_weights(self, true_regime: int) -> np.ndarray:
        return self.weights_per_regime[true_regime]


# =============================================================================
# Environment
# =============================================================================

class SyntheticHRMDPEnv(gym.Env):
    """
    Hidden-Regime MDP for portfolio allocation.

    Observation
    -----------
    base : flatten(last `window` rows of returns)            shape = window × N
    + regime tail per variant:
        baseline : nothing
        hard     : [HMM regime label]                        +1
        soft     : [HMM regime probabilities]                +4
        oracle   : [true regime label] (cheats — used only by Oracle agent)
    Action
    ------
    Continuous portfolio weights, projected onto the bounded simplex
    {sum(w)=1, w_min ≤ w_i ≤ w_max}.
    Reward
    ------
    reward_t = w_t · r_t − 0.5·λ·w_t·Σ_running·w_t           (mean-variance)
    """

    def __init__(self, p_switch: float = 0.05,
                 episode_len: int = 500,
                 variant: str = "baseline",
                 window: int = 20,
                 hmm: SyntheticRegimeHMM | None = None,
                 seed: int | None = None):
        super().__init__()
        assert variant in ("baseline", "hard", "soft", "oracle")
        self.p_switch    = p_switch
        self.episode_len = episode_len
        self.variant     = variant
        self.window      = window
        self.hmm         = hmm
        self.rng         = np.random.default_rng(seed)
        self.chain       = MarkovRegimeChain(N_REGIMES, p_switch, self.rng)

        if variant == "baseline":      tail = 0
        elif variant == "hard":        tail = 1
        elif variant == "soft":        tail = N_REGIMES
        else:                           tail = 1                # oracle gets true label

        self.obs_dim = window * N_ASSETS + tail
        self.action_space      = spaces.Box(W_MIN, W_MAX, shape=(N_ASSETS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                             shape=(self.obs_dim,), dtype=np.float32)

        # Filled in reset()
        self.return_history = None
        self.regime_history = None
        self.step_count     = None
        self.portfolio_value         = None
        self.asset_memory            = None
        self.portfolio_return_memory = None
        self.actions_memory          = None

    # ── Generative dynamics ───────────────────────────────────────────────
    def _sample_return(self, regime: int) -> np.ndarray:
        return self.rng.multivariate_normal(DAILY_MU[regime], REGIME_COV[regime])

    # ── Constraints (projection onto bounded simplex) ─────────────────────
    @staticmethod
    def _project_weights(w: np.ndarray) -> np.ndarray:
        """Bisection projection onto {sum=1, w_min ≤ w_i ≤ w_max}."""
        w        = np.asarray(w, dtype=np.float32)
        mu_min   = float(w.min()) - W_MAX - 1.0
        mu_max   = float(w.max()) - W_MIN + 1.0
        for _ in range(50):
            mu      = (mu_min + mu_max) / 2.0
            clipped = np.clip(w - mu, W_MIN, W_MAX)
            s       = clipped.sum()
            if abs(s - 1.0) < 1e-6:   break
            if s > 1.0:                mu_min = mu
            else:                      mu_max = mu
        return np.clip(w - mu, W_MIN, W_MAX).astype(np.float32)

    # ── Observation builder ───────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        recent = np.asarray(self.return_history[-self.window:], dtype=np.float32)
        base   = recent.flatten()

        if self.variant == "baseline":
            return base

        if self.variant == "oracle":
            return np.concatenate([base, [float(self.regime_history[-1])]]).astype(np.float32)

        if self.hmm is None:
            raise RuntimeError(f"variant={self.variant} requires a fitted HMM.")
        if self.variant == "hard":
            label = self.hmm.predict_label(recent)
            return np.concatenate([base, [float(label)]]).astype(np.float32)
        # soft
        probs = self.hmm.predict_proba(recent)
        return np.concatenate([base, probs]).astype(np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng   = np.random.default_rng(seed)
            self.chain = MarkovRegimeChain(N_REGIMES, self.p_switch, self.rng)

        # Burn-in `window` steps (regime fixed during burn-in to give the HMM
        # a clean signal in the very first observation).
        initial = self.chain.reset()
        self.return_history = []
        self.regime_history = []
        for _ in range(self.window):
            self.return_history.append(self._sample_return(initial))
            self.regime_history.append(initial)

        self.step_count                 = 0
        self.portfolio_value            = 1.0
        self.asset_memory               = [self.portfolio_value]
        self.portfolio_return_memory    = [0.0]
        self.actions_memory             = []
        return self._build_obs(), {}

    def step(self, action):
        weights        = self._project_weights(action)

        # Advance regime first, then sample today's return under the new regime
        regime_t       = self.chain.step()
        r_t            = self._sample_return(regime_t)
        self.return_history.append(r_t)
        self.regime_history.append(regime_t)

        # Realised one-step portfolio return
        port_ret       = float(weights @ r_t)

        # Running covariance estimate from the most recent window (excluding today)
        recent         = np.asarray(self.return_history[-(self.window + 1):-1])
        sigma_running  = np.cov(recent, rowvar=False) if recent.shape[0] >= 2 else np.eye(N_ASSETS) * 1e-6
        var_term       = float(weights @ sigma_running @ weights)

        # Turnover and concentration penalties (matches the real-data env reward)
        turnover       = (np.sum(np.abs(weights - self.actions_memory[-1]))
                          if self.actions_memory else 0.0)
        hhi            = float(np.sum(weights ** 2))
        reward         = (port_ret - 0.25 * var_term
                                  - 0.0001 * turnover
                                  - 0.005 * hhi) * 1000.0

        self.actions_memory.append(weights)
        self.portfolio_value         *= (1.0 + port_ret)
        self.asset_memory.append(self.portfolio_value)
        self.portfolio_return_memory.append(port_ret)

        self.step_count += 1
        terminal = self.step_count >= self.episode_len
        return self._build_obs(), reward, terminal, False, {"true_regime": regime_t}


# =============================================================================
# Convenience: generate offline data + fit HMM
# =============================================================================

def generate_training_returns(p_switch: float, n_steps: int = 5000,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Step the env with a trivial policy (equal weights) to collect a long
    time-series of (returns, true_regimes) for HMM fitting.
    """
    env = SyntheticHRMDPEnv(p_switch=p_switch, episode_len=n_steps,
                             variant="baseline", seed=seed)
    env.reset(seed=seed)
    eq_w = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32)
    returns, regimes = [], []
    done = False
    while not done:
        _, _, done, _, info = env.step(eq_w)
        returns.append(env.return_history[-1])
        regimes.append(info["true_regime"])
    return np.asarray(returns), np.asarray(regimes)


def fit_hmm_on_synthetic(p_switch: float, n_steps: int = 5000,
                          seed: int = 0) -> SyntheticRegimeHMM:
    returns, _ = generate_training_returns(p_switch, n_steps, seed)
    hmm        = SyntheticRegimeHMM(n_regimes=N_REGIMES, seed=42)
    hmm.fit(returns)
    return hmm
