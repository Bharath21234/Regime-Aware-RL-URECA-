"""
Switching Multi-Armed Bandit — Hidden-Regime MDP testbed.

K = 10 arms whose mean rewards are governed by a 4-state Markov chain over
latent regimes.  Each regime has a distinct best arm, by construction:

       arm:    0     1     2     3     4     5     6     7     8     9
  Regime 0:  0.90  0.70  0.30  0.40  0.20  0.50  0.10  0.30  0.40  0.50   ← best=0
  Regime 1:  0.20  0.40  0.85  0.60  0.30  0.40  0.20  0.50  0.40  0.50   ← best=2
  Regime 2:  0.30  0.20  0.40  0.50  0.90  0.70  0.40  0.50  0.40  0.50   ← best=4
  Regime 3:  0.10  0.30  0.20  0.30  0.40  0.50  0.95  0.70  0.40  0.50   ← best=6

Arms 8 and 9 are deliberately mediocre across all regimes — the "safe choice"
that a regime-blind learner converges to.

Environment exposes:
  - SwitchingBanditEnv      gym.Env with 4 variants (baseline / hard / soft / oracle)
  - MarkovRegimeChain       hidden regime dynamics
  - BanditRegimeHMM         GaussianHMM trained on per-arm rolling-mean features
  - Oracle                  pulls argmax_k M[z_t, k]   (perfect regime knowledge)
"""

from __future__ import annotations

import numpy as np
import gym
from gym import spaces
from hmmlearn.hmm import GaussianHMM


# =============================================================================
# Reward structure
# =============================================================================

K_ARMS    = 10
N_REGIMES = 4

REWARD_MEANS = np.array([
    [0.90, 0.70, 0.30, 0.40, 0.20, 0.50, 0.10, 0.30, 0.40, 0.50],   # Regime 0
    [0.20, 0.40, 0.85, 0.60, 0.30, 0.40, 0.20, 0.50, 0.40, 0.50],   # Regime 1
    [0.30, 0.20, 0.40, 0.50, 0.90, 0.70, 0.40, 0.50, 0.40, 0.50],   # Regime 2
    [0.10, 0.30, 0.20, 0.30, 0.40, 0.50, 0.95, 0.70, 0.40, 0.50],   # Regime 3
], dtype=np.float32)

REWARD_STD = 0.15

REGIME_NAMES = [f"Regime{i}" for i in range(N_REGIMES)]
BEST_ARM_PER_REGIME = REWARD_MEANS.argmax(axis=1)              # [0, 2, 4, 6]
BEST_MEAN_PER_REGIME = REWARD_MEANS.max(axis=1)                # [0.90, 0.85, 0.90, 0.95]


# =============================================================================
# Markov regime chain
# =============================================================================

class MarkovRegimeChain:
    """
    Symmetric 4-state Markov chain.  P[stay] = 1 − p_switch ; remaining mass
    spread equally across the other states.
    """
    def __init__(self, n_regimes: int = N_REGIMES, p_switch: float = 0.05,
                 rng: np.random.Generator | None = None):
        self.n_regimes = n_regimes
        self.p_switch  = p_switch
        self.rng       = rng if rng is not None else np.random.default_rng()
        T              = np.eye(n_regimes) * (1.0 - p_switch)
        T             += (p_switch / (n_regimes - 1)) * (1.0 - np.eye(n_regimes))
        self.T         = T
        self.state     = int(self.rng.integers(0, n_regimes))

    def reset(self, initial_state: int | None = None) -> int:
        self.state = (int(self.rng.integers(0, self.n_regimes))
                      if initial_state is None else initial_state)
        return self.state

    def step(self) -> int:
        self.state = int(self.rng.choice(self.n_regimes, p=self.T[self.state]))
        return self.state


# =============================================================================
# Per-arm feature extraction (used both for state and HMM input)
# =============================================================================

def per_arm_rolling_means(history: list, K: int, window: int) -> np.ndarray:
    """
    history : list of (action, reward, regime) tuples (regime kept for diagnostics)
    Returns a (K,) vector of rolling-mean reward per arm over the last `window`.
    Missing arms (no recent pulls) get the prior 0.5.
    """
    out = np.full(K, 0.5, dtype=np.float32)
    if not history:
        return out
    recent = history[-window:]
    for k in range(K):
        rewards_k = [r for a, r, _ in recent if a == k]
        if rewards_k:
            out[k] = float(np.mean(rewards_k))
    return out


def per_arm_pull_fraction(history: list, K: int, window: int) -> np.ndarray:
    """Fraction of pulls per arm in the last `window` steps."""
    out = np.zeros(K, dtype=np.float32)
    if not history:
        return out
    recent = history[-window:]
    for a, _, _ in recent:
        out[a] += 1
    return out / max(len(recent), 1)


# =============================================================================
# Bandit Regime HMM
# =============================================================================

class BanditRegimeHMM:
    """
    Gaussian HMM on K-dim per-arm rolling-mean feature vectors.
    Recovers the four regime profiles from the ROLLING reward statistics that
    the agent itself observes.  Labels are canonicalised so that label 0 ↔
    profile most similar to REWARD_MEANS[0], label 3 ↔ profile most similar
    to REWARD_MEANS[3] — this keeps the Hard Routing semantics consistent.
    """
    def __init__(self, n_regimes: int = N_REGIMES, K: int = K_ARMS, seed: int = 42):
        self.n_regimes = n_regimes
        self.K         = K
        self.model     = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=seed,
        )
        self.regime_order  = None
        self.inverse_order = None
        self._fitted       = False

    # ── canonicalisation: match each internal cluster centre to the closest
    #     row of the TRUE reward matrix.  We never use this in the agent's
    #     decision (only the canonical label is exposed), but it requires
    #     access to REWARD_MEANS which is fine because the testbed is
    #     fully synthetic and the means are part of the environment spec.
    def _canonicalise(self):
        # internal_means: shape (n_regimes, K)
        internal_means = self.model.means_
        # cost[i, j]: distance from internal cluster i to canonical regime j
        cost = np.linalg.norm(
            internal_means[:, None, :] - REWARD_MEANS[None, :, :], axis=-1
        )
        # Greedy assignment (Hungarian would be better but n_regimes is tiny)
        order = np.full(self.n_regimes, -1, dtype=int)
        used  = set()
        for i in np.argsort(cost.min(axis=1)):                  # most-confident first
            for j in np.argsort(cost[i]):
                if j not in used:
                    order[i] = j
                    used.add(j)
                    break
        self.regime_order  = order                              # internal -> canonical
        self.inverse_order = np.argsort(order)                  # canonical -> internal

    def fit(self, feature_seq: np.ndarray):
        """feature_seq : ndarray of shape (T, K) — per-arm rolling means."""
        self.model.fit(feature_seq)
        self._canonicalise()
        self._fitted = True

    def _check(self):
        if not self._fitted:
            raise RuntimeError("HMM not fitted; call .fit() first.")

    def predict_label(self, feature_seq: np.ndarray) -> int:
        """Most-likely canonical regime for the LAST step in the sequence."""
        self._check()
        if feature_seq.ndim == 1:
            feature_seq = feature_seq.reshape(1, -1)
        raw = self.model.predict(feature_seq)
        return int(self.regime_order[raw[-1]])

    def predict_proba(self, feature_seq: np.ndarray) -> np.ndarray:
        """Canonical-ordered (n_regimes,) probability vector for the last step."""
        self._check()
        if feature_seq.ndim == 1:
            feature_seq = feature_seq.reshape(1, -1)
        raw_proba = self.model.predict_proba(feature_seq)
        canonical = raw_proba[:, self.inverse_order]
        return canonical[-1].astype(np.float32)


# =============================================================================
# Oracle policy — perfect regime knowledge
# =============================================================================

class Oracle:
    """argmax_k M[z_t, k].  Achieves zero expected regret by construction."""
    def __init__(self):
        self.best_arm_per_regime = BEST_ARM_PER_REGIME.copy()

    def select_arm(self, true_regime: int) -> int:
        return int(self.best_arm_per_regime[true_regime])


# =============================================================================
# Environment
# =============================================================================

class SwitchingBanditEnv(gym.Env):
    """
    Hidden-Regime MDP version of a Multi-Armed Bandit.

    variant : 'baseline' | 'hard' | 'soft' | 'oracle'

    Observation
    -----------
    base : per-arm rolling mean (K,) ⊕ per-arm rolling pull fraction (K,)
    + variant tail:
        baseline : nothing
        hard     : [HMM regime label]                 +1
        soft     : [HMM regime probabilities]         +n_regimes
        oracle   : [true regime label] (cheats)       +1

    Action
    ------
    Discrete ∈ {0, …, K-1}

    Reward
    ------
    r_t ~ Normal(REWARD_MEANS[z_t, a_t], REWARD_STD²)

    Info dict every step contains:
      true_regime_at_action : the regime the reward was drawn from (z_t)
      oracle_arm            : argmax_k M[z_t, k]
      oracle_mean           : M[z_t, oracle_arm]
      agent_mean            : M[z_t, a_t]
      regret                : oracle_mean − agent_mean   (instantaneous, expected)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 p_switch: float = 0.05,
                 n_steps: int = 1000,
                 variant: str = "baseline",
                 hmm: BanditRegimeHMM | None = None,
                 window: int = 50,
                 seed: int | None = None):
        super().__init__()
        assert variant in ("baseline", "hard", "soft", "oracle")
        self.p_switch = p_switch
        self.n_steps  = n_steps
        self.variant  = variant
        self.hmm      = hmm
        self.window   = window
        self.K        = K_ARMS
        self.rng      = np.random.default_rng(seed)
        self.chain    = MarkovRegimeChain(N_REGIMES, p_switch, self.rng)

        if   variant == "baseline":   tail = 0
        elif variant == "hard":       tail = 1
        elif variant == "soft":       tail = N_REGIMES
        else:                          tail = 1                # oracle

        base_dim     = 2 * self.K
        self.obs_dim = base_dim + tail
        self.action_space      = spaces.Discrete(self.K)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                             shape=(self.obs_dim,), dtype=np.float32)

        self.history             = None
        self._cached_features    = None
        self.t                    = None
        self.current_regime       = None

    # ── observation construction ─────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        means  = per_arm_rolling_means(self.history, self.K, self.window)
        pulls  = per_arm_pull_fraction(self.history, self.K, self.window)
        base   = np.concatenate([means, pulls]).astype(np.float32)

        if self.variant == "baseline":
            return base

        if self.variant == "oracle":
            return np.concatenate([base, [float(self.current_regime)]]).astype(np.float32)

        # hard / soft — need HMM and at least `window` steps of history
        if self.variant == "hard":
            if self.hmm is None or len(self._cached_features) == 0:
                return np.concatenate([base, [0.0]]).astype(np.float32)
            label = self.hmm.predict_label(np.array(self._cached_features))
            return np.concatenate([base, [float(label)]]).astype(np.float32)

        # soft
        if self.hmm is None or len(self._cached_features) == 0:
            uniform = np.full(N_REGIMES, 1.0 / N_REGIMES, dtype=np.float32)
            return np.concatenate([base, uniform]).astype(np.float32)
        probs = self.hmm.predict_proba(np.array(self._cached_features))
        return np.concatenate([base, probs]).astype(np.float32)

    # ── gym interface ────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng   = np.random.default_rng(seed)
            self.chain = MarkovRegimeChain(N_REGIMES, self.p_switch, self.rng)

        self.history          = []
        self._cached_features = []
        self.t                = 0
        self.current_regime   = self.chain.reset()
        return self._build_obs(), {}

    def step(self, action: int):
        action          = int(action)
        regime_at_act   = self.current_regime
        oracle_arm      = int(BEST_ARM_PER_REGIME[regime_at_act])
        oracle_mean     = float(BEST_MEAN_PER_REGIME[regime_at_act])
        agent_mean      = float(REWARD_MEANS[regime_at_act, action])

        reward          = float(self.rng.normal(agent_mean, REWARD_STD))
        regret          = oracle_mean - agent_mean

        # Record history with the regime at decision time (z_t)
        self.history.append((action, reward, regime_at_act))

        # Update cached HMM features once the rolling window is fully populated
        if len(self.history) >= self.window:
            self._cached_features.append(
                per_arm_rolling_means(self.history, self.K, self.window)
            )

        # Advance regime for the next step
        self.current_regime = self.chain.step()
        self.t += 1
        terminal = self.t >= self.n_steps

        return self._build_obs(), reward, terminal, False, {
            "true_regime_at_action": regime_at_act,
            "oracle_arm":             oracle_arm,
            "oracle_mean":            oracle_mean,
            "agent_mean":             agent_mean,
            "regret":                 regret,
        }


# =============================================================================
# HMM training helper — uniform-exploration trajectory
# =============================================================================

def fit_hmm_on_uniform(p_switch: float,
                        n_steps: int = 10_000,
                        window: int = 50,
                        seed: int = 42) -> BanditRegimeHMM:
    """
    Build a uniform-policy trajectory and fit a 4-component GaussianHMM
    on the per-arm rolling-mean feature time series.
    """
    env = SwitchingBanditEnv(p_switch=p_switch, n_steps=n_steps,
                              variant="baseline", window=window, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)

    feature_seq = []
    done = False
    while not done:
        a = int(rng.integers(0, env.K))
        _, _, done, _, _ = env.step(a)
        if env._cached_features:
            feature_seq.append(env._cached_features[-1])
    feat_arr = np.array(feature_seq)

    hmm = BanditRegimeHMM(n_regimes=N_REGIMES, K=env.K, seed=seed)
    hmm.fit(feat_arr)
    return hmm
