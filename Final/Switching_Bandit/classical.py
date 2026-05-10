"""
Classical (non-RL) bandit baselines for the switching-MAB testbed.

Each algorithm is a small class with a simple interface:

    alg = AlgorithmName(K, **kwargs)
    for t in 1..T:
        a = alg.select_arm(t)        # choose arm
        r = env.step(a)              # pull, get reward
        alg.update(a, r, t)          # update internal statistics

Implemented algorithms
----------------------
  UCB1                Auer, Cesa-Bianchi & Fischer (2002)
  EpsGreedy           ε-greedy (Sutton & Barto)
  ThompsonGaussian    Thompson Sampling with Gaussian conjugate prior
  SlidingWindowUCB    Garivier & Moulines (2011)
  DiscountedUCB       Garivier & Moulines (2011)
  EXP3S               Auer, Cesa-Bianchi, Freund & Schapire (2002) — switching variant
  CUSUMUCB            Liu, Lee & Shroff (2018) — change detection + UCB
  MUCB                Cao, Wen, Kveton & Xie (AISTATS 2019) — parameter-free CD-UCB
  GLRklUCB            Besson, Kaufmann, Maillard & Seznec (JMLR 2022) — GLR test + kl-UCB
"""

from __future__ import annotations

import math
import numpy as np


# =============================================================================
# Base
# =============================================================================

class _Bandit:
    name: str = "base"

    def __init__(self, K: int, **kwargs):
        self.K = K
        self.t = 0

    def select_arm(self, t: int) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float, t: int):
        raise NotImplementedError


# =============================================================================
# 1. UCB1
# =============================================================================

class UCB1(_Bandit):
    name = "UCB1"
    def __init__(self, K: int):
        super().__init__(K)
        self.counts = np.zeros(K, dtype=np.int64)
        self.means  = np.zeros(K, dtype=np.float64)

    def select_arm(self, t: int) -> int:
        # Pull each arm once first
        for k in range(self.K):
            if self.counts[k] == 0:
                return k
        ucb = self.means + np.sqrt(2 * np.log(max(t, 2)) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float, t: int):
        self.counts[arm] += 1
        n  = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n


# =============================================================================
# 2. ε-greedy
# =============================================================================

class EpsGreedy(_Bandit):
    name = "EpsGreedy"
    def __init__(self, K: int, epsilon: float = 0.1, seed: int = 0):
        super().__init__(K)
        self.epsilon = epsilon
        self.counts  = np.zeros(K, dtype=np.int64)
        self.means   = np.zeros(K, dtype=np.float64)
        self.rng     = np.random.default_rng(seed)

    def select_arm(self, t: int) -> int:
        if self.rng.random() < self.epsilon or self.counts.sum() < self.K:
            return int(self.rng.integers(0, self.K))
        return int(np.argmax(self.means))

    def update(self, arm, reward, t):
        self.counts[arm] += 1
        self.means[arm]  += (reward - self.means[arm]) / self.counts[arm]


# =============================================================================
# 3. Thompson Sampling (Gaussian)
# =============================================================================

class ThompsonGaussian(_Bandit):
    name = "Thompson"
    def __init__(self, K: int, prior_mu: float = 0.5, prior_var: float = 1.0,
                  obs_var: float = 0.15 ** 2, seed: int = 0):
        super().__init__(K)
        self.mu0    = prior_mu
        self.var0   = prior_var
        self.obs_var = obs_var
        self.counts = np.zeros(K, dtype=np.int64)
        self.sums   = np.zeros(K, dtype=np.float64)
        self.rng    = np.random.default_rng(seed)

    def _posterior(self, k):
        n          = self.counts[k]
        post_var   = 1.0 / (1.0 / self.var0 + n / self.obs_var)
        post_mean  = post_var * (self.mu0 / self.var0 + self.sums[k] / self.obs_var)
        return post_mean, post_var

    def select_arm(self, t):
        samples = np.empty(self.K)
        for k in range(self.K):
            mu, var = self._posterior(k)
            samples[k] = self.rng.normal(mu, math.sqrt(var))
        return int(np.argmax(samples))

    def update(self, arm, reward, t):
        self.counts[arm] += 1
        self.sums[arm]   += reward


# =============================================================================
# 4. Sliding-Window UCB     (Garivier & Moulines 2011)
# =============================================================================

class SlidingWindowUCB(_Bandit):
    name = "SW-UCB"
    def __init__(self, K: int, window: int = 200, xi: float = 0.6):
        super().__init__(K)
        self.window = window
        self.xi     = xi
        self.history: list[tuple[int, float]] = []        # (arm, reward) pairs

    def _window_stats(self):
        recent = self.history[-self.window:]
        counts = np.zeros(self.K)
        sums   = np.zeros(self.K)
        for a, r in recent:
            counts[a] += 1
            sums[a]   += r
        means = np.divide(sums, counts, out=np.full(self.K, 0.5),
                          where=counts > 0)
        return counts, means

    def select_arm(self, t):
        counts, means = self._window_stats()
        if (counts == 0).any():
            return int(np.where(counts == 0)[0][0])
        tau    = min(t, self.window)
        bonus  = np.sqrt(self.xi * np.log(max(tau, 2)) / counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward, t):
        self.history.append((arm, reward))


# =============================================================================
# 5. Discounted UCB         (Garivier & Moulines 2011)
# =============================================================================

class DiscountedUCB(_Bandit):
    name = "D-UCB"
    def __init__(self, K: int, gamma: float = 0.99, xi: float = 0.6):
        super().__init__(K)
        self.gamma  = gamma
        self.xi     = xi
        self.disc_counts = np.zeros(K)
        self.disc_sums   = np.zeros(K)

    def select_arm(self, t):
        if (self.disc_counts == 0).any():
            return int(np.where(self.disc_counts == 0)[0][0])
        n_eff = self.disc_counts
        means = self.disc_sums / np.maximum(n_eff, 1e-9)
        N_t   = n_eff.sum()
        bonus = np.sqrt(self.xi * np.log(max(N_t, 2)) / n_eff)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward, t):
        self.disc_counts *= self.gamma
        self.disc_sums   *= self.gamma
        self.disc_counts[arm] += 1.0
        self.disc_sums[arm]   += reward


# =============================================================================
# 6. EXP3.S                 (Auer et al. 2002) — adversarial switching
# =============================================================================

class EXP3S(_Bandit):
    name = "EXP3.S"
    def __init__(self, K: int, gamma: float = 0.1, alpha: float | None = None,
                  T: int = 1000, seed: int = 0):
        super().__init__(K)
        self.gamma   = gamma
        self.alpha   = alpha if alpha is not None else 1.0 / max(T, 1)
        self.weights = np.ones(K)
        self.rng     = np.random.default_rng(seed)
        self.last_p  = None

    def _probs(self):
        w_sum = self.weights.sum()
        return ((1 - self.gamma) * self.weights / w_sum
                + self.gamma / self.K)

    def select_arm(self, t):
        p = self._probs()
        self.last_p = p
        return int(self.rng.choice(self.K, p=p))

    def update(self, arm, reward, t):
        # Estimated reward (importance-weighted, clipped to [0,1])
        r_clip = float(np.clip(reward, 0.0, 1.0))
        x_hat  = np.zeros(self.K)
        x_hat[arm] = r_clip / self.last_p[arm]
        # Multiplicative update with smoothing toward uniform
        new_weights = self.weights * np.exp(self.gamma * x_hat / self.K)
        smoothing   = (math.e * self.alpha / self.K) * self.weights.sum()
        self.weights = new_weights + smoothing


# =============================================================================
# 7. CUSUM-UCB              (Liu, Lee, Shroff 2018) — change detect + UCB
# =============================================================================

class CUSUMUCB(_Bandit):
    name = "CUSUM-UCB"
    def __init__(self, K: int, h: float = 4.0, eps: float = 0.05,
                  alpha: float = 0.1, M: int = 50):
        super().__init__(K)
        self.h     = h         # CUSUM detection threshold
        self.eps   = eps       # slack
        self.alpha = alpha     # forced exploration
        self.M     = M         # warm-up before CUSUM kicks in
        self.history = [[] for _ in range(K)]    # rewards observed for each arm
        self.last_change = [0 for _ in range(K)] # step index of last reset

    def _cusum(self, k):
        rewards = self.history[k]
        if len(rewards) < self.M:
            return False
        warmup    = rewards[:self.M]
        post      = np.array(rewards[self.M:], dtype=float)
        if len(post) == 0:
            return False
        ref       = float(np.mean(warmup))
        # Two-sided CUSUM
        gp = gn = 0.0
        for r in post:
            gp = max(0.0, gp + (r - ref) - self.eps)
            gn = max(0.0, gn - (r - ref) - self.eps)
            if gp > self.h or gn > self.h:
                return True
        return False

    def _stats(self):
        counts = np.array([len(h) for h in self.history], dtype=float)
        means  = np.array([np.mean(h) if h else 0.5 for h in self.history])
        return counts, means

    def select_arm(self, t):
        counts, means = self._stats()
        if (counts == 0).any():
            return int(np.where(counts == 0)[0][0])
        # ε-exploration
        if np.random.random() < self.alpha:
            return int(np.random.integers(0, self.K))
        bonus = np.sqrt(np.log(max(t, 2)) / counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward, t):
        self.history[arm].append(float(reward))
        if self._cusum(arm):
            # Reset this arm's stats — change detected
            self.history[arm] = []
            self.last_change[arm] = t


# =============================================================================
# 8. M-UCB                  (Cao et al. AISTATS 2019)
# =============================================================================

class MUCB(_Bandit):
    name = "M-UCB"
    def __init__(self, K: int, w: int = 80, b: float = 2.0, gamma: float = 0.1):
        super().__init__(K)
        self.w     = w        # detection window length
        self.b     = b        # detection threshold
        self.gamma = gamma    # forced exploration prob
        self.history: list[list[float]] = [[] for _ in range(K)]

    def _detect(self, k):
        h = self.history[k]
        if len(h) < self.w:
            return False
        first  = np.array(h[-self.w : -self.w // 2])
        second = np.array(h[-self.w // 2:])
        return abs(first.mean() - second.mean()) > self.b * math.sqrt(2 / self.w)

    def select_arm(self, t):
        if np.random.random() < self.gamma:
            return int(np.random.integers(0, self.K))
        counts = np.array([len(h) for h in self.history], dtype=float)
        if (counts == 0).any():
            return int(np.where(counts == 0)[0][0])
        means = np.array([np.mean(h) for h in self.history])
        bonus = np.sqrt(2 * math.log(max(t, 2)) / counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward, t):
        self.history[arm].append(float(reward))
        if self._detect(arm):
            self.history[arm] = []


# =============================================================================
# 9. GLR-klUCB              (Besson et al. JMLR 2022) — simplified Gaussian variant
# =============================================================================

class GLRklUCB(_Bandit):
    """
    Simplified Generalised Likelihood Ratio + UCB variant.
    Uses Gaussian GLR (rather than Bernoulli klUCB) since rewards here are
    Gaussian.  Resets per-arm statistics when the GLR detector fires.
    """
    name = "GLR-UCB"
    def __init__(self, K: int, threshold: float = 6.0, gamma: float = 0.1):
        super().__init__(K)
        self.threshold = threshold
        self.gamma     = gamma
        self.history: list[list[float]] = [[] for _ in range(K)]

    def _glr(self, k):
        h = np.array(self.history[k])
        n = len(h)
        if n < 20:
            return False
        # Test every split-point; report max GLR statistic
        best = 0.0
        for s in range(5, n - 5):
            mu1 = h[:s].mean();   v1 = h[:s].var() + 1e-6
            mu2 = h[s:].mean();   v2 = h[s:].var() + 1e-6
            mu  = h.mean();        v  = h.var()  + 1e-6
            ll0 = -0.5 * np.sum((h - mu) ** 2 / v) - 0.5 * n * math.log(v)
            ll1 = (-0.5 * np.sum((h[:s] - mu1) ** 2 / v1) - 0.5 * s * math.log(v1)
                   - 0.5 * np.sum((h[s:] - mu2) ** 2 / v2) - 0.5 * (n - s) * math.log(v2))
            stat = ll1 - ll0
            if stat > best:
                best = stat
        return best > self.threshold

    def select_arm(self, t):
        if np.random.random() < self.gamma:
            return int(np.random.integers(0, self.K))
        counts = np.array([len(h) for h in self.history], dtype=float)
        if (counts == 0).any():
            return int(np.where(counts == 0)[0][0])
        means = np.array([np.mean(h) for h in self.history])
        bonus = np.sqrt(2 * math.log(max(t, 2)) / counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward, t):
        self.history[arm].append(float(reward))
        # GLR is expensive: only check every few steps
        if len(self.history[arm]) % 25 == 0 and self._glr(arm):
            self.history[arm] = []


# =============================================================================
# Registry — easy iteration in main.py / sweep.py
# =============================================================================

CLASSICAL_ALGORITHMS = {
    "UCB1":      lambda K, T, seed: UCB1(K),
    "EpsGreedy": lambda K, T, seed: EpsGreedy(K, epsilon=0.1, seed=seed),
    "Thompson":  lambda K, T, seed: ThompsonGaussian(K, seed=seed),
    "SW-UCB":    lambda K, T, seed: SlidingWindowUCB(K, window=200, xi=0.6),
    "D-UCB":     lambda K, T, seed: DiscountedUCB(K, gamma=0.99, xi=0.6),
    "EXP3.S":    lambda K, T, seed: EXP3S(K, gamma=0.1, T=T, seed=seed),
    "CUSUM-UCB": lambda K, T, seed: CUSUMUCB(K),
    "M-UCB":     lambda K, T, seed: MUCB(K),
    "GLR-UCB":   lambda K, T, seed: GLRklUCB(K),
}
