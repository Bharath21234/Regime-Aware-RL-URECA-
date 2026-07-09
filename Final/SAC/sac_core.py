"""Shared SAC machinery for the regime-aware RL portfolio variant suite.

Mirrors the A2C (Final/) and PPO (Final/PPO/) suites: identical environments,
rewards, and actor trunk architectures (Baseline / Hard / Soft MoE / Learned
Router) — ONLY the training algorithm differs. SAC specifics:

  - Off-policy: replay buffer across episodes, twin Q-critics with target
    networks (clipped double-Q), soft target updates (tau).
  - Maximum-entropy objective with automatic temperature (alpha) tuning
    against target entropy = -act_dim.
  - Squashed-Gaussian policy: the variant trunks produce (mu, sigma) exactly
    as in A2C/PPO; SAC wraps them with tanh squashing + affine scaling into
    the per-asset action box [MIN_WEIGHT, MAX_WEIGHT], with the standard
    tanh/affine log-prob correction. (A2C/PPO sample an UNBOUNDED Gaussian
    and rely on the env's simplex projection; unbounded actions are ill-posed
    for SAC's Q-maximisation — the tanh bound is required for the algorithm
    to be well-defined. The env's bounded-simplex projection still applies
    on top, identically to the other suites. This is the one deliberate
    deviation from the A2C/PPO action parameterisation, and it is documented
    in the comparison notes.)
  - No L2 logit penalty by default (l2_coef=0.0): SAC's entropy term already
    regularises the policy, and applying the A2C suite's asymmetric L2
    (0.5 Hard vs 0.01 Soft) would import the known confound. Uniform and
    configurable if needed.

Verified by tests_sac.py before any cluster run (mirrors the PPO test-first
pattern).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

MIN_WEIGHT = -0.05
MAX_WEIGHT = 0.20
LOG_2 = float(np.log(2.0))


# ═════════════════════════════════════════════════════════════════════════════
# Replay buffer
# ═════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device
        self.s = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.s2 = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.m = np.zeros(self.capacity, dtype=np.float32)   # 1 - done
        self.ptr, self.size = 0, 0

    def add(self, s, a, r, s2, m):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.m[i] = s, a, r, s2, m
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=self.device)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.m)


# ═════════════════════════════════════════════════════════════════════════════
# Twin Q-critic — same 2x256 ReLU body as the suite's V-critics, but Q(s, a)
# ═════════════════════════════════════════════════════════════════════════════
class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Squashed-Gaussian wrapper around a variant trunk
# ═════════════════════════════════════════════════════════════════════════════
class SquashedSACActor(nn.Module):
    """Wraps a variant trunk (Baseline/Hard/Soft/Router actor, which returns
    (mu, sigma) exactly as in the A2C/PPO suites) into a SAC policy.

    forward(x)  -> (deterministic_action_in_env_space, sigma)
                   Deliberately mirrors the other suites' actor.forward
                   signature so the mains' greedy-evaluation loop
                   (`mean, _ = actor(s); env.step(mean)`) works unchanged.
    sample(x)   -> (action_in_env_space, log_prob)  via rsample + tanh
                   correction; used only inside train_sac.
    """
    def __init__(self, trunk, lo=MIN_WEIGHT, hi=MAX_WEIGHT):
        super().__init__()
        self.trunk = trunk
        self.lo, self.hi = lo, hi
        self.half_span = (hi - lo) / 2.0
        self.mid = (hi + lo) / 2.0

    def _scale(self, t):                        # tanh-space [-1,1] -> env box
        return self.mid + self.half_span * t

    def forward(self, x):
        mu, sigma = self.trunk(x)
        return self._scale(torch.tanh(mu)), sigma

    def sample(self, x):
        mu, sigma = self.trunk(x)
        dist = torch.distributions.Normal(mu, sigma)
        u = dist.rsample()
        t = torch.tanh(u)
        # log|d tanh/du| = log(1-tanh^2(u)); numerically stable form:
        # log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= (2.0 * (LOG_2 - u - torch.nn.functional.softplus(-2.0 * u))).sum(-1)
        log_prob -= np.log(self.half_span) * u.shape[-1]   # affine scaling Jacobian
        return self._scale(t), log_prob

    def pre_squash_mean(self, x):
        return self.trunk(x)[0]


# ═════════════════════════════════════════════════════════════════════════════
# SAC training loop
# ═════════════════════════════════════════════════════════════════════════════
def train_sac(env, trunk, epochs=300, device="cpu", tag="SAC",
              lr=3e-4, gamma=0.99, tau=0.005, batch_size=256,
              buffer_capacity=500_000, warmup_steps=5_000,
              updates_per_step=1, l2_coef=0.0, log_every=10):
    """Train a SAC agent whose policy trunk is one of the suite's variant
    actors. Returns (wrapped_actor, (q1, q2), episode_rewards) — the wrapped
    actor's forward() emits deterministic env-space actions, so the mains'
    greedy evaluation code runs unchanged.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SquashedSACActor(trunk).to(device)
    q1 = QCritic(obs_dim, act_dim).to(device)
    q2 = QCritic(obs_dim, act_dim).to(device)
    q1_t = copy.deepcopy(q1)
    q2_t = copy.deepcopy(q2)
    for p in list(q1_t.parameters()) + list(q2_t.parameters()):
        p.requires_grad_(False)

    pi_opt = optim.Adam(actor.parameters(), lr=lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # automatic temperature.
    # Target entropy must live in the same units as the policy's log-probs,
    # which include the affine tanh->[lo,hi] Jacobian: a constant
    # act_dim*log(half_span) (~ -79 for 38 assets at half_span 0.125).
    # The naive -act_dim target sits ABOVE the maximum entropy achievable on
    # the action box (act_dim*log(2*half_span) ~ -52.7), which made alpha
    # diverge (1 -> 5.7e8 in 40 epochs, Kaggle calib 2026-07-09) and
    # collapsed the policy to entropy-only near-uniform allocations.
    # -act_dim + act_dim*log(half_span) is the standard -dim(A) target
    # expressed in tanh-space, and is always achievable.
    target_entropy = float(act_dim) * (float(np.log(actor.half_span)) - 1.0)
    print(f"[SAC] target_entropy = {target_entropy:.1f} "
          f"(act_dim={act_dim}, half_span={actor.half_span})")
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    a_opt = optim.Adam([log_alpha], lr=lr)

    buf = ReplayBuffer(obs_dim, act_dim, buffer_capacity, device)
    total_steps = 0
    rewards_history = []

    print(f"Starting {tag} Training on {device} for {epochs} epochs "
          f"(SAC: off-policy, twin-Q, auto-alpha)...")
    for ep in range(epochs):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            s_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=device).unsqueeze(0)
            if total_steps < warmup_steps:
                action = np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, size=act_dim
                                           ).astype(np.float32)
            else:
                with torch.no_grad():
                    a_t, _ = actor.sample(s_t)
                action = a_t.cpu().numpy()[0]

            next_state, reward, done, _, _ = env.step(action)
            buf.add(state, action, reward, next_state, 1.0 - float(done))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if buf.size >= max(batch_size, warmup_steps):
                for _ in range(updates_per_step):
                    bs, ba, br, bs2, bm = buf.sample(batch_size)
                    alpha = log_alpha.exp().detach()

                    # ── critic update ──────────────────────────────────────
                    with torch.no_grad():
                        a2, logp2 = actor.sample(bs2)
                        q_targ = torch.min(q1_t(bs2, a2), q2_t(bs2, a2))
                        y = br + gamma * bm * (q_targ - alpha * logp2)
                    q_loss = (q1(bs, ba) - y).pow(2).mean() \
                           + (q2(bs, ba) - y).pow(2).mean()
                    q_opt.zero_grad(set_to_none=True)
                    q_loss.backward()
                    q_opt.step()

                    # ── actor update ───────────────────────────────────────
                    for p in list(q1.parameters()) + list(q2.parameters()):
                        p.requires_grad_(False)
                    a_new, logp_new = actor.sample(bs)
                    q_new = torch.min(q1(bs, a_new), q2(bs, a_new))
                    pi_loss = (alpha * logp_new - q_new).mean()
                    if l2_coef > 0:
                        pi_loss = pi_loss + l2_coef * actor.pre_squash_mean(bs).pow(2).mean()
                    pi_opt.zero_grad(set_to_none=True)
                    pi_loss.backward()
                    pi_opt.step()
                    for p in list(q1.parameters()) + list(q2.parameters()):
                        p.requires_grad_(True)

                    # ── temperature update ─────────────────────────────────
                    a_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
                    a_opt.zero_grad(set_to_none=True)
                    a_loss.backward()
                    a_opt.step()

                    # ── soft target update ─────────────────────────────────
                    with torch.no_grad():
                        for p, pt in zip(q1.parameters(), q1_t.parameters()):
                            pt.mul_(1 - tau).add_(tau * p)
                        for p, pt in zip(q2.parameters(), q2_t.parameters()):
                            pt.mul_(1 - tau).add_(tau * p)

        rewards_history.append(ep_reward)
        if ep % log_every == 0:
            print(f"[{tag}] Ep {ep:04d} | Reward: {ep_reward:.4f} | "
                  f"alpha: {log_alpha.exp().item():.4f} | "
                  f"PortVal: ${env.portfolio_value:,.2f}")

    return actor, (q1, q2), rewards_history
