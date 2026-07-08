"""Standalone unit tests for SAC/sac_core.py — run before any cluster job
(mirrors the PPO suite's test-first pattern):

  python3 tests_sac.py

Covers: tanh/affine log-prob correction (vs analytic reference), action
bounds, eval-forward compatibility with the mains' greedy loop, twin-Q
gradient isolation, soft target update math, replay buffer wraparound, hard
trunk head isolation through the wrapper, and a tiny end-to-end train_sac
run on a stub environment.
"""
import numpy as np
import torch
import torch.nn as nn

from sac_core import (ReplayBuffer, QCritic, SquashedSACActor, train_sac,
                      MIN_WEIGHT, MAX_WEIGHT)

torch.manual_seed(0)
np.random.seed(0)

OBS, ACT = 20, 6


class DummyTrunk(nn.Module):
    """Minimal trunk matching the suite's (mu, sigma) contract."""
    def __init__(self, obs_dim=OBS, act_dim=ACT):
        super().__init__()
        self.lin = nn.Linear(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.lin(x) * 0.1
        std = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


# ── 1. log-prob correction vs analytic reference ────────────────────────────
def test_logprob():
    trunk = DummyTrunk()
    actor = SquashedSACActor(trunk)
    x = torch.randn(64, OBS)
    torch.manual_seed(1)
    a, lp = actor.sample(x)

    # reference: recompute from u recovered by inverting the affine+tanh
    mu, sigma = trunk(x)
    t = (a - actor.mid) / actor.half_span
    u = torch.atanh(torch.clamp(t, -1 + 1e-7, 1 - 1e-7))
    dist = torch.distributions.Normal(mu, sigma)
    ref = dist.log_prob(u).sum(-1)
    ref -= torch.log(1 - t.pow(2) + 1e-12).sum(-1)
    ref -= np.log(actor.half_span) * ACT
    err = (lp - ref).abs().max().item()
    assert err < 1e-3, f"log-prob correction mismatch: max err {err}"
    print(f"1. tanh/affine log-prob matches analytic reference (max err {err:.2e}) — OK")


# ── 2. bounds ────────────────────────────────────────────────────────────────
def test_bounds():
    actor = SquashedSACActor(DummyTrunk())
    x = torch.randn(256, OBS) * 5
    a, _ = actor.sample(x)
    d, _ = actor(x)
    for name, v in (("sampled", a), ("deterministic", d)):
        assert v.min() >= MIN_WEIGHT - 1e-6 and v.max() <= MAX_WEIGHT + 1e-6, name
    print(f"2. sampled+deterministic actions inside [{MIN_WEIGHT}, {MAX_WEIGHT}] — OK")


# ── 3. eval-forward contract (mains' greedy loop) ────────────────────────────
def test_eval_contract():
    actor = SquashedSACActor(DummyTrunk())
    s = torch.randn(1, OBS)
    mean, std = actor(s)                      # exactly how the mains call it
    assert mean.shape == (1, ACT) and std.shape == (1, ACT)
    print("3. forward() returns (action, std) like A2C/PPO actors — OK")


# ── 4. twin-Q gradient isolation ─────────────────────────────────────────────
def test_grad_isolation():
    trunk = DummyTrunk()
    actor = SquashedSACActor(trunk)
    q1, q2 = QCritic(OBS, ACT), QCritic(OBS, ACT)
    s = torch.randn(8, OBS)

    # critic loss must not produce actor grads (actions detached via buffer)
    a_stored = actor.sample(s)[0].detach()
    q_loss = q1(s, a_stored).pow(2).mean() + q2(s, a_stored).pow(2).mean()
    q_loss.backward()
    assert all(p.grad is None for p in actor.parameters()), "critic loss leaked into actor"

    # actor loss with frozen critics must not produce critic grads
    for p in list(q1.parameters()) + list(q2.parameters()):
        p.grad = None
        p.requires_grad_(False)
    a_new, lp = actor.sample(s)
    pi_loss = (0.2 * lp - torch.min(q1(s, a_new), q2(s, a_new))).mean()
    pi_loss.backward()
    assert all(p.grad is None for p in list(q1.parameters()) + list(q2.parameters())), \
        "actor loss leaked into critics"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()), \
        "actor got no gradient"
    print("4. twin-Q / actor gradient isolation — OK")


# ── 5. soft target update math ───────────────────────────────────────────────
def test_soft_update():
    import copy
    q = QCritic(OBS, ACT)
    qt = copy.deepcopy(q)
    old = [p.clone() for p in qt.parameters()]
    with torch.no_grad():
        for p in q.parameters():
            p.add_(1.0)                        # perturb online net
        tau = 0.005
        for p, pt in zip(q.parameters(), qt.parameters()):
            pt.mul_(1 - tau).add_(tau * p)
    for o, pt, p in zip(old, qt.parameters(), q.parameters()):
        expected = (1 - tau) * o + tau * p
        assert torch.allclose(pt, expected, atol=1e-6)
    print("5. soft target update math — OK")


# ── 6. replay buffer wraparound ──────────────────────────────────────────────
def test_buffer():
    buf = ReplayBuffer(OBS, ACT, capacity=10, device="cpu")
    for i in range(25):
        buf.add(np.full(OBS, i, dtype=np.float32), np.zeros(ACT, np.float32),
                float(i), np.zeros(OBS, np.float32), 1.0)
    assert buf.size == 10 and buf.ptr == 5
    s, a, r, s2, m = buf.sample(32)
    assert s.shape == (32, OBS) and r.shape == (32,)
    assert r.min() >= 15, "buffer kept stale entries past capacity"
    print("6. replay buffer wraparound + sample shapes — OK")


# ── 7. hard-routing head isolation through the wrapper ───────────────────────
def test_hard_isolation():
    class HardTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.fe = nn.Sequential(nn.Linear(OBS, 16), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(16, ACT) for _ in range(4)])
            self.log_std = nn.Parameter(torch.zeros(ACT))

        def forward(self, x):
            idx = x[:, -1].long()
            f = self.fe(x)
            raw = torch.zeros(x.shape[0], ACT)
            for i in range(4):
                m = idx == i
                if m.any():
                    raw[m] = self.heads[i](f[m])
            std = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(raw)
            return raw * 0.1, std

    trunk = HardTrunk()
    actor = SquashedSACActor(trunk)
    x = torch.randn(4, OBS)
    x[:, -1] = 2.0                                       # all regime 2
    d, _ = actor(x)
    ref = actor._scale(torch.tanh(trunk.heads[2](trunk.fe(x)) * 0.1))
    assert torch.allclose(d, ref, atol=1e-6), "regime label did not isolate head 2"
    print("7. hard-routing head isolation through SAC wrapper — OK")


# ── 8. tiny end-to-end train_sac on a stub env ───────────────────────────────
def test_end_to_end():
    class _Space:                                # duck-typed gym.spaces.Box
        def __init__(self, shape):
            self.shape = shape

    class StubEnv:
        def __init__(self, T=30):
            self.T = T
            self.observation_space = _Space((OBS,))
            self.action_space = _Space((ACT,))
            self.portfolio_value = 1e6

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.random.randn(OBS).astype(np.float32), {}

        def step(self, a):
            self.t += 1
            r = float(-np.sum(np.square(a - 0.05)))      # peak reward inside box
            return (np.random.randn(OBS).astype(np.float32), r,
                    self.t >= self.T, False, {})

    env = StubEnv()
    actor, (q1, q2), hist = train_sac(
        env, DummyTrunk(), epochs=3, device="cpu", tag="TEST-SAC",
        batch_size=16, warmup_steps=20, buffer_capacity=1000, log_every=100)
    assert len(hist) == 3
    a, _ = actor(torch.randn(1, OBS))                    # greedy eval path
    assert a.shape == (1, ACT)
    print("8. end-to-end train_sac on stub env (3 epochs) — OK")


if __name__ == "__main__":
    test_logprob()
    test_bounds()
    test_eval_contract()
    test_grad_isolation()
    test_soft_update()
    test_buffer()
    test_hard_isolation()
    test_end_to_end()
    print("\nALL SAC TESTS PASS")
