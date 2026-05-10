"""
Shared A2C trainer + RL evaluator for the walk-forward pipeline.

Same slide-by-1 rolling buffer as the existing portfolio code, with two
small generalisations:

  1. variant ∈ {'baseline', 'hard', 'soft', 'lstm'}  — selects the actor
  2. The env passed in already has the right state-tail (regime info or
     LSTM seq buffer) so this trainer is variant-agnostic.

Returns
-------
trained actor + critic + per-epoch reward history.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from envs_models import make_actor, Critic


def _device_select() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_a2c(env, variant: str, num_assets: int,
              base_dim: int | None = None,
              seq_len: int = 20,
              epochs: int = 200, gamma: float = 0.99, lr: float = 1e-4,
              value_coef: float = 0.5, entropy_coef: float = 0.01,
              batch_size: int = 20, l2_coef: float = 0.5,
              device: str | None = None,
              verbose: bool = False) -> tuple:

    device = device or _device_select()
    obs_dim = env.observation_space.shape[0]
    actor   = make_actor(variant, obs_dim, num_assets, device,
                          base_dim=base_dim, seq_len=seq_len)
    critic  = Critic(obs_dim).to(device)
    opt     = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    history = []

    for ep in range(epochs):
        state, _   = env.reset()
        done        = False
        ep_reward   = 0.0
        s_buf, w_buf, r_buf, m_buf = [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(mean.cpu(), std.cpu()).sample().to(device)

            next_state, reward, done, _, _ = env.step(w_raw.cpu().numpy()[0])
            s_buf.append(s_t)
            w_buf.append(w_raw)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            state      = next_state
            ep_reward += reward

            if len(r_buf) >= batch_size:
                bs = torch.cat(s_buf)
                bw = torch.cat(w_buf)

                mean_b, std_b = actor(bs)
                vals          = critic(bs).squeeze()
                dist_b        = torch.distributions.Normal(mean_b, std_b)
                lp            = dist_b.log_prob(bw).sum(-1)
                ent           = dist_b.entropy().sum(-1)

                with torch.no_grad():
                    ns_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                    nv   = critic(ns_t).squeeze() if not done else torch.zeros(1, device=device)

                rets, R = [], nv
                for r, m in zip(reversed(r_buf), reversed(m_buf)):
                    R = r + gamma * R * m
                    rets.insert(0, R)
                rets_t = torch.stack(rets).squeeze()
                adv    = rets_t - vals

                loss = (
                    -(lp * adv.detach()).mean()
                    + value_coef   * adv.pow(2).mean()
                    - entropy_coef * ent.mean()
                    + l2_coef      * (mean_b ** 2).mean()
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                opt.step()

                # slide-by-1
                for buf in (s_buf, w_buf, r_buf, m_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf = [], [], [], []

        history.append(ep_reward)
        if verbose and ep % 20 == 0:
            print(f"      [{variant} ep {ep:03d}]  reward={ep_reward:.3f}  "
                  f"port_val=${env.portfolio_value:,.0f}")

    return actor, critic, history


def evaluate_actor_greedy(actor, env, device: str | None = None) -> dict:
    """Greedy rollout (mean of Gaussian).  Returns trajectory dict."""
    from envs_models import enforce_portfolio_constraints
    device = device or _device_select()
    state, _ = env.reset()
    done       = False
    actor.eval()
    with torch.no_grad():
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            mean, _ = actor(s_t)
            weights = mean.cpu().numpy()[0]
            state, _, done, _, _ = env.step(weights)

    return {
        "asset_memory":             list(env.asset_memory),
        "portfolio_return_memory":  list(env.portfolio_return_memory),
        "date_memory":              list(env.date_memory),
    }
