"""
A2C training loop with slide-by-1 rolling buffer.

Identical hyper-parameters to the real-data portfolio code so that any
performance gap between synthetic and real settings is attributable to the
data, not to a hyper-parameter difference.
"""

import torch
import torch.optim as optim

from models import make_actor, Critic


def train_a2c(env, variant: str, device: str,
              epochs: int = 200, gamma: float = 0.99, lr: float = 1e-4,
              value_coef: float = 0.5, entropy_coef: float = 0.01,
              batch_size: int = 20, l2_coef: float = 0.5,
              verbose: bool = False) -> tuple:

    obs_dim = env.observation_space.shape[0]
    actor   = make_actor(variant, obs_dim, device)
    critic  = Critic(obs_dim).to(device)
    opt     = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    history = []

    for ep in range(epochs):
        state, _  = env.reset()
        done       = False
        ep_reward  = 0.0
        s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(
                    mean.cpu(), std.cpu()
                ).sample().to(device)

            next_state, reward, done, _, _ = env.step(w_raw.cpu().numpy()[0])
            s_buf.append(s_t)
            w_buf.append(w_raw)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            mean_buf.append(mean)
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
                    nv   = (critic(ns_t).squeeze() if not done
                            else torch.zeros(1, device=device))

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
                for buf in (s_buf, w_buf, r_buf, m_buf, mean_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        history.append(ep_reward)
        if verbose and ep % 20 == 0:
            print(f"    [{variant} ep {ep:03d}]  reward={ep_reward:.3f}  "
                  f"port_val={env.portfolio_value:.4f}")

    return actor, critic, history
