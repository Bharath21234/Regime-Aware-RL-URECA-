"""
A2C training loop for categorical (discrete-action) bandit RL.

Same slide-by-1 rolling buffer and hyperparameters as the portfolio code,
adapted for a Categorical(logits=…) policy instead of Normal(mean, std).
"""

import torch
import torch.optim as optim

from models import make_actor, Critic


def train_a2c(env, variant: str, device: str,
              epochs: int = 200, gamma: float = 0.99, lr: float = 1e-3,
              value_coef: float = 0.5, entropy_coef: float = 0.05,
              batch_size: int = 32, verbose: bool = False) -> tuple:

    obs_dim = env.observation_space.shape[0]
    actor   = make_actor(variant, obs_dim, device)
    critic  = Critic(obs_dim).to(device)
    opt     = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    history = []

    for ep in range(epochs):
        state, _   = env.reset()
        done        = False
        ep_reward   = 0.0
        s_buf, a_buf, r_buf, m_buf = [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(s_t)
                dist   = torch.distributions.Categorical(logits=logits)
                a      = int(dist.sample().item())

            next_state, reward, done, _, _ = env.step(a)

            s_buf.append(s_t)
            a_buf.append(a)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            state      = next_state
            ep_reward += reward

            if len(r_buf) >= batch_size:
                bs = torch.cat(s_buf)
                ba = torch.tensor(a_buf, dtype=torch.long, device=device)

                logits_b = actor(bs)
                vals     = critic(bs).squeeze()
                dist_b   = torch.distributions.Categorical(logits=logits_b)
                lp       = dist_b.log_prob(ba)
                ent      = dist_b.entropy()

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
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                opt.step()

                # slide-by-1
                for buf in (s_buf, a_buf, r_buf, m_buf):
                    buf.pop(0)

            if done:
                s_buf, a_buf, r_buf, m_buf = [], [], [], []

        history.append(ep_reward)
        if verbose and ep % 20 == 0:
            print(f"    [{variant} ep {ep:03d}]  reward={ep_reward:.3f}")

    return actor, critic, history


def evaluate_actor(actor, env, device: str, eval_seed: int) -> tuple:
    """
    Greedy evaluation: pick argmax of the categorical logits each step.
    Returns:
      total_reward      : sum of realised rewards
      cumulative_regret : list of running cumulative regret per step
      arms_played       : list of arm indices picked
    """
    state, _ = env.reset(seed=eval_seed)
    done       = False
    total_r    = 0.0
    cum_regret = []
    arms       = []
    running    = 0.0

    actor.eval()
    with torch.no_grad():
        while not done:
            s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits = actor(s_t)
            a      = int(torch.argmax(logits, dim=-1).item())
            state, r, done, _, info = env.step(a)
            total_r += r
            running += info["regret"]
            cum_regret.append(running)
            arms.append(a)

    return total_r, cum_regret, arms
