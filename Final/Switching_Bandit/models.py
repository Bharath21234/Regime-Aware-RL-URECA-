"""
Actor + Critic for the switching-MAB testbed.

Categorical policy (softmax over K arms) — same backbone for all variants;
only the head-routing differs:

  BaselineActor  ignores the regime tail
  HardActor      4 specialist heads, hard-routed by x[:, -1].long()
  SoftMoEActor   4 specialist heads, blended by x[:, -4:] HMM probabilities
"""

import torch
import torch.nn as nn

from bandit_env import K_ARMS, N_REGIMES

HIDDEN = 128


class _Backbone(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BaselineActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.head     = nn.Linear(HIDDEN, K_ARMS)

    def forward(self, x):
        return self.head(self.backbone(x))                    # logits


class HardActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, K_ARMS) for _ in range(N_REGIMES)])

    def forward(self, x):
        regime_idx = x[:, -1].long()
        feats      = self.backbone(x)
        logits     = torch.zeros(x.shape[0], K_ARMS, device=x.device)
        for i in range(N_REGIMES):
            mask = (regime_idx == i)
            if mask.any():
                logits[mask] = self.heads[i](feats[mask])
        return logits


class SoftMoEActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, K_ARMS) for _ in range(N_REGIMES)])

    def forward(self, x):
        probs = x[:, -N_REGIMES:]                              # (batch, 4)
        feats = self.backbone(x)
        logits = sum(probs[:, i:i+1] * self.heads[i](feats) for i in range(N_REGIMES))
        return logits


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_actor(variant: str, obs_dim: int, device: str):
    if variant == "baseline":
        return BaselineActor(obs_dim).to(device)
    if variant == "hard":
        return HardActor(obs_dim).to(device)
    if variant == "soft":
        return SoftMoEActor(obs_dim).to(device)
    raise ValueError(f"Unknown variant '{variant}'.")
