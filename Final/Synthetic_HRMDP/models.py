"""
Actor + Critic models for the Synthetic HR-MDP experiments.

Three actor variants share identical backbones; the only difference is how
they incorporate (or ignore) the regime tail in the observation:

  BaselineActor  — single Gaussian head, ignores the regime tail (none in obs)
  HardActor      — 4 specialist heads, hard-routed by x[:, -1].long()
  SoftMoEActor   — 4 specialist heads blended by x[:, -4:] regime probabilities

Action  : continuous, shape (N_ASSETS,) — projected to bounded simplex by env
Mean    : tanh-bounded then scaled by 0.1 to encourage small initial weights
"""

import torch
import torch.nn as nn

from synthetic_env import N_ASSETS, N_REGIMES

HIDDEN = 256


class _SharedBackbone(nn.Module):
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
        self.backbone = _SharedBackbone(input_dim)
        self.head     = nn.Linear(HIDDEN, N_ASSETS)
        self.log_std  = nn.Parameter(torch.zeros(N_ASSETS))

    def forward(self, x):
        feats = self.backbone(x)
        mean  = self.head(feats) * 0.1
        std   = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class HardActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = _SharedBackbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, N_ASSETS) for _ in range(N_REGIMES)])
        self.log_std  = nn.Parameter(torch.zeros(N_ASSETS))

    def forward(self, x):
        regime_idx = x[:, -1].long()
        feats      = self.backbone(x)
        raw        = torch.zeros(x.shape[0], N_ASSETS, device=x.device)
        for i in range(N_REGIMES):
            mask = (regime_idx == i)
            if mask.any():
                raw[mask] = self.heads[i](feats[mask])
        mean = raw * 0.1
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class SoftMoEActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = _SharedBackbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, N_ASSETS) for _ in range(N_REGIMES)])
        self.log_std  = nn.Parameter(torch.zeros(N_ASSETS))

    def forward(self, x):
        probs = x[:, -N_REGIMES:]                                  # (batch, 4)
        feats = self.backbone(x)
        raw   = sum(probs[:, i:i+1] * self.heads[i](feats) for i in range(N_REGIMES))
        mean  = raw * 0.1
        std   = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


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
