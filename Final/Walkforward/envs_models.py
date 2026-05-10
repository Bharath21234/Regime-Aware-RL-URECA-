"""
Environments + Actor / Critic models for the walk-forward evaluation.

Three environments (all share the same flat 1D state base; only the regime
tail differs — keeping the variant comparison airtight):

  FlatPortfolioEnv         baseline, no regime tail
  HardRegimePortfolioEnv   appends single regime label
  MixturePortfolioEnv      appends 4-dim regime probability vector

LSTMContextEnv             same as FlatPortfolioEnv but returns a (seq_len, base_dim)
                           sequence of recent observations as a single state.
                           Used by LSTMContextActor as a regime-AGNOSTIC RL baseline.

Five actors:
  BaselineActor            single Gaussian head (no regime info)
  HardActor                4 specialist heads, hard-routed by regime label
  SoftMoEActor             4 specialist heads, blended by HMM probs
  LSTMContextActor         LSTM over a sequence (context-conditioned RL baseline)
  Critic                   shared 2-layer MLP value head
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gym
from gym import spaces

from data_loader import N_ASSETS as _N_ASSETS_DEFAULT


# =============================================================================
# Portfolio constraints (identical to existing code)
# =============================================================================

MIN_WEIGHT = -0.05
MAX_WEIGHT =  0.20

def enforce_portfolio_constraints(weights: np.ndarray) -> np.ndarray:
    weights = np.array(weights, dtype=np.float32)
    mu_min  = float(np.min(weights)) - MAX_WEIGHT - 1.0
    mu_max  = float(np.max(weights)) - MIN_WEIGHT + 1.0
    mu = 0.0
    for _ in range(50):
        mu      = (mu_min + mu_max) / 2.0
        clipped = np.clip(weights - mu, MIN_WEIGHT, MAX_WEIGHT)
        s       = clipped.sum()
        if abs(s - 1.0) < 1e-6:
            break
        if s > 1.0:   mu_min = mu
        else:          mu_max = mu
    return np.clip(weights - mu, MIN_WEIGHT, MAX_WEIGHT).astype(np.float32)


# =============================================================================
# Base flat-state environment
# =============================================================================

TECH_INDICATORS = ["macd", "rsi", "cci", "adx"]


class _BasePortfolioEnv(gym.Env):
    """
    Common machinery: state base = flatten(cov || tech) ; reward = MV utility −
    turnover − HHI penalty (× 1000).  Subclasses override `_state_tail()`.
    """

    def __init__(self, df: pd.DataFrame, regime_df: pd.DataFrame | None,
                 stock_dim: int, initial_amount: float = 1_000_000,
                 reward_scaling: float = 1000.0):
        super().__init__()
        self.df              = df
        self.regime_df       = regime_df
        self.stock_dim       = stock_dim
        self.initial_amount  = initial_amount
        self.reward_scaling  = reward_scaling
        self.unique_dates    = sorted(df.date.unique())
        self.day             = 0

        self.action_space = spaces.Box(
            low=MIN_WEIGHT, high=MAX_WEIGHT,
            shape=(stock_dim,), dtype=np.float32,
        )
        # observation_space size depends on tail — set after building first state
        first_obs = self._build_state()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=first_obs.shape, dtype=np.float32,
        )

    # ── subclass hook ────────────────────────────────────────────────────
    def _state_tail(self, date) -> np.ndarray:
        return np.array([], dtype=np.float32)

    # ── flat base state ──────────────────────────────────────────────────
    def _state_base(self, date) -> np.ndarray:
        data  = self.df[self.df.date == date].sort_values('tic')
        covs  = np.nan_to_num(np.array(data["cov_list"].iloc[0])).astype(np.float32)
        techs = np.nan_to_num(
            np.array([data[t].values for t in TECH_INDICATORS])
        ).astype(np.float32)
        return np.vstack([covs, techs]).flatten()

    def _build_state(self) -> np.ndarray:
        date = self.unique_dates[self.day]
        base = self._state_base(date)
        tail = self._state_tail(date)
        return np.concatenate([base, tail]).astype(np.float32)

    # ── gym interface ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value         = self.initial_amount
        self.asset_memory            = [self.initial_amount]
        self.portfolio_return_memory = [0.0]
        self.date_memory             = [self.unique_dates[0]]
        self.actions_memory          = []
        self.state                   = self._build_state()
        return self.state, {}

    def step(self, actions):
        weights   = enforce_portfolio_constraints(actions)
        last_date = self.unique_dates[self.day]
        last_data = self.df[self.df.date == last_date].sort_values('tic')
        covs      = np.nan_to_num(np.array(last_data["cov_list"].iloc[0]))

        turnover  = (np.sum(np.abs(weights - self.actions_memory[-1]))
                     if self.actions_memory else 0.0)
        self.actions_memory.append(weights)

        self.day += 1
        terminal  = self.day >= len(self.unique_dates) - 1
        new_data  = self.df[self.df.date == self.unique_dates[self.day]].sort_values('tic')

        ret = float(np.sum(((new_data.close.values / last_data.close.values) - 1) * weights))
        var = float(np.dot(weights, np.dot(covs, weights)))
        hhi = float(np.sum(weights ** 2))

        reward = (ret - 0.25 * var - 0.0001 * turnover - 0.005 * hhi) * self.reward_scaling

        self.portfolio_value *= (1 + ret)
        self.asset_memory.append(self.portfolio_value)
        self.portfolio_return_memory.append(ret)
        self.date_memory.append(self.unique_dates[self.day])

        self.state = self._build_state()
        return self.state, reward, terminal, False, {}


# =============================================================================
# Three flat-state variants
# =============================================================================

class FlatPortfolioEnv(_BasePortfolioEnv):
    """Baseline — no regime info appended."""
    def _state_tail(self, date):
        return np.array([], dtype=np.float32)


class HardRegimePortfolioEnv(_BasePortfolioEnv):
    """Hard routing — single regime label appended."""
    def _state_tail(self, date):
        row = self.regime_df[self.regime_df.date == date]
        label = float(row["regime"].values[0]) if not row.empty else 0.0
        return np.array([label], dtype=np.float32)


class MixturePortfolioEnv(_BasePortfolioEnv):
    """Soft MoE — 4 regime probabilities appended."""
    def _state_tail(self, date):
        row = self.regime_df[self.regime_df.date == date]
        if row.empty:
            return np.full(4, 0.25, dtype=np.float32)
        return np.array([row[f"prob_{i}"].values[0] for i in range(4)],
                          dtype=np.float32)


# =============================================================================
# LSTM-Context env  — returns last `seq_len` flat-state observations stacked
# =============================================================================

class LSTMContextEnv(FlatPortfolioEnv):
    """
    State : (seq_len × base_dim,) — flatten of last `seq_len` observations.
    The actor reshapes back to (1, seq_len, base_dim) for the LSTM.
    """
    def __init__(self, df, stock_dim, initial_amount=1_000_000, seq_len: int = 20,
                 reward_scaling: float = 1000.0):
        self.seq_len    = seq_len
        self._obs_buf   = []
        super().__init__(df=df, regime_df=None, stock_dim=stock_dim,
                          initial_amount=initial_amount, reward_scaling=reward_scaling)

    def _build_state(self):
        date = self.unique_dates[self.day]
        cur  = self._state_base(date)
        if not self._obs_buf:
            self._obs_buf = [cur.copy() for _ in range(self.seq_len)]
        else:
            self._obs_buf.append(cur)
            if len(self._obs_buf) > self.seq_len:
                self._obs_buf = self._obs_buf[-self.seq_len:]
        return np.array(self._obs_buf, dtype=np.float32).flatten()

    def reset(self, seed=None, options=None):
        self._obs_buf = []
        return super().reset(seed=seed, options=options)


# =============================================================================
# Actor + Critic models
# =============================================================================

HIDDEN = 256


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
    def __init__(self, input_dim: int, num_assets: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.head     = nn.Linear(HIDDEN, num_assets)
        self.log_std  = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        mean = self.head(self.backbone(x)) * 0.1
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class HardActor(nn.Module):
    def __init__(self, input_dim: int, num_assets: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, num_assets) for _ in range(4)])
        self.log_std  = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        regime_idx = x[:, -1].long()
        feats      = self.backbone(x)
        raw        = torch.zeros(x.shape[0], self.heads[0].out_features, device=x.device)
        for i in range(4):
            mask = (regime_idx == i)
            if mask.any():
                raw[mask] = self.heads[i](feats[mask])
        mean = raw * 0.1
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class SoftMoEActor(nn.Module):
    def __init__(self, input_dim: int, num_assets: int):
        super().__init__()
        self.backbone = _Backbone(input_dim)
        self.heads    = nn.ModuleList([nn.Linear(HIDDEN, num_assets) for _ in range(4)])
        self.log_std  = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        probs = x[:, -4:]
        feats = self.backbone(x)
        raw   = sum(probs[:, i:i+1] * self.heads[i](feats) for i in range(4))
        mean  = raw * 0.1
        std   = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
        return mean, std


class LSTMContextActor(nn.Module):
    """
    LSTM over a length-`seq_len` window of base observations.
    Critically REGIME-AGNOSTIC — meant to test whether an LSTM can implicitly
    recover regime structure WITHOUT an explicit HMM.
    """
    def __init__(self, base_dim: int, num_assets: int, seq_len: int = 20,
                 hidden: int = HIDDEN):
        super().__init__()
        self.base_dim = base_dim
        self.seq_len  = seq_len
        self.lstm     = nn.LSTM(base_dim, hidden, batch_first=True)
        self.head     = nn.Linear(hidden, num_assets)
        self.log_std  = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        # x: (batch, seq_len * base_dim)
        b = x.shape[0]
        x = x.view(b, self.seq_len, self.base_dim)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        mean   = self.head(last) * 0.1
        std    = torch.clamp(torch.exp(self.log_std), 1e-3, 1.0).unsqueeze(0).expand_as(mean)
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


def make_actor(variant: str, obs_dim: int, num_assets: int,
                device: str, base_dim: int | None = None,
                seq_len: int = 20):
    """variant ∈ {'baseline', 'hard', 'soft', 'lstm'}"""
    if variant == "baseline":
        return BaselineActor(obs_dim, num_assets).to(device)
    if variant == "hard":
        return HardActor(obs_dim, num_assets).to(device)
    if variant == "soft":
        return SoftMoEActor(obs_dim, num_assets).to(device)
    if variant == "lstm":
        if base_dim is None:
            raise ValueError("base_dim must be provided for LSTM variant.")
        return LSTMContextActor(base_dim, num_assets, seq_len=seq_len).to(device)
    raise ValueError(f"Unknown variant '{variant}'.")
