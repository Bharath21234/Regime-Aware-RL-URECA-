"""
Energy Storage Arbitrage via Regime-Aware A2C
=============================================
Problem  : Operate a grid-scale battery.  Buy cheap electricity (charge),
           sell when prices spike (discharge).  Maximise profit minus
           degradation over the 2022-2024 out-of-sample test window.

Data     : Synthetic daily electricity prices calibrated to US wholesale
           market statistics (mean ~$45/MWh, four latent regimes).
           A GaussianHMM must RECOVER regimes from rolling price features —
           the generator uses fat-tailed log-normals + seasonal overlays so
           recovery is genuinely non-trivial.

Battery  : 40 MWh capacity | 10 MWh max per day | 90 % round-trip efficiency
           Degradation cost $5/MWh cycled.

State    : [SoC_norm, price_z, roll_mean_z, roll_std_z, roll_max_z,
            sin_month, cos_month, sin_dow, cos_dow, trend_z]
           + regime tail (0 / 1 / 4 extra dims → baseline / hard / soft)

Action   : scalar a ∈ [−1, 1]
           a > 0 → charge  a × MAX_ENERGY_MWH from the grid
           a < 0 → discharge  |a| × MAX_ENERGY_MWH to the grid

Reward   : (revenue − cost − degradation) × reward_scaling per step

Variants : baseline (single head) | hard (argmax routing) | soft (MoE blend)
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gym import spaces
import gym
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[Energy] Using device: {DEVICE}")

# ── Battery / market parameters ───────────────────────────────────────────────
BATTERY_CAPACITY_MWH  = 40.0   # usable capacity
MAX_ENERGY_MWH        = 10.0   # max charge or discharge per day
EFFICIENCY            = 0.90   # round-trip; one-way = sqrt(0.90)
ONE_WAY_EFF           = EFFICIENCY ** 0.5
DEGRADATION_PER_MWH   = 5.0    # $/MWh cycled (industry: $5–$15)
INITIAL_AMOUNT        = 1_000_000   # notional starting capital ($)
REWARD_SCALING        = 1e-3   # scale rewards into ~[-10, +10] range

# ── Date splits ───────────────────────────────────────────────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2023-12-31"

ROLL_WIN = 20   # rolling-feature window (days)

# =============================================================================
# 1.  Synthetic price generator
# =============================================================================

def generate_prices(seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic daily electricity prices for 2015-2024.

    Four latent regimes (Markov chain):
      0  Off-peak / renewable surplus   ~LogNormal( $20, 8 )
      1  Normal                         ~LogNormal( $45, 12 )
      2  High demand                    ~LogNormal( $90, 25 )
      3  Scarcity spike                 ~LogNormal($300, 120 )

    Overlaid with seasonal and day-of-week deterministic multipliers.
    Fat-tailed log-normals ensure the fitted GaussianHMM is misspecified
    (i.e. regime recovery is a genuine inference task, not trivial).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(TRAIN_START, "2024-01-01", freq="D")[:-1]
    T     = len(dates)

    # ── Regime transition matrix ─────────────────────────────────────────
    P = np.array([
        [0.88, 0.11, 0.01, 0.00],   # from off-peak
        [0.04, 0.88, 0.07, 0.01],   # from normal
        [0.05, 0.17, 0.75, 0.03],   # from high-demand
        [0.20, 0.50, 0.25, 0.05],   # from scarcity (fast reversion)
    ])

    regimes = np.zeros(T, dtype=int)
    regimes[0] = 1
    for t in range(1, T):
        regimes[t] = rng.choice(4, p=P[regimes[t - 1]])

    # ── Regime-specific log-normal prices ────────────────────────────────
    params = [(20, 8), (45, 12), (90, 25), (300, 120)]
    prices = np.zeros(T)
    for r, (mu, sigma) in enumerate(params):
        mask = regimes == r
        n    = mask.sum()
        if n == 0:
            continue
        log_mu    = np.log(mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2))
        log_sigma = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        prices[mask] = rng.lognormal(log_mu, log_sigma, n)

    # ── Seasonal multiplier ───────────────────────────────────────────────
    month = dates.month.values
    seasonal = (
        1.0
        + 0.18 * ((month >= 6) & (month <= 8)).astype(float)   # summer AC
        + 0.12 * ((month == 12) | (month == 1)).astype(float)  # winter heat
        - 0.08 * ((month >= 3) & (month <= 5)).astype(float)   # spring mild
    )

    # ── Day-of-week: weekends 15 % cheaper ───────────────────────────────
    dow        = dates.dayofweek.values
    dow_factor = 1.0 - 0.15 * (dow >= 5).astype(float)

    prices = np.maximum(prices * seasonal * dow_factor, 1.0)   # floor $1

    df = pd.DataFrame({
        "date":         dates,
        "price":        prices,
        "regime_true":  regimes,
        "month":        month,
        "dayofweek":    dow,
    })
    return df


# =============================================================================
# 2.  HMM regime detection
# =============================================================================

class ElectricityRegimeHMM:
    """
    4-regime Gaussian HMM fitted to rolling price features.

    Features (4):
      0  20-day rolling mean price      (price level)
      1  20-day rolling price std       (volatility)
      2  20-day rolling max price       (spike exposure)
      3  20-day price momentum          (trend: mean_t - mean_{t-20})

    Regimes sorted by (mean − volatility) → 0=cheap/stable … 3=expensive/spikey
    """

    def __init__(self, n_regimes: int = 4, win: int = ROLL_WIN):
        self.n_regimes    = n_regimes
        self.win          = win
        self.model        = GaussianHMM(
            n_components=n_regimes, covariance_type="diag",
            n_iter=300, random_state=42
        )
        self.scaler       = StandardScaler()
        self.regime_order = None

    # ── feature builder ──────────────────────────────────────────────────────
    def _features(self, prices: pd.Series) -> np.ndarray:
        w           = self.win
        roll_mean   = prices.rolling(w).mean()
        roll_std    = prices.rolling(w).std().fillna(0)
        roll_max    = prices.rolling(w).max()
        momentum    = roll_mean - roll_mean.shift(w).fillna(roll_mean)
        X = np.column_stack([
            roll_mean.values,
            roll_std.values,
            roll_max.values,
            momentum.values,
        ])
        X = np.nan_to_num(X)
        return X

    def fit(self, train_prices: pd.Series):
        X        = self._features(train_prices)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        means    = self.model.means_
        scores   = means[:, 0] - means[:, 1]        # level − volatility
        self.regime_order   = np.argsort(scores)
        self.inverse_order  = np.argsort(self.regime_order)

    def predict(self, prices: pd.Series) -> np.ndarray:
        X = self._features(prices)
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)
        return self.regime_order[raw]

    def predict_proba(self, prices: pd.Series) -> np.ndarray:
        X = self._features(prices)
        X_scaled = self.scaler.transform(X)
        proba_raw = self.model.predict_proba(X_scaled)
        return proba_raw[:, self.inverse_order]


# =============================================================================
# 3.  Data pipeline
# =============================================================================

print("Generating synthetic electricity price data...")
price_df = generate_prices(seed=0)

train_df = price_df[(price_df.date >= TRAIN_START) & (price_df.date <= TRAIN_END)].copy()
test_df  = price_df[(price_df.date >= TEST_START)  & (price_df.date <= TEST_END)].copy()

print(f"Train: {len(train_df)} days   Test: {len(test_df)} days")

# Fit HMM on training prices only
print("Fitting ElectricityRegimeHMM on training data...")
hmm = ElectricityRegimeHMM(n_regimes=4, win=ROLL_WIN)
hmm.fit(train_df["price"])

# Predict regimes for FULL series (needed for test env)
full_labels = hmm.predict(price_df["price"])
full_proba  = hmm.predict_proba(price_df["price"])

price_df["regime"]  = full_labels
for i in range(4):
    price_df[f"prob_{i}"] = full_proba[:, i]

# Re-split with regime columns attached
train_df = price_df[(price_df.date >= TRAIN_START) & (price_df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = price_df[(price_df.date >= TEST_START)  & (price_df.date <= TEST_END)].reset_index(drop=True)

# Pre-compute rolling features for state (on full series to avoid leakage at split boundary)
price_series = price_df["price"]
roll_mean_full = price_series.rolling(ROLL_WIN).mean().fillna(price_series.expanding().mean())
roll_std_full  = price_series.rolling(ROLL_WIN).std().fillna(0)
roll_max_full  = price_series.rolling(ROLL_WIN).max().fillna(price_series.expanding().max())
trend_full     = roll_mean_full - roll_mean_full.shift(ROLL_WIN).fillna(roll_mean_full)

price_df["roll_mean"] = roll_mean_full.values
price_df["roll_std"]  = roll_std_full.values
price_df["roll_max"]  = roll_max_full.values
price_df["trend"]     = trend_full.values

# Global normalisation stats from training set only (prevent test leakage)
_train_mask = (price_df.date >= TRAIN_START) & (price_df.date <= TRAIN_END)
_stats = price_df[_train_mask][["price", "roll_mean", "roll_std", "roll_max", "trend"]].agg(["mean", "std"])
PRICE_MU,     PRICE_S     = _stats["price"]
RM_MU,        RM_S        = _stats["roll_mean"]
RSTD_MU,      RSTD_S      = _stats["roll_std"]
RMAX_MU,      RMAX_S      = _stats["roll_max"]
TREND_MU,     TREND_S     = _stats["trend"]


def _znorm(val, mu, s):
    return (val - mu) / (s + 1e-8)


train_df = price_df[(price_df.date >= TRAIN_START) & (price_df.date <= TRAIN_END)].reset_index(drop=True)
test_df  = price_df[(price_df.date >= TEST_START)  & (price_df.date <= TEST_END)].reset_index(drop=True)


# =============================================================================
# 4.  Battery environment
# =============================================================================

BASE_STATE_DIM = 10   # fixed regardless of variant

class BatteryEnv(gym.Env):
    """
    Daily battery storage environment.

    variant : 'baseline' | 'hard' | 'soft'
    State (base 10-dim):
      0  SoC_norm        — battery level / CAPACITY  ∈ [0,1]
      1  price_z         — z-scored current price
      2  roll_mean_z     — z-scored 20-day rolling mean
      3  roll_std_z      — z-scored 20-day rolling std
      4  roll_max_z      — z-scored 20-day rolling max
      5  sin_month
      6  cos_month
      7  sin_dow
      8  cos_dow
      9  trend_z         — z-scored 20-day momentum

    + regime tail for hard (1) / soft (4) variants.
    """

    def __init__(self, df: pd.DataFrame, variant: str = "baseline",
                 initial_soc: float = BATTERY_CAPACITY_MWH / 2):
        super().__init__()
        self.df          = df.reset_index(drop=True)
        self.variant     = variant
        self.initial_soc = initial_soc
        self.dates       = self.df["date"].values
        self.n_days      = len(self.df)

        tail = {"baseline": 0, "hard": 1, "soft": 4}[variant]
        obs_dim = BASE_STATE_DIM + tail

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ── state builder ────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        row   = self.df.iloc[self.day]
        soc_n = self.soc / BATTERY_CAPACITY_MWH

        base = np.array([
            soc_n,
            _znorm(row.price,     PRICE_MU, PRICE_S),
            _znorm(row.roll_mean, RM_MU,    RM_S),
            _znorm(row.roll_std,  RSTD_MU,  RSTD_S),
            _znorm(row.roll_max,  RMAX_MU,  RMAX_S),
            np.sin(2 * np.pi * row.month      / 12),
            np.cos(2 * np.pi * row.month      / 12),
            np.sin(2 * np.pi * row.dayofweek  / 7),
            np.cos(2 * np.pi * row.dayofweek  / 7),
            _znorm(row.trend,     TREND_MU,  TREND_S),
        ], dtype=np.float32)

        if self.variant == "hard":
            return np.concatenate([base, [float(row.regime)]])
        elif self.variant == "soft":
            probs = np.array([row[f"prob_{i}"] for i in range(4)], dtype=np.float32)
            return np.concatenate([base, probs])
        return base

    # ── gym interface ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day              = 0
        self.soc              = self.initial_soc
        self.portfolio_value  = INITIAL_AMOUNT   # tracks cumulative P&L on top of base
        self.daily_pnl        = []               # per-step net P&L ($)
        self.asset_memory     = [INITIAL_AMOUNT]
        self.portfolio_return_memory = [0.0]
        self.date_memory      = [self.dates[0]]
        return self._obs(), {}

    def step(self, action):
        price     = float(self.df.iloc[self.day]["price"])
        a         = float(np.clip(action, -1.0, 1.0))
        energy_MW = a * MAX_ENERGY_MWH   # > 0 = charge, < 0 = discharge

        if energy_MW > 0:
            # Charging: limited by remaining capacity
            energy_in   = min(energy_MW * ONE_WAY_EFF, BATTERY_CAPACITY_MWH - self.soc)
            energy_drawn = energy_in / ONE_WAY_EFF    # from grid
            cost         = energy_drawn * price
            rev          = 0.0
            self.soc    += energy_in
        else:
            # Discharging: limited by available SoC
            energy_out  = min(-energy_MW / ONE_WAY_EFF, self.soc)
            energy_sold  = energy_out * ONE_WAY_EFF   # to grid
            rev          = energy_sold * price
            cost         = 0.0
            energy_drawn = 0.0
            self.soc    -= energy_out

        degradation = DEGRADATION_PER_MWH * abs(energy_MW)
        pnl         = rev - cost - degradation
        reward      = pnl * REWARD_SCALING

        self.portfolio_value += pnl
        ret = pnl / (INITIAL_AMOUNT + 1e-8)

        self.daily_pnl.append(pnl)
        self.asset_memory.append(self.portfolio_value)
        self.portfolio_return_memory.append(ret)

        self.day += 1
        terminal  = self.day >= self.n_days - 1
        if not terminal:
            self.date_memory.append(self.dates[self.day])

        return self._obs(), reward, terminal, False, {}


# Instantiate environments (module-level, reused across seeds)
env_train = {v: BatteryEnv(train_df, variant=v) for v in ("baseline", "hard", "soft")}
env_test  = {v: BatteryEnv(test_df,  variant=v) for v in ("baseline", "hard", "soft")}


# =============================================================================
# 5.  Actor / Critic models
# =============================================================================

HIDDEN = 256


class _SharedNet(nn.Module):
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
        self.backbone  = _SharedNet(input_dim)
        self.head      = nn.Linear(HIDDEN, 1)
        self.log_std   = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, x):
        mean = torch.tanh(self.head(self.backbone(x))).squeeze(-1)
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 0.5).expand(x.shape[0])
        return mean, std


class HardActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone  = _SharedNet(input_dim)
        self.heads     = nn.ModuleList([nn.Linear(HIDDEN, 1) for _ in range(4)])
        self.log_std   = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, x):
        regime_idx = x[:, -1].long()
        feats      = self.backbone(x)
        raw        = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(4):
            mask = (regime_idx == i)
            if mask.any():
                raw[mask] = self.heads[i](feats[mask])
        mean = torch.tanh(raw).squeeze(-1)
        std  = torch.clamp(torch.exp(self.log_std), 1e-3, 0.5).expand(x.shape[0])
        return mean, std


class SoftMoEActor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone  = _SharedNet(input_dim)
        self.heads     = nn.ModuleList([nn.Linear(HIDDEN, 1) for _ in range(4)])
        self.log_std   = nn.Parameter(torch.tensor([-1.0]))

    def forward(self, x):
        probs = x[:, -4:]          # [batch, 4]
        feats = self.backbone(x)
        raw   = sum(probs[:, i:i+1] * self.heads[i](feats) for i in range(4))
        mean  = torch.tanh(raw).squeeze(-1)
        std   = torch.clamp(torch.exp(self.log_std), 1e-3, 0.5).expand(x.shape[0])
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


def _make_actor(variant: str, obs_dim: int):
    if variant == "baseline":
        return BaselineActor(obs_dim).to(DEVICE)
    elif variant == "hard":
        return HardActor(obs_dim).to(DEVICE)
    return SoftMoEActor(obs_dim).to(DEVICE)


# =============================================================================
# 6.  A2C training  (slide-by-1 rolling buffer — identical to portfolio code)
# =============================================================================

def train_a2c(env: BatteryEnv, variant: str,
              epochs: int = 200, gamma: float = 0.99, lr: float = 1e-4,
              value_coef: float = 0.5, entropy_coef: float = 0.01,
              batch_size: int = 20, l2_coef: float = 0.5) -> tuple:

    obs_dim = env.observation_space.shape[0]
    actor   = _make_actor(variant, obs_dim)
    critic  = Critic(obs_dim).to(DEVICE)
    opt     = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr
    )
    history = []

    print(f"  Training [{variant}] for {epochs} epochs...")
    for ep in range(epochs):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0
        s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mean, std = actor(s_t)
                w_raw = torch.distributions.Normal(
                    mean.cpu(), std.cpu()
                ).sample().to(DEVICE)

            action_np = w_raw.cpu().numpy()         # shape (1,)
            next_state, reward, done, _, _ = env.step(action_np)

            s_buf.append(s_t)
            w_buf.append(w_raw.unsqueeze(-1) if w_raw.dim() == 1 else w_raw)
            r_buf.append(reward)
            m_buf.append(1.0 - float(done))
            mean_buf.append(mean)
            state      = next_state
            ep_reward += reward

            if len(r_buf) >= batch_size:
                bs = torch.cat(s_buf)
                bw = torch.cat(w_buf).squeeze(-1)
                br = torch.tensor(r_buf, dtype=torch.float32, device=DEVICE)
                bm = torch.tensor(m_buf, dtype=torch.float32, device=DEVICE)

                mean_b, std_b = actor(bs)
                vals          = critic(bs).squeeze()
                dist_b        = torch.distributions.Normal(mean_b, std_b)
                lp            = dist_b.log_prob(bw)
                ent           = dist_b.entropy()

                with torch.no_grad():
                    ns_t = torch.tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)
                    nv = (critic(ns_t).squeeze()
                          if not done else torch.zeros(1, device=DEVICE))

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
                for buf in (s_buf, w_buf, r_buf, m_buf, mean_buf):
                    buf.pop(0)

            if done:
                s_buf, w_buf, r_buf, m_buf, mean_buf = [], [], [], [], []

        history.append(ep_reward)
        if ep % 20 == 0:
            print(f"    ep {ep:03d}  reward={ep_reward:8.3f}  "
                  f"P&L=${env.portfolio_value - INITIAL_AMOUNT:+,.0f}")

    return actor, critic, history


# =============================================================================
# 7.  Metrics & evaluation
# =============================================================================

def compute_metrics(daily_pnl: list, asset_memory: list,
                    periods_per_year: int = 252) -> dict:
    pnl    = np.array(daily_pnl)
    values = np.array(asset_memory)

    total_profit       = float(values[-1] - INITIAL_AMOUNT)
    total_return_pct   = (values[-1] / INITIAL_AMOUNT - 1) * 100

    mean_p = pnl.mean()
    std_p  = pnl.std(ddof=1) + 1e-8
    annualised_sharpe  = (mean_p / std_p) * np.sqrt(periods_per_year)

    peak             = np.maximum.accumulate(values)
    max_drawdown_pct = ((peak - values) / (peak + 1e-8)).max() * 100

    downside = pnl[pnl < 0]
    down_std = (np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8) + 1e-8
    sortino_ratio = (mean_p / down_std) * np.sqrt(periods_per_year)

    n_years = len(pnl) / periods_per_year
    calmar  = (total_return_pct / 100) / (max_drawdown_pct / 100 + 1e-8) / n_years

    active_days    = np.sum(np.abs(pnl) > 0.01)
    utilisation_pct = active_days / len(pnl) * 100

    return {
        "Total Profit ($)":      round(total_profit,        2),
        "Total Return (%)":      round(total_return_pct,    4),
        "Annualised Sharpe":     round(annualised_sharpe,   4),
        "Max Drawdown (%)":      round(max_drawdown_pct,    4),
        "Sortino Ratio":         round(sortino_ratio,       4),
        "Calmar Ratio":          round(calmar,              4),
        "Utilisation (%)":       round(utilisation_pct,     2),
        "N Trading Days":        int(len(pnl)),
        "Final Value ($)":       round(float(values[-1]),   2),
    }


def evaluate(actor: nn.Module, env: BatteryEnv) -> dict:
    state, _ = env.reset()
    done      = False
    actor.eval()
    with torch.no_grad():
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mean, _ = actor(s_t)
            action_np = mean.cpu().numpy()
            state, _, done, _, _ = env.step(action_np)
    return compute_metrics(env.daily_pnl, env.asset_memory)


# =============================================================================
# 8.  Plotting
# =============================================================================

def plot_training_curves(histories: dict, save_path: str = "results/training_curves.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = {"baseline": "steelblue", "hard": "darkorange", "soft": "green"}
    for variant, h in histories.items():
        ma = pd.Series(h).rolling(20).mean()
        ax.plot(h,  alpha=0.25, color=colors[variant])
        ax.plot(ma, lw=2, color=colors[variant], label=f"{variant} (20-ep MA)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Episode Reward")
    ax.set_title("A2C Training — Energy Storage Arbitrage")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_regime_detection(save_path: str = "results/regimes.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    dates = test_df["date"].values

    ax = axes[0]
    ax.plot(dates, test_df["price"], color="steelblue", lw=0.8, alpha=0.7)
    ax.set_ylabel("Price ($/MWh)"); ax.set_title("Test Period: Prices & HMM Regimes")
    ax.grid(alpha=0.3)

    colors_r = {0: "royalblue", 1: "gold", 2: "darkorange", 3: "crimson"}
    labels_r = {0: "Off-peak", 1: "Normal", 2: "High demand", 3: "Scarcity"}
    ax2 = axes[1]
    for r in range(4):
        mask = test_df["regime"] == r
        ax2.scatter(dates[mask], [r] * mask.sum(),
                    c=colors_r[r], s=8, label=labels_r[r], alpha=0.6)
    ax2.set_ylabel("HMM Regime"); ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels([labels_r[i] for i in range(4)])
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_cumulative_pnl(all_curves: dict, save_path: str = "results/cumulative_pnl.png"):
    """
    all_curves : {label: asset_memory_list}
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    styles = {
        "Baseline A2C": ("steelblue",  "-",  2.0),
        "Hard Routing": ("darkorange", "-",  2.0),
        "Soft MoE":     ("green",      "-",  2.0),
        "Threshold":    ("grey",       "--", 1.4),
        "Time-of-Day":  ("brown",      "--", 1.4),
        "Momentum":     ("purple",     "-.", 1.4),
        "Oracle (LP)":  ("black",      ":",  1.6),
        "Do Nothing":   ("lightgrey",  "--", 1.0),
    }
    dates = test_df["date"].values
    for label, asset_mem in all_curves.items():
        cum_profit = (np.array(asset_mem) - INITIAL_AMOUNT) / 1000   # in $K
        d          = dates[:len(cum_profit)]
        col, ls, lw = styles.get(label, ("grey", "-", 1))
        ax.plot(d, cum_profit, color=col, linestyle=ls, linewidth=lw, label=label)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Cumulative Profit ($K)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Energy Storage Arbitrage: Regime-Aware RL vs Baselines (2022–2023)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_metrics_over_time(env: BatteryEnv, title_prefix: str,
                            save_path: str, rolling_window: int = 20):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pnl    = np.array(env.daily_pnl)
    values = np.array(env.asset_memory)
    dates  = env.date_memory

    cum_profit  = (values - INITIAL_AMOUNT) / 1000   # $K
    peak        = np.maximum.accumulate(values)
    drawdown    = -(peak - values) / (peak + 1e-8) * 100

    rs = pd.Series(pnl)
    roll_mean   = rs.rolling(rolling_window).mean()
    roll_std    = rs.rolling(rolling_window).std() + 1e-8
    roll_sharpe = np.concatenate([[np.nan], (roll_mean / roll_std * np.sqrt(252)).values])

    def _roll_sortino(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w - 1, len(arr)):
            win  = arr[i - w + 1: i + 1]
            m    = win.mean()
            d    = win[win < 0]
            dstd = (np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 1e-8) + 1e-8
            out[i] = m / dstd * np.sqrt(252)
        return out

    roll_sortino = np.concatenate([[np.nan], _roll_sortino(pnl, rolling_window)])

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"{title_prefix} — Metrics Over Time (Out-of-Sample Test)",
                 fontsize=13, fontweight="bold")

    plot_dates = list(dates)[:len(values)]

    axes[0].plot(plot_dates, cum_profit, color="steelblue", lw=1.5)
    axes[0].fill_between(plot_dates, 0, cum_profit,
                         where=(cum_profit >= 0), alpha=0.2, color="green")
    axes[0].fill_between(plot_dates, 0, cum_profit,
                         where=(cum_profit < 0), alpha=0.2, color="red")
    axes[0].axhline(0, color="grey", ls="--", lw=0.8)
    axes[0].set_ylabel("Cumulative Profit ($K)"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Cumulative Profit", fontweight="bold")

    axes[1].plot(plot_dates, roll_sharpe, color="purple", lw=1.5,
                 label=f"{rolling_window}-day rolling")
    axes[1].axhline(0, color="grey", ls="--", lw=0.8)
    axes[1].axhline(1, color="green", ls=":", lw=0.8, alpha=0.7, label="Sharpe=1")
    axes[1].set_ylabel("Sharpe (ann.)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title(f"Rolling {rolling_window}-Day Sharpe", fontweight="bold")

    axes[2].fill_between(plot_dates, 0, drawdown[:len(plot_dates)],
                         color="red", alpha=0.35)
    axes[2].plot(plot_dates, drawdown[:len(plot_dates)], color="darkred", lw=1, alpha=0.8)
    axes[2].set_ylabel("Drawdown (%)"); axes[2].grid(alpha=0.3)
    axes[2].set_title("Running Max Drawdown", fontweight="bold")

    axes[3].plot(plot_dates, roll_sortino, color="darkorange", lw=1.5,
                 label=f"{rolling_window}-day rolling")
    axes[3].axhline(0, color="grey", ls="--", lw=0.8)
    axes[3].axhline(1, color="green", ls=":", lw=0.8, alpha=0.7, label="Sortino=1")
    axes[3].set_ylabel("Sortino (ann.)"); axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)
    axes[3].set_title(f"Rolling {rolling_window}-Day Sortino", fontweight="bold")
    axes[3].set_xlabel("Date")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# =============================================================================
# 9.  run_experiment  (importable by run_all.py)
# =============================================================================

def run_experiment(variant: str, seed: int = 0,
                   out_dir: str = "results") -> dict:
    """
    Train + evaluate one variant for one seed.
    Returns metrics dict (including 'rewards' key with epoch history).
    Per-seed plots written to out_dir/.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    actor, critic, history = train_a2c(env_train[variant], variant)

    metrics = evaluate(actor, env_test[variant])
    metrics["seed"]         = seed
    metrics["rewards"]      = history
    metrics["asset_memory"] = list(env_test[variant].asset_memory)

    plot_metrics_over_time(
        env_test[variant],
        title_prefix=f"{variant.capitalize()} — Seed {seed}",
        save_path=os.path.join(out_dir, f"{variant}_seed{seed}_metrics.png"),
    )
    return metrics


# =============================================================================
# 10.  Standalone entry point
# =============================================================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_regime_detection()

    histories = {}
    actors    = {}
    all_metrics = {}

    for v in ("baseline", "hard", "soft"):
        print(f"\n{'='*55}\n  Variant: {v.upper()}\n{'='*55}")
        results           = run_experiment(v, seed=0, out_dir="results")
        histories[v]      = results["rewards"]
        all_metrics[v]    = {k: val for k, val in results.items()
                             if k not in ("rewards", "seed")}

    plot_training_curves(histories)

    print("\n" + "="*55)
    print("  OUT-OF-SAMPLE METRICS (seed=0)")
    print("="*55)
    cols = ["Total Profit ($)", "Total Return (%)", "Annualised Sharpe",
            "Max Drawdown (%)", "Sortino Ratio", "Calmar Ratio"]
    print(f"  {'Metric':<22s}  {'Baseline':>10s}  {'Hard':>10s}  {'Soft MoE':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for c in cols:
        row = "  " + f"{c:<22s}"
        for v in ("baseline", "hard", "soft"):
            row += f"  {all_metrics[v].get(c, 'N/A'):>10}"
        print(row)
    print("="*55)
