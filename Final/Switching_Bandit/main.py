"""
Single-experiment runner for the Switching Multi-Armed Bandit testbed.

For one (p_switch, seed) combination:
  1. Fit the GaussianHMM on a uniform-policy trajectory.
  2. Train baseline / hard / soft A2C agents.
  3. Evaluate ALL agents on a SHARED evaluation trajectory:
       - shared eval seed  →  identical regime sequence
       - 3 RL variants (greedy)
       - Oracle (cheats; argmax over the true regime row)
       - All classical baselines from classical.CLASSICAL_ALGORITHMS
  4. Compute cumulative regret per step using EXPECTED reward differences
     (de-noised: oracle_mean − agent_mean rather than realised rewards).
  5. Save curves + per-experiment plots.

Stand-alone use:  python main.py --p_switch 0.05 --seed 0 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bandit_env import (
    SwitchingBanditEnv, Oracle, fit_hmm_on_uniform,
    K_ARMS, N_REGIMES, REWARD_MEANS, BEST_ARM_PER_REGIME, BEST_MEAN_PER_REGIME,
)
from train import train_a2c, evaluate_actor
from classical import CLASSICAL_ALGORITHMS


DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


# =============================================================================
# Shared evaluation trajectory
# =============================================================================

def _generate_eval_trajectory(p_switch: float, n_steps: int, eval_seed: int):
    """
    Pre-roll the regime sequence so all algorithms (RL + classical + oracle)
    see exactly the same z_t sequence and the same reward draws conditional
    on (z_t, a_t).  Each algorithm picks its own a_t, then the env consumes
    the same shared rng draws to produce r_t.
    """
    env = SwitchingBanditEnv(p_switch=p_switch, n_steps=n_steps,
                              variant="oracle", seed=eval_seed)
    env.reset(seed=eval_seed)
    return env


def _evaluate_oracle(p_switch: float, n_steps: int, eval_seed: int):
    env       = _generate_eval_trajectory(p_switch, n_steps, eval_seed)
    oracle    = Oracle()
    cum_regret = []
    running    = 0.0
    total_r    = 0.0
    state, _ = env.reset(seed=eval_seed)
    done = False
    while not done:
        a = oracle.select_arm(env.current_regime)
        state, r, done, _, info = env.step(a)
        total_r += r
        running += info["regret"]                  # always 0 for oracle
        cum_regret.append(running)
    return total_r, cum_regret


def _evaluate_classical(alg_factory, p_switch: float, n_steps: int, eval_seed: int):
    env  = SwitchingBanditEnv(p_switch=p_switch, n_steps=n_steps,
                                variant="baseline", seed=eval_seed)
    env.reset(seed=eval_seed)
    alg  = alg_factory(K_ARMS, n_steps, eval_seed)

    total_r    = 0.0
    cum_regret = []
    running    = 0.0
    done       = False
    t          = 0
    while not done:
        a = alg.select_arm(t)
        _, r, done, _, info = env.step(a)
        alg.update(a, r, t)
        total_r += r
        running += info["regret"]
        cum_regret.append(running)
        t += 1
    return total_r, cum_regret


# =============================================================================
# Single experiment
# =============================================================================

def run_single_experiment(p_switch: float = 0.05,
                          seed: int = 0,
                          epochs: int = 200,
                          train_n_steps: int = 1000,
                          eval_n_steps: int = 2000,
                          hmm_uniform_steps: int = 10_000,
                          window: int = 50,
                          device: str = DEVICE,
                          verbose: bool = False) -> dict:
    """
    Returns
    -------
    dict with:
      'p_switch', 'seed'
      'cum_regret'   : {method: list[float]}    (length eval_n_steps)
      'final_regret' : {method: float}
      'total_reward' : {method: float}
      'histories'    : {variant: list[float]}   (RL training curves)
    """
    # ── 1. Fit HMM (deterministic per p_switch — independent of RL seed) ──
    hmm = fit_hmm_on_uniform(p_switch, n_steps=hmm_uniform_steps,
                              window=window, seed=42)

    # ── 2. Set RL seeds and train each variant ────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    histories = {}
    actors    = {}
    for variant in ("baseline", "hard", "soft"):
        train_env = SwitchingBanditEnv(
            p_switch=p_switch, n_steps=train_n_steps,
            variant=variant, hmm=hmm, window=window,
            seed=seed * 100 + (hash(variant) % 97),
        )
        if verbose:
            print(f"  Training [{variant}]  p_switch={p_switch}  seed={seed}")
        actor, _, hist = train_a2c(train_env, variant, device,
                                    epochs=epochs, verbose=verbose)
        actors[variant]    = actor
        histories[variant] = hist

    # ── 3. Shared evaluation trajectory (eval_seed deterministic per seed) ─
    eval_seed = seed * 10_000 + 7

    cum_regret    = {}
    total_reward  = {}

    # Oracle
    tot_r, cum_r = _evaluate_oracle(p_switch, eval_n_steps, eval_seed)
    cum_regret["Oracle"]   = cum_r
    total_reward["Oracle"] = tot_r

    # 3 RL variants (greedy)
    for variant in ("baseline", "hard", "soft"):
        eval_env = SwitchingBanditEnv(
            p_switch=p_switch, n_steps=eval_n_steps,
            variant=variant, hmm=hmm, window=window, seed=eval_seed,
        )
        tot_r, cum_r, _ = evaluate_actor(actors[variant], eval_env, device, eval_seed)
        label = {"baseline": "Baseline A2C",
                 "hard":     "Hard Routing",
                 "soft":     "Soft MoE"}[variant]
        cum_regret[label]   = cum_r
        total_reward[label] = tot_r

    # All classical baselines
    for name, factory in CLASSICAL_ALGORITHMS.items():
        tot_r, cum_r = _evaluate_classical(factory, p_switch, eval_n_steps, eval_seed)
        cum_regret[name]   = cum_r
        total_reward[name] = tot_r

    final_regret = {m: float(c[-1]) if c else 0.0 for m, c in cum_regret.items()}

    return {
        "p_switch":     p_switch,
        "seed":         seed,
        "cum_regret":   {m: list(map(float, c)) for m, c in cum_regret.items()},
        "final_regret": final_regret,
        "total_reward": {m: float(v) for m, v in total_reward.items()},
        "histories":    histories,
    }


# =============================================================================
# Plots
# =============================================================================

# Plotting style: RL variants thick coloured, classical thin/dashed grey-ish
RL_STYLES = {
    "Oracle":        {"color": "black",      "lw": 2.0, "ls": ":",  "zorder": 5},
    "Soft MoE":      {"color": "green",      "lw": 2.4, "ls": "-",  "zorder": 6},
    "Hard Routing":  {"color": "darkorange", "lw": 2.0, "ls": "-",  "zorder": 4},
    "Baseline A2C":  {"color": "steelblue",  "lw": 2.0, "ls": "-",  "zorder": 3},
}
CLASSICAL_STYLES = {
    "UCB1":      {"color": "#888888", "lw": 1.0, "ls": "--"},
    "EpsGreedy": {"color": "#aa6e8b", "lw": 1.0, "ls": "--"},
    "Thompson":  {"color": "#9467bd", "lw": 1.2, "ls": "--"},
    "SW-UCB":    {"color": "#1f77b4", "lw": 1.0, "ls": "-."},
    "D-UCB":     {"color": "#ff7f0e", "lw": 1.0, "ls": "-."},
    "EXP3.S":    {"color": "#d62728", "lw": 1.0, "ls": "-."},
    "CUSUM-UCB": {"color": "#8c564b", "lw": 1.2, "ls": ":"},
    "M-UCB":     {"color": "#e377c2", "lw": 1.2, "ls": ":"},
    "GLR-UCB":   {"color": "#2ca02c", "lw": 1.2, "ls": ":"},
}


def _style_for(method):
    if method in RL_STYLES:        return RL_STYLES[method]
    if method in CLASSICAL_STYLES: return CLASSICAL_STYLES[method]
    return {"color": "grey", "lw": 1.0, "ls": "-"}


def plot_single_experiment(result: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"p{result['p_switch']:.3f}_seed{result['seed']}"

    # ── Cumulative regret curves (all methods) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for method, curve in result["cum_regret"].items():
        ax.plot(curve, label=method, **_style_for(method))
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Cumulative regret  (oracle mean − agent mean)", fontsize=11)
    ax.set_title(f"Switching MAB — Cumulative Regret  "
                 f"(p_switch={result['p_switch']}, seed={result['seed']})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"regret_{suffix}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Training curves (RL variants only) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for variant, h in result["histories"].items():
        col = RL_STYLES[{"baseline": "Baseline A2C", "hard": "Hard Routing",
                          "soft": "Soft MoE"}[variant]]["color"]
        ax.plot(h, alpha=0.3, color=col)
        import pandas as pd
        ma = pd.Series(h).rolling(20).mean()
        ax.plot(ma, lw=2, color=col, label=f"{variant} (20-ep MA)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Episode reward")
    ax.set_title(f"A2C Training — Switching MAB  "
                 f"(p_switch={result['p_switch']}, seed={result['seed']})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_{suffix}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def _save_result(result, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_switch", type=float, default=0.05)
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--eval_steps",  type=int, default=2000)
    parser.add_argument("--out_dir",  type=str,   default="results")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print(f"[Switching MAB]  device={DEVICE}  "
          f"p_switch={args.p_switch}  seed={args.seed}  epochs={args.epochs}")

    result = run_single_experiment(
        p_switch          = args.p_switch,
        seed              = args.seed,
        epochs            = args.epochs,
        train_n_steps     = args.train_steps,
        eval_n_steps      = args.eval_steps,
        verbose           = args.verbose,
    )

    suffix = f"p{args.p_switch:.3f}_seed{args.seed}"
    _save_result(result, os.path.join(args.out_dir, f"experiment_{suffix}.json"))
    plot_single_experiment(result, args.out_dir)

    print("\n" + "=" * 70)
    print(f"  Final cumulative regret  (p_switch={args.p_switch}, seed={args.seed})")
    print("=" * 70)
    sorted_methods = sorted(result["final_regret"].items(), key=lambda kv: kv[1])
    print(f"  {'Method':<18s}  {'Final regret':>14s}  {'Total reward':>14s}")
    print(f"  {'-'*18}  {'-'*14}  {'-'*14}")
    for m, r in sorted_methods:
        print(f"  {m:<18s}  {r:14.3f}  {result['total_reward'][m]:14.3f}")
    print("=" * 70)
    print(f"\nSaved → {args.out_dir}/experiment_{suffix}.json")
