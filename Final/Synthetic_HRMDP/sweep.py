"""
Sweep over regime-switching rates × seeds → produce paper Figure 1.

For each p_switch in P_SWITCH_GRID and each seed in 0..n_seeds-1:
  run main.run_single_experiment → record final regret per variant.

Then plot:
  Figure 1   regret vs. switching rate, mean ± 95% bootstrap CI per variant
             (Oracle ceiling at 0 plotted as a dashed line).

Outputs (under Final/Synthetic_HRMDP/results/):
  experiment_p<p>_seed<s>.json     — every individual run
  sweep_results.json               — flat list with all final-regret rows
  aggregate_stats.json             — mean / std / 95% CI per (p_switch, variant)
  figure1_regret_vs_pswitch.png    — the headline figure
  per_pswitch_regret_curves.png    — supplementary: regret-over-time by p_switch
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import run_single_experiment, plot_single_experiment, DEVICE


# Default difficulty grid — log-spaced from "almost no switching" up to
# "regimes change every other day on average".  Tunable via --p_switch_grid.
DEFAULT_P_SWITCH_GRID = [0.005, 0.02, 0.05, 0.10, 0.20, 0.50]
VARIANTS              = ("baseline", "hard", "soft")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, ci: float = 0.95):
    rng    = np.random.default_rng(0)
    boot   = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def _ser(x):
        if isinstance(x, dict):  return {k: _ser(v) for k, v in x.items()}
        if isinstance(x, list):  return [_ser(v) for v in x]
        if hasattr(x, "tolist"): return x.tolist()
        return x
    with open(path, "w") as f:
        json.dump(_ser(obj), f, indent=2)


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

def run_sweep(p_switch_grid, n_seeds: int, epochs: int,
              train_len: int, eval_len: int,
              save_per_seed_plots: bool, out_dir: str) -> tuple[list, dict]:

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    # nested dict: results_by_p[p_switch][variant] -> list of arrays (one per seed)
    regret_curves = defaultdict(lambda: defaultdict(list))

    n_runs = len(p_switch_grid) * n_seeds
    run_idx = 0

    for p in p_switch_grid:
        for s in range(n_seeds):
            run_idx += 1
            print(f"\n[{run_idx}/{n_runs}]  p_switch={p}   seed={s}")

            result = run_single_experiment(
                p_switch=p, seed=s,
                epochs=epochs,
                train_episode_len=train_len,
                eval_episode_len=eval_len,
                device=DEVICE,
                verbose=False,
            )

            # Per-seed JSON + plots
            suffix = f"p{p:.3f}_seed{s}"
            _save_json(result, os.path.join(out_dir, f"experiment_{suffix}.json"))
            if save_per_seed_plots:
                plot_single_experiment(result, save_dir=out_dir)

            # Aggregate row (final regret per variant)
            row = {"p_switch": p, "seed": s,
                   **{f"final_regret_{v}":     result["final_regret"][v]
                      for v in VARIANTS},
                   **{f"final_logwealth_{v}":  result["final_log_wealth"][v]
                      for v in VARIANTS + ("oracle",)}}
            rows.append(row)

            # Cache regret-over-time arrays for the supplementary figure
            for v in VARIANTS:
                regret_curves[p][v].append(np.asarray(result["regrets"][v]))

            print(f"   final regret  →  baseline {row['final_regret_baseline']:+.4f}"
                  f"   hard {row['final_regret_hard']:+.4f}"
                  f"   soft {row['final_regret_soft']:+.4f}")

    _save_json(rows, os.path.join(out_dir, "sweep_results.json"))
    return rows, regret_curves


# -----------------------------------------------------------------------------
# Aggregate stats
# -----------------------------------------------------------------------------

def aggregate(rows: list, p_switch_grid) -> dict:
    """For each (p_switch, variant) compute mean / std / 95% bootstrap CI."""
    stats = {}
    for p in p_switch_grid:
        stats[str(p)] = {}
        for v in VARIANTS:
            vals = np.array([r[f"final_regret_{v}"]
                              for r in rows if r["p_switch"] == p], dtype=float)
            if vals.size == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            stats[str(p)][v] = {
                "mean":   float(vals.mean()),
                "std":    float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "ci95_lo": lo, "ci95_hi": hi,
                "n": int(vals.size),
            }
    return stats


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------

def plot_figure1(stats: dict, p_switch_grid, save_path: str):
    colors = {"baseline": "steelblue", "hard": "darkorange", "soft": "green"}
    labels = {"baseline": "Baseline A2C",
              "hard":     "Hard Routing",
              "soft":     "Soft MoE"}

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.array(p_switch_grid, dtype=float)

    for v in VARIANTS:
        means = np.array([stats[str(p)][v]["mean"]    for p in p_switch_grid])
        los   = np.array([stats[str(p)][v]["ci95_lo"] for p in p_switch_grid])
        his   = np.array([stats[str(p)][v]["ci95_hi"] for p in p_switch_grid])
        ax.plot(xs, means, "o-", color=colors[v], lw=2, ms=7, label=labels[v])
        ax.fill_between(xs, los, his, color=colors[v], alpha=0.15)

    ax.axhline(0.0, color="black", ls="--", lw=1.0, label="Oracle ceiling (regret = 0)")
    ax.set_xscale("log")
    ax.set_xlabel("Per-step regime switch probability  p_switch", fontsize=12)
    ax.set_ylabel("Final regret  (log-wealth: oracle − agent)", fontsize=12)
    ax.set_title("Figure 1 — Regret vs. Regime-Switching Rate (Synthetic HR-MDP)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_regret_curves_per_p(regret_curves, p_switch_grid, save_path: str):
    """Supplementary: small-multiples of regret curves, one panel per p_switch."""
    n   = len(p_switch_grid)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows),
                              sharey=False)
    axes = np.array(axes).reshape(-1)

    colors = {"baseline": "steelblue", "hard": "darkorange", "soft": "green"}
    labels = {"baseline": "Baseline", "hard": "Hard", "soft": "Soft MoE"}

    for ax, p in zip(axes, p_switch_grid):
        for v in VARIANTS:
            curves = regret_curves[p][v]
            if not curves:  continue
            min_len = min(len(c) for c in curves)
            arr = np.stack([c[:min_len] for c in curves])    # (seeds, T)
            mean = arr.mean(axis=0); std = arr.std(axis=0)
            xs = np.arange(min_len)
            ax.plot(xs, mean, color=colors[v], lw=1.6, label=labels[v])
            ax.fill_between(xs, mean - std, mean + std, color=colors[v], alpha=0.15)
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.set_title(f"p_switch = {p}")
        ax.set_xlabel("Step"); ax.set_ylabel("Regret"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n:]:    ax.set_visible(False)
    plt.suptitle("Regret-over-time by switching rate (mean ± std across seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------

def print_comparison_table(stats: dict, p_switch_grid):
    print("\n" + "=" * 86)
    print("  Final Regret  (lower = better)   |   95% bootstrap CI in [ ]")
    print("=" * 86)
    hdr = f"  {'p_switch':>9s}  "
    for v in VARIANTS:
        hdr += f"  {v.capitalize():>26s}"
    print(hdr)
    print("  " + "-" * 9 + "  " + ("  " + "-" * 26) * 3)
    for p in p_switch_grid:
        row = f"  {p:>9.3f}  "
        for v in VARIANTS:
            s = stats[str(p)][v]
            cell = f"{s['mean']:+.4f}  [{s['ci95_lo']:+.3f},{s['ci95_hi']:+.3f}]"
            row += f"  {cell:>26s}"
        print(row)
    print("=" * 86)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int,   default=4)
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--train_len", type=int,   default=500)
    parser.add_argument("--eval_len",  type=int,   default=2000)
    parser.add_argument("--p_switch_grid", type=float, nargs="+",
                        default=DEFAULT_P_SWITCH_GRID,
                        help="Space-separated list of p_switch values.")
    parser.add_argument("--out_dir",   type=str,   default="results")
    parser.add_argument("--no_per_seed_plots", action="store_true",
                        help="Skip per-seed wealth/regret/training PNGs.")
    args = parser.parse_args()

    print(f"[Sweep]   device={DEVICE}")
    print(f"          p_switch grid = {args.p_switch_grid}")
    print(f"          seeds = {args.seeds}   epochs = {args.epochs}")
    print(f"          train_len = {args.train_len}   eval_len = {args.eval_len}")

    rows, regret_curves = run_sweep(
        p_switch_grid       = args.p_switch_grid,
        n_seeds             = args.seeds,
        epochs              = args.epochs,
        train_len           = args.train_len,
        eval_len            = args.eval_len,
        save_per_seed_plots = not args.no_per_seed_plots,
        out_dir             = args.out_dir,
    )

    stats = aggregate(rows, args.p_switch_grid)
    _save_json(stats, os.path.join(args.out_dir, "aggregate_stats.json"))

    print_comparison_table(stats, args.p_switch_grid)

    plot_figure1(
        stats, args.p_switch_grid,
        save_path=os.path.join(args.out_dir, "figure1_regret_vs_pswitch.png"),
    )
    plot_regret_curves_per_p(
        regret_curves, args.p_switch_grid,
        save_path=os.path.join(args.out_dir, "per_pswitch_regret_curves.png"),
    )

    print(f"\nDone.  All outputs in ./{args.out_dir}/")
