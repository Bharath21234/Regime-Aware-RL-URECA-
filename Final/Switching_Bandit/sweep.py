"""
Multi-seed sweep over regime-switching rates → headline figures and tables.

For each (p_switch, seed):
  call main.run_single_experiment → record final regret per method.

Outputs:
  Final/Switching_Bandit/results/
    experiment_p<p>_seed<s>.json   — per-run JSON (all curves + histories)
    sweep_results.json              — flat list of all final-regret rows
    aggregate_stats.json            — mean / std / 95% bootstrap CI per (p, method)
    figure_regret_vs_pswitch.png    — headline: final regret vs p_switch
    regret_curves_by_pswitch.png    — supplementary: small multiples
    comparison_table.txt            — paper-ready table
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import (
    run_single_experiment, plot_single_experiment, DEVICE,
    RL_STYLES, CLASSICAL_STYLES, _style_for,
)


DEFAULT_P_SWITCH_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

ALL_METHODS = (
    ["Oracle", "Soft MoE", "Hard Routing", "Baseline A2C"]
    + ["UCB1", "EpsGreedy", "Thompson", "SW-UCB", "D-UCB",
       "EXP3.S", "CUSUM-UCB", "M-UCB", "GLR-UCB"]
)
RL_METHODS = ["Soft MoE", "Hard Routing", "Baseline A2C"]


# =============================================================================
# Helpers
# =============================================================================

def _bootstrap_ci(arr, n_boot=2000, ci=0.95):
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


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(p_switch_grid, n_seeds, epochs, train_steps, eval_steps,
              save_per_seed_plots, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows           = []                          # flat row per (p, seed)
    regret_curves  = defaultdict(lambda: defaultdict(list))   # [p][method] -> list of curves

    n_runs  = len(p_switch_grid) * n_seeds
    run_idx = 0

    for p in p_switch_grid:
        for s in range(n_seeds):
            run_idx += 1
            print(f"\n[{run_idx}/{n_runs}]  p_switch={p}  seed={s}")
            result = run_single_experiment(
                p_switch      = p,
                seed          = s,
                epochs        = epochs,
                train_n_steps = train_steps,
                eval_n_steps  = eval_steps,
                device        = DEVICE,
                verbose       = False,
            )

            suffix = f"p{p:.3f}_seed{s}"
            _save_json(result, os.path.join(out_dir, f"experiment_{suffix}.json"))
            if save_per_seed_plots:
                plot_single_experiment(result, save_dir=out_dir)

            row = {"p_switch": p, "seed": s,
                   **{f"final_regret_{m}":  result["final_regret"][m]   for m in ALL_METHODS},
                   **{f"total_reward_{m}":  result["total_reward"][m]   for m in ALL_METHODS}}
            rows.append(row)

            for m in ALL_METHODS:
                regret_curves[p][m].append(np.asarray(result["cum_regret"][m]))

            top3 = sorted(((m, result["final_regret"][m]) for m in ALL_METHODS),
                          key=lambda kv: kv[1])[:5]
            print(f"   top-5 (lowest regret):  " +
                  "  ".join(f"{m}={v:.2f}" for m, v in top3))

    _save_json(rows, os.path.join(out_dir, "sweep_results.json"))
    return rows, regret_curves


# =============================================================================
# Aggregate stats
# =============================================================================

def aggregate(rows, p_switch_grid):
    stats = {}
    for p in p_switch_grid:
        stats[str(p)] = {}
        for m in ALL_METHODS:
            vals = np.array([r[f"final_regret_{m}"]
                              for r in rows if r["p_switch"] == p], dtype=float)
            if vals.size == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            stats[str(p)][m] = {
                "mean":   float(vals.mean()),
                "std":    float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "ci95_lo": lo, "ci95_hi": hi,
                "n":     int(vals.size),
            }
    return stats


# =============================================================================
# Headline figure: final regret vs p_switch
# =============================================================================

def plot_figure_regret_vs_pswitch(stats, p_switch_grid, save_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    xs = np.asarray(p_switch_grid, dtype=float)

    # Plot RL methods (thick) + classical (thin)
    plot_order = (["Oracle"] + RL_METHODS
                  + ["UCB1", "EpsGreedy", "Thompson", "SW-UCB", "D-UCB",
                     "EXP3.S", "CUSUM-UCB", "M-UCB", "GLR-UCB"])

    for m in plot_order:
        means = np.array([stats[str(p)][m]["mean"]    for p in p_switch_grid])
        los   = np.array([stats[str(p)][m]["ci95_lo"] for p in p_switch_grid])
        his   = np.array([stats[str(p)][m]["ci95_hi"] for p in p_switch_grid])
        st    = _style_for(m)
        ax.plot(xs, means, marker="o", ms=5, label=m, **st)
        ax.fill_between(xs, los, his, color=st["color"], alpha=0.10)

    ax.set_xscale("log")
    ax.set_xlabel("Per-step regime switch probability  $p_{switch}$  (log)", fontsize=12)
    ax.set_ylabel("Final cumulative regret  (lower = better)", fontsize=12)
    ax.set_title("Switching MAB — Regret vs. Switching Rate  "
                 "(mean ± 95% bootstrap CI)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# =============================================================================
# Supplementary: regret-curves-by-pswitch (small multiples)
# =============================================================================

def plot_regret_curves_by_p(regret_curves, p_switch_grid, save_path):
    n    = len(p_switch_grid)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharey=False)
    axes = np.array(axes).reshape(-1)

    plot_order = ["Oracle", "Soft MoE", "Hard Routing", "Baseline A2C",
                  "GLR-UCB", "M-UCB", "SW-UCB", "Thompson", "UCB1"]

    for ax, p in zip(axes, p_switch_grid):
        for m in plot_order:
            curves = regret_curves[p][m]
            if not curves:  continue
            min_len = min(len(c) for c in curves)
            arr     = np.stack([c[:min_len] for c in curves])
            mean    = arr.mean(axis=0)
            std     = arr.std(axis=0)
            xs      = np.arange(min_len)
            st      = _style_for(m)
            ax.plot(xs, mean, label=m, **{**st, "lw": st.get("lw", 1.0)})
            ax.fill_between(xs, mean - std, mean + std, color=st["color"], alpha=0.10)
        ax.set_title(f"$p_{{switch}}$ = {p}")
        ax.set_xlabel("Step"); ax.set_ylabel("Cumulative regret")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[n:]:    ax.set_visible(False)
    plt.suptitle("Cumulative regret over time, by switching rate (mean ± std)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# =============================================================================
# Comparison table
# =============================================================================

def write_comparison_table(stats, p_switch_grid, save_path):
    lines = []
    lines.append("=" * 110)
    lines.append("  SWITCHING MAB — Final cumulative regret  (mean ± std,  95% bootstrap CI)")
    lines.append("=" * 110)
    header = f"  {'Method':<14s}  " + "  ".join(f"{f'p={p}':>17s}" for p in p_switch_grid)
    lines.append(header)
    lines.append("  " + "-" * 14 + "  " + "  ".join("-" * 17 for _ in p_switch_grid))

    plot_order = ["Oracle", "Soft MoE", "Hard Routing", "Baseline A2C",
                  "UCB1", "EpsGreedy", "Thompson",
                  "SW-UCB", "D-UCB", "EXP3.S",
                  "CUSUM-UCB", "M-UCB", "GLR-UCB"]
    for m in plot_order:
        cells = []
        for p in p_switch_grid:
            s = stats[str(p)].get(m)
            if s is None:
                cells.append(f"{'N/A':>17s}")
            else:
                cells.append(f"{s['mean']:7.2f}±{s['std']:5.2f}  ")
        lines.append(f"  {m:<14s}  " + "  ".join(cells))
    lines.append("=" * 110)

    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved {save_path}")


# =============================================================================
# Significance test:  Soft MoE vs each other method (Welch t-test)
# =============================================================================

def print_significance(rows, p_switch_grid):
    from scipy import stats as scipy_stats
    print(f"\n{'='*90}")
    print("  Welch's t-test  —  Soft MoE  vs  each baseline   (final regret across seeds)")
    print(f"{'='*90}")
    print(f"  {'Method':<14s}  " + "  ".join(f"{f'p={p}':>10s}" for p in p_switch_grid))
    print(f"  {'-'*14}  " + "  ".join("-" * 10 for _ in p_switch_grid))

    soft_vals_per_p = {p: np.array([r[f"final_regret_Soft MoE"]
                                     for r in rows if r["p_switch"] == p])
                       for p in p_switch_grid}

    for m in ALL_METHODS:
        if m == "Soft MoE":   continue
        cells = []
        for p in p_switch_grid:
            others = np.array([r[f"final_regret_{m}"] for r in rows if r["p_switch"] == p])
            soft   = soft_vals_per_p[p]
            if soft.size < 2 or others.size < 2:
                cells.append(f"{'-':>10s}")
                continue
            _, pval = scipy_stats.ttest_ind(soft, others, equal_var=False)
            sig     = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            cells.append(f"{pval:.3f}{sig:>4s}")
        print(f"  {m:<14s}  " + "  ".join(cells))

    print(f"  * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
    print(f"{'='*90}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int,  default=4)
    parser.add_argument("--epochs",       type=int,  default=200)
    parser.add_argument("--train_steps",  type=int,  default=1000)
    parser.add_argument("--eval_steps",   type=int,  default=2000)
    parser.add_argument("--p_switch_grid", type=float, nargs="+",
                        default=DEFAULT_P_SWITCH_GRID)
    parser.add_argument("--out_dir",      type=str,  default="results")
    parser.add_argument("--no_per_seed_plots", action="store_true")
    args = parser.parse_args()

    print(f"[Sweep]   device={DEVICE}")
    print(f"          p_switch grid = {args.p_switch_grid}")
    print(f"          seeds={args.seeds}   epochs={args.epochs}   "
          f"train_steps={args.train_steps}   eval_steps={args.eval_steps}")

    rows, regret_curves = run_sweep(
        p_switch_grid       = args.p_switch_grid,
        n_seeds             = args.seeds,
        epochs              = args.epochs,
        train_steps         = args.train_steps,
        eval_steps          = args.eval_steps,
        save_per_seed_plots = not args.no_per_seed_plots,
        out_dir             = args.out_dir,
    )

    stats = aggregate(rows, args.p_switch_grid)
    _save_json(stats, os.path.join(args.out_dir, "aggregate_stats.json"))

    write_comparison_table(stats, args.p_switch_grid,
                            os.path.join(args.out_dir, "comparison_table.txt"))

    plot_figure_regret_vs_pswitch(
        stats, args.p_switch_grid,
        save_path=os.path.join(args.out_dir, "figure_regret_vs_pswitch.png"),
    )
    plot_regret_curves_by_p(
        regret_curves, args.p_switch_grid,
        save_path=os.path.join(args.out_dir, "regret_curves_by_pswitch.png"),
    )

    print_significance(rows, args.p_switch_grid)

    print(f"\nDone.  All outputs in ./{args.out_dir}/")
