"""
Multi-seed runner for Energy Storage Arbitrage.

Runs all three RL variants × N_SEEDS, then adds non-RL baselines,
prints a paper-ready comparison table with 95 % bootstrap CIs, and
saves a combined cumulative-profit plot.

Usage (from Final/Energy_Storage_Arbitrage/):
  python run_all.py                 # 20 seeds, all variants
  python run_all.py --seeds 3       # quick smoke-test
  python run_all.py --variant hard  # one RL variant only
"""

import argparse
import json
import os

import numpy as np
import torch

# ── Import shared data + functions from main (triggers data gen + HMM fit) ──
from main import (
    run_experiment, compute_metrics, plot_training_curves,
    plot_cumulative_pnl, INITIAL_AMOUNT, test_df,
)
from baselines import run_all_baselines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPORT_KEYS = [
    "Total Profit ($)", "Total Return (%)", "Annualised Sharpe",
    "Max Drawdown (%)", "Sortino Ratio", "Calmar Ratio", "Utilisation (%)",
]


def _ser(obj):
    if isinstance(obj, dict):   return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_ser(v) for v in obj]
    if hasattr(obj, "tolist"):  return obj.tolist()
    return obj


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_ser(data), f, indent=2)
    print(f"  -> {path}")


def _bootstrap_ci(arr, n_boot=2000, ci=0.95):
    rng  = np.random.default_rng(0)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# RL multi-seed loop
# ---------------------------------------------------------------------------

def run_rl_seeds(variants: list, n_seeds: int, out_root: str) -> dict:
    """Returns {variant: [row_dict, ...]}"""
    all_rows      = {v: [] for v in variants}
    all_histories = {v: [] for v in variants}
    last_asset    = {v: None for v in variants}   # for combined plot

    for v in variants:
        vdir = os.path.join(out_root, v)
        os.makedirs(vdir, exist_ok=True)
        print(f"\n{'#'*60}\n  RL variant: {v.upper()}  ({n_seeds} seeds)\n{'#'*60}")

        for seed in range(n_seeds):
            print(f"\n  seed {seed}/{n_seeds-1}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            metrics      = run_experiment(v, seed=seed, out_dir=vdir)
            rewards      = metrics.pop("rewards", [])
            asset_memory = metrics.pop("asset_memory", [])
            all_histories[v].append(rewards)
            last_asset[v] = asset_memory   # keep most-recent seed for plot

            save_json(metrics, os.path.join(vdir, f"seed_{seed}.json"))
            save_json({"rewards": rewards},
                      os.path.join(vdir, f"seed_{seed}_rewards.json"))
            all_rows[v].append(metrics)

            print(f"    Profit ${metrics['Total Profit ($)']:+,.0f}  "
                  f"Sharpe {metrics['Annualised Sharpe']:.3f}")

            save_json({"runs": all_rows[v]}, os.path.join(vdir, "summary.json"))

    # Training curves (last seed of each variant)
    last_histories = {v: h[-1] for v, h in all_histories.items() if h}
    if last_histories:
        plot_training_curves(last_histories,
                             save_path=os.path.join(out_root, "training_curves.png"))

    return all_rows, last_asset


# ---------------------------------------------------------------------------
# Aggregate printing
# ---------------------------------------------------------------------------

def print_aggregate(label: str, rows: list) -> dict:
    stats = {}
    w = 72
    print(f"\n{'='*w}")
    print(f"  {label}  ({len(rows)} seeds, 95 % bootstrap CI)")
    print(f"{'='*w}")
    hdr = (f"  {'Metric':<24s}  {'Mean':>10s}  {'±Std':>8s}  "
           f"{'Median':>8s}  {'95% CI lo':>10s}  {'95% CI hi':>10s}")
    print(hdr)
    print(f"  {'-'*24}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for k in REPORT_KEYS:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        lo, hi = _bootstrap_ci(arr)
        stats[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                    "median": float(np.median(arr)),
                    "ci95_lo": lo, "ci95_hi": hi, "n": len(arr)}
        print(f"  {k:<24s}  {arr.mean():10.3f}  {arr.std(ddof=1):8.3f}  "
              f"{np.median(arr):8.3f}  {lo:10.3f}  {hi:10.3f}")
    print(f"{'='*w}")
    return stats


def print_comparison_table(rl_rows: dict, baseline_metrics: dict):
    """Side-by-side table: RL (mean over seeds) vs baseline (single run)."""
    w = 90
    print(f"\n{'='*w}")
    print("  FULL COMPARISON TABLE (RL = mean across seeds)")
    print(f"{'='*w}")

    all_labels = (
        [f"{v.capitalize()} A2C" if v == "baseline" else
         "Hard Routing" if v == "hard" else "Soft MoE"
         for v in rl_rows]
        + list(baseline_metrics.keys())
    )

    # Header
    header = f"  {'Method':<18s}"
    for k in REPORT_KEYS[:5]:     # keep table width manageable
        short = k.split("(")[0].strip()[:11]
        header += f"  {short:>11s}"
    print(header)
    print(f"  {'-'*18}" + ("  " + "-"*11) * 5)

    # RL rows (mean over seeds)
    label_map = {"baseline": "Baseline A2C", "hard": "Hard Routing", "soft": "Soft MoE"}
    for v, rows in rl_rows.items():
        means = {k: np.mean([r[k] for r in rows if k in r]) for k in REPORT_KEYS}
        row_s = f"  {label_map[v]:<18s}"
        for k in REPORT_KEYS[:5]:
            row_s += f"  {means.get(k, float('nan')):11.3f}"
        print(row_s)

    # Baseline rows (single run)
    for name, m in baseline_metrics.items():
        row_s = f"  {name:<18s}"
        for k in REPORT_KEYS[:5]:
            row_s += f"  {m.get(k, float('nan')):11.3f}"
        print(row_s)

    print(f"{'='*w}")


def print_significance(rl_rows: dict):
    """Welch's t-test between Soft MoE and Hard Routing."""
    from scipy import stats as scipy_stats
    if "soft" not in rl_rows or "hard" not in rl_rows:
        return
    w = 72
    print(f"\n{'='*w}")
    print("  SIGNIFICANCE: Soft MoE vs Hard Routing (Welch t-test)")
    print(f"{'='*w}")
    print(f"  {'Metric':<24s}  {'Soft mean':>10s}  {'Hard mean':>10s}  "
          f"{'Δ':>10s}  {'p-val':>8s}  {'sig':>4s}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*4}")
    for k in REPORT_KEYS[:5]:
        s = np.array([r[k] for r in rl_rows["soft"] if k in r])
        h = np.array([r[k] for r in rl_rows["hard"] if k in r])
        if len(s) < 2 or len(h) < 2:
            continue
        _, p = scipy_stats.ttest_ind(s, h, equal_var=False)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {k:<24s}  {s.mean():10.3f}  {h.mean():10.3f}  "
              f"{s.mean()-h.mean():+10.3f}  {p:8.4f}  {sig:>4s}")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, default=20)
    parser.add_argument("--variant", choices=["baseline", "hard", "soft", "all"],
                        default="all")
    args = parser.parse_args()

    variants = (["baseline", "hard", "soft"] if args.variant == "all"
                else [args.variant])
    out_root = "results"
    os.makedirs(out_root, exist_ok=True)

    # ── RL experiments ───────────────────────────────────────────────────
    rl_rows, rl_asset_memory = run_rl_seeds(variants, args.seeds, out_root)

    # ── Aggregate stats per RL variant ───────────────────────────────────
    rl_stats = {}
    label_map = {"baseline": "Baseline A2C", "hard": "Hard Routing", "soft": "Soft MoE"}
    for v, rows in rl_rows.items():
        rl_stats[v] = print_aggregate(label_map[v], rows)
        save_json(rl_stats[v],
                  os.path.join(out_root, v, "aggregate_stats.json"))

    # ── Significance test ─────────────────────────────────────────────────
    if len(rl_rows) > 1:
        print_significance(rl_rows)

    # ── Baselines ─────────────────────────────────────────────────────────
    baseline_results = run_all_baselines()
    baseline_metrics = {}
    for name, res in baseline_results.items():
        baseline_metrics[name] = compute_metrics(
            res["daily_pnl"], res["asset_memory"]
        )
        save_json(baseline_metrics[name],
                  os.path.join(out_root, "baselines",
                               name.replace(" ", "_").replace("/", "") + ".json"))

    # ── Full comparison table ─────────────────────────────────────────────
    print_comparison_table(rl_rows, baseline_metrics)

    # ── Combined cumulative profit plot ───────────────────────────────────
    label_map  = {"baseline": "Baseline A2C", "hard": "Hard Routing", "soft": "Soft MoE"}
    plot_curves = {}
    for v, asset_mem in rl_asset_memory.items():
        if asset_mem:
            plot_curves[label_map[v]] = asset_mem
    for name, res in baseline_results.items():
        plot_curves[name] = res["asset_memory"]

    plot_cumulative_pnl(plot_curves,
                        save_path=os.path.join(out_root, "cumulative_pnl_all.png"))

    print(f"\nAll done. Results in ./{out_root}/")
