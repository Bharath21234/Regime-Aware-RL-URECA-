"""
Multi-seed experiment runner for regime-aware RL portfolio variants.

Runs Hard Routing (Select_1) and Soft MoE (Select_3) across N_SEEDS seeds.
Data is downloaded and preprocessed ONCE per variant (on module import),
then training is repeated N_SEEDS times with different random initialisations.

Output layout:
  Final/results/hard/seed_{n}.json          — scalar metrics
  Final/results/hard/seed_{n}_training.png  — training reward curve
  Final/results/hard/seed_{n}_metrics_over_time.png
  Final/results/hard/summary.json           — all seeds aggregated
  Final/results/moe/seed_{n}.json
  ...

Usage (run from the Final/ directory):
  python run.py                        # both variants, 20 seeds each
  python run.py --seeds 3              # quick smoke-test
  python run.py --variant hard         # hard routing only
  python run.py --variant moe          # soft MoE only
  python run.py --seeds 5 --variant hard
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise(obj):
    """Recursively convert numpy types to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if hasattr(obj, 'tolist'):          # numpy scalar / array
        return obj.tolist()
    return obj


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(_serialise(data), fh, indent=2)
    print(f"    saved {path}")


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

def run_variant(variant: str, n_seeds: int) -> list:
    """
    Import the variant module (triggers data download + HMM fit once),
    then loop over seeds calling run_experiment(seed, out_dir).

    Returns a list of per-seed metric dicts (without the 'rewards' key).
    """
    print(f"\n{'#'*70}")
    print(f"  Loading {variant.upper()} module  (data download happens here)...")
    print(f"{'#'*70}\n")

    if variant == 'hard':
        variant_dir = os.path.join(HERE, '3_Agent_Select_1')
        out_root    = os.path.join(HERE, 'results', 'hard')
        sys.path.insert(0, variant_dir)
        from Finrlmain import run_experiment          # noqa: E402
    elif variant == 'moe':
        variant_dir = os.path.join(HERE, '3_Agent_Select_3')
        out_root    = os.path.join(HERE, 'results', 'moe')
        sys.path.insert(0, variant_dir)
        from main_moe import run_experiment           # noqa: E402
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'hard' or 'moe'.")

    os.makedirs(out_root, exist_ok=True)
    summary_rows = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"  {variant.upper()} | seed {seed} / {n_seeds - 1}")
        print(f"{'='*60}")

        # Seed both frameworks before calling run_experiment
        torch.manual_seed(seed)
        np.random.seed(seed)

        metrics = run_experiment(seed=seed, out_dir=out_root)

        # Separate heavy rewards list so scalar JSON stays small
        rewards  = metrics.pop('rewards', [])
        scalar_m = dict(metrics)          # 'seed' key is already in here

        save_json(scalar_m, os.path.join(out_root, f'seed_{seed}.json'))
        save_json({'rewards': rewards},   os.path.join(out_root, f'seed_{seed}_rewards.json'))

        summary_rows.append(scalar_m)
        print(f"  -> {variant} seed {seed} done | "
              f"Return: {scalar_m.get('Total Return (%)', 'N/A'):.2f}% | "
              f"Sharpe: {scalar_m.get('Annualised Sharpe', 'N/A'):.4f}")

    save_json({'runs': summary_rows}, os.path.join(out_root, 'summary.json'))
    print(f"\n  All {n_seeds} seeds complete for {variant}.")
    return summary_rows


# ---------------------------------------------------------------------------
# Aggregate statistics helpers
# ---------------------------------------------------------------------------

REPORT_KEYS = ['Total Return (%)', 'Annualised Sharpe', 'Max Drawdown (%)', 'Sortino Ratio']


def _bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Return (lower, upper) bootstrap percentile CI for the mean."""
    rng     = np.random.default_rng(0)
    boots   = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi  = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi


def compute_aggregate(rows: list) -> dict:
    """Return a dict of per-metric statistics for later comparison / saving."""
    stats = {}
    for k in REPORT_KEYS:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        arr        = np.array(vals, dtype=float)
        lo, hi     = _bootstrap_ci(arr)
        stats[k]   = {
            'mean':   float(arr.mean()),
            'std':    float(arr.std(ddof=1)),
            'median': float(np.median(arr)),
            'min':    float(arr.min()),
            'max':    float(arr.max()),
            'ci95_lo': float(lo),
            'ci95_hi': float(hi),
            'n':      int(len(arr)),
        }
    return stats


def print_aggregate(variant: str, rows: list) -> dict:
    stats = compute_aggregate(rows)
    w = 70
    print(f"\n{'='*w}")
    print(f"  AGGREGATE STATS — {variant.upper()} ({len(rows)} seeds, 95% bootstrap CI)")
    print(f"{'='*w}")
    hdr = f"  {'Metric':<25s}  {'Mean':>9s}  {'±Std':>7s}  {'Median':>8s}  {'95% CI':>17s}  {'Min':>8s}  {'Max':>8s}"
    print(hdr)
    print(f"  {'-'*25}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*17}  {'-'*8}  {'-'*8}")
    for k, s in stats.items():
        ci_str = f"[{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}]"
        print(f"  {k:<25s}  {s['mean']:9.4f}  {s['std']:7.4f}  {s['median']:8.4f}  "
              f"{ci_str:>17s}  {s['min']:8.4f}  {s['max']:8.4f}")
    print(f"{'='*w}")
    return stats


def print_comparison(variant_rows: dict):
    """Welch's t-test (hard vs moe) for each metric. Prints p-values.

    variant_rows: {'hard': [row_dict, ...], 'moe': [row_dict, ...]}
    """
    from scipy import stats as scipy_stats

    if 'hard' not in variant_rows or 'moe' not in variant_rows:
        return

    hard_rows = variant_rows['hard']
    moe_rows  = variant_rows['moe']

    w = 70
    print(f"\n{'='*w}")
    print("  CROSS-VARIANT COMPARISON: Hard Routing vs Soft MoE")
    print(f"  (Welch's two-sided t-test, H0: means are equal)")
    print(f"{'='*w}")
    print(f"  {'Metric':<25s}  {'Hard mean':>10s}  {'MoE mean':>9s}  {'Δ (MoE-Hard)':>13s}  {'p-value':>9s}  {'sig':>4s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*9}  {'-'*13}  {'-'*9}  {'-'*4}")

    for k in REPORT_KEYS:
        h_vals = np.array([r[k] for r in hard_rows if k in r], dtype=float)
        m_vals = np.array([r[k] for r in moe_rows  if k in r], dtype=float)
        if len(h_vals) < 2 or len(m_vals) < 2:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(h_vals, m_vals, equal_var=False)
        delta = m_vals.mean() - h_vals.mean()
        sig   = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"  {k:<25s}  {h_vals.mean():10.4f}  {m_vals.mean():9.4f}  "
              f"{delta:+13.4f}  {p_val:9.4f}  {sig:>4s}")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-seed experiment runner for regime-aware RL variants.'
    )
    parser.add_argument(
        '--seeds', type=int, default=20,
        help='Number of random seeds (runs seed 0 … seeds-1). Default: 20.'
    )
    parser.add_argument(
        '--variant', choices=['hard', 'moe', 'both'], default='both',
        help="Which variant to run: 'hard', 'moe', or 'both'. Default: both."
    )
    args = parser.parse_args()

    variants = ['hard', 'moe'] if args.variant == 'both' else [args.variant]

    # ── Run all variants ──────────────────────────────────────────────────
    collected = {}          # variant → {'rows': [...], 'stats': {...}}
    for v in variants:
        rows              = run_variant(v, args.seeds)
        stats             = print_aggregate(v, rows)
        collected[v]      = {'rows': rows, 'stats': stats}
        # Save aggregate stats JSON next to the per-seed files
        agg_path = os.path.join(HERE, 'results', v, 'aggregate_stats.json')
        save_json(stats, agg_path)

    # ── Cross-variant significance test (only when both variants ran) ─────
    if len(collected) == 2:
        print_comparison({v: d['rows'] for v, d in collected.items()})

    print(f"\nDone. Results in {os.path.join(HERE, 'results')}/")
