"""
Walk-forward evaluation orchestrator.

For each window in WALK_FORWARD_WINDOWS (8 windows):
  - load train/test data + fit the per-window HMM (data_loader.split_window)
  - run the 3 non-RL baselines  (Equal-Weight, Markowitz MVO, S&P 500 B&H)
  - train + evaluate the 4 RL variants (Baseline, Hard, Soft, LSTM-Context)
    with `--seeds` random seeds each
  - compute Sharpe / Return / Max-Drawdown / Sortino per (window × method × seed)

Then aggregate across seeds (per window) and across windows (per method),
run pairwise Wilcoxon signed-rank tests on per-window Sharpe values,
compute bootstrap 95 % CIs on mean Sharpe, and write the paper-ready
results table.

Outputs (in --out_dir, default `results/`):
  per_run_metrics.json            row per (window, method, seed)
  per_window_summary.json         row per (window, method)  (seeds averaged)
  aggregate_table.json            method-level mean ± std + bootstrap CI
  results_table.txt               paper-ready text table
  pvalue_matrix.txt               Wilcoxon signed-rank p-values
  sharpe_boxplot.png              per-window Sharpe distribution box plots
  cumulative_returns_window<i>.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import (
    WALK_FORWARD_WINDOWS, split_window, ASSETS, N_ASSETS,
)
from envs_models import (
    FlatPortfolioEnv, HardRegimePortfolioEnv, MixturePortfolioEnv, LSTMContextEnv,
)
from train import train_a2c, evaluate_actor_greedy, _device_select
from baselines import NON_RL_BASELINES, INITIAL_AMOUNT


DEVICE = _device_select()


# =============================================================================
# Metrics (same definitions as the existing code)
# =============================================================================

def compute_metrics(asset_memory, portfolio_return_memory,
                    initial_amount=INITIAL_AMOUNT, periods_per_year: int = 252) -> dict:
    rets   = np.asarray(portfolio_return_memory[1:])
    values = np.asarray(asset_memory)

    total_return_pct = (values[-1] / initial_amount - 1) * 100
    mu_r   = float(np.mean(rets)) if len(rets) else 0.0
    sd_r   = float(np.std(rets, ddof=1)) + 1e-8 if len(rets) > 1 else 1e-8
    sharpe = (mu_r / sd_r) * np.sqrt(periods_per_year)

    peak     = np.maximum.accumulate(values)
    max_dd   = float(((peak - values) / (peak + 1e-8)).max() * 100) if len(values) else 0.0

    downside = rets[rets < 0]
    d_std    = (np.sqrt(np.mean(downside ** 2)) if len(downside) else 1e-8) + 1e-8
    sortino  = (mu_r / d_std) * np.sqrt(periods_per_year)

    return {
        "Sharpe":           round(float(sharpe),  4),
        "Return (%)":       round(float(total_return_pct), 4),
        "Max Drawdown (%)": round(float(max_dd),  4),
        "Sortino":          round(float(sortino), 4),
        "Final Value ($)":  round(float(values[-1]), 2),
    }


# =============================================================================
# RL variant runner (one window × one seed × one variant)
# =============================================================================

def _run_rl_variant(window: dict, variant: str, seed: int,
                     epochs: int, seq_len: int = 20,
                     verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_assets   = window["n_assets"]
    train_df   = window["train_df"]
    test_df    = window["test_df"]
    train_reg  = window["train_regime"]
    test_reg   = window["test_regime"]

    if variant == "baseline":
        train_env = FlatPortfolioEnv(train_df, None, n_assets)
        test_env  = FlatPortfolioEnv(test_df,  None, n_assets)
        base_dim  = None
    elif variant == "hard":
        train_env = HardRegimePortfolioEnv(train_df, train_reg, n_assets)
        test_env  = HardRegimePortfolioEnv(test_df,  test_reg,  n_assets)
        base_dim  = None
    elif variant == "soft":
        train_env = MixturePortfolioEnv(train_df, train_reg, n_assets)
        test_env  = MixturePortfolioEnv(test_df,  test_reg,  n_assets)
        base_dim  = None
    elif variant == "lstm":
        train_env = LSTMContextEnv(train_df, n_assets, seq_len=seq_len)
        test_env  = LSTMContextEnv(test_df,  n_assets, seq_len=seq_len)
        base_dim  = train_env._state_base(train_env.unique_dates[0]).shape[0]
    else:
        raise ValueError(variant)

    actor, _, history = train_a2c(
        train_env, variant, num_assets=n_assets,
        base_dim=base_dim, seq_len=seq_len,
        epochs=epochs, device=DEVICE, verbose=verbose,
    )
    rollout = evaluate_actor_greedy(actor, test_env, device=DEVICE)

    return {**rollout, "history": history}


# =============================================================================
# One full window: all baselines + all RL variants
# =============================================================================

RL_VARIANTS = [
    ("baseline", "Baseline (no gate)"),
    ("hard",     "Hard Routing"),
    ("soft",     "Soft MoE (ours)"),
    ("lstm",     "LSTM-Context"),
]


def evaluate_window(window: dict, n_seeds: int, epochs: int,
                     verbose: bool = False) -> list:
    """Returns list of per-run dicts (one row per (method, seed))."""
    rows = []
    label = window["label"]

    # ── Non-RL baselines (deterministic — single 'seed') ─────────────────
    for name, fn in NON_RL_BASELINES.items():
        if verbose: print(f"      [baseline] {name}")
        traj    = fn(window)
        metrics = compute_metrics(traj["asset_memory"], traj["portfolio_return_memory"])
        rows.append({
            "window":  label, "method": name, "seed": 0, **metrics,
            "asset_memory": traj["asset_memory"],
            "date_memory":  [str(d) for d in traj["date_memory"]],
        })

    # ── RL variants (n_seeds each) ────────────────────────────────────────
    for variant_key, variant_name in RL_VARIANTS:
        for seed in range(n_seeds):
            if verbose:
                print(f"      [RL]  {variant_name:<22s} seed={seed}")
            t0     = time.time()
            traj   = _run_rl_variant(window, variant_key, seed, epochs, verbose=False)
            metrics = compute_metrics(traj["asset_memory"], traj["portfolio_return_memory"])
            rows.append({
                "window":  label, "method": variant_name, "seed": seed, **metrics,
                "asset_memory": traj["asset_memory"],
                "date_memory":  [str(d) for d in traj["date_memory"]],
                "elapsed_s":    round(time.time() - t0, 1),
            })

    return rows


# =============================================================================
# Aggregation + statistical tests
# =============================================================================

METHOD_ORDER = (
    ["Equal-Weight", "Markowitz MVO", "S&P 500 B&H"]
    + [name for _, name in RL_VARIANTS]
)


def per_window_summary(rows: list) -> pd.DataFrame:
    """For each (window, method): seed-averaged metrics."""
    df = pd.DataFrame(rows)
    cols = ["Sharpe", "Return (%)", "Max Drawdown (%)", "Sortino"]
    grp  = df.groupby(["window", "method"])[cols].mean().reset_index()
    return grp


def _bootstrap_ci(arr: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float]:
    rng    = np.random.default_rng(0)
    if len(arr) == 0:
        return float("nan"), float("nan")
    boot   = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def aggregate_table(per_window_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-window numbers to per-method mean ± std + 95% CI."""
    rows = []
    for m in METHOD_ORDER:
        sub = per_window_df[per_window_df.method == m]
        if sub.empty:
            continue
        s_arr = sub["Sharpe"].values
        lo, hi = _bootstrap_ci(s_arr)
        rows.append({
            "Method":            m,
            "Sharpe (mean)":      round(float(s_arr.mean()),         4),
            "Sharpe (std)":       round(float(s_arr.std(ddof=1)),    4) if len(s_arr) > 1 else 0.0,
            "Sharpe CI95 lo":     round(lo, 4),
            "Sharpe CI95 hi":     round(hi, 4),
            "Return (%) mean":    round(float(sub["Return (%)"].mean()),       4),
            "Return (%) std":     round(float(sub["Return (%)"].std(ddof=1)),  4) if len(sub) > 1 else 0.0,
            "Max DD (%) mean":    round(float(sub["Max Drawdown (%)"].mean()), 4),
            "Sortino mean":       round(float(sub["Sortino"].mean()),          4),
            "n_windows":          len(sub),
        })
    return pd.DataFrame(rows)


def wilcoxon_pvalues(per_window_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank on per-window Sharpe."""
    from scipy.stats import wilcoxon
    methods = [m for m in METHOD_ORDER
                if m in per_window_df.method.unique()]
    out = pd.DataFrame(index=methods, columns=methods, dtype=object)

    for a in methods:
        for b in methods:
            if a == b:
                out.loc[a, b] = "—"
                continue
            sa = (per_window_df[per_window_df.method == a]
                   .sort_values("window")["Sharpe"].values)
            sb = (per_window_df[per_window_df.method == b]
                   .sort_values("window")["Sharpe"].values)
            n  = min(len(sa), len(sb))
            sa, sb = sa[:n], sb[:n]
            # Need at least 2 paired observations and some non-zero diff
            try:
                _, p = wilcoxon(sa, sb, zero_method="wilcox", alternative="two-sided")
                out.loc[a, b] = round(float(p), 4)
            except (ValueError, IndexError):
                out.loc[a, b] = float("nan")
    return out


# =============================================================================
# Plots
# =============================================================================

def plot_sharpe_boxplot(per_window_df: pd.DataFrame, save_path: str):
    methods = [m for m in METHOD_ORDER if m in per_window_df.method.unique()]
    data    = [per_window_df[per_window_df.method == m]["Sharpe"].values for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True)
    palette = ["#7fa6c1", "#a4a8b3", "#444444",
               "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_ylabel("Per-window Sharpe ratio", fontsize=12)
    ax.set_title("Walk-Forward Sharpe Distribution Across 8 Test Windows",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_window_cumret(rows: list, window_label: str, save_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {
        "Equal-Weight": "#7fa6c1", "Markowitz MVO": "#a4a8b3", "S&P 500 B&H": "#444444",
        "Baseline (no gate)": "steelblue",
        "Hard Routing":       "darkorange",
        "Soft MoE (ours)":    "green",
        "LSTM-Context":       "purple",
    }
    for r in rows:
        if r["window"] != window_label or "asset_memory" not in r:
            continue
        vals = np.asarray(r["asset_memory"])
        cum  = (vals / INITIAL_AMOUNT - 1) * 100
        ax.plot(cum, label=f"{r['method']} (seed={r['seed']})",
                color=palette.get(r["method"], "grey"),
                lw=1.2, alpha=0.6 if r["method"] in
                  ("Baseline (no gate)", "Hard Routing", "Soft MoE (ours)", "LSTM-Context")
                else 1.6)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"Cumulative Return — Window {window_label}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Final results table (paper-ready)
# =============================================================================

def write_results_table(agg_df: pd.DataFrame, save_path: str):
    lines = []
    w = 96
    lines.append("=" * w)
    lines.append("  WALK-FORWARD RESULTS  (mean ± std across 8 test windows;  RL methods seed-averaged within window)")
    lines.append("=" * w)
    hdr = (f"  {'Method':<22s}  {'Sharpe (mean ± std)':>22s}  "
            f"{'Return % (mean ± std)':>22s}  {'Max DD %':>10s}  {'Sortino':>10s}")
    lines.append(hdr)
    lines.append("  " + "-" * 22 + "  " + "-" * 22 + "  " + "-" * 22 + "  " +
                  "-" * 10 + "  " + "-" * 10)

    for _, r in agg_df.iterrows():
        lines.append(
            f"  {r['Method']:<22s}  "
            f"{r['Sharpe (mean)']:7.3f} ± {r['Sharpe (std)']:5.3f}  "
            f"   "
            f"{r['Return (%) mean']:7.2f} ± {r['Return (%) std']:5.2f}  "
            f"   "
            f"{r['Max DD (%) mean']:10.2f}  "
            f"{r['Sortino mean']:10.3f}"
        )
    lines.append("=" * w)
    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text + "\n\n")
        f.write("Bootstrap 95% CI on mean Sharpe (per method):\n")
        for _, r in agg_df.iterrows():
            f.write(f"  {r['Method']:<22s}  [{r['Sharpe CI95 lo']:+.4f}, {r['Sharpe CI95 hi']:+.4f}]\n")
    print(text)
    print(f"Saved {save_path}")


def write_pvalue_table(pmat: pd.DataFrame, save_path: str):
    cols = list(pmat.columns)
    lines = ["Pairwise Wilcoxon signed-rank p-values on per-window Sharpe ratios:",
              "(symmetric — same numbers above and below diagonal)",
              ""]
    hdr = "  " + " " * 22 + "  " + "  ".join(f"{c[:18]:>18s}" for c in cols)
    lines.append(hdr)
    for r in cols:
        row_vals = []
        for c in cols:
            v = pmat.loc[r, c]
            if isinstance(v, float):
                star = "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""
                cell = f"{v:.4f}{star}"
            else:
                cell = str(v)
            row_vals.append(f"{cell:>18s}")
        lines.append(f"  {r[:22]:<22s}  " + "  ".join(row_vals))
    lines.append("\n  * p<0.05  ** p<0.01  *** p<0.001")
    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"Saved {save_path}")


# =============================================================================
# Main
# =============================================================================

def _save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def _ser(x):
        if isinstance(x, dict):    return {k: _ser(v) for k, v in x.items()}
        if isinstance(x, list):    return [_ser(v) for v in x]
        if hasattr(x, "tolist"):   return x.tolist()
        return x
    with open(path, "w") as f:
        json.dump(_ser(obj), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, default=4)
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--windows", type=int, default=len(WALK_FORWARD_WINDOWS),
                          help="Run only the first N windows (debug/smoke-test).")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n[Walkforward]   device={DEVICE}   seeds={args.seeds}   epochs={args.epochs}")
    print(f"               running {args.windows}/{len(WALK_FORWARD_WINDOWS)} windows")

    all_rows = []
    for win_tuple in WALK_FORWARD_WINDOWS[:args.windows]:
        print(f"\n{'#'*70}")
        print(f"  WINDOW {win_tuple[0]}   train {win_tuple[1]}…{win_tuple[2]}   "
              f"test {win_tuple[3]}…{win_tuple[4]}")
        print(f"{'#'*70}")
        win = split_window(win_tuple)
        print(f"     train_df={win['train_df'].shape}   test_df={win['test_df'].shape}   "
              f"n_assets={win['n_assets']}")

        rows = evaluate_window(win, n_seeds=args.seeds, epochs=args.epochs,
                                verbose=args.verbose)
        all_rows.extend(rows)
        # Per-window cumulative-return plot
        plot_window_cumret(
            rows, win["label"],
            save_path=os.path.join(args.out_dir, f"cumulative_returns_{win['label']}.png"),
        )

    # ── Save per-run rows (drop heavy asset_memory for slim version too) ─
    _save_json(all_rows, os.path.join(args.out_dir, "per_run_metrics.json"))

    # ── Per-window summary (seeds averaged) ──────────────────────────────
    pw_df = per_window_summary(all_rows)
    pw_df.to_csv(os.path.join(args.out_dir, "per_window_summary.csv"), index=False)
    _save_json(pw_df.to_dict(orient="records"),
                os.path.join(args.out_dir, "per_window_summary.json"))

    # ── Aggregate table ──────────────────────────────────────────────────
    agg_df = aggregate_table(pw_df)
    agg_df.to_csv(os.path.join(args.out_dir, "aggregate_table.csv"), index=False)
    _save_json(agg_df.to_dict(orient="records"),
                os.path.join(args.out_dir, "aggregate_table.json"))
    write_results_table(agg_df, os.path.join(args.out_dir, "results_table.txt"))

    # ── Pairwise Wilcoxon ────────────────────────────────────────────────
    pmat = wilcoxon_pvalues(pw_df)
    pmat.to_csv(os.path.join(args.out_dir, "pvalue_matrix.csv"))
    write_pvalue_table(pmat, os.path.join(args.out_dir, "pvalue_matrix.txt"))

    # ── Box plot ─────────────────────────────────────────────────────────
    plot_sharpe_boxplot(pw_df, os.path.join(args.out_dir, "sharpe_boxplot.png"))

    print(f"\nDone.  All outputs in ./{args.out_dir}/")
