"""Paired seed-matched statistics + Ledoit-Wolf/Jobson-Korkie-Memmel
Sharpe-difference tests (ICAIF items #20/#21, added 2026-07-04).

Part A — paired tests. All architecture comparisons share seeds, so
observations are paired (Soft-seed-0 vs Hard-seed-0, ...). The results_log's
Welch (unpaired) tests discard that structure; paired t-tests on seed-matched
differences remove between-seed variance. Applied to:
  A1. Pooled single-period Runs A+B+C (9 campaign-seed pairs, Table A5 data)
  A2. Run B alone (3 pairs, from results/seed_*.json + results/moe_run_3590623)
  A3. PPO (3 pairs per comparison, from PPO/results/*)

Part B — Sharpe-difference test on DAILY returns (Jobson-Korkie with Memmel
correction; the standard finance test, ~n=1000 daily obs instead of n=3 seed
obs). Needs daily return series, which only the walk-forward saves
(asset_memory). Applied to walkforward results dir (default the superseded
results_gae_2seed — rerun with Walkforward/results_batchfix_2seed when it
lands):
  python3 analysis_paired_stats.py [walkforward_results_dir]
"""
import json
import os
import sys
import numpy as np
from scipy import stats

WF_DIR = sys.argv[1] if len(sys.argv) > 1 else "Walkforward/results_gae_2seed"

# ── Table A5 per-seed data (Runs A, B, C — verified against paper appendix) ──
# (run, seed): (hard_return, hard_sharpe, hard_maxdd, hard_sortino,
#               soft_return, soft_sharpe, soft_maxdd, soft_sortino)
A5 = {
    ("A", 0): (19.79, 0.463, 29.17, 0.469, 18.79, 0.497, 16.98, 0.516),
    ("A", 1): (12.40, 0.355, 28.90, 0.363, 37.09, 0.658, 28.72, 0.709),
    ("A", 2): (32.38, 0.736, 24.06, 0.759, 13.15, 0.393, 22.56, 0.411),
    ("B", 0): (25.49, 0.5655, 30.83, -0.1013, 28.45, 0.6910, 17.98, 0.4893),
    ("B", 1): (-13.88, -0.1026, 46.35, 0.5440, 17.67, 0.4500, 19.28, 0.7256),
    ("B", 2): (3.07, 0.2019, 52.52, 0.2022, 22.79, 0.5628, 24.83, 0.6109),
    ("C", 0): (9.18, 0.2978, 32.11, 0.0993, 75.40, 1.0714, 23.63, 0.3091),
    ("C", 1): (-0.63, 0.0984, 33.25, 0.2983, 10.07, 0.3249, 24.79, 0.6767),
    ("C", 2): (23.47, 0.5231, 48.35, 0.5180, 29.24, 0.6674, 35.42, 1.1100),
}
METRICS = ["Return (%)", "Sharpe", "Max DD (%)", "Sortino"]


def paired_report(name, a, b, la, lb, use_wilcoxon=True):
    """a, b: dict metric -> np.array of seed-matched values. Tests a > b
    (a < b for Max DD)."""
    print(f"\n  {name}  ({la} vs {lb}, n={len(next(iter(a.values())))} pairs)")
    print(f"    {'Metric':<14} {'mean diff':>10} {'paired t p1':>12} "
          f"{'Wilcoxon p1':>12} {'old Welch p1':>13}")
    for m in METRICS:
        d = a[m] - b[m]
        alt = "less" if "DD" in m else "greater"
        tt = stats.ttest_rel(a[m], b[m], alternative=alt)
        if use_wilcoxon and len(d) >= 5:
            wc = stats.wilcoxon(d, alternative=alt)
            wc_p = f"{wc.pvalue:.4f}"
        else:
            wc_p = "  n<5"
        _, welch_p = stats.ttest_ind(a[m], b[m], equal_var=False, alternative=alt)
        star = " *" if tt.pvalue < 0.05 else ("  " if tt.pvalue > 0.1 else " .")
        print(f"    {m:<14} {d.mean():>+10.3f} {tt.pvalue:>12.4f}{star}"
              f"{wc_p:>11} {welch_p:>13.4f}")


def load_dir(d, n=3):
    out = {m: [] for m in METRICS}
    keymap = {"Return (%)": "Total Return (%)", "Sharpe": "Annualised Sharpe",
              "Max DD (%)": "Max Drawdown (%)", "Sortino": "Sortino Ratio"}
    for s in range(n):
        j = json.load(open(os.path.join(d, f"seed_{s}.json")))
        for m in METRICS:
            out[m].append(j[keymap[m]])
    return {m: np.array(v) for m, v in out.items()}


print("=" * 76)
print("PART A — PAIRED SEED-MATCHED TESTS (one-tailed in the claimed direction)")
print("=" * 76)

# A1: pooled Runs A+B+C, 9 pairs
hard = {m: np.array([A5[k][i] for k in sorted(A5)]) for i, m in enumerate(METRICS)}
soft = {m: np.array([A5[k][i + 4] for k in sorted(A5)]) for i, m in enumerate(METRICS)}
paired_report("A1. Pooled Runs A+B+C (Table I data)", soft, hard, "Soft", "Hard")

# A2: Run B alone (local JSONs; falls back silently if files missing)
try:
    hard_b = load_dir("results")                       # Run B hard (loose files)
    soft_b = load_dir("results/moe_run_3590623")
    paired_report("A2. Run B alone (Table II Hard/Soft)", soft_b, hard_b,
                  "Soft", "Hard", use_wilcoxon=False)
except FileNotFoundError as e:
    print(f"\n  A2 skipped: {e}")

# A3: PPO
try:
    hp = load_dir("PPO/results/hard_ppo")
    mp = load_dir("PPO/results/moe_ppo")
    rp = load_dir("PPO/results/router_ppo")
    paired_report("A3a. PPO Soft vs Hard", mp, hp, "Soft-PPO", "Hard-PPO",
                  use_wilcoxon=False)
    paired_report("A3b. PPO Soft vs Router", mp, rp, "Soft-PPO", "Router-PPO",
                  use_wilcoxon=False)
except FileNotFoundError as e:
    print(f"\n  A3 skipped: {e}")


# ── Part B: JK-Memmel Sharpe-difference test on daily returns ────────────────
def jk_memmel(r1, r2):
    """Jobson-Korkie (1981) test with Memmel (2003) correction for
    H0: SR1 == SR2, on paired daily return series (non-annualised SRs)."""
    n = len(r1)
    s1, s2 = r1.mean() / r1.std(ddof=1), r2.mean() / r2.std(ddof=1)
    rho = np.corrcoef(r1, r2)[0, 1]
    theta = (2 * (1 - rho) + 0.5 * (s1**2 + s2**2) - s1 * s2 * (1 + rho**2)) / n
    z = (s1 - s2) / np.sqrt(theta)
    p2 = 2 * (1 - stats.norm.cdf(abs(z)))
    return s1 * np.sqrt(252), s2 * np.sqrt(252), z, p2


print("\n" + "=" * 76)
print(f"PART B — JK-MEMMEL SHARPE TESTS ON DAILY RETURNS  (dir: {WF_DIR})")
if "gae_2seed" in WF_DIR:
    print("  !! superseded old-mechanics data — pipeline validation only;")
    print("  !! rerun with Walkforward/results_batchfix_2seed for paper numbers")
print("=" * 76)

path = os.path.join(WF_DIR, "per_run_metrics.json")
if not os.path.exists(path):
    print(f"  skipped: {path} not found")
    sys.exit(0)
rows = json.load(open(path))


def daily(method, seed=None):
    """Concatenate daily returns across all 8 windows for one method (+seed)."""
    out = []
    for w in ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]:
        rs = [r for r in rows if r["window"] == w and r["method"] == method
              and (seed is None or r.get("seed") == seed)]
        if not rs:
            return None
        am = np.mean([np.array(r["asset_memory"]) / r["asset_memory"][0]
                      for r in rs], axis=0)   # seed-avg wealth curve
        out.append(am[1:] / am[:-1] - 1)
    return np.concatenate(out)


pairs = [("Soft MoE (ours)", "Hard Routing"),
         ("Soft MoE (ours)", "Baseline (no gate)"),
         ("Soft MoE (ours)", "Equal-Weight"),
         ("Hard Routing", "Baseline (no gate)")]
print(f"\n  {'Comparison':<42} {'SR1':>7} {'SR2':>7} {'z':>7} {'p (2t)':>8}")
for m1, m2 in pairs:
    r1, r2 = daily(m1), daily(m2)
    if r1 is None or r2 is None or len(r1) != len(r2):
        print(f"  {m1} vs {m2}: series unavailable/misaligned, skipped")
        continue
    s1, s2, z, p = jk_memmel(r1, r2)
    star = " *" if p < 0.05 else ""
    print(f"  {m1 + ' vs ' + m2:<42} {s1:>7.3f} {s2:>7.3f} {z:>+7.2f} {p:>8.4f}{star}")
print(f"\n  (daily obs per series: {len(daily('Equal-Weight'))}; seed-averaged"
      f" wealth curves; annualised SRs shown, test on daily values)")
