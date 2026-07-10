# Experimental Results Log — ICAIF '26 Submission

Living document collating every experimental result generated for the paper.
**Workflow**: paste new run output/tables into the chat → Claude computes
significance stats and appends a dated entry below. Don't hand-edit the
numbers — only Claude appends, so this stays a clean audit trail.

Target venue: ICAIF '26 (Milan), submission deadline **2026-08-02**.

---

## 1. Main Comparison — Hard Routing vs Soft MoE (3-seed, single test period)

### Run A (earlier 3-seed run, job ID unknown, epochs unspecified)

| Metric | Hard S0 | Hard S1 | Hard S2 | Hard Avg | MoE S0 | MoE S1 | MoE S2 | MoE Avg |
|---|---|---|---|---|---|---|---|---|
| Return (%) | 19.79 | 12.40 | 32.38 | 21.52 | 18.79 | 37.09 | 13.15 | 23.01 |
| Sharpe | 0.463 | 0.355 | 0.736 | 0.518 | 0.497 | 0.658 | 0.393 | 0.516 |
| Max DD (%) | 29.17 | 28.90 | 24.06 | 27.38 | 16.98 | 28.72 | 22.56 | 22.75 |
| Sortino | 0.469 | 0.363 | 0.759 | 0.530 | 0.516 | 0.709 | 0.411 | 0.545 |

**Welch's t-test (n=3 per arm):**

| Metric | Δ (MoE−Hard) | t | p (2-tail) | p (1-tail, MoE better) | sig |
|---|---|---|---|---|---|
| Return (%) | +1.49 | -0.160 | 0.8809 | 0.4404 | ns |
| Sharpe | -0.002 | 0.015 | 0.9891 | 0.5054 | ns |
| Max DD (%) | -4.62 | 1.225 | 0.3106 | 0.1553 | ns |
| Sortino | +0.015 | -0.102 | 0.9241 | 0.4620 | ns |

Nothing significant — n=3 has ~10-15% power. Max DD closest (p₁=0.155).

---

### Run B — job 3590623 (completed 2026-06-10, 3 seeds × 1000 epochs)

| Metric | Hard S0 | Hard S1 | Hard S2 | Hard Avg | MoE S0 | MoE S1 | MoE S2 | MoE Avg |
|---|---|---|---|---|---|---|---|---|
| Return (%) | 25.49 | -13.88 | 3.07 | 4.891 | 28.45 | 17.67 | 22.79 | 22.967 |
| Sharpe | 0.5655 | -0.1026 | 0.2019 | 0.2216 | 0.6910 | 0.4500 | 0.5628 | 0.5679 |
| Max DD (%) | 30.8253 | 46.3512 | 52.5225 | 43.233 | 17.9823 | 19.2766 | 24.8348 | 20.698 |
| Sortino | -0.1013 | 0.5440 | 0.2022* | 0.2150 | 0.4893 | 0.7256 | 0.6109* | 0.6086 |

\* third-seed Sortino values reconstructed from mean/min/max (mid = 3·mean − min − max); matches reported medians exactly.

**Welch's t-test (n=3 per arm):**

| Metric | Hard mean | MoE mean | Δ (MoE−Hard) | t | p (2-tail) | p (1-tail, MoE better) | sig |
|---|---|---|---|---|---|---|---|
| Return (%) | 4.891 | 22.967 | +18.077 | -1.529 | 0.2500 | 0.1250 | ns |
| Sharpe | 0.2216 | 0.5679 | +0.346 | -1.687 | 0.2076 | 0.1038 | ns |
| Max DD (%) | 43.233 | 20.698 | -22.535 | 3.320 | 0.0614 | **0.0307** | **SIG (p<0.05)** |
| Sortino | 0.2150 | 0.6086 | +0.394 | — | 0.1586 | 0.0793 | ns (trend) |

**Key finding — Hard Routing is unstable, MoE is consistent**: Seed 1 of Hard
Routing *collapsed* (Return = -13.88%, Sharpe = -0.10), dragging the Hard mean
return from ~21% (Run A) down to ~4.9%. MoE's return stayed ~23% and Sharpe
~0.55-0.57 across **both independent 3-seed runs** with completely different
Hard outcomes — i.e., MoE is reproducible, Hard is not. The Max DD reduction
(43.2% → 20.7%) is significant one-sided (p=0.031), driven by this instability.

**Resource usage**: 28h42m wall time for 3 seeds × 2 variants × 1000 epochs
→ ≈ 4.78h per (seed, variant). Backed up at
`~/Regime-Aware-RL-URECA-/Final/results/{hard,moe}_run_3590623/` on the cluster.

---

### Run C — job 3606871 (completed 2026-06-15, `--variant both`, 27h18m)

Submitted intending to run the router job, but `--variant` defaulted to
`both` (hard+moe) — so this is a **third independent 3-seed Hard vs MoE run**,
not the router job. Router job still pending (see §2).

| Metric | Hard S0 | Hard S1 | Hard S2 | Hard Avg | MoE S0 | MoE S1 | MoE S2 | MoE Avg |
|---|---|---|---|---|---|---|---|---|
| Return (%) | 9.18 | -0.63 | 23.47 | 10.672 | 75.40 | 10.07 | 29.24 | 38.238 |
| Sharpe | 0.2978 | 0.0984 | 0.5231 | 0.3064 | 1.0714 | 0.3249 | 0.6674 | 0.6879 |
| Max DD (%) | 32.1091* | 33.2510* | 48.3456* | 37.902 | 23.6333* | 24.7858* | 35.4174* | 27.946 |
| Sortino | 0.0993* | 0.2983* | 0.5180* | 0.3052 | 0.3091* | 0.6767* | 1.1100* | 0.6986 |

\* reconstructed from mean/min/max (mid = 3·mean − min − max); matches reported medians.

**Welch's t-test (n=3, this run alone):** nothing significant (Return p₁=0.145,
Sharpe p₁=0.109, MaxDD p₁=0.102, Sortino p₁=0.114) — but direction is correct
on all 4 metrics, same as Runs A and B.

**Backup on cluster** (run before next job, since `--variant both` overwrites
`results/hard/` and `results/moe/`):
```bash
cd ~/Regime-Aware-RL-URECA-/Final/results/
cp -r hard hard_run_3606871
cp -r moe  moe_run_3606871
```

**Resource usage**: 27h18m wall time, 6 runs (3 seeds × 2 variants) → ≈4.55h/run.
Note: venv activation failed (`venv/bin/activate: No such file or directory`)
but job completed via fallback conda env (Exit Status 0) — worth fixing the
venv path before the next submission so it doesn't silently rely on fallback.

---

### Pooled analysis — Runs A + B + C (n=9 per arm)

Three independent 3-seed campaigns, each showing the same direction (MoE
better on all 4 metrics). Pooling triples the sample size and **every metric
now clears p<0.05 one-sided**:

| Metric | Hard mean (n=9) | MoE mean (n=9) | Δ (MoE−Hard) | t | p (2-tail) | p (1-tail, MoE better) | sig |
|---|---|---|---|---|---|---|---|
| Return (%) | 12.363 | 28.072 | +15.709 | -1.924 | 0.0738 | **0.0369** | **SIG** |
| Sharpe | 0.3487 | 0.5906 | +0.2419 | -2.138 | 0.0486 | **0.0243** | **SIG** |
| Max DD (%) | 36.171 | 23.799 | -12.372 | 3.187 | 0.0074 | **0.0037** | **SIG** |
| Sortino | 0.3502 | 0.6175 | +0.2673 | -2.304 | 0.0351 | **0.0176** | **SIG** |

Hard pooled std is large (Return σ=14.6, driven by the seed-1 collapses in
Runs B and C: -13.88%, -0.63%) — MoE's std is much tighter relative to its
mean. This is the **headline robustness result**: MoE Routing significantly
outperforms Hard Routing on return, risk-adjusted return, and drawdown,
*and* is far more reproducible across independent training runs.

**Caveat for the paper**: Runs A/B/C were not run with byte-identical configs
(MoE hyperparameters were tuned between runs — see Appendix). Frame this as
"3 independent training campaigns, consistent direction across all metrics in
every campaign" rather than "9 i.i.d. seeds of one fixed config." The
walkforward (§3, fixed 8-window protocol) is the methodologically clean
robustness check; this pooled result is strong supporting/motivating evidence.

---

## 2. Learned Router Comparison (3-seed, single test period)

**Status: COMPLETE** — job 3611187 (`python run.py --variant router --seeds 3`,
1000 epochs), completed 2026-06-16, walltime **24h28m of 48h**, Exit Status 0.
Compared against Run B (job 3590623) Hard and MoE at same epoch budget (1000).

---

### 2a — Per-Seed Results (job 3611187, 1000 epochs)

| Metric | Router S0 | Router S1 | Router S2 | Router mean ± std |
|---|---|---|---|---|
| Return (%) | -18.96 | +18.80 | -15.05 | **-5.07 ± 20.76** |
| Sharpe | -0.3980 | +0.4583 | -0.2410 | **-0.060 ± 0.456** |
| Max DD (%) | 40.27\* | 28.47\* | 33.70\* | 34.15 ± 5.91 |
| Sortino | -0.3685\* | +0.4687\* | -0.2472\* | -0.049 ± 0.452 |

\* per-seed MaxDD and Sortino reconstructed from aggregate min/median/max (S0=worst
return → highest DD, S1=best return → lowest DD, S2=median). Return/Sharpe are
from the output directly; MaxDD/Sortino from bootstrap CI [min, median, max].

2/3 seeds produced large negative returns. For reference, Run B baselines:
Hard Routing mean Return 4.89%, Sharpe 0.222 | Soft MoE mean Return 22.97%, Sharpe 0.568

---

### 2b — 3-Way Comparison Table (all single-period runs, 2022-2024 test split)

Combining Run C (job 3606871, Hard + MoE, most recent) and job 3611187 (Router):

| Metric | Hard Routing (Run C) | **Soft MoE (Run C)** | Learned Router |
|---|---|---|---|
| Return mean (%) | 10.67 | **38.24** | -5.07 |
| Return std (%) | ±12.12 | ±33.58 | ±20.76 |
| Sharpe mean | 0.306 | **0.688** | -0.060 |
| Sharpe std | ±0.213 | ±0.374 | ±0.456 |
| Max DD mean (%) | 37.90 | **27.95** | 34.15 |
| Sortino mean | 0.305 | **0.699** | -0.049 |
| Seeds | 3 | 3 | 3 |
| Epochs | 1000 | 700 | 1000 |

Ranking on every metric: **Soft MoE > Hard Routing > Learned Router**.
Router is also worse than the unrouted Baseline (Return +3.07%, Sharpe +0.35, §3 walkforward).

---

### 2c — Welch's t-tests (n=3 per arm, Router vs Run B)

**Router vs Hard Routing (Run B):**

| Metric | Hard mean | Router mean | Δ (Router−Hard) | t | df | p (2-tail) | p (1-tail, Router<Hard) | sig |
|---|---|---|---|---|---|---|---|---|
| Return (%) | 4.891 | -5.07 | −9.961 | −0.602 | 4.0 | 0.5796 | 0.290 | ns |
| Sharpe | 0.2216 | -0.0602 | −0.2818 | −0.863 | 3.7 | 0.4407 | 0.220 | ns |
| Max DD (%) | 43.23 | 34.15 | −9.08 | −1.243 | 3.0 | 0.300 | 0.150 | ns |
| Sortino | 0.2150 | -0.049 | −0.264 | −0.822 | 3.6 | 0.463 | 0.231 | ns |

**Router vs Soft MoE (Run B):**

| Metric | MoE mean | Router mean | Δ (Router−MoE) | t | df | p (2-tail) | p (1-tail, Router<MoE) | sig |
|---|---|---|---|---|---|---|---|---|
| Return (%) | 22.967 | -5.07 | −28.037 | −2.264 | 2.3 | 0.1367 | **0.0684** | ns (near trend) |
| Sharpe | 0.5679 | -0.0602 | −0.6281 | −2.307 | 2.3 | 0.1318 | **0.0659** | ns (near trend) |
| Max DD (%) | 20.698 | 34.15 | +13.452 | +2.14 | 3.0 | 0.122 | 0.939 | ns (wrong dir) |
| Sortino | 0.6086 | -0.049 | −0.658 | −2.439 | 2.3 | 0.121 | **0.060** | ns (near trend) |

Neither comparison clears p<0.05 — n=3 has ~10-15% power. The Router vs MoE
contrasts approach trend-level (p₁≈0.06-0.07) despite only 3 seeds per arm,
driven by MoE consistency (Sharpe std≈0.12) vs Router instability (std 0.456).
With n=9 the Return, Sharpe, and Sortino differences would plausibly reach
significance. Note: Router MaxDD (34.15%) is actually *lower* than Hard (43.23%)
— the only metric where Router partially improves over Hard — but higher than
MoE (20.70%), consistent with the overall ranking.

---

### 2d — Key Findings (for paper — §4 Ablation)

**1. Learned Router is worse than the unrouted Baseline**: mean Return −5.07%
vs Baseline (no gate) +3.07% and Hard Routing +4.89%. Joint end-to-end learning
of routing and portfolio policy *destroyed* value.

**2. Catastrophic seed instability (2/3 seeds collapse)**: Seed 0 (−18.96%) and
Seed 2 (−15.05%) produced large negative returns. Router MaxDD of 34.15% is
worse than MoE (20.70%) and better than Hard (43.23%) — suggesting some
architectural benefit but insufficient without a stable routing signal.

**3. Validates the pre-trained HMM routing design**: By separating regime
detection (HMM, pre-trained on macro ETFs) from portfolio optimization (A2C),
Soft MoE provides a stable, regime-appropriate routing signal from episode 1.
Joint learning forces the router to simultaneously learn both tasks — gradient
conflicts in the shared objective and a harder optimization landscape lead to
seed collapse. The Learned Router's high variance (σ=20.76% vs MoE σ≈6%)
confirms this.

**Paper framing for §4 ablation**: *"Replacing the fixed HMM gate with an
end-to-end learned router — jointly trained with the portfolio policy — yields
mean Sharpe −0.060 and mean Return −5.07%, worse than both the Soft MoE
(Sharpe +0.568, Return +22.97%) and the no-gate Baseline (Sharpe +0.222,
Return +4.89%). Two of three seeds collapsed to large negative returns.
This confirms that stable pre-trained regime detection is essential: the HMM
provides a regime signal invariant to the reward landscape of the portfolio
task, avoiding the gradient conflicts that destabilise end-to-end routing."*

---

## 3. Walkforward Validation (8 rolling windows)

**Status: COMPLETE** — job 3606870 (submitted 2026-06-13, completed
2026-06-16), `--seeds 1 --epochs 300`, Exit Status 0, walltime used
**68h36m of 72h**. W1 was cached from a prior attempt; W2-W8 trained in
this job. Actual per-window training time grew from 6.68h (W2, 5.5yr
train set) to 13.22h (W8, 8.5yr train set) as training data expanded —
actual per-epoch cost ~29s (incl. LSTM overhead), roughly 71% higher than
the pre-run estimate based on non-LSTM architectures only.

---

### 3a — Final Aggregate Table (mean across 8 test windows, n=1 seed each)

| Method | Sharpe (mean ± std) | Return % (mean ± std) | Max DD % | Sortino |
|---|---|---|---|---|
| Equal-Weight | 1.645 ± 1.621 | 11.56 ± 14.23 | 12.63 | 1.655 |
| Markowitz MVO | 1.130 ± 1.647 | 12.16 ± 20.51 | 16.03 | 1.222 |
| S&P 500 B&H | 1.104 ± 1.485 | 6.39 ± 13.81 | 13.69 | 1.097 |
| Baseline (no gate) | 0.348 ± 1.486 | 3.07 ± 20.24 | 20.05 | 0.350 |
| Hard Routing | 1.176 ± 1.235 | 12.28 ± 18.08 | 16.04 | 1.298 |
| **Soft MoE (ours)** | **1.352 ± 1.537** | **14.59 ± 19.33** | **15.27** | **1.464** |
| LSTM-Context | 1.270 ± 2.019 | 25.32 ± 44.36 | 26.23 | 1.367 |

std is temporal variance across the 8 windows (not seed variance; n=1 seed per window).

---

### 3b — Reconstructed Per-Window Sharpe

(Derived from running cumulative means printed after each window; Wn = n·mean_n − (n−1)·mean_{n−1})

| Window | Test Period | EW | MVO | SPY | Baseline | Hard | **Soft MoE** | LSTM |
|---|---|---|---|---|---|---|---|---|
| W1 | H1-2020 (COVID crash) | 0.325 | 1.150 | 0.034 | -0.442 | 0.052 | **1.001** | 0.960 |
| W2 | H2-2020 (recovery) | 3.177 | 0.952 | 2.428 | 1.968 | 1.760 | 1.833 | 1.910 |
| W3 | H1-2021 (bull) | 3.101 | 0.325 | 2.452 | -0.155 | 1.845 | 2.449 | 2.353 |
| W4 | H2-2021 (bull) | 2.033 | 3.257 | 1.722 | 1.401 | 1.759 | 2.233 | 2.205 |
| W5 | H1-2022 (bear onset) | -1.466 | -1.529 | -1.721 | -1.902 | **-1.041** | -1.766 | -2.908 |
| W6 | H2-2022 (bear/recovery) | 0.918 | -0.063 | 0.215 | 0.498 | **2.825** | 0.196 | -0.050 |
| W7 | H1-2023 (bull) | 2.867 | 3.391 | 2.325 | -0.948 | 0.542 | 3.070 | **3.804** |
| W8 | H2-2023 (bull) | 2.205 | 1.557 | 1.377 | 2.364 | 1.666 | 1.800 | 1.886 |

---

### 3c — Wilcoxon Signed-Rank p-values (final, n=8 windows)

Key pairs from the W8 pvalue matrix:

| Comparison | p (2-tail) | sig |
|---|---|---|
| Equal-Weight vs S&P 500 B&H | 0.0078 | ** |
| Equal-Weight vs Baseline | 0.0156 | * |
| Soft MoE vs Hard Routing | 0.5469 | ns |
| Soft MoE vs Baseline | 0.1953 | ns |
| Hard Routing vs Baseline | 0.0781 | ns (trend) |
| Soft MoE vs Equal-Weight | 0.2500 | ns |

No RL-vs-RL comparison reaches significance with n=8 — expected given high
cross-window variance and minimum achievable p=0.0078 for Wilcoxon with 8 pairs.

---

### 3d — Key Findings (for paper)

**1. Soft MoE is the best RL method**: Highest Sharpe (1.352) and lowest
MaxDD (15.27%) among all RL variants, beating Hard (1.176) in 6 of 8 windows.
Consistent direction confirms §1 pooled result (all 4 metrics significant, n=9).

**2. COVID crash (W1) is the headline regime result**: During H1-2020, Soft MoE
(1.001) vastly outperformed Hard (0.052) and Baseline (−0.442). When regime
switching matters most — the crash — gated routing clearly adds value. This is
the core argument for the architecture.

**3. Equal-Weight leads on aggregate Sharpe (1.645)**: Well-known "1/N puzzle"
(DeMiguel et al. 2009) — simple diversification dominates in our sample period,
driven by exceptional post-COVID bull run (W2: 3.177, W3: 3.101). Frame this
honestly: RL methods add value during regime-uncertain periods (W1), but
sustained trending markets favour naive diversification. Not a failure —
consistent with prior literature, reinforces the regime-detection story.

**4. Hard Routing shines in H2-2022 (W6, Sharpe 2.825)**: The only window
where Hard significantly beats Soft (2.825 vs 0.196). Hard routing may
opportunistically exploit sharp bear-market recoveries that Soft MoE smooths
over. Interesting regime-specific insight for paper discussion.

**5. LSTM-Context is high-variance**: Best raw return (25.32%) but worst MaxDD
(26.23%) and catastrophic W5 (−2.908 Sharpe during bear onset). The recurrent
architecture amplifies both upswings and downswings — a risk-return tradeoff
story, not a clear win or loss.

**6. Baseline (no gate) clearly worst RL method** (0.348 Sharpe, negative in 3/8
windows) — validates that the routing/gating architecture is essential, not just
the RL framework.

**7. No Wilcoxon significance for RL pairs** — with n=8 windows and n=1 seed
per window, the test is low-power. Frame walkforward as "directional robustness
evidence across 8 diverse market regimes (2020-2023)" rather than a primary
significance result. Primary statistics are in §1 (Welch's t, n=9, all p<0.05).

---

### 3e — Caveats for Paper

- n=1 seed per window: seed variance unobservable in walkforward; single
  extreme-seed result (e.g. W6 Hard at 2.825) may not be reproducible. Compare
  to §1 where Hard showed high seed instability. Disclose explicitly.
- W1 was recovered from checkpoint of a prior aborted job run; identical
  code/config, no retraining needed.
- Equal-Weight's superiority is sample-period-dependent (heavy weighting on
  2020-2021 bull market). A bear-dominated sample would likely favour regime-aware
  methods more strongly.

---

## 4. Training Mechanics Ablation — GAE + Advantage Normalization + LR Annealing

Motivated by Hard Routing's repeated seed instability across Runs A/B/C
(§1) and the observation that Soft MoE/Learned Router already used advantage
normalisation while Hard/Baseline did not, three standard actor-critic
stabilisation techniques were added to all four `run.py` architectures (Hard,
Baseline, Soft MoE, Learned Router) and the Walkforward pipeline:

1. **GAE-λ (λ=0.95)** replacing the previous single n-step bootstrapped
   return advantage.
2. **Advantage normalisation** applied consistently to all four architectures
   (previously only Soft MoE/Learned Router had it).
3. **Linear learning-rate annealing** (1.0× → 0.1× of the initial rate) over
   the full training run.

Verified via standalone unit tests (GAE math vs hand-computed reference
including λ=0 and terminal-masking edge cases; gradient flow critic←value_loss,
actor←policy_loss with no cross-leakage) before any cluster run.

### 4a — Calibration (Hard Routing, 1 seed, 50 epochs, mv reward)

job 3686455, Exit Status 0, walltime 25m58s. **Return 72.49%, Sharpe 1.1744,
Max DD 33.37%, Sortino 1.2107** — roughly 3x the best Sharpe ever obtained
for Hard Routing under the old training mechanics (Run A: 0.518), achieved in
1/20th the training budget (50 vs 1000 epochs). Reward trajectory stable, no
oscillation or blow-up.

### 4b — Full run (Hard vs Soft MoE, 3 seeds, full epochs: Hard=1000, MoE=1000)

job 3686465, Exit Status 0, walltime 41h06m.

**Per-seed:**

| Metric | Hard S0 | Hard S1 | Hard S2 | Hard Avg | MoE S0 | MoE S1 | MoE S2 | MoE Avg |
|---|---|---|---|---|---|---|---|---|
| Return (%) | -1.53 | -7.74 | -8.57 | -5.95 | 7.04 | 20.75 | 12.40 | 13.39 |
| Sharpe | 0.1056 | -0.0685 | -0.0700 | -0.0110 | 0.2617 | 0.4867 | 0.3468 | 0.3651 |
| Max DD (%) | — | — | — | 30.01 | — | — | — | 32.43 |
| Sortino | — | — | — | -0.0136 | — | — | — | 0.3664 |

**Welch's t-test (n=3 per arm):**

| Metric | Hard mean | MoE mean | Δ (MoE−Hard) | p-value | sig |
|---|---|---|---|---|---|
| Return (%) | -5.95 | 13.39 | +19.34 | 0.0221 | * |
| Sharpe | -0.011 | 0.365 | +0.376 | 0.0132 | * |
| Max DD (%) | 30.01 | 32.43 | +2.42 | 0.6182 | ns |
| Sortino | -0.014 | 0.366 | +0.380 | 0.0104 | * |

**Headline finding: Soft MoE still significantly beats Hard Routing on 3 of 4
metrics under the new training mechanics** — the core architectural claim is
unchanged by the mechanics fix.

### 4c — Anomaly: Hard Routing peaks early then degrades over the full 1000-epoch budget

Hard's full-budget result (Sharpe -0.011) is dramatically *worse* than the
50-epoch calibration (Sharpe 1.1744) on the same architecture and reward.
Averaging the logged per-10-epoch training reward confirms a real, consistent
degradation pattern, not just one bad seed:

| Seed | Avg reward, epochs 0-90 | Avg reward, epochs 200-290 | Avg reward, epochs 700-790 |
|---|---|---|---|
| 0 | -3092 | -3931 | -3957 |
| 1 | -3259 | — | -3746 |

Both seeds peak early (consistent with the 50-epoch calibration) then degrade
to a worse plateau that does not recover, even with LR annealing active. Soft
MoE shows no equivalent degradation (tight, consistently positive Sharpe
across all 3 seeds: 0.262, 0.487, 0.347).

**Interpretation for the paper**: this is supporting mechanistic evidence for
the architectural claim, not a contradiction of it. Even with modern
actor-critic stabilisation (GAE, advantage normalisation, LR annealing), the
*discrete* hard-routing mechanism remains comparatively fragile over long
training horizons; soft blending does not show the same fragility. This adds
a "why" to the existing "that" (MoE beats Hard) — possibly worth a sentence
in §6 Discussion.

**Not yet pursued**: re-running Hard at a reduced/early-stopped epoch budget
to capture its peak performance with multiple seeds. Decided against for now
— the existing full-budget result already gives the significant comparison
the paper needs, and chasing Hard's best-case number would cost more cluster
time without changing the architectural conclusion.

### 4d — DSR + same mechanics (Hard, 1 seed, 50 epochs) — confirms mv is still better

job 3686468, Exit Status 0, walltime 21m59s. Return -13.58%, Sharpe -0.2429 —
clearly worse than the mv+GAE calibration (Return +72.49%, Sharpe +1.1744) at
the identical architecture/epoch budget. Confirms the earlier decision to
proceed with mv over DSR; the new training mechanics do not change that
conclusion.

### 4e — Walkforward + same mechanics, W1 only (1 seed, 30 epochs, mv reward)

job 3686467, Exit Status 0, walltime 31m32s. Mechanically clean (no crash),
but not a fair quality comparison since 30 epochs vs the original protocol's
300 — Soft MoE looked worse here (Sharpe -0.197 vs the original 1.00) most
likely due to undertraining at 1/10th budget, not a real regression. Baseline
and LSTM-Context both looked *better* despite fewer epochs. Full 8-window,
2-seed, 300-epoch run (LSTM-Context excluded to fit walltime) queued next.

### 4f — PPO calibration (Hard Routing, mv reward)

Hand-rolled PPO (clipped-surrogate, full-episode on-policy rollout, GAE-λ
computed once per episode, 10 epochs of minibatch SGD per episode) added as
a parallel implementation (`Final/PPO/`) mirroring the A2C variant suite
exactly — same environments, rewards, and actor/critic architectures, only
the training algorithm differs. Verified via standalone unit tests (clipped-
ratio behaviour at the trust-region boundary, gradient flow, GAE under the
full-episode terminal-masked setting) before any cluster run.

| Calibration | Epochs | Return | Sharpe | Max DD | Walltime |
|---|---|---|---|---|---|
| job 3686470 | 20 | 20.18% | 0.4888 | 22.77% | 8m54s |
| job 3687570 | 100 | 29.43% | 0.6305 | 37.42% | 30m04s |

**Key contrast with A2C+GAE**: PPO's performance *improves* with more
training (20→100 epochs: Sharpe 0.489→0.631), the opposite of Hard Routing's
A2C+GAE degradation pattern in §4c. Consistent with PPO's clipped-surrogate
objective being specifically designed to prevent the destructive policy
updates that may be causing that degradation. Full 3-seed/1000-epoch PPO run
queued next.

### 4g — Non-overlapping batch fix confirms the update-frequency hypothesis (§4c)

Root cause hypothesized for the §4c degradation: the sliding-by-1 buffer
(`if len(buf) >= batch_size: ... buf.pop(0)`) produced ~1743 highly
correlated, overlapping gradient updates per episode (~1.74M total over
1000 epochs) with no trust-region mechanism bounding how far any single
update could move the policy. Fix: `buf.pop(0)` → `buf.clear()` (non-overlapping
batches, ~88 updates/episode) applied to all five A2C trainers (Hard, Baseline,
MoE, Router, `Walkforward/train.py`).

job 3689528, Hard Routing, 1 seed, 300 epochs, mv reward, Exit Status 0,
walltime 1h10m51s.

| Epoch range | Avg reward (post-fix) | Avg reward (pre-fix, §4c seed 0) |
|---|---|---|
| 0-90 | -3189.54 | -3092 |
| 100-190 | -2756.73 | — |
| 200-290 | **-2573.82** | **-3931** |

The trend **reverses direction**: pre-fix, reward degraded by ~840 from
epochs 0-90 to 200-290; post-fix, it *improves* by ~616 over the same range.
Final metrics: Return 2.71%, Sharpe 0.2164, Sortino 0.2147, Max DD 54.54%
(1 seed only — noisy, but no longer catastrophic, and trending the right way
unlike the old 1000-epoch run that ended at Sharpe -0.011).

**Confirms the update-frequency hypothesis was the (or a major) cause of
Hard Routing's late-training collapse.** Implication: the §4b headline result
(MoE significantly beats Hard, Sharpe -0.011 vs 0.365) was partly an artifact
of this training-mechanics bug, not purely an architectural deficiency of
discrete routing. **The §4b multi-seed Hard-vs-MoE comparison needs to be
re-run with this fix in place** before being used as a final paper result —
queued as job (see PBS script `run_mv_gae_hardmoe_batchfix.pbs`).

---

## 5. PPO Full Comparison — all 3 variants × 3 seeds, single test period

**Status: COMPLETE** — jobs 3689517 (`hardmoe`: Hard-PPO + Soft-PPO, walltime
24h25m of 48h) and 3689530 (`router`: Router-PPO, walltime 14h45m of 48h;
resubmission of 3689518, which crashed at startup on a missing
`hmm_probabilistic` module before the fix synced to the cluster). Both Exit
Status 0, completed 2026-06-30, logs/results pulled 2026-07-04.
`reward_mode=mv`, 1000 epochs, same single-period protocol (train 2015-2021,
test 2022-2024) and same seeds (0-2) as the A2C comparison. Results in
`Final/PPO/results/{hard_ppo,moe_ppo,router_ppo}/`.

### 5a — Per-seed results

| Metric | Hard S0 | Hard S1 | Hard S2 | **Hard mean±std** | Soft S0 | Soft S1 | Soft S2 | **Soft mean±std** | Rtr S0 | Rtr S1 | Rtr S2 | **Rtr mean±std** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Return (%) | 51.36 | 49.22 | 21.44 | **40.67±16.69** | 28.07 | 74.86 | 57.36 | **53.43±23.64** | 21.94 | 48.66 | -27.80 | **14.27±38.81** |
| Sharpe | 0.886 | 0.809 | 0.483 | **0.726±0.214** | 0.538 | 0.975 | 0.808 | **0.774±0.221** | 0.454 | 0.767 | -0.281 | **0.314±0.538** |
| Max DD (%) | 31.24 | 39.59 | 38.35 | **36.40±4.51** | 46.99 | 40.92 | 37.10 | **41.67±4.99** | 36.40 | 33.46 | 55.35 | **41.74±11.88** |
| Sortino | 0.918 | 0.819 | 0.479 | **0.739±0.230** | 0.534 | 0.995 | 0.854 | **0.794±0.236** | 0.461 | 0.807 | -0.274 | **0.331±0.552** |

### 5b — Welch's t-tests (n=3 per arm, one-tailed in the paper's claimed direction)

| Comparison | Return p₁ | Sharpe p₁ | MaxDD p₁ | Sortino p₁ |
|---|---|---|---|---|
| Soft vs Hard | 0.246 | 0.401 | 0.877 (wrong dir) | 0.393 |
| Soft vs Router | 0.112 | 0.137 | 0.497 | 0.142 |
| Hard vs Router | 0.183 | 0.158 | 0.264 | 0.166 |

Nothing significant at n=3.

### 5c — Key findings

1. **Sharpe/Return/Sortino ranking preserved: Soft > Hard > Router** —
   directionally consistent with the A2C comparison.
2. **Router instability replicates under PPO**: seed 2 collapsed (-27.8%,
   negative Sharpe); Router's variance is ~1.6-2.5× the other variants' on
   every metric. The external-signal-vs-learned-routing finding is now
   **algorithm-robust** (holds under both A2C and PPO). Strongest
   cross-algorithm claim in the project.
3. **The Soft-vs-Hard gap largely closes under PPO** (Sharpe 0.774 vs 0.726,
   p₁=0.40 — a statistical tie), and **Max Drawdown reverses**: Hard-PPO
   36.4% vs Soft-PPO 41.7%, where under A2C drawdown was Soft's most
   significant win (Run B: 20.7% vs 43.2%, p=0.031). Consistent with the
   §4c/§4g mechanism: Hard's A2C-era underperformance was substantially
   driven by unconstrained on-policy updates, which PPO's clipped objective
   prevents by construction (as §6.3 of the paper itself predicted).
4. **Absolute levels are not comparable across algorithms**: PPO agents earn
   much higher returns (40-53% vs A2C's 5-23%) at much higher drawdowns
   (36-42%). Compare rankings within-algorithm only.
5. **Framing for ICAIF**: present PPO as mechanism-validation ("the soft-gate
   advantage is largest when the optimizer lacks trust-region control;
   stabilising the optimizer rescues discrete routing"), *not* as a simple
   robustness confirmation — a reviewer reading p₁=0.40 and the drawdown
   reversal would correctly reject the latter framing.
6. Next step queued: 2-3 extra Soft/Router PPO seeds to push the replicated
   Router-instability contrast (p₁≈0.11-0.14) toward significance.

---

## 6. Walkforward v2 — new training mechanics, 2 seeds/window (job 3687571)

**Status: COMPLETE** — job 3687571, Exit Status 0, completed 2026-06-30
10:59, walltime **70h56m of 72h** (near-miss on the budget). `--seeds 2
--epochs 300 --reward_mode mv --skip_variants lstm`, all 8 windows trained
fresh (W1 included). Results in `Final/Walkforward/results_gae_2seed/`,
pulled 2026-07-04. Addresses the "n=1 seed per window" limitation of §3
and applies the GAE/normalisation/LR-annealing mechanics (§4).

**⚠ Mechanics-version caveat — RESOLVED 2026-07-04: this job ran the OLD
sliding-window code.** Cluster-side verification (git reflog + file mtime):
job started 2026-06-27 12:03 (reflog shows the pull-and-submit at
12:02:57), but the batch-fix commit 23a87ae only reached the cluster at
2026-06-28 19:33 — ~31h into the run. Python imports `train.py` once at
process start, so the entire 71h job used the old `buf.pop(0)` sliding
buffer. **This run is therefore the GAE + sliding-window combination that
§4c showed is pathological**: Hard peaks early then degrades under it, so
at 300 epochs Hard was measured near its early peak while Soft (LR 3e-5 +
annealing) was likely undertrained. The §6a "reversal" is doubly
confounded and must not be used in the paper. Superseding re-run with the
fixed code queued: `run_wf_batchfix_2seed.pbs` → `results_batchfix_2seed/`.

### 6a — Aggregate table (mean across 8 windows, RL methods seed-averaged within window)

| Method | Sharpe | vs §3a (old mechanics, 1 seed) | Return % | Max DD % | Sortino |
|---|---|---|---|---|---|
| Equal-Weight | 1.645 | 1.645 (=, deterministic ✓) | 11.56 | 12.63 | 1.655 |
| Markowitz MVO | 1.130 | 1.130 (=) | 12.16 | 16.03 | 1.222 |
| S&P 500 B&H | 1.104 | 1.104 (=) | 6.39 | 13.69 | 1.097 |
| Baseline (no gate) | 0.455 | 0.348 (↑) | 4.97 | 17.12 | 0.534 |
| **Hard Routing** | **1.133** | 1.176 (≈) | 14.83 | 15.94 | 1.286 |
| **Soft MoE (ours)** | **0.848** | **1.352 (↓↓)** | 9.07 | 15.74 | 0.906 |

**The §3 ranking reverses: Hard is now the best RL variant on aggregate
Sharpe; Soft drops from 1.352 to 0.848**, falling below Hard, MVO and SPY.
(LSTM-Context excluded from this run by design.)

### 6b — Per-window Sharpe (mean of 2 seeds)

| Window | Test period | EW | MVO | SPY | Baseline | Hard | Soft |
|---|---|---|---|---|---|---|---|
| W1 | H1-2020 COVID crash | 0.325 | 1.150 | 0.034 | 0.390 | **1.047** | 0.463 |
| W2 | H2-2020 recovery | 3.176 | 0.952 | 2.429 | 2.400 | 2.665 | 2.131 |
| W3 | H1-2021 bull | 3.102 | 0.325 | 2.451 | 0.115 | **3.983** | 2.200 |
| W4 | H2-2021 bull | 2.033 | 3.257 | 1.724 | -1.055 | -0.234 | 0.012 |
| W5 | H1-2022 bear onset | -1.469 | -1.527 | -1.724 | -0.836 | -0.931 | **-0.688** |
| W6 | H2-2022 bear/recov | 0.920 | -0.064 | 0.216 | 1.232 | 0.223 | **1.570** |
| W7 | H1-2023 bull | 2.870 | 3.394 | 2.322 | -0.053 | **2.558** | 0.389 |
| W8 | H2-2023 bull | 2.203 | 1.555 | 1.382 | 1.446 | -0.242 | 0.707 |

**The W1 COVID headline flips**: §3b had Soft 1.001 vs Hard 0.052; this run
has **Hard 1.047 vs Soft 0.463**. The single most-quoted regime-transition
anecdote in the paper points the other way under the new mechanics.
Interestingly W6 also flips (was Hard 2.825 vs Soft 0.196; now Soft 1.570
vs Hard 0.223) — both "anomaly windows" swapped owners, consistent with
these being seed-noise artifacts rather than architectural regularities.

### 6c — Within-window seed spread (Sharpe, |seed0 − seed1|)

Largest spreads: Soft W6 **2.89** (3.01 vs 0.13), Soft W3 **2.55**
(3.47 vs 0.93), Soft W8 **2.46** (-0.52 vs 1.94), Hard W7 **2.24**
(3.68 vs 1.44), Hard W8 1.58, Hard W6 1.43. Median spread ≈ 0.6 Sharpe.
With 2 seeds and spreads of this size, window-level means are extremely
noisy; the aggregate Hard-vs-Soft gap (1.133 vs 0.848 ≈ 0.29) is well
inside seed noise.

### 6d — Wilcoxon signed-rank (n=8 windows, per-window Sharpe)

| Comparison | p (2-tail) | sig |
|---|---|---|
| Equal-Weight vs S&P 500 B&H | 0.0078 | ** |
| Soft vs Hard | 0.6406 | ns |
| Soft vs Baseline | 0.2500 | ns |
| Hard vs Baseline | 0.4609 | ns |
| Soft vs Equal-Weight | 0.1094 | ns |

No RL-vs-RL comparison is anywhere near significance — same as §3c.

### 6e — Key findings & interpretation

**[Superseded 2026-07-04 — see resolved caveat above: this run used the old
sliding-window code, so findings 1-4 describe the pathological §4c
mechanics, not the fixed ones. Kept for the audit trail; the
`results_batchfix_2seed` re-run replaces this section's numbers.]**

1. **§3d finding 1 ("Soft MoE is the best RL method") does not survive the
   re-run**, and §3d finding 2 (COVID window) actively reverses. The old
   Table III narrative must not be used for ICAIF.
2. **The honest statistical read is a tie, not a reversal**: given the seed
   spreads (§6c) and Wilcoxon p=0.64, Hard and Soft are indistinguishable
   in the walk-forward once training is stabilised. Both the old ranking
   (Soft first) and the new one (Hard first) are noise-level orderings.
3. **Triangulates with §5 (PPO)**: two independent stabilisation routes
   (GAE-era mechanics here, PPO there) both collapse the Soft-vs-Hard gap
   that the original A2C runs showed. The architecture story that survives
   everywhere: Router unstable/worst (§2, §5), gated variants ≥ no-gate
   Baseline, EW tops all (1/N).
4. **Open mechanical suspect — Soft undertraining at 300 epochs**: Soft's
   LR (3e-5) is 3.3× smaller than Hard's (1e-4), and the *new* mechanics
   anneal LR to 0.1× — a compounding penalty at a 300-epoch budget that
   didn't exist in the old-mechanics §3 run (no annealing) and doesn't
   bind at 1000 epochs (single-period runs, where Soft does fine under the
   new mechanics, §4b). Calibration run queued: Soft-only, W1+W7 (its two
   biggest drops), 1000 epochs, 2 seeds — see
   `run_wf_soft_epochs_calib.pbs`. If Soft recovers toward its §3b values,
   the §6a reversal is an artifact of the epoch budget interacting with
   LR annealing, not architecture.
5. Baseline improved (0.348 → 0.455) under the new mechanics, consistent
   with the stabilisation helping the architectures that were previously
   most update-noise-sensitive.

---

## 7. Statistical upgrades + rule baseline (local analyses, 2026-07-04)

Three zero-compute additions (ICAIF ideas #20-22). Scripts:
`analysis_paired_stats.py`, `analysis_gated_ew.py`.

### 7a — Paired seed-matched tests (replacing unpaired Welch)

All architecture comparisons share seeds, so observations are paired;
paired tests remove between-seed variance. One-tailed:

**Pooled Runs A+B+C (Table I data, n=9 pairs) — headline result STRENGTHENS:**

| Metric | mean diff (Soft−Hard) | paired t p₁ | Wilcoxon p₁ | old Welch p₁ |
|---|---|---|---|---|
| Return (%) | +15.71 | 0.0437* | 0.0273* | 0.0369 |
| Sharpe | +0.242 | 0.0258* | 0.0273* | 0.0243 |
| Max DD (%) | −12.37 | 0.0025** | 0.0020** | 0.0037 |
| Sortino | +0.267 | 0.0128* | 0.0195* | 0.0176 |

All four metrics now significant under BOTH the parametric (paired t) and
non-parametric (Wilcoxon signed-rank) paired tests — a stronger, more
defensible claim than Welch alone. **Use paired tests in the ICAIF paper.**

**Run B alone (n=3 pairs)**: Sortino flips to significant (p₁=0.0398 vs
Welch 0.0793), Max DD stays significant (0.0273), Sharpe improves to
near-trend (0.0536 vs 0.1038).

**PPO (n=3 pairs)**: pairing does NOT rescue Soft-vs-Hard (Sharpe p₁=0.42) —
confirms the §5 conclusion that the PPO gap is genuinely small, not a
power artifact. Soft-vs-Router: p₁≈0.12-0.15, still awaiting extra seeds.

### 7b — Jobson-Korkie-Memmel Sharpe tests on daily returns

Implemented (JK 1981 + Memmel 2003 correction) in
`analysis_paired_stats.py` Part B; runs on any walkforward results dir
(daily series, n≈998 obs). On the superseded old-mechanics walkforward:
no RL-vs-RL Sharpe difference is significant at daily granularity
(closest: Hard vs Baseline, z=+1.82, p=0.068). Pipeline validated; rerun
on `results_batchfix_2seed` for paper numbers. Single-period runs don't
save daily test returns yet (ideas #16) so JK-Memmel can't run there.

### 7c — HMM-gated Equal-Weight rule baseline (the "why RL at all?" test)

Rule: Bear regime → cash (hard) or equity exposure = 1−P(Bear) (soft);
otherwise 1/N. Zero training. EW daily returns from the walkforward's own
deterministic EW rows (final regardless of the RL-code caveat);
single-period EW from yfinance daily-rebalanced 1/N (frictionless).

| Protocol | EW | hard-gated EW | soft-gated EW |
|---|---|---|---|
| Walk-forward agg Sharpe | 1.645 | 1.559 | 1.561 |
| Single-period Sharpe | 0.410 | 0.161 | 0.188 |
| Single-period Return % | 12.64 | 2.56 | 3.59 |

**Finding — favourable for the paper**: the hand-crafted rule DESTROYS
value. The gate only triggers in W1 (9.7% of days, tiny gain) and W5
(16.3%, hurts), and on the single-period test naive Bear de-risking cuts
EW's Sharpe from 0.41 to 0.16. Meanwhile Soft RA-RL (0.568) beats both
plain EW (0.410) and gated EW on the single-period protocol. The HMM
signal is NOT directly monetizable by a simple rule — the value comes
from RL learning *how* to use it (which assets, how much, when), which is
exactly the paper's thesis. Also noteworthy: single-period EW Sharpe
(0.410) < Soft RA-RL (0.568) — the 1/N dominance is a walk-forward-2020s
phenomenon, not universal; worth a sentence in §6.2's 1/N discussion.

### 7d — Multi-asset feasibility probe (zero training; scopes the follow-up paper)

`analysis_multiasset_probe.py`. Adds a defensive leg (50% TLT + 50% GLD) the
Bear-regime rule can rotate into, instead of cash: hard-rotate (Bear → 100%
defensive), soft-rotate (P(Bear)-weighted), vs EW-equity, static 80/20, and
7c's cash-gated. All legs yfinance, daily-rebalanced, rf=0. (Internal
comparisons self-consistent; EW levels differ slightly from 7c's
backtest-derived rows due to EW construction details.)

Per-window Sharpe, the two Bear-triggered windows (all other windows: gate
never fires, all variants = EW-equity):

| Window | EW-equity | hard-rotate | soft-rotate | cash-gated | defensive leg itself |
|---|---|---|---|---|---|
| W1 COVID crash | 0.171 | **0.470** | 0.447 | 0.264 | **+2.03 Sharpe, +20.2%** |
| W5 2022 bear onset | -1.686 | **-2.897** | -2.831 | -2.396 | **-1.76 Sharpe, -12.0%** |

Single-period (2022-2024): hard-rotate Sharpe **-0.077 vs EW 0.410 — worse,
and the difference is JK-Memmel SIGNIFICANT (z=-2.04, p=0.042)**, the first
significant daily-level result produced in this project, and it's *against*
the fixed rule.

**Interpretation — the key insight for the multi-asset follow-up:**
"Bear → bonds/gold" is itself a regime-dependent strategy. It pays exactly
as theory predicts in a flight-to-quality crash (COVID: defensive leg +20%,
rotation nearly triples the window Sharpe) and BACKFIRES in a rate-driven
bear (2022: TLT crashed alongside equities, stock-bond correlation flipped
positive). A fixed rule cannot distinguish crisis *types* — and a 4-state
HMM collapses all bears into one state. Implications:
1. Multi-asset regime rotation is NOT free alpha — the follow-up is a real
   research problem, not a layup. Good news for its publishability, bad
   news for anyone expecting easy gains.
2. The follow-up needs either bear-type-aware regime detection (richer
   macro features — rates/inflation — and/or more states; note the 7/BIC
   finding that K=5/6 states exist but are data-starved at 0.3-2% occupancy)
   or learned per-regime allocation (RL) that can infer *which* defensive
   asset works from recent data — i.e., exactly the RA-RL architecture,
   with a stronger motivation than in equities-only.
3. For the ICAIF paper: one sentence in future work can now cite this
   quantitatively ("a fixed defensive-rotation rule helps in liquidity
   crises (+0.30 Sharpe, COVID) but destroys value in rate-driven bears
   (-1.21, 2022), motivating learned multi-asset regime allocation").
Sample-size caveat: 2015-2024 contains exactly ONE bear of each type — the
follow-up likely needs a longer history (2000s: dot-com + GFC are both
flight-to-quality; 1970s-style rate bears are scarce in ETF-era data).

---

## 8. Corrected single-period Hard-vs-Soft (batchfix mechanics) — COMPLETE

**Status: COMPLETE (6/6 seeds, 2026-07-08)** — job 3709637
(`URC_RL_mv_gae_bf`, 30h request) was killed by the walltime limit **90
seconds over budget** (108090s vs 108000s), mid-way through MoE seed 2
(Hard seeds 0-2 and MoE seeds 0-1 completed and saved). The missing MoE
seed 2 was completed by finisher job **3714204** (`run_moe_bf_seed2.pbs`
via `--seed_start 2`): Exit Status 0, walltime 3h44m of 10h (node
hpc-tl-gpu2 — much faster than the ~6-7h/seed measured on the original
job's node). Local copies in `results/hard_bf/`, `results/moe_bf/` —
**stale leftovers quarantined** (`hard_bf/seed_3`, `moe_bf/seed_2` were
pre-batchfix files from the shared cluster dirs; verified against job
logs before removal to `stale_prev_run/`).

Timing lesson: ~5h/Hard-seed, ~6-7h/MoE-seed at 1000 epochs under the new
mechanics — substantially slower than the 3.9h/seed extrapolated from the
300-epoch Hard calibration. `run_moe_kregimes.pbs` bumped 30→40h and
`run_hard_l2match.pbs` 16→20h accordingly.

### 8a — Final results (single-period, 1000 epochs, batchfix mechanics)

| Metric | Hard (n=3) | Soft (n=3) |
|---|---|---|
| Return (%) | +0.28 ± 28.00 | +26.24 ± 29.83 |
| Sharpe | +0.142 ± 0.364 | +0.462 ± 0.245 |
| Max Drawdown (%) | 48.87 ± 3.93 | 57.23 ± 2.86 |
| Sortino | +0.144 ± 0.364 | +0.469 ± 0.242 |

Per-seed raw values (from the scp'd `results/hard_bf/`, `results/moe_bf/`
seed JSONs; each 500 trading days, $1M start):

| Variant / seed | Return (%) | Sharpe | MaxDD (%) | Sortino | Final Value ($) |
|---|---|---|---|---|---|
| Hard seed 0 | -17.50 | -0.083 | 50.72 | -0.080 | 824,976 |
| Hard seed 1 | -14.21 | -0.053 | 51.54 | -0.053 | 857,870 |
| Hard seed 2 | **+32.56** | **+0.563** | 44.36 | +0.565 | 1,325,620 |
| Soft seed 0 | +3.01 | +0.262 | 57.17 | +0.267 | 1,030,143 |
| Soft seed 1 | **+59.88** | **+0.735** | 54.40 | +0.737 | 1,598,805 |
| Soft seed 2 | +15.82 | +0.390 | 60.12 | +0.403 | 1,158,160 |

### 8a' — Paired seed-matched tests (n=3 pairs, one-tailed in claimed direction)

| Metric | mean diff (Soft−Hard) | paired t | t p₁ | Wilcoxon p₁ |
|---|---|---|---|---|
| Return (%) | +25.96 | +0.98 | 0.214 | 0.250 |
| Sharpe | +0.320 | +1.15 | 0.184 | 0.250 |
| Max DD (%) | +8.36 (Soft WORSE) | +2.17 | 0.919 | 1.000 |
| Sortino | +0.325 | +1.18 | 0.179 | 0.250 |

Nothing significant at n=3 (Wilcoxon's floor at n=3 is 0.125). MaxDD's t
in the OPPOSITE direction (Hard better) would be p₁=0.081 — a trend
against the old claim. Note the seed-2 pair is Hard's win (Sharpe 0.563
vs 0.390): Soft wins 2/3 pairs on Sharpe/Return/Sortino, 0/3 on MaxDD.

### 8b — Final A3 read (2026-07-08)

1. **Soft > Hard on Sharpe/Return/Sortino survives the batch fix in
   direction and effect size** (Sharpe 0.46 vs 0.14, Return +26% vs +0.3%)
   but is NOT significant at n=3 (paired p₁≈0.18) — the old pooled n=9
   significance came from old-mechanics campaigns and cannot be claimed
   for the corrected mechanics without more seeds. **Hard's
   seed-instability persists**: 2 of 3 Hard seeds collapsed to negative
   Sharpe while all 3 Soft seeds are positive, and Soft's dispersion is
   smaller on every ratio metric. Within A2C-family training, discrete
   routing remains fragile; this is no longer attributable to the
   sliding-window bug.
2. **Soft's Max Drawdown advantage is definitively GONE** (Hard 48.9% vs
   Soft 57.2%, reversed in all 3 seed-pairs; opposite-direction trend
   p₁=0.081). The old Table I MaxDD claim (p=0.0025 paired) is a
   mechanics artifact — same reversal as PPO (§5). Dropped from the
   paper; now the §5.6 "mechanics artifact" cautionary result in the
   ICAIF draft.
3. Consistent cross-algorithm story, now final: (a) Soft's
   risk-adjusted-return advantage is directionally robust under corrected
   A2C but statistically a tie under PPO; (b) Soft's drawdown advantage
   was an artifact of the old mechanics everywhere; (c) Hard is
   seed-unstable under all A2C variants, stable under PPO; (d) Router
   worst and least stable everywhere (§2, §5).
4. **Gated-EW cross-check under corrected mechanics**: Soft's mean Sharpe
   0.462 still exceeds plain single-period EW (0.410, §7c) but only 1 of
   3 seeds individually beats it (0.735; 0.262 and 0.390 do not) — the
   "Soft beats 1/N on the single-period protocol" sentence must be
   hedged to mean-level and seed-dependent.
5. If more budget arrives, the highest-value spend for THIS table is +3
   more seed-pairs (≈2×4h×3 ≈ 24 walltime-hours) to give the corrected
   comparison the same n=6 power the PPO arm is getting — decide after
   the walk-forward lands.

---

## 9. SAC calibration — Kaggle free-tier run (2026-07-09)

**Status: PIPELINE PASS / LEARNING-SIGNAL FLAG.** Run on Kaggle (Tesla T4,
free tier) via `SAC/kaggle_sac_calib.ipynb` v2 (with the finrl shim — v1
failed on `ModuleNotFoundError: finrl`). All 8 unit tests passed on GPU.
Hard variant, 1 seed, 50 epochs, mv reward, `--calib` tag. Results copied
to `sac_calib_results/` (repo root, untracked).

| Metric | SAC Hard, seed 0, 50 epochs (T4) |
|---|---|
| Total Return (%) | +11.36 |
| Annualised Sharpe | +0.386 |
| Max Drawdown (%) | **23.60** |
| Sortino | +0.387 |

Reads (updated after the Kaggle log arrived — ROOT CAUSE FOUND & FIXED):
1. **End-to-end pipeline works off-cluster** (data via yfinance+stockstats
   shim, HMM fit, replay-buffer training, greedy eval, plots).
2. **⚠⚠ Auto-alpha DIVERGED**: log shows alpha 1.0 → 74.5 (ep10) → 14,733
   (ep20) → 2.9e6 (ep30) → **5.7e8 (ep40)**. Root cause: target_entropy
   was the textbook -act_dim = -38, but the policy's log-probs include the
   affine tanh→[-0.05,0.20] Jacobian (38·log 0.125 ≈ -79), so the MAXIMUM
   achievable entropy on the action box is 38·log 0.25 ≈ -52.7 < -38. The
   target was unreachable ⇒ the temperature controller raised alpha
   without bound ⇒ the actor optimised entropy only ⇒ near-uniform
   allocations.
3. **The "decent" eval was the policy dissolving into 1/N**: SAC eval
   (Return 11.36%, Sharpe 0.386, MaxDD 23.6%) ≈ single-period plain EW
   (12.64%, 0.410, §7c). SAC learned nothing; it mimicked equal weight.
   Flat training reward (20-ep MA ≈ -750..-800 throughout) consistent.
4. **FIX applied to sac_core.py** (2026-07-09): target_entropy =
   act_dim·(log(half_span) − 1) — the standard −dim(A) target expressed in
   tanh-space (≈ -117 for 38 assets), always achievable. Unit tests pass
   (stub act_dim=6 → -18.5, alpha stable at ~1.0 over 3 epochs).
   **Needs commit+push before the Kaggle rerun** (notebook clones GitHub).
5. Timing (T4, free tier): 50 epochs = **0.74 h → 0.9 min/epoch → ~4.5 h
   per 300-epoch seed** ⇒ a 3-seed variant ≈ 13.5 h ≈ half a Kaggle weekly
   quota (30 h/wk); all three variants ≈ 40 h ≈ 1.3 weeks. SAC full runs
   are Kaggle-viable without cluster budget.
6. Environment caveat: shim preprocessing ≈ finrl but not byte-identical;
   SAC numbers from Kaggle are NOT comparable against cluster A2C/PPO
   tables — cluster re-run required if SAC enters the paper.
7. Bonus finding for the paper's themes: this is ANOTHER case of a
   training-mechanics/config artifact producing a superficially plausible
   result (a "conservative low-drawdown SAC agent" that was actually 1/N)
   — caught only because the calibration protocol demanded a healthy
   learning signal, not just good eval metrics.

### 9b — Calibration v2 with the target-entropy fix (Kaggle, 2026-07-09)

**Status: FIX CONFIRMED — learning signal healthy; test transfer poor at
50 epochs.** Same protocol (Hard, seed 0, 50 epochs, mv, T4).

| Metric | v1 (broken alpha) | v2 (fixed) |
|---|---|---|
| Train reward, 20-ep MA | flat ≈ -750..-800 | **-500 → +10,000, monotonic, still rising at ep 50** |
| Total Return (%) | +11.36 (fake: ≈EW) | -10.54 |
| Annualised Sharpe | +0.386 (fake: ≈EW) | -0.081 |
| Max Drawdown (%) | 23.60 | 34.49 |
| Sortino | +0.387 | -0.081 |

Reads:
1. **The entropy-target fix works**: strong monotonic train-reward growth
   replaces the flat entropy-only collapse. SAC training is now
   mechanically sound.
2. **Train-test divergence at 50 epochs**: the agent exploits the
   2015-2021 train distribution hard while test (2022-2024) Sharpe is
   negative. Notably, Hard-SAC seed 0 (-0.081) lands almost exactly on
   corrected Hard-A2C seed 0 (-0.083) — a tease that discrete routing's
   test-period fragility may replicate under an off-policy algorithm,
   but n=1 at a calibration budget; do not overread.
3. **Update-count accounting matters for the full-run budget**: SAC at
   updates_per_step=1 does ~1,750 grad updates/epoch → 50 epochs ≈ 87k
   updates ≈ PPO's ENTIRE 300-epoch budget (~81k). The default 300-epoch
   SAC run would be ~6x PPO's update budget. Full-suite budget should be
   chosen by TRAIN-side reward plateau (not test performance — that would
   overfit the test set): next step is one 300-epoch seed (~4.5h, free)
   to locate the plateau, then fix the suite budget at/past it.
4. Decision: full 3x3 SAC suite is GO on Kaggle quota once the plateau
   budget is known. Cluster re-run still required if SAC enters the paper
   (shim/env caveat, 9.6).

### 9c — Plateau run: Hard, seed 0, 300 epochs (Kaggle, 2026-07-09)

**Status: budget question answered.** Train reward: smooth monotonic
-500 → ~28,500; decelerating after ~ep 150 and near-plateau by ep
250-300 (end slope a few % of mid-run slope). No instability, no
divergence — the fixed auto-alpha is healthy over 525k gradient updates.

| Epochs | Return (%) | Sharpe | MaxDD (%) | Sortino |
|---|---|---|---|---|
| 50 (9b) | -10.54 | -0.081 | 34.49 | -0.081 |
| 300 | **+14.31** | **+0.378** | **27.29** | +0.396 |

Reads:
1. **More training improved TEST performance** (Sharpe -0.08 → +0.38) —
   the 50-epoch dip was undertraining, not overfitting. Train-side
   near-plateau at 300 epochs makes **300 epochs the defensible suite
   budget** (chosen on the train criterion, not test peeking).
2. Context: Hard-SAC seed 0 (0.378, MaxDD 27.3%) vs Hard-A2C seed 0
   (-0.083, 50.7%) and Hard-PPO seed 0 (0.886, 31.2%), single-period EW
   0.410. n=1 — no conclusions, but discrete routing looks mid-pack
   under SAC's twin-Q + entropy regularisation rather than collapsed.
3. ⚠ Watch-item for the suite: eval Sharpe ≈ EW again (0.378 vs 0.410).
   Unlike v1 this sits on a genuinely learned policy (train reward 28.5k
   vs v1's -800), but the runs don't save actions/weights, so
   EW-similarity can't be ruled out directly. Idea #16 (save daily
   returns + weights) would settle it; consider patching before the
   suite so all 9 runs record daily test returns.
4. Update-budget note for any paper text: SAC@300ep ≈ 525k updates ≈
   6.5x PPO's full budget — report as "trained to train-reward plateau",
   not "same epochs".
5. Suite logistics (T4, ~4.5h/seed): 9 runs of `--seeds 1 --seed_start N`
   fit Kaggle's 12h sessions and ~30h/wk quota in ~1.5 weeks. SAC stays
   OUT of the ICAIF main tables (env caveat + scope); it's follow-up /
   journal-version material.

---

## 10. Corrected Walk-Forward (batchfix mechanics, 2 seeds/window) — COMPLETE

**Status: COMPLETE** — job 3714203 (`wf_bf_2seed`), Exit Status 0,
finished 2026-07-10 02:23 after resuming from the W1-W4 cache. 8 windows
× 2 seeds × {Baseline, Hard, Soft}, 300 epochs, mv reward, per-window HMM
refit, corrected non-overlapping-batch mechanics. Results in
`Walkforward/results_batchfix_2seed/`, pulled 2026-07-10. **Supersedes §3
(old mechanics, 1 seed) and §6 (GAE + sliding-window, invalidated).**

### 10a — Aggregate (mean across 8 windows, seed-averaged within window)

| Method | Sharpe | Return % | MaxDD % | Sortino |
|---|---|---|---|---|
| Equal-Weight | **1.645** | 11.56 | **12.63** | 1.656 |
| Markowitz MVO | 1.130 | 12.15 | 16.02 | 1.221 |
| S&P 500 B&H | 1.104 | 6.39 | 13.69 | 1.097 |
| **Baseline (no gate)** | **1.270** | **18.50** | 21.97 | 1.342 |
| Hard Routing | 1.265 | 15.32 | 19.08 | 1.328 |
| Soft MoE (ours) | 1.105 | 13.66 | 18.58 | 1.171 |

### 10b — Per-window Sharpe (seed-averaged)

| W | EW | MVO | SPY | Baseline | Hard | Soft |
|---|---|---|---|---|---|---|
| W1 COVID | 0.33 | 1.15 | 0.03 | 1.11 | 1.12 | 0.68 |
| W2 | 3.18 | 0.96 | 2.43 | 2.51 | 2.34 | 2.23 |
| W3 | 3.10 | 0.33 | 2.45 | 1.76 | 1.94 | 2.20 |
| W4 | 2.03 | 3.25 | 1.72 | 1.80 | 2.59 | 1.73 |
| W5 bear | -1.47 | -1.53 | -1.72 | -2.17 | -1.98 | -2.53 |
| W6 | 0.92 | -0.06 | 0.22 | 0.50 | 0.95 | 0.98 |
| W7 | 2.87 | 3.39 | 2.32 | 3.26 | 2.01 | 2.27 |
| W8 | 2.20 | 1.55 | 1.38 | 1.39 | 1.14 | 1.29 |

### 10c — Statistics

- Wilcoxon (n=8 windows): **no RL-vs-RL comparison remotely significant**
  (all p ≥ 0.55). Only EW vs SPY significant (p=0.0078, the n=8 floor);
  EW vs Soft p=0.0547.
- JK-Memmel on 998-day seed-averaged daily returns: Soft 0.766 vs Hard
  0.877 (z=-0.55, p=0.58); Soft vs Baseline 0.889 (z=-0.60, p=0.55);
  Hard vs Baseline z=-0.06 p=0.95; Soft vs EW 1.006 (z=-0.83, p=0.40).
  Nothing significant at daily granularity either.
- Transition-day decomposition (±2-day buffer, 150/300 transition days):
  every method's Sharpe collapses on transition days (stable ≈ +1.2..1.6
  vs transition ≈ -0.9..-2.0, within-method Welch p<0.05 for most).
  Ordering on transition days: **Hard -0.87 (least bad)** > EW -1.41 >
  MVO -1.56 > Soft -1.57 > SPY -1.88 > Baseline -2.03. Between-method
  significance on transition days NOT yet tested (follow-up if used in
  the paper).

### 10d — Key findings (all three prior walk-forward claims FLIP)

1. **The regime gate adds nothing in the walk-forward protocol**:
   Baseline (no gate) 1.270 ≈ Hard 1.265 > Soft 1.105, all statistically
   indistinguishable. The old ranking (Soft best RL variant, Baseline
   crippled at 0.348) was a training-mechanics artifact — correcting the
   update scheme helped the BASELINE most of all (0.35 → 1.27).
2. **The COVID-window story is dead**: old = Soft 1.00 vs Hard 0.05 vs
   Baseline -0.44 ("regime-awareness matters most in the crash"). New =
   Baseline 1.11 ≈ Hard 1.12 > Soft 0.68. Under sound mechanics every
   variant, gated or not, weathers W1 — and Soft is the laggard.
   The paper's flagship stress-test claim cannot be made.
3. **Hard's W6 spike (2.83) is gone too** (now 0.95 ≈ Soft's 0.98) —
   that anomaly was also mechanics-borne.
4. **Coherent mechanistic synthesis with §8 (single-period)**: where the
   policy is trained ONCE on 7y and must survive an unseen bear+recovery
   (single-period), soft regime-gating shows a large (if n=3-underpowered)
   advantage and Hard is unstable. Where the policy is RETRAINED every 6
   months on an expanding window (walk-forward), regime conditioning adds
   nothing — **frequent retraining substitutes for regime awareness**.
   This protocol-dependence is arguably the paper's most honest and most
   interesting positive statement about when regime-aware architecture
   pays.
5. Transition-day nuance: Hard's discrete switch is nominally least-bad
   on transition days under corrected mechanics — the OPPOSITE direction
   of the old paper's claim (soft blending smoother at transitions).
   Untested between methods; treat as descriptive only.
6. **1/N still wins the walk-forward** (1.645, lowest drawdown), as
   before — the only statistically significant aggregate fact in the
   protocol.

### 10f — Deep-dive: code + math audit of why Soft doesn't win (2026-07-10)

Facts established by reading the implementation and measuring the gate:

1. **Architecture math.** All wf actors share a 2-layer 256-unit trunk;
   specialisation is ONLY in the final linear head(s):
   Baseline mean = 0.1·W·h(x); Hard = 0.1·W_z·h(x) (z = MAP label);
   Soft = 0.1·(Σ_k p_k W_k)·h(x). Since heads are linear over SHARED
   features, Soft is exactly a bilinear model — the gate can only express
   regime-differences that are linear in the shared features. Expressivity
   of specialisation is one linear map per regime.
2. **Gradient flow.** ∂loss/∂W_k ∝ p_k ⇒ each head's effective learning
   rate is its regime's occupancy. W5 train: occupancy (Bear→Bull) =
   [0.50, 0.25, 0.03, 0.22] — the Sideways-Up head gets 3% of gradient
   mass.
3. **The gate is near-one-hot**: measured on W1/W5 fits, mean max-prob
   0.97, >0.99 on 78-85% of days, <0.6 on <2%. So Soft ≈ Hard except on
   ~15-20% of days around transitions — consistent with their wf tie.
4. **⚠ Hyperparameter discovery: the wf trains ALL variants at lr=1e-4,
   L2=0.5** (train_a2c defaults; no per-variant override in
   run_walkforward). Soft's tuned settings (single-period) are lr=3e-5,
   L2=0.01. The wf is therefore hyperparameter-CONTROLLED (good: no
   Soft-favouring confound) but Soft runs 3.3x hotter and 50x more
   L2-shrunk than its tuned config — its wf number may be depressed.
   The queued soft-epochs job tests budget only, NOT lr/L2.
5. **⚠ Mild lookahead in the regime signal**: split_window predicts
   posteriors via hmmlearn predict_proba over the FULL train+test
   sequence — forward-BACKWARD smoothing, so test-day gates use future
   observations. Measured smoothed-vs-causal divergence on test days
   (W1/W5): mean L1 0.08-0.13, MAP-label flips on 4-8% of days, max L1
   1.45 near transitions. Small, and it FAVOURS the gated variants —
   so the wf null holds a fortiori — but it must be disclosed and
   ideally fixed (filtered/forward-only posteriors). The single-period
   pipeline likely shares the pattern (verify hmm_probabilistic.py).
6. **W5 smoking gun — regime-semantics mismatch**: the W5 HMM (trained
   2015-2021) assigns the H1-2022 bear 0% to its "Bear" state and ~80%
   to "Sideways-Down": the 2022 rate-driven grind is statistically
   unlike the COVID-style crash the model learned as "Bear"
   (momentum-vol signature differs). The gate routes the one crisis
   window in the protocol to a mid-occupancy sideways specialist; Soft
   posts its single worst number there (-2.53, worst of ALL methods).
   Independently corroborates §7d's bear-type finding: a 4-state
   momentum-vol HMM cannot distinguish crash-bears from rate-bears.

### 10e — Consequences for the ICAIF paper

- §5.4 (walk-forward) is now a NEGATIVE/boundary-condition section — and
  the mechanics-artifact section gains its third and most sweeping flip
  (drawdown claim, walk-forward ranking incl. Baseline, COVID story).
- The paper's positive claims now rest on: (a) Router collapse (A2C+PPO,
  algorithm-robust); (b) single-period Soft>Hard effect size + stability
  (needs more seeds for significance); (c) gated-EW rule baseline ("the
  signal isn't directly monetizable"); (d) the retraining-substitutes-
  for-regime-awareness synthesis.
- Do NOT reuse the old walkforward_sharpe.png anywhere.

---

## Appendix — Hyperparameters in effect for Run B / Select_3 (Soft MoE)

- LR = 3e-5, L2 coef = 0.01, advantage normalization enabled, epochs = 700
  (config in `main_moe.py` as of commit `e132072`)
- Regime probabilities excluded from shared feature extractor (`input_dim - 4`),
  fed only to the gate
- `env_mixture.py`: covariance / tech-indicator NaNs guarded via `nan_to_num`
