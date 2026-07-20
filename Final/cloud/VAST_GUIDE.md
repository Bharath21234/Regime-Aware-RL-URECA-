# Vast.ai Runbook — Finishing the ICAIF Experiment Program (~$25–27)

End-to-end guide for running the remaining experiments on rented A40 GPUs.
Written 2026-07-16. Companion script: [`vast_setup.sh`](vast_setup.sh).

## What gets run, and what it costs

| Run | Box | Hours | ~Cost @ $0.29/h |
|---|---|---|---|
| Multi-asset **baseline** (3 seeds) | 1 | ~19.5 | $5.70 |
| Multi-asset **hard** (3 seeds) | 2 | ~19.5 | $5.70 |
| Multi-asset **moe/soft** (3 seeds) | 3 | ~19.5 | $5.70 |
| **Soft-v2** remediation (3 seeds) | reuse a box, day 2 | ~19.5 | $5.70 |
| **Baseline seed-2** finisher (single-asset) | same box, after Soft-v2 | ~6.5 | $1.90 |
| **Total** | | ~85 GPU-h | **~$24.70** |

Timeline from launch: trio done in ~20h; Soft-v2 + seed 2 done ~26h after
that. **All compute finished ~2.5 days after you start.** Hands-on time:
about an hour total.

Rules of the road:
- The multi-asset runs are a **trio — run all three arms or none**
  (pre-committed interpretation in `results_log.md` §15; a gated arm
  without its baseline is uninterpretable).
- Billing is per-second while an instance exists. **Destroy** boxes the
  moment their results are downloaded. "Stop" still bills storage.
- If the balance hits $0, running instances are stopped mid-job. Load the
  full amount up front.

## Step 1 — Account and credit (5 min)

1. Sign up at https://cloud.vast.ai and verify your email.
2. Billing → Add Credit → **$27** by card.

## Step 2 — Template

Console → **Templates** → select the official **PyTorch** template (CUDA
image, torch preinstalled). This is the image your instances boot with.

## Step 3 — Rent three boxes

On the Search / Create page set the filters:

- **GPU: A40**, count **1×** (not 2×/4× — the code is single-GPU; extra
  GPUs are pure waste). If A40s are scarce: RTX A6000 or RTX 3090 at a
  similar price are fine — note which one you used. Avoid T4/P100 (2–3×
  slower). Don't pay up for A100/H100 — this workload is CPU-loop-bound
  and won't run meaningfully faster.
- **On-Demand** — NOT "Interruptible" (interruptible can be preempted
  mid-run and these are ~20h jobs).
- **Reliability ≥ 99%**, disk slider **~40 GB**, CPU cores ≥ 4.
- Sort by price ascending. Rent the cheapest sane host. Repeat ×3.

## Step 4 — Open a terminal on each box

Instances tab → wait for status **Running** → **Open** → Jupyter loads →
**New → Terminal**.

## Step 5 — Setup (one paste per box, ~5 min)

```bash
bash <(curl -sL https://raw.githubusercontent.com/Bharath21234/Regime-Aware-RL-URECA-/main/Final/cloud/vast_setup.sh)
```

The script clones the repo, **asserts the GPU is visible** (if it fails
here, the host is broken — destroy it and rent another), installs the
package set, and prints the run commands.

## Step 6 — Launch one arm per box

```bash
# Box 1
MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 3 \
    --reward_mode mv --epochs 1000 --tag ma > ma_baseline.out 2>&1 &

# Box 2
MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant hard --seeds 3 \
    --reward_mode mv --epochs 1000 --tag ma > ma_hard.out 2>&1 &

# Box 3
MULTI_ASSET=1 nohup python -u run.py --variant moe --seeds 3 \
    --reward_mode mv --epochs 1000 --tag ma > ma_moe.out 2>&1 &
```

Sanity-check the first minutes of each log:

```bash
tail -f ma_baseline.out     # Ctrl-C stops the tail, not the job
```

You should see `MULTI_ASSET=1: universe extended to 42 assets
(+TLT/IEF/GLD/SHY)`, the data download, then epoch lines. After that,
**close the tab; laptop can be off** — `nohup` keeps the job alive.
Reconnect any time via the Open button.

## Step 7 — Collect results (~20h later)

Each log ends with an aggregate stats table. On each box:

```bash
cd /workspace/Regime-Aware-RL-URECA- && zip -r ma_results.zip Final/results/*_ma Final/*.out
```

Download the zip via the Jupyter file browser (right-click the file →
Download).

## Step 8 — Day-2 sequence (reuse ONE box, destroy the other two)

Keep the cheapest box; destroy the rest. On the kept box:

```bash
cd /workspace/Regime-Aware-RL-URECA-/Final

# Soft-v2 (~19.5h)
SOFT_V2=1 nohup python -u run.py --variant moe --seeds 3 --reward_mode mv \
    --epochs 1000 --tag softv2 > softv2.out 2>&1 &

# When softv2.out shows the final table, run the seed-2 finisher (~6.5h):
L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 1 --seed_start 2 \
    --reward_mode mv --epochs 1000 > baseline_s2.out 2>&1 &
```

Collect the same way:

```bash
cd /workspace/Regime-Aware-RL-URECA- && zip -r day2_results.zip \
    Final/results/moe_softv2 Final/results/baseline Final/*.out
```

## Step 9 — Destroy everything

Instances page → trash icon → **Destroy** (not Stop). Billing ends the
moment the instance is destroyed. Verify the Instances page is empty.

## Step 10 — Hand off for analysis

Send the zips (`ma_results.zip` ×3, `day2_results.zip`) for analysis:
paired stats against the §15 pre-committed interpretation, 42-asset
equal-weight/Markowitz/S&P benchmarks recomputed for the comparison
table, results_log §16, and the paper table updates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Setup script dies at the GPU assert | Broken host — destroy, rent another |
| Yahoo/yfinance download errors in the first ~10 min | Rerun the command once; if it persists, the host IP is rate-limited — destroy, rent a different host |
| `nvidia-smi` shows ~0% GPU utilisation | Normal for this workload (small nets, CPU env loop) — check the log is advancing instead |
| Log stalls mid-training for >30 min | `ps aux \| grep python` — if the process died, check the log tail for a traceback and rerun; seeds already completed are saved |
| Instance disappeared / "outbid" | You rented Interruptible by mistake — re-rent as On-Demand |
| Balance ran out, instances stopped | Add credit; instances restart from the Instances page, but the in-flight seed restarts from scratch — completed seeds are safe |

## Cost-control checklist

- [ ] On-Demand, 1× GPU, ≥99% reliability, ~40 GB disk
- [ ] One arm per box; log shows `MULTI_ASSET=1 ... 42 assets`
- [ ] Zips downloaded before destroying anything
- [ ] Two boxes destroyed after day 1; last box destroyed after day 2
- [ ] Instances page empty at the end
