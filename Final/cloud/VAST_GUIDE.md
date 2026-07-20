# Vast.ai Runbook — Finishing the ICAIF Experiment Program (~$27)

End-to-end guide, updated 2026-07-16 for what's actually rentable: no 1×
A40s were available with the PyTorch template, so the plan uses **two 2×
A40 boxes** ($0.577/hr = same ~$0.29 per GPU) and runs **everything in
parallel**. Companion script: [`vast_setup.sh`](vast_setup.sh).

## The plan at a glance

| Box | GPU 0 | GPU 1 | Hours |
|---|---|---|---|
| **A** (2× A40) | Multi-asset **baseline** (3 seeds) | Multi-asset **hard** (3 seeds) | ~20 |
| **B** (2× A40) | Multi-asset **moe/soft** (3 seeds) | **Soft-v2** (3 seeds) | ~20 |
| **B** afterwards | Baseline **seed-2** finisher (single-asset) | — | ~6.5 |

Cost: 2 boxes × ~20h × $0.577 ≈ $23, plus ~6.5h × $0.577 ≈ $3.75 for the
finisher → **~$27 total. All compute done ~27 hours after launch.**

Rules of the road:
- The multi-asset runs are a **trio — all three arms must run**
  (pre-committed interpretation in `results_log.md` §15).
- Billing is per-second while an instance exists. **Destroy** boxes the
  moment results are downloaded — "Stop" still bills storage.
- If the balance hits $0, instances stop mid-job. Load $27 up front.

## Step 1 — Account, credit, template (done once)

1. Sign up at https://cloud.vast.ai, verify email.
2. Billing → Add Credit → **$27** by card.
3. Templates → **PyTorch (Vast)** (NOT PyTorch NGC — its CUDA 13 image
   is incompatible with hosts whose Max CUDA is 12.x).

## Step 2 — Rent the two boxes

Search page with the template selected:
- \#GPUs: **2X** · **On-Demand** (never Interruptible — these are 20h
  jobs with no checkpointing) · GPU: **A40** · sort **Price (inc.)**
- Left panel: Container Size **40 GB**. Leave other sliders at defaults.
- Rent the two ~$0.577/hr 2× A40 listings (reliability ≈ 98.8–99.2%).
  Caveat: if both are on the **same host id**, a host failure kills
  everything at once — if Show More Results offers a 2× A40 on a
  *different* host at similar price, prefer it for the second box.

If 2× A40s vanish too: 1× RTX A6000 / RTX 3090 On-Demand ≤ $0.35/hr are
acceptable substitutes (note which model was used); avoid T4/P100
(2–3× slower) and don't pay A100/H100 prices — this workload is
CPU-loop-bound and won't benefit.

## Step 3 — Setup (once per box, ~5 min)

Instances tab → wait for **Running** → **Open** → Jupyter → New →
Terminal, then paste:

```bash
bash <(curl -sL https://raw.githubusercontent.com/Bharath21234/Regime-Aware-RL-URECA-/main/Final/cloud/vast_setup.sh)
```

It clones the repo, asserts a GPU is visible, and installs packages.
Then confirm **both** GPUs are present:

```bash
nvidia-smi   # must list two A40s
```

## Step 4 — Launch (one paste per box)

Each job is pinned to its own GPU with `CUDA_VISIBLE_DEVICES`. Both
hosts have ~128 CPU cores, so two jobs per box don't contend.

**Box A:**
```bash
cd /workspace/Regime-Aware-RL-URECA-/Final
CUDA_VISIBLE_DEVICES=0 MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 3 --reward_mode mv --epochs 1000 --tag ma > ma_baseline.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant hard --seeds 3 --reward_mode mv --epochs 1000 --tag ma > ma_hard.out 2>&1 &
```

**Box B:**
```bash
cd /workspace/Regime-Aware-RL-URECA-/Final
CUDA_VISIBLE_DEVICES=0 MULTI_ASSET=1 nohup python -u run.py --variant moe --seeds 3 --reward_mode mv --epochs 1000 --tag ma > ma_moe.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 SOFT_V2=1 nohup python -u run.py --variant moe --seeds 3 --reward_mode mv --epochs 1000 --tag softv2 > softv2.out 2>&1 &
```

Sanity-check the first minutes:

```bash
tail -f ma_baseline.out    # Ctrl-C stops the tail, not the job
```

Expected early lines: `MULTI_ASSET=1: universe extended to 42 assets
(+TLT/IEF/GLD/SHY)` on the ma_* jobs, and `SOFT_V2 active: 2-layer heads
+ inverse-occupancy rescale` on softv2. Then data download, then epoch
lines. Once both logs on a box are advancing, **close the tab — laptop
can be off**; `nohup` keeps everything running. Reconnect via Open.

## Step 5 — Collect (~20h later)

Each log ends with an aggregate stats table (`Done. Results in ...`).
On each box:

```bash
cd /workspace/Regime-Aware-RL-URECA- && zip -r results_$(hostname).zip Final/results/*_ma Final/results/moe_softv2 Final/*.out
```

(One of the two patterns won't exist on each box — zip warns and
continues; that's fine.) Download the zip via the Jupyter file browser
(right-click → Download).

## Step 6 — Seed-2 finisher, then destroy

1. **Destroy Box A** (Instances → trash icon → Destroy; NOT Stop).
2. On Box B:

```bash
cd /workspace/Regime-Aware-RL-URECA-/Final
L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 1 --seed_start 2 --reward_mode mv --epochs 1000 > baseline_s2.out 2>&1 &
```

(~6.5h, single-asset — completes the corrected Baseline at n=3.)

3. When it finishes:

```bash
cd /workspace/Regime-Aware-RL-URECA- && zip -r seed2.zip Final/results/baseline Final/baseline_s2.out
```

Download, then **destroy Box B**. Verify the Instances page is empty.

## Step 7 — Hand off for analysis

Send all zips for analysis: paired stats against the §15 pre-committed
interpretation, 42-asset EW/Markowitz/S&P benchmarks recomputed for the
comparison table, Soft-v2 vs Soft (§13 pre-commitment), results_log
§16, and the paper updates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Setup script dies at the GPU assert | Broken host — destroy, rent another |
| `nvidia-smi` shows one GPU on a 2× box | Wrong listing or broken host — destroy, re-rent |
| Yahoo/yfinance download errors in first ~10 min | Rerun the command once; if persistent, host IP is rate-limited — destroy, rent a different host |
| GPU utilisation ~0% in `nvidia-smi` | Normal (small nets, CPU env loop) — check the log is advancing instead |
| Log stalls >30 min | `ps aux \| grep python`; if the process died, check the traceback and relaunch that one command — completed seeds are saved |
| Instance disappeared / "outbid" | You rented Interruptible by mistake — re-rent On-Demand |
| Balance ran out, instances stopped | Add credit, restart from Instances page; in-flight seed restarts, completed seeds are safe |

## Cost-control checklist

- [ ] $27 loaded before renting
- [ ] 2 boxes, On-Demand, 2× A40, 40 GB container
- [ ] `nvidia-smi` shows 2 GPUs on each box
- [ ] 4 logs advancing: ma_baseline, ma_hard, ma_moe, softv2
- [ ] Zips downloaded before destroying anything
- [ ] Box A destroyed after day 1; Box B destroyed after seed-2
- [ ] Instances page empty at the end
