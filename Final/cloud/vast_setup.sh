#!/usr/bin/env bash
# ============================================================================
# One-paste setup for a rented GPU box (vast.ai / RunPod), mirroring the NTU
# cluster stack. Pick an A40 instance (same GPU as the cluster) with a
# PyTorch/CUDA template image, open the shell, and paste this whole file.
#
#   bash <(curl -sL https://raw.githubusercontent.com/Bharath21234/Regime-Aware-RL-URECA-/main/Final/cloud/vast_setup.sh)
#
# or clone first and `bash Final/cloud/vast_setup.sh`. Billing is per-minute
# while the instance is on -- DESTROY the instance as soon as results are
# downloaded, don't just stop the jupyter tab.
# ============================================================================
set -e

cd /workspace 2>/dev/null || cd ~

if [ ! -d Regime-Aware-RL-URECA- ]; then
    git clone https://github.com/Bharath21234/Regime-Aware-RL-URECA-.git
fi
cd Regime-Aware-RL-URECA-
git pull

# Template images ship torch+CUDA; verify before installing anything.
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "NO GPU VISIBLE -- wrong instance/image, destroy and re-rent"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY

# Same package set as the cluster venv. FinRL is pinned to the SAME git
# commit as requirements.txt (fee45af) -- a different FinRL version could
# preprocess data differently and silently break comparability with the
# cluster runs.
pip install -q "finrl @ git+https://github.com/AI4Finance-Foundation/FinRL.git@fee45af12ee0af490cd8e091514173b571dcd9ed" \
    hmmlearn gymnasium gym pandas matplotlib scikit-learn scipy yfinance stockstats

# Fail fast: verify the exact imports the training code uses, and that
# market data is downloadable from this host's IP (Yahoo rate-limits some
# datacenter IPs). If either check fails, DESTROY this box and rent another.
python3 - <<'PY'
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import hmmlearn, gym, stockstats
df = YahooDownloader(start_date="2024-01-02", end_date="2024-01-10",
                     ticker_list=["SPY", "TLT"]).fetch_data()
assert len(df) > 5 and set(df.tic.unique()) == {"SPY", "TLT"}, f"bad download: {len(df)} rows"
print(f"SANITY OK: finrl imports fine, downloaded {len(df)} rows for SPY+TLT")
PY

cd Final
mkdir -p results

cat <<'EOF'

============================ READY -- run ONE per box ============================
Soft-v2 (3 seeds, ~19.5h on A40):
  SOFT_V2=1 nohup python -u run.py --variant moe --seeds 3 --reward_mode mv \
      --epochs 1000 --tag softv2 > softv2.out 2>&1 &

Baseline seed-2 finisher (~6.5h):
  L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 1 --seed_start 2 \
      --reward_mode mv --epochs 1000 > baseline_s2.out 2>&1 &

Router corrected-A2C (3 seeds, ~19.5h):
  nohup python -u run.py --variant router --seeds 3 --reward_mode mv \
      --epochs 1000 --tag bf > router_bf.out 2>&1 &

Multi-asset (42 assets incl TLT/IEF/GLD/SHY; run all three arms, one per box;
pre-committed interpretation in results_log 15 -- run ONLY as the full trio):
  MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant baseline --seeds 3 \
      --reward_mode mv --epochs 1000 --tag ma > ma_baseline.out 2>&1 &
  MULTI_ASSET=1 L2_COEF=0.01 nohup python -u run.py --variant hard --seeds 3 \
      --reward_mode mv --epochs 1000 --tag ma > ma_hard.out 2>&1 &
  MULTI_ASSET=1 nohup python -u run.py --variant moe --seeds 3 \
      --reward_mode mv --epochs 1000 --tag ma > ma_moe.out 2>&1 &

Watch:      tail -f <logfile>
Collect:    cd .. && zip -r results_$(hostname).zip Final/results Final/*.out
            then download the zip via the vast.ai file browser / scp, and
            DESTROY the instance.
==================================================================================
EOF
