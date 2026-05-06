#!/usr/bin/env bash
# ============================================================
# One-shot setup for GPU cluster (CUDA).
# Run once after git clone:
#   bash setup_cluster.sh
# ============================================================
set -e

# ── 1. Create and activate a virtual environment ──────────────
python3 -m venv .venv
source .venv/bin/activate

# ── 2. Install PyTorch with CUDA ──────────────────────────────
# Adjust the --index-url if your cluster uses a different CUDA version.
# Check with:  nvidia-smi | head -3
#   CUDA 12.1 → cu121   CUDA 11.8 → cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 3. Install everything else from requirements.txt ──────────
# Skip torch/torchvision/torchaudio (already installed above),
# ElegantRL (not used by our code), and MPS-only wheels.
pip install \
    finrl \
    hmmlearn \
    gymnasium \
    gym \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    scipy \
    yfinance \
    stockstats \
    pyfolio-reloaded \
    empyrical-reloaded

echo ""
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Then run:  cd Final && python run.py --seeds 20"
