#!/bin/bash
# =============================================================================
# userdata.sh — EC2 instance bootstrap script for XTTS training
#
# Designed for: Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)
# Instance:     g4dn.xlarge (Tesla T4)
#
# This script runs automatically on instance launch via EC2 User Data.
# It assumes the custom TTS AMI is used — all dependencies are pre-installed.
# It only pulls code, downloads the dataset, and launches training.
#
# Required EC2 launch parameters (pass as environment via user data or SSM):
#   CHARACTER   — character name matching S3 dataset path (e.g. thrall)
#   S3_BUCKET   — S3 bucket name (e.g. amzn-s3-voices-dataset)
#   GITHUB_REPO — GitHub repo URL (e.g. https://github.com/nikotpbn/tts.git)
#
# Usage (override defaults below or pass via launch template):
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override these at launch via EC2 user data
# ---------------------------------------------------------------------------

CHARACTER="${CHARACTER:-thrall}"
S3_BUCKET="${S3_BUCKET:-amzn-s3-voices-dataset}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/nikotpbn/tts.git}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"

PROJECT_DIR="/home/ubuntu/tts"
VENV_DIR="/home/ubuntu/venv"
LOG_FILE="/home/ubuntu/bootstrap.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=============================================="
echo "TTS Training Bootstrap"
echo "Started: $(date -u)"
echo "Character: $CHARACTER"
echo "S3 Bucket: $S3_BUCKET"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1. Pull latest code from GitHub
# ---------------------------------------------------------------------------

echo "[1/5] Pulling latest code from GitHub..."

if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 2. Activate venv (pre-installed in AMI)
# ---------------------------------------------------------------------------

echo "[2/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Apply known TTS patch (idempotent)
sed -i 's/config.audio.dvae_sample_rate/config.audio.sample_rate/g' \
    "$VENV_DIR/lib/python3.11/site-packages/TTS/tts/layers/xtts/trainer/gpt_trainer.py" \
    2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Download dataset from S3
# ---------------------------------------------------------------------------

echo "[3/5] Downloading dataset from S3..."

mkdir -p "$PROJECT_DIR/data/processed/$CHARACTER"

aws s3 cp \
    "s3://$S3_BUCKET/characters/$CHARACTER/processed/" \
    "$PROJECT_DIR/data/processed/$CHARACTER/" \
    --recursive

echo "Dataset downloaded."

# ---------------------------------------------------------------------------
# 4. Launch training inside tmux
# ---------------------------------------------------------------------------

echo "[4/5] Launching training..."

tmux new-session -d -s train \
    "source $VENV_DIR/bin/activate && \
     cd $PROJECT_DIR && \
     python -u scripts/train_xtts.py \
         --character $CHARACTER \
         --epochs $EPOCHS \
         --batch-size $BATCH_SIZE \
         --s3-bucket $S3_BUCKET \
     2>&1 | tee scripts/training.log"

echo "Training launched in tmux session 'train'."
echo "Attach with: tmux attach -t train"

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------

echo "[5/5] Bootstrap complete: $(date -u)"
echo "Monitor training: tail -f $PROJECT_DIR/scripts/training.log"