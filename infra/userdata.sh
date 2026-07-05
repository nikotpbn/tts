#!/bin/bash
# =============================================================================
# userdata.sh — EC2 instance bootstrap script for XTTS training.
#
# This script is loaded and templated by launch_training.py at launch time.
# Placeholders ({{VAR}}) are substituted before being passed to EC2.
#
# Do NOT run this script directly — use: make train CHARACTER=<character>
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (injected by launch_training.py)
# ---------------------------------------------------------------------------

CHARACTER="{{CHARACTER}}"
EPOCHS="{{EPOCHS}}"
BATCH_SIZE="{{BATCH_SIZE}}"
S3_BUCKET="{{S3_BUCKET}}"
GITHUB_REPO="{{GITHUB_REPO}}"
SNS_TOPIC_ARN="{{SNS_TOPIC_ARN}}"
CLOUDWATCH_LOG_GROUP="{{CLOUDWATCH_LOG_GROUP}}"
AWS_REGION="{{AWS_DEFAULT_REGION}}"

PROJECT_DIR="/home/ubuntu/tts"
VENV_DIR="/home/ubuntu/venv"
LOG_FILE="/home/ubuntu/bootstrap.log"
RUN_ID="$(date -u +%Y-%m-%d_%H-%M-%S)"

# ---------------------------------------------------------------------------
# Start logging to file immediately — before anything else
# ---------------------------------------------------------------------------

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "TTS Training Bootstrap"
echo "Started: $(date -u)"
echo "Character: $CHARACTER"
echo "Run ID: $RUN_ID"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="

# ---------------------------------------------------------------------------
# CloudWatch log streaming (best effort — never kills the script)
# ---------------------------------------------------------------------------

echo "[CW] Installing CloudWatch agent..."
apt-get install -y amazon-cloudwatch-agent 2>/dev/null || echo "[CW] Agent install failed, continuing..."

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << CWCONFIG || true
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/home/ubuntu/bootstrap.log",
                        "log_group_name": "$CLOUDWATCH_LOG_GROUP",
                        "log_stream_name": "$CHARACTER/$RUN_ID",
                        "timezone": "UTC"
                    }
                ]
            }
        }
    }
}
CWCONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s \
    2>/dev/null || echo "[CW] Agent start failed, continuing without CloudWatch streaming..."

echo "[CW] CloudWatch setup complete."

# ---------------------------------------------------------------------------
# SNS notification helper
# ---------------------------------------------------------------------------

notify() {
    local subject="$1"
    local message="$2"
    aws sns publish \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "$subject" \
        --message "$message" \
        --region "$AWS_REGION" || true
}

# ---------------------------------------------------------------------------
# Spot interruption detector (background)
# ---------------------------------------------------------------------------

(
    while true; do
        RESPONSE=$(curl -s -w "\n%{http_code}" \
            http://169.254.169.254/latest/meta-data/spot/interruption-notice 2>/dev/null)
        HTTP_CODE=$(echo "$RESPONSE" | tail -1)
        BODY=$(echo "$RESPONSE" | head -1)

        if [ "$HTTP_CODE" -eq 200 ]; then
            REASON=$(echo "$BODY" | python3 -c \
                "import sys,json; d=json.load(sys.stdin); print(d.get('action', 'unknown'))" \
                2>/dev/null || echo "unknown")
            echo "[WARNING] Spot interruption notice received. Action: $REASON"
            notify \
                "TTS Training INTERRUPTED — $CHARACTER" \
                "Spot instance reclaimed by AWS.\nAction: $REASON\nCharacter: $CHARACTER\nRun ID: $RUN_ID\nCheck CloudWatch: $CLOUDWATCH_LOG_GROUP/$CHARACTER/$RUN_ID"
            break
        fi
        sleep 5
    done
) &
SPOT_MONITOR_PID=$!

# ---------------------------------------------------------------------------
# Trap errors — notify and terminate on failure
# ---------------------------------------------------------------------------

trap '{
    echo "[ERROR] Bootstrap failed at line $LINENO"
    kill $SPOT_MONITOR_PID 2>/dev/null || true
    notify \
        "TTS Training FAILED — $CHARACTER" \
        "Training failed for character: $CHARACTER\nRun ID: $RUN_ID\nCheck CloudWatch: $CLOUDWATCH_LOG_GROUP/$CHARACTER/$RUN_ID"
    shutdown -h now
}' ERR

# ---------------------------------------------------------------------------
# 1. Pull latest code from GitHub
# ---------------------------------------------------------------------------

echo "[1/5] Pulling latest code..."
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR" && git pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
fi
echo "[1/5] Code ready."

# ---------------------------------------------------------------------------
# 2. Activate venv and apply known patches
# ---------------------------------------------------------------------------

echo "[2/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

sed -i 's/config.audio.dvae_sample_rate/config.audio.sample_rate/g' \
    "$VENV_DIR/lib/python3.11/site-packages/TTS/tts/layers/xtts/trainer/gpt_trainer.py" \
    2>/dev/null || true

echo "[2/5] Virtual environment ready."

# ---------------------------------------------------------------------------
# 3. Download dataset from S3
# ---------------------------------------------------------------------------

echo "[3/5] Downloading dataset from S3..."
echo "[3/5] Bucket: $S3_BUCKET"
echo "[3/5] Source: s3://$S3_BUCKET/characters/$CHARACTER/processed/"
echo "[3/5] Dest:   $PROJECT_DIR/data/processed/$CHARACTER/"
echo "[3/5] AWS CLI version: $(aws --version 2>&1)"
echo "[3/5] AWS identity: $(aws sts get-caller-identity 2>&1)"

mkdir -p "$PROJECT_DIR/data/processed/$CHARACTER"

aws s3 sync \
    "s3://$S3_BUCKET/characters/$CHARACTER/processed/" \
    "$PROJECT_DIR/data/processed/$CHARACTER/"

echo "[3/5] Download complete."

# ---------------------------------------------------------------------------
# 4. Run training
# ---------------------------------------------------------------------------

echo "[4/5] Starting training..."
cd "$PROJECT_DIR"
python -u scripts/train_xtts.py \
    --character "$CHARACTER" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --s3-bucket "$S3_BUCKET" \
    2>&1 | tee scripts/training.log

# ---------------------------------------------------------------------------
# 5. Notify success and terminate
# ---------------------------------------------------------------------------

kill $SPOT_MONITOR_PID 2>/dev/null || true

echo "[5/5] Training complete."
notify \
    "TTS Training Complete — $CHARACTER" \
    "Training finished for character: $CHARACTER\nRun ID: $RUN_ID\nModel saved to: s3://$S3_BUCKET/characters/$CHARACTER/models/\nCloudWatch logs: $CLOUDWATCH_LOG_GROUP/$CHARACTER/$RUN_ID"

echo "Shutting down instance..."
shutdown -h now