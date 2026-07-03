# ---------------------------------------------------------------------------
# TTS Pipeline Makefile
#
# Usage:
#   make scrape CHARACTER=thrall
#   make transform CHARACTER=thrall
#   make upload CHARACTER=thrall
#   make dataset CHARACTER=thrall              # scrape + transform + upload
#   make dataset CHARACTER=thrall MAX=100      # with custom download cap
#   make dataset CHARACTER=thrall WHISPER=medium
#
# Defaults:
#   CHARACTER = thrall
#   MAX       = 50
#   WHISPER   = large-v2
# ---------------------------------------------------------------------------

CHARACTER ?= thrall
MAX       ?= 50
WHISPER   ?= large-v2
EPOCHS      ?= 100
BATCH_SIZE  ?= 2

.PHONY: help install scrape transform upload dataset clean

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "TTS Pipeline"
	@echo "------------"
	@echo "  make install                        Install local dependencies"
	@echo "  make scrape CHARACTER=thrall        Scrape raw audio files"
	@echo "  make transform CHARACTER=thrall     Transform + transcribe audio"
	@echo "  make upload CHARACTER=thrall        Upload processed dataset to S3"
	@echo "  make dataset CHARACTER=thrall       Full pipeline: scrape + transform + upload"
	@echo ""
	@echo "Optional overrides:"
	@echo "  MAX=100                             Max files to download (default: 50)"
	@echo "  WHISPER=medium                      Whisper model size (default: large-v2)"
	@echo ""

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install:
	@echo ">>> Installing dependencies..."
	pip install -r requirements.txt
	playwright install chromium
	@echo ">>> Done. Remember to install FFmpeg via: brew install ffmpeg"

# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

scrape:
	@echo ">>> Scraping audio for character: $(CHARACTER) (max: $(MAX))"
	python scripts/scrape.py --character $(CHARACTER) --max $(MAX)

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

transform:
	@echo ">>> Transforming audio for character: $(CHARACTER) (whisper: $(WHISPER))"
	python scripts/transform.py --character $(CHARACTER) --whisper-model $(WHISPER)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

upload:
	@echo ">>> Uploading processed dataset for character: $(CHARACTER) to S3..."
	python scripts/upload.py --character $(CHARACTER)

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

dataset: scrape transform upload
	@echo ">>> Pipeline complete for character: $(CHARACTER)"

# ---------------------------------------------------------------------------
# Clean (removes processed output for a character — raw data is preserved)
# ---------------------------------------------------------------------------

clean:
	@echo ">>> Cleaning processed data for character: $(CHARACTER)"
	@read -p "Are you sure? This deletes data/processed/$(CHARACTER)/ [y/N]: " confirm && \
	[ "$$confirm" = "y" ] && rm -rf data/processed/$(CHARACTER) || echo "Aborted."

# ---------------------------------------------------------------------------
# Download VOD for manual clipping
# ---------------------------------------------------------------------------
download:
	@echo ">>> Downloading VOD for character: $(CHARACTER)"
	python scripts/download_vod.py --character $(CHARACTER) --url $(URL)

# ---------------------------------------------------------------------------
# Complete training pipeline (sync dataset + launch training - requires AWS credentials)
# ---------------------------------------------------------------------------
train:
	@echo ">>> Syncing dataset and launching training for: $(CHARACTER)"
	python scripts/upload.py --character $(CHARACTER)
	python scripts/launch_training.py --character $(CHARACTER) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)