# TTS Pipeline

A pipeline for scraping, processing, and fine-tuning XTTS voice models on character audio datasets. Built for solo or small-team use with a clean separation between local processing and cloud training.

---

## Project Structure

```
tts/
├── .env                        # AWS credentials (never commit)
├── .gitignore
├── Makefile                    # Pipeline orchestration
├── README.md
├── requirements.txt            # Local dependencies
├── config/                     # Per-character training configs + AWS policy docs
├── data/
│   ├── raw/<character>/        # Scraped .ogg files + download_map.json
│   ├── processed/<character>/  # wavs/ + metadata.csv + dataset_meta.json
│   └── inference/<character>/  # Generated audio output
├── models/
│   ├── base/                   # XTTS base weights (downloaded once, reused)
│   └── <character>/
│       └── <run_id>/           # best_model.pth + config.json per training run
├── infra/
│   └── userdata.sh             # EC2 bootstrap script
└── scripts/
    ├── scrape.py               # Audio scraper (local only)
    ├── transform.py            # Audio transformation + transcription (local only)
    ├── upload.py               # S3 dataset upload (local only)
    ├── train_xtts.py           # XTTS fine-tuning (EC2 only)
    └── inference.py            # Speech synthesis from fine-tuned model (local)
```

---

## Architecture Overview

```
LOCAL                               AWS
─────                               ───
scrape.py                           EC2 g4dn.xlarge (spot)
    ↓                                   ↓
transform.py                    ←── S3 bucket (dataset + models)
    ↓                                   ↓
upload.py ─────────────────────→    train_xtts.py
                                        ↓
inference.py ←─────────────────── best_model.pth (uploaded to S3)
```

- **Local machine** handles all data work: scraping, audio conversion, Whisper transcription, S3 upload, and inference.
- **EC2** is reserved exclusively for GPU training. No scraping or transformation runs there.
- **S3** is the handoff point between local and cloud — clean, versioned, always the source of truth.

---

## Local Setup

### Requirements

- Python 3.11+
- FFmpeg (system level)
- AWS CLI

```bash
# macOS
brew install ffmpeg awscli

# Ubuntu/Debian
sudo apt install ffmpeg awscli
```

### Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### Environment variables

Create a `.env` file in the project root:

```dotenv
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your_bucket_name
AWS_DEFAULT_REGION=eu-central-1
```

The IAM user behind these credentials should have a **scoped S3 policy** — not `AmazonS3FullAccess`. Minimum required actions:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutObject",
    "s3:GetObject",
    "s3:DeleteObject",
    "s3:ListBucket"
  ],
  "Resource": ["arn:aws:s3:::your-bucket", "arn:aws:s3:::your-bucket/*"]
}
```

The IAM policy JSON template lives in `config/iam_policy.json` for reference.

---

## Pipeline Usage

All steps are orchestrated via `make`. Run from the project root.

### Full pipeline (scrape + transform + upload)

```bash
make dataset CHARACTER=thrall
```

### Individual steps

```bash
make scrape CHARACTER=thrall MAX=100
make transform CHARACTER=thrall WHISPER=large-v2
make upload CHARACTER=thrall
```

### Clean processed data (raw files are preserved)

```bash
make clean CHARACTER=thrall
```

### Available options

| Option      | Default    | Description                          |
| ----------- | ---------- | ------------------------------------ |
| `CHARACTER` | `thrall`   | Character name                       |
| `MAX`       | `50`       | Max files to download per scrape run |
| `WHISPER`   | `large-v2` | Whisper model size for transcription |

---

## Pipeline Scripts

### `scrape.py`

Scrapes audio files from [wago.tools](https://wago.tools) using Playwright.

- Maintains a `download_map.json` ledger per character to skip already-downloaded files (idempotent)
- Randomized delay between downloads (20–60s) to respect rate limits
- 3-attempt retry with exponential backoff per file
- Logs to `data/raw/<character>/scrape.log`
- **Run locally only** — requires a visible browser window (`headless=False`)

### `transform.py`

Converts raw audio to LJSpeech-format training data.

- Converts `.ogg`/`.mp3`/`.wav` → mono 22050Hz 16-bit PCM WAV
- Filters clips under 1.5s (attack sounds, death sounds, etc.)
- Splits long clips (> 10s) on silence boundaries using pydub — no mid-word cuts
- Transcribes each chunk with OpenAI Whisper (`large-v2` recommended)
- Filters garbage transcriptions (< 3 words or < 10 characters)
- Idempotent — tracks processed files in `processed_files.json`, only processes new files on re-runs
- Writes `metadata.csv` (LJSpeech format: `id|text|text`) and `dataset_meta.json`
- Logs to `data/processed/<character>/transform.log`

### `upload.py`

Uploads the processed dataset to S3.

- Uploads all files from `data/processed/<character>/` to `s3://<bucket>/characters/<character>/processed/`
- Always overwrites existing files (latest local version wins)
- Logs to `data/processed/<character>/upload.log`

### `train_xtts.py`

Fine-tunes XTTS on a character dataset. **Run on EC2 only.**

- Downloads base XTTS weights to `models/base/` once — reused across all characters and runs
- Saves checkpoints to `models/<character>/<run_id>/`
- Uploads `best_model.pth` and `config.json` to S3 automatically at end of training
- Pre-flight wav validation before training starts

```bash
python scripts/train_xtts.py --character thrall --epochs 100 --batch-size 2
```

### `inference.py`

Generates speech from a fine-tuned model. **Run locally.**

- Auto-selects the latest training run if `--run-id` is not specified
- Auto-selects the best reference audio clip from the processed dataset
- Output saved to `data/inference/<character>/<timestamp>.wav`

```bash
python scripts/inference.py --character thrall --text "For the Horde!"
python scripts/inference.py --character thrall --text "For the Horde!" --run-id 2026-05-23_22-16-25
python scripts/inference.py --character thrall --text "For the Horde!" --reference data/processed/thrall/wavs/wg_thrall_hor21_001.wav
```

### Dataset versioning

Every transform run writes a `dataset_meta.json` alongside the dataset:

```json
{
  "character": "thrall",
  "whisper_model": "large-v2",
  "created_at": "2026-05-23T15:00:00",
  "sample_count": 47,
  "raw_files_total": 50,
  "max_duration_ms": 10000,
  "sample_rate": 22050
}
```

This file is uploaded to S3 with the dataset and logged by the training script, so every trained model is traceable to a specific dataset version.

---

## Cloud Setup (EC2)

### Instance

- **Type:** `g4dn.xlarge` (Tesla T4, 16GB VRAM, 4 vCPUs, 16GB RAM)
- **Pricing:** ~$0.25/hr spot in `eu-central-1` (max spot price set to $0.30)
- **Storage:** 100GB gp3 EBS
- **IAM:** Instance profile with scoped S3 read/write access — no credentials stored on instance

### AMI

**Custom AMI built on: Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)**

The custom AMI has all dependencies pre-installed. New instances are ready to train in under 2 minutes — no manual setup required.

Chosen over Amazon Linux because:

- Ubuntu 24.04 is the standard for deep learning workloads — broad ecosystem support
- Ships CUDA and cuDNN pre-validated for the T4 — no manual driver install
- "Base" variant gives a clean slate — no opinionated framework versions pre-installed
- `apt install ffmpeg` works on the first try (Amazon Linux has no FFmpeg in its default repos)

### Dependency stack (baked into AMI)

```bash
# System
ffmpeg 6.x             # Audio decoding for torchaudio and pydub
python3.11             # Coqui TTS does not support Python 3.12+

# Python (in ~/venv)
torch==2.3.0+cu121     # Pinned — 2.4+ switched torchaudio backend to torchcodec
torchaudio==2.3.0+cu121
TTS==0.21.3            # Coqui TTS
transformers==4.44.0   # Pinned — 5.x dropped BeamSearchScorer used by XTTS
awscli                 # S3 dataset download
```

### Known patches (applied in AMI and userdata.sh)

TTS 0.21.3 has an internal bug where `GPTTrainer` references `config.audio.dvae_sample_rate`
which no longer exists in `XttsAudioConfig`. Applied with:

```bash
sed -i 's/config.audio.dvae_sample_rate/config.audio.sample_rate/g' \
    ~/venv/lib/python3.11/site-packages/TTS/tts/layers/xtts/trainer/gpt_trainer.py
```

### Launching a training run

1. Launch `g4dn.xlarge` spot instance from the custom AMI
2. Paste `infra/userdata.sh` into the User Data field with your variables set
3. Instance boots, pulls code from GitHub, downloads dataset from S3, and starts training automatically
4. On completion, `best_model.pth` is uploaded to S3 and the instance can be terminated

### Manual training (SSH/EC2 Instance Connect)

```bash
tmux new -s train
source ~/venv/bin/activate
cd ~/tts
python -u scripts/train_xtts.py --character thrall 2>&1 | tee scripts/training.log
```

Detach: `Ctrl+B, D` — Reattach: `tmux attach -t train`

### Dataset download on EC2

```bash
aws s3 cp s3://<bucket>/characters/<character>/processed/ \
    ~/tts/data/processed/<character>/ --recursive
```

### Model download locally (for inference)

```bash
aws s3 cp s3://<bucket>/characters/<character>/models/ \
    models/<character>/ --recursive
```

---

## S3 Structure

```
s3://<bucket>/
└── characters/
    └── <character>/
        ├── processed/
        │   ├── wavs/               # All .wav training clips
        │   ├── metadata.csv        # LJSpeech format
        │   ├── dataset_meta.json   # Dataset version info
        │   ├── processed_files.json
        │   └── *.log
        └── models/
            └── <run_id>/
                ├── best_model.pth  # Fine-tuned checkpoint
                └── config.json     # Training config
```

---

## LJSpeech Format

`metadata.csv` follows the standard LJSpeech format:

```
clip_id|transcription|normalized_transcription
```

Example:

```
wg_thrall_hor18_001|Lead the way, Dark Lady. We will follow.|Lead the way, Dark Lady. We will follow.
```

---

## Future Work

- [ ] Makefile target to launch EC2 spot instance automatically
- [ ] Per-character config files in `config/` (epochs, batch size, whisper model)
- [ ] Silence-aware splitting with Whisper word timestamps for more precise cuts
- [ ] Twitch bot integration — pipe chat messages through inference.py
- [ ] Multi-character support in a single training run
- [ ] Automatic instance termination after training completes
