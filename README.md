# TTS Pipeline

A pipeline for scraping, processing, and fine-tuning XTTS voice models on character audio datasets. Built for solo or small-team use with a clean separation between local processing and cloud training.

---

## Project Structure

```
tts/
├── .env                        # AWS credentials (never commit)
├── .env.example                # Environment variable template
├── .gitignore
├── Makefile                    # Pipeline orchestration
├── README.md
├── requirements.txt            # Dependencies (local + inference)
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
│   └── userdata.sh             # EC2 bootstrap script (templated by launch_training.py)
└── scripts/
    ├── scrape.py               # Audio scraper (local only)
    ├── download_vod.py         # VOD audio downloader via yt-dlp (local only)
    ├── transform.py            # Audio transformation + transcription (local only)
    ├── upload.py               # S3 dataset sync (local only)
    ├── launch_training.py      # EC2 spot instance launcher (local only)
    ├── train_xtts.py           # XTTS fine-tuning (EC2 only)
    └── inference.py            # Speech synthesis from fine-tuned model (local)
```

---

## Architecture Overview

```
LOCAL                               AWS
─────                               ───
scrape.py / download_vod.py         EC2 g4dn.xlarge (spot)
    ↓                                   ↑
transform.py                        launch_training.py
    ↓                                   ↓
upload.py ─────────────────────→    userdata.sh (bootstrap)
                                        ↓
                                    train_xtts.py
                                        ↓
                                    S3 (model upload)
                                        ↓
inference.py ←─────────────────── best_model.pth
```

- **Local machine** handles all data work: scraping, audio conversion, Whisper transcription, S3 sync, EC2 launch, and inference.
- **EC2** is reserved exclusively for GPU training — fully automated via `userdata.sh`.
- **S3** is the handoff point between local and cloud — dataset in, model out.
- **SNS** sends email notification on training completion or failure.
- **CloudWatch** streams training logs in real time.

---

## Local Setup

### Requirements

- Python 3.11 — Coqui TTS does not support 3.12+
- FFmpeg (system level)
- AWS CLI

```bash
# macOS
brew install python@3.11 ffmpeg awscli

# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv ffmpeg awscli
```

### Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# AWS Credentials (IAM user: tts-pipeline-local)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=eu-central-1

# S3
S3_BUCKET=your_bucket_name

# EC2 Training
AMI_ID=your_custom_ami_id
INSTANCE_TYPE=g4dn.xlarge
SUBNET_ID=your_subnet_id
SECURITY_GROUP_ID=your_security_group_id
INSTANCE_PROFILE_ARN=arn:aws:iam::your_account_id:instance-profile/tts-ec2-training-role
SPOT_MAX_PRICE=0.30

# Notifications
SNS_TOPIC_ARN=arn:aws:sns:your-region:your_account_id:tts-training-notifications

# Monitoring
CLOUDWATCH_LOG_GROUP=/tts/training

# GitHub
GITHUB_REPO=your_github_repo
```

### IAM setup

Two IAM identities are required:

**Local IAM user (`tts-pipeline-local`)** — used by local scripts to upload to S3 and launch EC2:

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

Also needs `ec2:RunInstances` and `iam:PassRole` to launch training instances.

**EC2 IAM role (`tts-ec2-training-role`)** — attached to training instances, no credentials stored:

- S3 read/write on dataset bucket
- `sns:Publish` on the training notifications topic
- `logs:CreateLogStream`, `logs:PutLogEvents` on the CloudWatch log group

Policy templates live in `config/` for reference.

---

## Pipeline Usage

All steps are orchestrated via `make`. Run from the project root.

### Full dataset pipeline (scrape + transform + upload)

```bash
make dataset CHARACTER=thrall
```

### Launch automated training

```bash
make train CHARACTER=thrall
```

This syncs the dataset to S3 then launches a spot EC2 instance that trains automatically, uploads the model to S3, notifies you by email, and self-terminates.

### Individual steps

```bash
make scrape CHARACTER=thrall MAX=100
make transform CHARACTER=thrall WHISPER=large-v2
make upload CHARACTER=thrall
make download CHARACTER=xqc URL="https://www.youtube.com/watch?v=..."
```

### Clean processed data (raw files are preserved)

```bash
make clean CHARACTER=thrall
```

### Available options

| Option       | Default    | Description                          |
| ------------ | ---------- | ------------------------------------ |
| `CHARACTER`  | `thrall`   | Character name                       |
| `MAX`        | `50`       | Max files to download per scrape run |
| `WHISPER`    | `large-v2` | Whisper model size for transcription |
| `EPOCHS`     | `100`      | Training epochs                      |
| `BATCH_SIZE` | `2`        | Training batch size                  |
| `URL`        |            | VOD URL for `make download`          |

---

## Pipeline Scripts

### `scrape.py`

Scrapes audio files from [wago.tools](https://wago.tools) using Playwright.

- Maintains a `download_map.json` ledger per character to skip already-downloaded files (idempotent)
- Randomized delay between downloads (20–60s) to respect rate limits
- 3-attempt retry with exponential backoff per file
- Logs to `data/raw/<character>/scrape.log`
- **Run locally only** — requires a visible browser window (`headless=False`)

### `download_vod.py`

Downloads VOD audio from YouTube or Twitch for manual clipping.

- Uses `yt-dlp` — supports YouTube, Twitch, and most streaming platforms
- Downloads to `data/raw/<character>/vods/`
- Supports partial downloads via `--start` and `--end` timestamps
- After downloading, manually clip clean speech segments in Audacity and save to `data/raw/<character>/`

```bash
python scripts/download_vod.py --character xqc --url "https://www.youtube.com/watch?v=..."
python scripts/download_vod.py --character xqc --url "..." --start 00:10:00 --end 01:00:00
```

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

Syncs the processed dataset to S3.

- Uses `aws s3 sync` — only uploads new or changed files, skips unchanged ones
- Syncs `data/processed/<character>/` to `s3://<bucket>/characters/<character>/processed/`
- Logs to `data/processed/<character>/upload.log`

### `launch_training.py`

Launches a spot EC2 instance for automated training.

- Loads `infra/userdata.sh`, substitutes `{{PLACEHOLDER}}` variables, and passes it to EC2
- Configures spot instance with AMI, instance type, IAM role, security group from `.env`
- Tags the instance with character name and run ID for easy identification
- After launching, prints the CloudWatch log tail command and S3 model path

```bash
python scripts/launch_training.py --character thrall --epochs 100 --batch-size 2
```

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
- Auto-selects the longest reference audio clip from the processed dataset for best conditioning
- Output saved to `data/inference/<character>/<timestamp>.wav`
- Requires `models/base/vocab.json` — download once with:
  ```bash
  mkdir -p models/base
  curl -L "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json" -o models/base/vocab.json
  ```

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
- **Storage:** 100GB gp3 EBS (deleted on termination)
- **IAM:** Instance profile `tts-ec2-training-role` — no credentials stored on instance

### AMI

**Custom AMI built on: Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)**

The custom AMI has all dependencies pre-installed. New instances are ready to train in under 2 minutes — no manual setup required.

Chosen over Amazon Linux because:

- Ubuntu 24.04 is the standard for deep learning workloads — broad ecosystem support
- Ships CUDA and cuDNN pre-validated for the T4 — no manual driver install
- "Base" variant gives a clean slate — no opinionated framework versions pre-installed
- `apt install ffmpeg` works on the first try (Amazon Linux has no FFmpeg in its default repos)

### Dependency stack (baked into AMI)

```
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

### Automated training flow

```bash
make train CHARACTER=thrall
```

1. `upload.py` syncs dataset to S3
2. `launch_training.py` reads `infra/userdata.sh`, substitutes variables, launches spot instance
3. Instance boots from custom AMI — no setup time
4. `userdata.sh` pulls latest code from GitHub
5. Downloads dataset from S3
6. Runs `train_xtts.py` — streams logs to CloudWatch
7. Uploads `best_model.pth` to S3
8. Sends SNS email notification (success or failure)
9. Instance self-terminates

### Monitoring training

```bash
# Stream all training logs in real time
aws logs tail /tts/training --follow

# Stream logs for a specific character
aws logs tail /tts/training --log-stream-name-prefix thrall/ --follow

# Stream logs for a specific run
aws logs tail /tts/training --log-stream-name thrall/2026-05-23_22-16-25 --follow
```

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
aws s3 sync s3://<bucket>/characters/<character>/processed/ \
    ~/tts/data/processed/<character>/
```

### Model download locally (for inference)

```bash
aws s3 sync s3://<bucket>/characters/<character>/models/ \
    models/<character>/
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

- [ ] Per-character config files in `config/` (epochs, batch size, whisper model, reference clip)
- [ ] Audio denoising in transform pipeline (`noisereduce` or `deepfilternet`)
- [ ] Audio loudness normalization before export
- [ ] Silence-aware splitting with Whisper word timestamps for more precise cuts
- [ ] Twitch bot integration — pipe chat messages through `inference.py`
- [ ] Multi-character support in a single training run
