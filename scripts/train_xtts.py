"""
train_xtts.py — XTTS fine-tuning script.

Usage:
    python scripts/train_xtts.py --character thrall
    python scripts/train_xtts.py --character thrall --epochs 50 --batch-size 4

Notes:
    - Run this script on EC2 (g4dn.xlarge or similar GPU instance).
    - Dataset must be available at data/processed/<character>/ before running.
    - Base model weights are downloaded once to models/base/ and reused across runs.
    - Checkpoints are saved to models/<character>/<run_id>/
    - best_model.pth and config.json are uploaded to S3 at end of training.
"""

import os
import wave
import argparse
import boto3
from datetime import datetime, timezone

import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune XTTS on a character dataset.")
    parser.add_argument("--character", type=str, required=True, help="Character name (matches data/processed/<character>/)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: current timestamp)")
    parser.add_argument("--s3-bucket", type=str, default=None, help="S3 bucket for uploading trained model (optional)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_paths(character: str, run_id: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, "data", "processed", character)
    meta_file = os.path.join(data_path, "metadata.csv")
    base_weights_path = os.path.join(project_root, "models", "base")
    run_output_path = os.path.join(project_root, "models", character, run_id)

    os.makedirs(base_weights_path, exist_ok=True)
    os.makedirs(run_output_path, exist_ok=True)

    return data_path, meta_file, base_weights_path, run_output_path

# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def validate_wavs(samples: list, max_duration: float = 11.5) -> list:
    bad = []
    for s in samples:
        path = s["audio_file"]
        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / sr
                if sr != 22050:
                    bad.append((path, f"bad SR: {sr}"))
                elif frames == 0:
                    bad.append((path, "zero frames"))
                elif duration > max_duration:
                    bad.append((path, f"too long: {duration:.1f}s"))
        except Exception as e:
            bad.append((path, str(e)))
    return bad

# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def upload_model_to_s3(run_output_path: str, character: str, run_id: str, bucket: str):
    print(f"> Uploading trained model to S3...")
    s3 = boto3.client("s3")

    files_to_upload = ["best_model.pth", "config.json"]

    for filename in files_to_upload:
        local_path = os.path.join(run_output_path, filename)
        if not os.path.exists(local_path):
            print(f"  [WARNING] {filename} not found, skipping.")
            continue

        s3_key = f"characters/{character}/models/{run_id}/{filename}"
        try:
            s3.upload_file(local_path, bucket, s3_key)
            print(f"  Uploaded: {filename} -> s3://{bucket}/{s3_key}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload {filename}: {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Generate run ID from timestamp if not provided
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    # 0. Force deterministic CUDA behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # 1. Resolve paths
    data_path, meta_file, base_weights_path, run_output_path = get_paths(args.character, run_id)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}\nRun transform.py and upload to S3 first.")

    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"metadata.csv not found: {meta_file}")

    print(f"> Character:    {args.character}")
    print(f"> Run ID:       {run_id}")
    print(f"> Data path:    {data_path}")
    print(f"> Base weights: {base_weights_path}")
    print(f"> Output:       {run_output_path}")

    # 2. Download base model weights (once — reused across all runs)
    dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_norm_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    tokenizer_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    xtts_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    if not os.path.exists(os.path.join(base_weights_path, "model.pth")):
        print("> Downloading base model weights...")
        ModelManager._download_model_files(
            [dvae_link, mel_norm_link, tokenizer_link, xtts_link],
            base_weights_path,
            progress_bar=True
        )
    else:
        print("> Base model weights already present, skipping download.")

    # 3. Dataset config
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=args.character,
        path=data_path,
        meta_file_train=meta_file,
        language="en",
    )

    # 4. Model architecture & audio config
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=os.path.join(base_weights_path, "mel_stats.pth"),
        dvae_checkpoint=os.path.join(base_weights_path, "dvae.pth"),
        xtts_checkpoint=os.path.join(base_weights_path, "model.pth"),
        tokenizer_file=os.path.join(base_weights_path, "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050, output_sample_rate=24000
    )

    # 5. Training config
    config = GPTTrainerConfig(
        output_path=run_output_path,
        model_args=model_args,
        run_name=f"{args.character}_xtts_finetune",
        project_name="XTTS_trainer",
        run_description=f"Fine tuning XTTS on the {args.character} dataset",
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_loader_workers=0,
        epochs=args.epochs,
        print_step=50,
        save_step=500,
        save_n_checkpoints=2,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000, 150000, 300000],
            "gamma": 0.5,
            "last_epoch": -1,
        },
    )

    # 6. Load samples
    train_samples, eval_samples = load_tts_samples(
        config_dataset, eval_split=True, eval_split_size=0.1
    )

    print(f"> Loaded {len(train_samples)} train samples, {len(eval_samples)} eval samples")

    # 7. Pre-flight wav validation
    print("> Validating wav files...")
    bad = validate_wavs(train_samples + eval_samples)

    if bad:
        print(f"[WARNING] {len(bad)} problematic files:")
        for path, reason in bad:
            print(f"  {path}: {reason}")
        raise SystemExit("Fix wav issues before training.")
    else:
        print("> All wav files OK")

    # 8. Initialize model and start training
    print("> Handing off to Trainer...")
    model = GPTTrainer.init_from_config(config)

    trainer = Trainer(
        TrainerArgs(),
        config,
        run_output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    # 9. Upload trained model to S3
    s3_bucket = args.s3_bucket or os.getenv("S3_BUCKET")
    if s3_bucket:
        upload_model_to_s3(run_output_path, args.character, run_id, s3_bucket)
    else:
        print("> [WARNING] No S3 bucket specified. Skipping model upload.")
        print(">           Pass --s3-bucket or set S3_BUCKET environment variable.")

if __name__ == "__main__":
    main()