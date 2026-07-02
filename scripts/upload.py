"""
upload.py — Sync processed TTS dataset to S3.

Usage:
    python scripts/upload.py --character thrall

Notes:
    - Run this script LOCALLY after transform.py has completed.
    - Syncs data/processed/<character>/ to:
      s3://<S3_BUCKET>/characters/<character>/processed/
    - Only uploads new or changed files (aws s3 sync).
    - Requires .env in project root with AWS credentials.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Sync processed TTS dataset to S3.")
    parser.add_argument("--character", type=str, required=True, help="Character name")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("upload")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(character: str, logger: logging.Logger):
    project_root = Path(__file__).resolve().parent.parent

    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")

    if not bucket:
        logger.error("S3_BUCKET not set. Check your .env file.")
        sys.exit(1)

    local_dir = project_root / "data" / "processed" / character
    if not local_dir.exists():
        logger.error(f"Processed directory does not exist: {local_dir}")
        logger.error("Run transform.py first.")
        sys.exit(1)

    s3_uri = f"s3://{bucket}/characters/{character}/processed/"

    logger.info(f"Character:  {character}")
    logger.info(f"Source:     {local_dir}/")
    logger.info(f"Dest:       {s3_uri}")
    logger.info(f"Region:     {region}")
    logger.info("Syncing (only new or changed files will be uploaded)...")

    result = subprocess.run(
        [
            "aws", "s3", "sync",
            str(local_dir) + "/",
            s3_uri,
            "--region", region,
        ],
        capture_output=False,  # stream output directly to terminal
    )

    if result.returncode != 0:
        logger.error("Sync failed.")
        sys.exit(1)

    logger.info("Sync complete.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    log_path = project_root / "data" / "processed" / args.character / "upload.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(log_path))
    upload(character=args.character, logger=logger)