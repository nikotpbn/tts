"""
upload.py — Upload processed TTS dataset to S3.

Usage:
    python scripts/upload.py --character thrall

Notes:
    - Run this script LOCALLY after transform.py has completed.
    - Uploads data/processed/<character>/ to:
      s3://<S3_BUCKET>/characters/<character>/processed/
    - Existing files in S3 are always overwritten (latest local version wins).
    - Requires a .env file in the project root with:
        AWS_ACCESS_KEY_ID=...
        AWS_SECRET_ACCESS_KEY=...
        S3_BUCKET=...
        AWS_DEFAULT_REGION=...  (optional)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Upload processed TTS dataset to S3.")
    parser.add_argument("--character", type=str, required=True, help="Character name")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("upload")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

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

    # Load .env from project root
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    bucket = os.getenv("S3_BUCKET")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not all([bucket, aws_access_key, aws_secret_key]):
        logger.error("Missing required environment variables. Check your .env file.")
        logger.error("Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET")
        sys.exit(1)

    local_dir = project_root / "data" / "processed" / character
    if not local_dir.exists():
        logger.error(f"Processed directory does not exist: {local_dir}")
        logger.error("Run transform.py first.")
        sys.exit(1)

    s3_prefix = f"characters/{character}/processed"

    logger.info(f"Character:  {character}")
    logger.info(f"Source:     {local_dir}")
    logger.info(f"Bucket:     {bucket}")
    logger.info(f"S3 prefix:  s3://{bucket}/{s3_prefix}/")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )

        # Collect all files to upload
        files = list(local_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        logger.info(f"Files to upload: {len(files)}")

        uploaded = 0
        failed = 0

        for file_path in files:
            # Preserve directory structure relative to processed/<character>/
            relative = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative}"

            try:
                s3.upload_file(
                    Filename=str(file_path),
                    Bucket=bucket,
                    Key=s3_key,
                )
                logger.info(f"  Uploaded: {relative} -> s3://{bucket}/{s3_key}")
                uploaded += 1

            except (BotoCoreError, ClientError) as e:
                logger.error(f"  Failed to upload {relative}: {e}")
                failed += 1

        logger.info(f"Upload complete. Uploaded: {uploaded} | Failed: {failed}")

        if failed > 0:
            sys.exit(1)

    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 connection failed: {e}")
        sys.exit(1)


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
