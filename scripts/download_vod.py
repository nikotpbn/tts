"""
download_vod.py — Download VOD audio from YouTube or Twitch for manual clipping.

Usage:
    python scripts/download_vod.py --character xqc --url https://www.youtube.com/watch?v=...
    python scripts/download_vod.py --character xqc --url https://www.twitch.tv/videos/...

Notes:
    - Run this script LOCALLY.
    - Audio is downloaded to data/raw/<character>/vods/
    - After downloading, manually clip clean speech segments in Audacity
      and save them to data/raw/<character>/
    - Then run: make dataset CHARACTER=<character>
    - Requires yt-dlp: pip install yt-dlp
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download VOD audio for manual clipping."
    )
    parser.add_argument(
        "--character", type=str, required=True, help="Character/streamer name"
    )
    parser.add_argument(
        "--url", type=str, required=True, help="YouTube or Twitch VOD URL"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="wav",
        choices=["wav", "mp3"],
        help="Output audio format (default: wav)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start timestamp for partial download e.g. 00:10:00",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End timestamp for partial download e.g. 01:00:00",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("download_vod")
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
# Download
# ---------------------------------------------------------------------------


def download_vod(
    character: str, url: str, fmt: str, start: str, end: str, logger: logging.Logger
):
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed. Run: pip install yt-dlp")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "raw" / character / "vods"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_template = str(output_dir / f"%(id)s_{timestamp}.%(ext)s")

    logger.info(f"Character:  {character}")
    logger.info(f"URL:        {url}")
    logger.info(f"Format:     {fmt}")
    logger.info(f"Output dir: {output_dir}")

    if start or end:
        logger.info(f"Clipping:   {start or 'start'} -> {end or 'end'}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": fmt,
                "preferredquality": "0" if fmt == "wav" else "192",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    # Partial download via timestamp range
    if start or end:
        download_sections = ""
        if start and end:
            download_sections = f"*{start}-{end}"
        elif start:
            download_sections = f"*{start}-inf"
        elif end:
            download_sections = f"*0-{end}"

        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(download_sections,)]
        )
        ydl_opts["force_keyframes_at_cuts"] = True

    logger.info("Starting download...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown")
            duration = info.get("duration", 0)
            logger.info(
                f'Downloaded: "{title}" ({duration//60:.0f}m {duration%60:.0f}s)'
            )
            logger.info(f"Saved to:   {output_dir}")
            logger.info("Next step: open in Audacity, clip clean speech segments,")
            logger.info(f"           save clips to data/raw/{character}/")
            logger.info(f"           then run: make dataset CHARACTER={character}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "data" / "raw" / args.character / "vods"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / "download.log")

    logger = setup_logger(log_path)
    download_vod(
        character=args.character,
        url=args.url,
        fmt=args.format,
        start=args.start,
        end=args.end,
        logger=logger,
    )
