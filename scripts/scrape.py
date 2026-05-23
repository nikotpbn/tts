"""
scrape.py — Audio scraper for TTS dataset collection.

Usage:
    python scrape.py --character thrall --max 100

Notes:
    - Run this script LOCALLY. It launches a visible browser window (headless=False)
      and is not intended for EC2 or headless environments.
    - AWS EC2 is reserved exclusively for model training.
    - Downloads are saved to: data/raw/<character_name>/
    - A download_map.json ledger is maintained per character to skip already-downloaded files.
    - Logs are written to: data/raw/<character_name>/scrape.log
"""

import os
import time
import random
import json
import logging
import argparse
from playwright.sync_api import sync_playwright

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape audio files for TTS dataset.")
    parser.add_argument(
        "--character",
        type=str,
        default="thrall",
        help="Character name (used for folder and search query)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=50,
        help="Maximum number of files to download this run",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — persists across terminal sessions
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Download map
# ---------------------------------------------------------------------------


def load_download_map(map_file: str) -> dict:
    if os.path.exists(map_file):
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_download_map(map_file: str, download_map: dict):
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(download_map, f, indent=4)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


def scrape_audio(character: str, max_downloads: int, logger: logging.Logger):
    output_dir = os.path.join(PROJECT_ROOT, "data", "raw", character)
    os.makedirs(output_dir, exist_ok=True)

    map_file = os.path.join(output_dir, "download_map.json")
    download_map = load_download_map(map_file)

    base_url = f"https://wago.tools/files?search=sound%2Fcreature%2F{character}"

    logger.info(f"Starting scrape for character: {character}")
    logger.info(f"Max downloads this run: {max_downloads}")
    logger.info(f"Output directory: {output_dir}")

    with sync_playwright() as p:
        logger.info("Launching browser...")
        browser = p.chromium.launch(
            headless=False
        )  # Intentionally headless=False — run locally only
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        success_count = 0
        current_page = 1

        while success_count < max_downloads:
            page_url = f"{base_url}&page={current_page}"
            logger.info(f"Navigating to page {current_page}: {page_url}")
            page.goto(page_url)

            try:
                page.wait_for_selector("table tbody tr", timeout=15000)
                time.sleep(3)
            except Exception:
                logger.warning(
                    f"No table found on page {current_page}. End of results."
                )
                break

            rows = page.locator("table tbody tr").all()
            logger.info(f"Found {len(rows)} rows on page {current_page}.")

            if not rows:
                logger.info("No rows found. Stopping pagination.")
                break

            for index, row in enumerate(rows):
                if success_count >= max_downloads:
                    logger.info(f"Reached download cap of {max_downloads}. Stopping.")
                    break

                try:
                    download_btn = row.locator(
                        "a[href*='/api/'][href*='download']"
                    ).first

                    if not download_btn.is_visible():
                        continue

                    href = download_btn.get_attribute("href")
                    if not href:
                        continue

                    # Skip if already downloaded and file exists on disk
                    if href in download_map:
                        existing_file = os.path.join(output_dir, download_map[href])
                        if os.path.exists(existing_file):
                            logger.debug(
                                f"Skipping already downloaded: {download_map[href]}"
                            )
                            continue

                    # Retry loop
                    downloaded = False
                    for attempt in range(1, 4):
                        try:
                            with page.expect_download(timeout=10000) as download_info:
                                download_btn.click()

                            download = download_info.value
                            file_name = download.suggested_filename

                            if file_name.endswith((".ogg", ".mp3", ".wav")):
                                file_path = os.path.join(output_dir, file_name)
                                download.save_as(file_path)

                                download_map[href] = file_name
                                save_download_map(map_file, download_map)

                                success_count += 1
                                logger.info(
                                    f"[{success_count}/{max_downloads}] Downloaded: {file_name}"
                                )
                                downloaded = True
                            break

                        except Exception as e:
                            logger.warning(
                                f"Attempt {attempt}/3 failed for row {index + 1}: {e}"
                            )
                            if attempt < 3:
                                time.sleep(2**attempt)  # Exponential backoff: 2s, 4s

                    if not downloaded:
                        logger.error(
                            f"Failed to download row {index + 1} after 3 attempts. Skipping."
                        )

                    if downloaded and success_count < max_downloads:
                        delay = random.randint(20, 60)
                        logger.info(f"Rate limiting: waiting {delay}s...")
                        time.sleep(delay)

                except Exception as e:
                    logger.warning(f"Unexpected error on row {index + 1}: {e}")

            if success_count < max_downloads:
                current_page += 1

        logger.info(
            f"Scrape complete. Downloaded {success_count} files to '{output_dir}'."
        )
        browser.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "data", "raw", args.character)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "scrape.log")

    logger = setup_logger(log_path)
    scrape_audio(character=args.character, max_downloads=args.max, logger=logger)
