"""
transform.py — Audio transformation pipeline for TTS dataset construction.

Converts raw scraped audio files (.ogg, .mp3, .wav) into LJSpeech-format
training data: mono 22050Hz 16-bit WAV files + metadata.csv

Usage:
    python transform.py --character thrall
    python transform.py --character thrall --whisper-model large-v2

Notes:
    - Run this script LOCALLY. Whisper runs on CPU here; EC2 is reserved for training.
    - Input:  data/raw/<character>/
    - Output: data/processed/<character>/wavs/ + metadata.csv + dataset_meta.json
    - Idempotent: already-processed files are skipped on re-runs.
    - Audio is split on silence boundaries to keep clips within 1-10s (LJSpeech standard).
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

import ssl
import certifi

import whisper
from pydub import AudioSegment
from pydub.silence import detect_silence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)
SAMPLE_RATE = 22050
MIN_DURATION_MS = 1500  # 1.5s — discard clips shorter than this
MAX_DURATION_MS = 10000  # 10s  — LJSpeech standard upper bound
SILENCE_THRESH_DB = -40  # dBFS threshold for silence detection
MIN_SILENCE_MS = 300  # minimum silence length to consider as a split point

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform raw audio into LJSpeech TTS dataset."
    )
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        help="Character name (matches data/raw/<character>/)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v2)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("transform")
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
# Audio splitting
# ---------------------------------------------------------------------------


def split_on_silence_boundaries(
    audio: AudioSegment, base_name: str, logger: logging.Logger
) -> list[tuple[str, AudioSegment]]:
    """
    Split an AudioSegment into chunks of MAX_DURATION_MS or less,
    cutting at silence boundaries where possible.

    Returns a list of (chunk_id, AudioSegment) tuples.
    """
    duration_ms = len(audio)

    # No split needed
    if duration_ms <= MAX_DURATION_MS:
        return [(f"{base_name}_001", audio)]

    logger.debug(
        f"  Splitting {base_name} ({duration_ms/1000:.1f}s) on silence boundaries..."
    )

    # Detect silence regions: list of [start_ms, end_ms]
    silence_regions = detect_silence(
        audio, min_silence_len=MIN_SILENCE_MS, silence_thresh=SILENCE_THRESH_DB
    )

    # Build a list of candidate split points (midpoint of each silence region)
    split_points = [int((start + end) / 2) for start, end in silence_regions]

    chunks = []
    chunk_index = 1
    cursor = 0

    while cursor < duration_ms:
        window_end = cursor + MAX_DURATION_MS

        if window_end >= duration_ms:
            # Last chunk — take whatever remains
            chunk = audio[cursor:]
            if len(chunk) >= MIN_DURATION_MS:
                chunks.append((f"{base_name}_{chunk_index:03d}", chunk))
            break

        # Find the best silence boundary within the window
        candidates = [p for p in split_points if cursor < p <= window_end]

        if candidates:
            # Cut at the last silence boundary within the window
            cut_point = candidates[-1]
            logger.debug(
                f"  Chunk {chunk_index}: {cursor/1000:.1f}s → {cut_point/1000:.1f}s (silence boundary)"
            )
        else:
            # No silence found — hard cut at window end (better than nothing)
            cut_point = window_end
            logger.warning(
                f"  Chunk {chunk_index}: no silence found, hard cut at {cut_point/1000:.1f}s"
            )

        chunk = audio[cursor:cut_point]
        if len(chunk) >= MIN_DURATION_MS:
            chunks.append((f"{base_name}_{chunk_index:03d}", chunk))

        cursor = cut_point
        chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_audio(character: str, whisper_model_name: str, logger: logging.Logger):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_dir = os.path.join(project_root, "data", "raw", character)
    output_dir = os.path.join(project_root, "data", "processed", character)
    wav_dir = os.path.join(output_dir, "wavs")
    csv_path = os.path.join(output_dir, "metadata.csv")
    meta_path = os.path.join(output_dir, "dataset_meta.json")
    processed_log_path = os.path.join(output_dir, "processed_files.json")

    os.makedirs(wav_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.error("Run scrape.py first to populate raw data.")
        sys.exit(1)

    # Load idempotency ledger — tracks which source files have already been processed
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r", encoding="utf-8") as f:
            processed_files = json.load(f)
    else:
        processed_files = {}

    # Load existing metadata lines so we can append without losing prior entries
    existing_metadata = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) == 3:
                        existing_metadata[parts[0]] = line

    logger.info(f"Character:     {character}")
    logger.info(f"Whisper model: {whisper_model_name}")
    logger.info(f"Input dir:     {input_dir}")
    logger.info(f"Output dir:    {output_dir}")

    # Gather raw audio files
    all_files = [
        f for f in os.listdir(input_dir) if f.endswith((".ogg", ".mp3", ".wav"))
    ]
    new_files = [f for f in all_files if f not in processed_files]

    logger.info(
        f"Total raw files: {len(all_files)} | Already processed: {len(processed_files)} | New: {len(new_files)}"
    )

    if not new_files:
        logger.info("Nothing new to process. Exiting.")
        return

    logger.info(f"Loading Whisper model '{whisper_model_name}'...")
    model = whisper.load_model(whisper_model_name)
    logger.info("Whisper model loaded.")

    new_metadata = dict(existing_metadata)
    skipped = 0
    processed = 0
    failed = 0

    for index, filename in enumerate(new_files):
        file_base = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)

        logger.info(f"[{index + 1}/{len(new_files)}] Processing: {filename}")

        # Step A: Load and convert to mono 22050Hz 16-bit PCM
        try:
            audio = AudioSegment.from_file(input_path)
            audio = (
                audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            )
        except Exception as e:
            logger.error(f"  Failed to load/convert {filename}: {e}")
            failed += 1
            continue

        # Step B: Discard if entire clip is too short
        if len(audio) < MIN_DURATION_MS:
            logger.info(f"  Skipping {filename}: too short ({len(audio)}ms)")
            processed_files[filename] = {
                "status": "skipped_short",
                "duration_ms": len(audio),
            }
            skipped += 1
            continue

        # Step C: Split into LJSpeech-compliant chunks on silence boundaries
        chunks = split_on_silence_boundaries(audio, file_base, logger)
        logger.debug(f"  Produced {len(chunks)} chunk(s)")

        chunk_results = []

        for chunk_id, chunk_audio in chunks:
            wav_filename = f"{chunk_id}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            # Export chunk
            try:
                chunk_audio.export(wav_path, format="wav")
            except Exception as e:
                logger.error(f"  Failed to export chunk {chunk_id}: {e}")
                continue

            # Step D: Transcribe with Whisper
            try:
                result = model.transcribe(wav_path)
                text = result["text"].strip()

                if not text:
                    logger.warning(
                        f"  Empty transcription for {chunk_id}, skipping chunk."
                    )
                    os.remove(wav_path)
                    continue

                # Filter garbage transcriptions
                words = text.split()
                if len(words) < 3 or len(text) < 10:
                    logger.warning(
                        f'  Discarding {chunk_id}: low quality transcription: "{text}"'
                    )
                    os.remove(wav_path)
                    continue

                metadata_line = f"{chunk_id}|{text}|{text}"
                new_metadata[chunk_id] = metadata_line
                chunk_results.append(chunk_id)

                logger.info(f'  [{chunk_id}] "{text}"')

            except Exception as e:
                logger.error(f"  Transcription failed for {chunk_id}: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                continue

        # Mark source file as processed
        processed_files[filename] = {
            "status": "processed",
            "chunks": chunk_results,
            "duration_ms": len(audio),
        }

        # Persist idempotency ledger after each file
        with open(processed_log_path, "w", encoding="utf-8") as f:
            json.dump(processed_files, f, indent=4)

        processed += 1

    # Step E: Write metadata.csv
    logger.info(f"Writing metadata.csv ({len(new_metadata)} entries)...")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_metadata.values()))

    # Step F: Write dataset_meta.json
    dataset_meta = {
        "character": character,
        "whisper_model": whisper_model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(new_metadata),
        "raw_files_total": len(all_files),
        "raw_files_processed_this_run": processed,
        "raw_files_skipped_this_run": skipped,
        "raw_files_failed_this_run": failed,
        "min_duration_ms": MIN_DURATION_MS,
        "max_duration_ms": MAX_DURATION_MS,
        "sample_rate": SAMPLE_RATE,
        "silence_thresh_db": SILENCE_THRESH_DB,
        "min_silence_ms": MIN_SILENCE_MS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=4)

    logger.info(f"dataset_meta.json written.")
    logger.info(
        f"Done. Processed: {processed} | Skipped: {skipped} | Failed: {failed} | Total samples: {len(new_metadata)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data", "processed", args.character)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "transform.log")

    logger = setup_logger(log_path)
    process_audio(
        character=args.character, whisper_model_name=args.whisper_model, logger=logger
    )
