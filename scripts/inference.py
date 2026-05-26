"""
inference.py — Generate speech from a fine-tuned XTTS model.

Usage:
    python scripts/inference.py --character thrall --text "For the Horde!"
    python scripts/inference.py --character thrall --text "For the Horde!" --run-id 2026-05-23_22-16-25
    python scripts/inference.py --character thrall --text "For the Horde!" --reference data/processed/thrall/wavs/wg_thrall_hor21_001.wav

Notes:
    - Run this script LOCALLY.
    - Requires the fine-tuned model to be present in models/<character>/<run_id>/
    - Download from S3 first if needed:
        aws s3 cp s3://<bucket>/characters/<character>/models/<run_id>/ models/<character>/<run_id>/ --recursive
    - Output is saved to data/inference/<character>/<timestamp>.wav
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using a fine-tuned XTTS model."
    )
    parser.add_argument("--character", type=str, required=True, help="Character name")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--run-id", type=str, default=None, help="Training run ID (default: latest)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio clip (default: auto-selected)",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language code (default: en)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_latest_run(models_dir: Path) -> str:
    """Return the most recent run folder by name."""
    runs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No training runs found in {models_dir}")
    return runs[-1].name


def get_default_reference(wavs_dir: Path) -> str:
    """Pick a default reference clip — longest wav available for best conditioning."""
    import wave as wav_module

    best = None
    best_duration = 0

    for wav_file in wavs_dir.glob("*.wav"):
        try:
            with wav_module.open(str(wav_file), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                if duration > best_duration:
                    best_duration = duration
                    best = wav_file
        except Exception:
            continue

    if best is None:
        raise FileNotFoundError(f"No wav files found in {wavs_dir}")

    return str(best)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Resolve model path
    models_dir = project_root / "models" / args.character
    if not models_dir.exists():
        print(
            f"[ERROR] No models found for character '{args.character}' at {models_dir}"
        )
        print("Download from S3 first:")
        print(
            f"  aws s3 cp s3://<bucket>/characters/{args.character}/models/ models/{args.character}/ --recursive"
        )
        sys.exit(1)

    run_id = args.run_id or get_latest_run(models_dir)
    run_path = models_dir / run_id

    checkpoint_path = run_path / "best_model.pth"
    config_path = run_path / "config.json"

    if not checkpoint_path.exists():
        print(f"[ERROR] best_model.pth not found at {checkpoint_path}")
        sys.exit(1)

    if not config_path.exists():
        print(f"[ERROR] config.json not found at {config_path}")
        sys.exit(1)

    # Resolve reference audio
    wavs_dir = project_root / "data" / "processed" / args.character / "wavs"
    if args.reference:
        reference_path = args.reference
        if not os.path.exists(reference_path):
            print(f"[ERROR] Reference audio not found: {reference_path}")
            sys.exit(1)
    else:
        if not wavs_dir.exists():
            print(f"[ERROR] Wavs directory not found: {wavs_dir}")
            print("Cannot auto-select reference audio.")
            sys.exit(1)
        reference_path = get_default_reference(wavs_dir)
        print(f"> Auto-selected reference: {os.path.basename(reference_path)}")

    # Output path
    output_dir = project_root / "data" / "inference" / args.character
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}.wav"

    print(f"> Character:  {args.character}")
    print(f"> Run ID:     {run_id}")
    print(f"> Model:      {checkpoint_path}")
    print(f"> Reference:  {reference_path}")
    print(f'> Text:       "{args.text}"')
    print(f"> Output:     {output_path}")

    # Load model
    print("> Loading model...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(config_path))

    model = Xtts.init_from_config(config)

    model.load_checkpoint(
        config,
        checkpoint_dir=str(run_path),
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(project_root / "models" / "base" / "vocab.json"),
        eval=True,
    )
    model.cuda() if __import__("torch").cuda.is_available() else model.cpu()
    print("> Model loaded.")

    # Generate speech
    print("> Synthesizing...")
    outputs = model.synthesize(
        text=args.text,
        config=config,
        speaker_wav=reference_path,
        language=args.language,
    )

    # Save output
    import soundfile as sf

    sf.write(str(output_path), outputs["wav"], config.audio.output_sample_rate)

    print(f"> Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
