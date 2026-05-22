import os
import whisper
from pydub import AudioSegment

# 1. Dynamically anchor paths to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 2. Scaffolding-aligned paths
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "thrall")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "thrall")
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
SAMPLE_RATE = 22050

# Ensure output directories exist
os.makedirs(WAV_DIR, exist_ok=True)

def process_audio():
    print("Loading Whisper 'base' model... (Will download on first initialization)")
    model = whisper.load_model("base") 
    
    metadata_lines = []
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Target input folder does not exist: {INPUT_DIR}")
        print("Please move your scraped raw files to 'data/raw/thrall/' before running.")
        return
        
    # Gather only audio targets
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".ogg", ".mp3", ".wav"))]
    total_files = len(files)
    
    print(f"Found {total_files} files in raw workspace to transform.")
    
    for index, filename in enumerate(files):
        file_base = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = f"{file_base}.wav"
        output_path = os.path.join(WAV_DIR, output_filename)
        
        print(f"[{index + 1}/{total_files}] Processing: {filename}")
        
        # Step A: Format Conversion via Pydub -> Mono, 22.05kHz, 16-bit PCM
        try:
            audio = AudioSegment.from_file(input_path)
            
            # We want our dataset to only contain files with more than 1.5s of audio.
            if len(audio) < 1500:  # 1500ms = 1.5 seconds
                print(f"  -> Skipping {filename}: Audio too short ({len(audio)}ms)")
                # Optionally, append to a blacklist.txt file here if you want a permanent record
                continue
            
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            audio.export(output_path, format="wav")
        except Exception as e:
            print(f"  -> Audio framing failure on {filename}: {e}")
            continue

        # Step B: Auto Transcription via Local Whisper Model
        try:
            result = model.transcribe(output_path)
            text = result["text"].strip()
            
            # Formulate standard LJSpeech pattern (id|raw_text|normalized_text)
            metadata_line = f"{file_base}|{text}|{text}"
            metadata_lines.append(metadata_line)
            
            print(f"  -> Text: '{text}'")
        except Exception as e:
            print(f"  -> Text translation failure on {filename}: {e}")
            
    # Step C: Persist metadata index
    print(f"\nWriting target dataset translation matrix to {CSV_PATH}")
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
        
    print("Dataset construction sequence successfully executed!")

if __name__ == "__main__":
    process_audio()